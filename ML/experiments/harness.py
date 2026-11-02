"""
Experiment harness for the dorsiflexion R² investigation.

Reproduces ``ML.training.load_subject`` exactly when called with default
options, but keeps every trial's 100 Hz arrays around so preprocessing,
target-definition and model variants can be swapped via an ``opts`` dict.

Pipeline (per subject)
----------------------
1.  read_flb (cached as pickle)
2.  classify / rescue / std-filter / split   — identical to build_dataset
3.  envelope at 1000 Hz on the full trial     — opts: bp_low, bp_high, lp
4.  EMG normalisation                         — opts: emg_norm
5.  trim 1 s transient, decimate ×10 (FIR, zero-phase)
6.  target definition                         — opts: target, passive_*
7.  plugins (list of 'module:function')       — can edit per-trial arrays
8.  sliding windows, 80/20 temporal split on test session, retest = test

Every per-trial record is a dict with (T = samples at 100 Hz):
    idx, comment, session ('test'|'retest'), split ('trainval'|'test'),
    env      (T, 4)  normalised envelopes (model features)
    env_raw  (T, 4)  un-normalised envelopes
    pos      (T,)    ankle position (rad)
    tq       (T,)    measured torque (Nm)
    passive  (T,)    passive torque that was subtracted
    y        (T,)    model target
    y_ref    (T,)    reference target = tq − session passive map (baseline)
    extra    (T, k)  optional extra features appended after position
"""

import os
import sys
import json
import time
import pickle
import importlib
import warnings

import numpy as np
from scipy.signal import decimate

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ML.config import (SUBJECTS, DATA_DIR, TRIAL_OVERRIDES,
                       PASSIVE_TORQUE_OVERRIDES)
from ML.preprocessing.flb_reader import read_flb
from ML.preprocessing import dataset_builder as db
from ML.preprocessing.emg_envelope import extract_envelope, detect_emg_columns

EXP_DIR     = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR   = os.environ.get('EMG_CACHE_DIR', os.path.join(EXP_DIR, '_cache'))
RESULTS_DIR = os.environ.get('EMG_RESULTS_DIR', os.path.join(EXP_DIR, 'results'))

DEFAULT_OPTS = {
    # envelope
    'bp_low': 30.0, 'bp_high': 300.0, 'lp': 2.0,
    # EMG normalisation: 'train_max' (baseline)
    'emg_norm': 'train_max',
    # target: 'passive' (baseline, session-specific map) | 'raw'
    'target': 'passive',
    # passive map source: 'pipeline' (baseline: overrides or auto-detect)
    'passive_source': 'pipeline',
    # windowing
    'window': 20, 'stride': 1, 'train_stride': 1,
    # position feature: 'raw' (baseline)
    'pos_feature': 'raw',
    # plugins: list of 'module.path:function' applied to the per-trial records
    'plugins': [],
    # free-form parameters for plugins
    'params': {},
}


# ── Raw trial cache ──────────────────────────────────────────────────────────

def _atomic_pickle(obj, path):
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _raw_cache_path(subject):
    return os.path.join(CACHE_DIR, f'{subject}_raw.pkl')


def get_raw_trials(subject):
    """Return list of (attrs_dict, ndarray[T, 6], columns) — cached."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _raw_cache_path(subject)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    fn, sid = SUBJECTS[subject]
    trials = read_flb(os.path.join(DATA_DIR, fn), subject_id=sid)
    packed = [(dict(df.attrs), df.values.astype(np.float64), list(df.columns))
              for df in trials]
    _atomic_pickle(packed, path)
    return packed


def _unpack(packed):
    import pandas as pd
    out = []
    for attrs, arr, cols in packed:
        df = pd.DataFrame(arr.copy(), columns=cols)
        df.attrs.update(attrs)
        out.append(df)
    return out


# ── Trial split (mirrors build_dataset exactly) ─────────────────────────────

def split_trials(trials, test_trial_indices=None, retest_trial_indices=None,
                 verbose=False):
    mvc_trials, passive_trials, active_trials = db.classify_trials(trials)
    if test_trial_indices is not None or retest_trial_indices is not None:
        forced = set()
        if test_trial_indices:
            forced.update(i - 1 for i in test_trial_indices)
        if retest_trial_indices:
            forced.update(i - 1 for i in retest_trial_indices)
        rescued, remaining = [], []
        for df in passive_trials:
            (rescued if df.attrs.get('trial_index') in forced else remaining).append(df)
        passive_trials = remaining
        active_trials.extend(rescued)

    active_trials = [df for df in active_trials
                     if df[db.TORQUE_COL].std() <= db._ACTIVE_STD_THRESHOLD]

    if test_trial_indices is not None and retest_trial_indices is not None:
        test_set = {i - 1 for i in test_trial_indices}
        retest_set = {i - 1 for i in retest_trial_indices}
    else:
        test_set, retest_set, unlabelled = set(), set(), set()
        for df in active_trials:
            idx, session = df.attrs.get('trial_index'), df.attrs.get('session')
            if session == 'retest':
                retest_set.add(idx)
            elif session == 'test':
                test_set.add(idx)
            else:
                unlabelled.add(idx)
        if not test_set and retest_set and unlabelled:
            test_set = unlabelled
    trainval = [df for df in active_trials if df.attrs.get('trial_index') in test_set]
    retest = [df for df in active_trials if df.attrs.get('trial_index') in retest_set]
    return mvc_trials, passive_trials, trainval, retest, test_set, retest_set


def pipeline_passive_maps(subject, passive_trials):
    """Session-specific passive maps exactly as build_dataset computes them."""
    overrides = PASSIVE_TORQUE_OVERRIDES.get(subject)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if overrides and 'test' in overrides:
            test_map = sorted(overrides['test'], key=lambda x: x[0])
        else:
            test_map = db.get_passive_torque_map(
                [df for df in passive_trials if df.attrs.get('session') != 'retest'])
        if overrides and 'retest' in overrides:
            retest_map = sorted(overrides['retest'], key=lambda x: x[0])
        else:
            rt = [df for df in passive_trials if df.attrs.get('session') == 'retest']
            retest_map = db.get_passive_torque_map(rt) if rt else test_map
    return test_map, retest_map


def passive_curve(entries, pos):
    """Vectorised lookup_passive_torque (identical semantics)."""
    if not entries:
        return np.zeros_like(pos, dtype=np.float64)
    return np.array([db.lookup_passive_torque(entries, float(p)) for p in pos])


# ── Per-trial processing ────────────────────────────────────────────────────

def _decim(x, factor=db.DOWNSAMPLE_FACTOR):
    return decimate(x, factor, ftype='fir', zero_phase=True)


def _process_trial(df, emg_cols, opts, trim=True):
    """Envelope at 1 kHz on full trial, then trim + decimate. Returns dict."""
    env_full = np.column_stack([
        extract_envelope(df[c].values, bp_low=opts['bp_low'],
                         bp_high=opts['bp_high'], lp_cutoff=opts['lp'])
        for c in emg_cols])
    pos = df[db.POSITION_COL].values.astype(np.float64)
    tq = df[db.TORQUE_COL].values.astype(np.float64)
    start = db.TRANSIENT_SAMPLES if trim else 0
    env_t, pos_t, tq_t = env_full[start:], pos[start:], tq[start:]
    env_ds = np.column_stack([_decim(env_t[:, k]) for k in range(env_t.shape[1])])
    pos_ds, tq_ds = _decim(pos_t), _decim(tq_t)
    n = min(len(env_ds), len(pos_ds), len(tq_ds))
    return {
        'idx': df.attrs.get('trial_index'),
        'comment': df.attrs.get('comment', ''),
        'env_full_max': env_full.max(axis=0),
        'env_raw': env_ds[:n], 'pos': pos_ds[:n], 'tq': tq_ds[:n],
    }


def _feature_cache_path(subject, opts):
    key = f"{subject}_bp{opts['bp_low']}-{opts['bp_high']}_lp{opts['lp']}"
    return os.path.join(CACHE_DIR, key + '.pkl')


def get_processed(subject, opts):
    """Processed (envelope + decimation) trials, cached per filter setting."""
    path = _feature_cache_path(subject, opts)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    trials = _unpack(get_raw_trials(subject))
    ov = TRIAL_OVERRIDES.get(subject, {})
    mvc, passive, trainval, retest, test_set, retest_set = split_trials(
        trials, ov.get('test'), ov.get('retest'))
    emg_cols = detect_emg_columns(trainval[0])
    out = {
        'subject': subject, 'emg_cols': emg_cols,
        'trainval': [_process_trial(df, emg_cols, opts) for df in trainval],
        'retest': [_process_trial(df, emg_cols, opts) for df in retest],
        'mvc': [_process_trial(df, emg_cols, opts, trim=False) for df in mvc],
        'passive_trials': passive,   # raw DataFrames (for passive maps)
    }
    for rec in out['trainval']:
        rec['session'], rec['split'] = 'test', 'trainval'
    for rec in out['retest']:
        rec['session'], rec['split'] = 'retest', 'test'
    for rec in out['mvc']:
        rec['session'], rec['split'] = 'mvc', 'mvc'
    os.makedirs(CACHE_DIR, exist_ok=True)
    _atomic_pickle(out, path)
    return out


def _load_plugin(spec):
    mod_name, fn_name = spec.split(':')
    return getattr(importlib.import_module(mod_name), fn_name)


def prepare(subject, opts=None, verbose=True):
    """Build per-trial records + windowed arrays for one subject/variant."""
    o = dict(DEFAULT_OPTS)
    o.update(opts or {})
    proc = get_processed(subject, o)
    import copy
    trainval = copy.deepcopy(proc['trainval'])
    retest = copy.deepcopy(proc['retest'])
    mvc = copy.deepcopy(proc['mvc'])

    # ── EMG normalisation (baseline: global max of 1 kHz envelope over train/val)
    emg_max = np.max(np.stack([r['env_full_max'] for r in trainval]), axis=0)
    emg_max = np.where(emg_max > 0, emg_max, 1.0)
    for r in trainval + retest + mvc:
        r['env'] = r['env_raw'] / emg_max

    # ── Passive maps
    test_map, retest_map = pipeline_passive_maps(subject, proc['passive_trials'])
    ctx = {
        'subject': subject, 'opts': o, 'emg_cols': proc['emg_cols'],
        'emg_max': emg_max, 'test_map': test_map, 'retest_map': retest_map,
        'mvc': mvc, 'passive_trials': proc['passive_trials'],
    }
    for r in trainval + retest:
        pmap = retest_map if r['session'] == 'retest' else test_map
        r['passive_ref'] = passive_curve(pmap, r['pos'])
        r['y_ref'] = r['tq'] - r['passive_ref']
        if o['target'] == 'raw':
            r['passive'] = np.zeros_like(r['tq'])
        else:
            r['passive'] = r['passive_ref'].copy()
        r['y'] = r['tq'] - r['passive']
        r['extra'] = None

    # ── Plugins (may modify env / y / passive / extra, or add ctx keys)
    for spec in o['plugins']:
        fn = _load_plugin(spec)
        fn(trainval, retest, ctx)

    return finalize(trainval, retest, ctx, verbose=verbose)


def _windows(feat, y, window, stride):
    T = len(feat)
    n = (T - window) // stride + 1
    if n <= 0:
        return np.zeros((0, window, feat.shape[1]), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
    starts = np.arange(n) * stride
    idx = starts[:, None] + np.arange(window)[None, :]
    X = feat[idx].astype(np.float32)
    last = starts + window - 1
    return X, y[last].astype(np.float32), last


def finalize(trainval, retest, ctx, verbose=True):
    o = ctx['opts']
    W, S = o['window'], o['stride']
    tr, va, te = [], [], []
    test_info = []
    for r in trainval + retest:
        # Position feature: 'raw' (baseline, per-sample measured angle — note it
        # deflects ~2-3 mrad per 10 Nm through footplate compliance, i.e. it
        # leaks torque at high-torque angles) or 'knot' (trial-constant
        # set-point angle = trial mean, no compliance signal).
        pos_feat = r['pos']
        if o.get('pos_feature') == 'knot':
            pos_feat = np.full_like(r['pos'], float(np.mean(r['pos'])))
        env_feat = r['env']
        if o.get('drop_emg'):
            env_feat = np.zeros_like(r['env'])
        cols = [env_feat.astype(np.float32), pos_feat[:, None].astype(np.float32)]
        if r.get('extra') is not None:
            cols.append(np.asarray(r['extra'], np.float32).reshape(len(r['pos']), -1))
        feat = np.concatenate(cols, axis=1)
        X, y, last = _windows(feat, r['y'], W, S)
        if r['split'] == 'trainval':
            n_tr = int(len(X) * db.TRAIN_FRAC)
            n_va = int(len(X) * (1.0 - db.TRAIN_FRAC))
            ts = o.get('train_stride', 1)
            tr.append((X[:n_tr:ts], y[:n_tr:ts]))
            va.append((X[n_tr:n_tr + n_va], y[n_tr:n_tr + n_va]))
        else:
            te.append((X, y))
            test_info.append({
                'idx': r['idx'], 'pos': r['pos'][last], 'y_ref': r['y_ref'][last],
                'offset': (r['y_ref'] - r['y'])[last],
                'mean_pos': float(np.mean(r['pos'])),
            })
    data = {
        'X_train': np.concatenate([x for x, _ in tr]),
        'y_train': np.concatenate([y for _, y in tr]),
        'X_val': np.concatenate([x for x, _ in va]),
        'y_val': np.concatenate([y for _, y in va]),
        'X_test': np.concatenate([x for x, _ in te]),
        'y_test': np.concatenate([y for _, y in te]),
        'test_pos': np.concatenate([t['pos'] for t in test_info]),
        'test_y_ref': np.concatenate([t['y_ref'] for t in test_info]),
        'test_offset': np.concatenate([t['offset'] for t in test_info]),
        'test_trial': np.concatenate([np.full(len(t['pos']), t['idx']) for t in test_info]),
        'operating_positions': sorted(set(t['mean_pos'] for t in test_info)),
        'trainval': trainval, 'retest': retest, 'ctx': ctx,
    }
    # EMG gain augmentation (training windows only): k extra copies with a
    # random per-window, per-channel gain exp(U(−a, a)) on the 4 EMG channels.
    a, k = o.get('aug_gain', 0.0), int(o.get('aug_copies', 0))
    if a > 0 and k > 0:
        rng = np.random.default_rng(o.get('aug_seed', 0))
        Xs, ys = [data['X_train']], [data['y_train']]
        for _ in range(k):
            g = np.exp(rng.uniform(-a, a, size=(len(data['X_train']), 1, 4))).astype(np.float32)
            Xa = data['X_train'].copy()
            Xa[:, :, :4] *= g
            Xs.append(Xa); ys.append(data['y_train'])
        data['X_train'] = np.concatenate(Xs)
        data['y_train'] = np.concatenate(ys)
        data['aug_k'] = k
    if o.get('pos_feature') == 'normalized':
        pi = 4
        lo = float(data['X_train'][:, :, pi].min()); hi = float(data['X_train'][:, :, pi].max())
        for k in ('X_train', 'X_val', 'X_test'):
            data[k][:, :, pi] = (data[k][:, :, pi] - lo) / (hi - lo)
    if verbose:
        print(f"[{ctx['subject']}] X_train {data['X_train'].shape}  X_val {data['X_val'].shape}  "
              f"X_test {data['X_test'].shape}")
    return data


# ── Metrics ─────────────────────────────────────────────────────────────────

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, np.float64); y_pred = np.asarray(y_pred, np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    rng = float(y_true.max() - y_true.min())
    return {
        'r2': float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan'),
        'vaf': float((1 - np.var(y_true - y_pred) / np.var(y_true)) * 100) if np.var(y_true) > 0 else float('nan'),
        'rmse': rmse, 'nrmse': rmse / rng if rng > 0 else float('nan'),
        'mae': float(np.mean(np.abs(y_true - y_pred))),
        'bias': float(np.mean(y_pred - y_true)),
        'corr': float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 0 else float('nan'),
        'std_true': float(np.std(y_true)), 'n': int(len(y_true)),
    }


def evaluate(data, y_pred):
    """Per-position metrics on (a) the variant's own target and (b) the
    reference target (pred + offset, offset = y_ref − y per sample)."""
    y_pred = np.asarray(y_pred, np.float64).ravel()
    known = np.array(data['operating_positions'])
    snapped = known[np.argmin(np.abs(data['test_pos'][:, None] - known[None, :]), axis=1)]
    y_t, y_r, off = data['y_test'], data['test_y_ref'], data['test_offset']
    out = {'overall': metrics(y_t, y_pred),
           'overall_ref': metrics(y_r, y_pred + off), 'per_position': {}}
    for p in known:
        m = snapped == p
        out['per_position'][f'{p:+.3f}'] = {
            'target': metrics(y_t[m], y_pred[m]),
            'ref': metrics(y_r[m], y_pred[m] + off[m]),
        }
    return out


# ── Models ──────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {}


def register_model(name):
    def deco(fn):
        MODEL_REGISTRY[name] = fn
        return fn
    return deco


@register_model('lstm8')
def _lstm8(n_steps, n_features, **kw):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
    return Sequential([
        Input(shape=(n_steps, n_features)),
        LSTM(8, activation='tanh', return_sequences=True), Dropout(0.3),
        LSTM(8, activation='tanh', return_sequences=False), Dropout(0.3),
        Dense(1)], name='ankle_torque_lstm')


def get_model(name, n_steps, n_features, **kw):
    if ':' in name:
        return _load_plugin(name)(n_steps=n_steps, n_features=n_features, **kw)
    if name not in MODEL_REGISTRY:
        importlib.import_module('ML.experiments.plugins.core')
    return MODEL_REGISTRY[name](n_steps=n_steps, n_features=n_features, **kw)


PROTOCOLS = {
    # identical to ML.training.train_model
    'orig': dict(batch=8, lr=0.003, lr_factor=0.7, lr_patience=10,
                 early=25, epochs=200),
    # faster screening protocol (same optimiser/loss, bigger batch)
    'fast': dict(batch=64, lr=0.003, lr_factor=0.5, lr_patience=6,
                 early=15, epochs=150),
}


def set_threads(n):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(n)
    tf.config.threading.set_inter_op_parallelism_threads(max(1, n // 2))


def train_eval(data, model_name='lstm8', protocol='fast', seed=0,
               sample_weight_fn=None, loss='mse', model_kw=None, verbose=0):
    import tensorflow as tf
    from tensorflow.keras.optimizers import Nadam
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
    tf.keras.utils.set_random_seed(seed)
    p = PROTOCOLS[protocol] if isinstance(protocol, str) else protocol
    Xtr, ytr = data['X_train'], data['y_train']
    model = get_model(model_name, Xtr.shape[1], Xtr.shape[2], **(model_kw or {}))
    model.compile(optimizer=Nadam(learning_rate=p['lr']), loss=loss)
    sw = sample_weight_fn(data) if sample_weight_fn else None
    val_data = (data['X_val'], data['y_val'])
    if sample_weight_fn:
        # weight validation the same way so early stopping / LR schedule follow
        # the weighted objective (review finding)
        sw_val = sample_weight_fn(dict(data, X_train=data['X_val'], y_train=data['y_val']))
        val_data = (data['X_val'], data['y_val'], sw_val)
    cbs = [ReduceLROnPlateau(monitor='val_loss', factor=p['lr_factor'],
                             patience=p['lr_patience'], min_lr=1e-6, verbose=0),
           EarlyStopping(monitor='val_loss', patience=p['early'],
                         restore_best_weights=True, verbose=0)]
    t0 = time.time()
    hist = model.fit(Xtr, ytr, sample_weight=sw,
                     validation_data=val_data,
                     epochs=p['epochs'], batch_size=p['batch'], callbacks=cbs,
                     shuffle=True, verbose=verbose)
    train_s = time.time() - t0
    y_pred = model.predict(data['X_test'], verbose=0, batch_size=4096).ravel()
    return y_pred, {'epochs': len(hist.history['loss']), 'train_s': train_s,
                    'best_val': float(min(hist.history['val_loss']))}


def summarize(results):
    """Mean over seeds of per-position R² (target & ref) + ensemble."""
    keys = list(results['seeds'][0]['eval']['per_position'].keys())
    s = {}
    for k in keys:
        r2t = [r['eval']['per_position'][k]['target']['r2'] for r in results['seeds']]
        r2r = [r['eval']['per_position'][k]['ref']['r2'] for r in results['seeds']]
        rm = [r['eval']['per_position'][k]['target']['rmse'] for r in results['seeds']]
        s[k] = {'r2_mean': float(np.mean(r2t)), 'r2_std': float(np.std(r2t)),
                'r2_ref_mean': float(np.mean(r2r)), 'rmse_mean': float(np.mean(rm))}
    return s


def run(subject, name, opts=None, model='lstm8', protocol='fast', seeds=(0, 1, 2),
        threads=None, loss='mse', model_kw=None, sample_weight=None,
        save_preds=True, out_dir=None):
    """Full experiment: prepare → train/eval for each seed → JSON + npz."""
    if threads:
        set_threads(threads)
    out_dir = out_dir or RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    data = prepare(subject, opts)
    sw_fn = _load_plugin(sample_weight) if sample_weight else None
    res = {'subject': subject, 'name': name, 'opts': data['ctx']['opts'],
           'model': model, 'protocol': protocol, 'loss': loss,
           'sample_weight': sample_weight, 'model_kw': model_kw, 'seeds': []}
    preds = []
    for sd in seeds:
        y_pred, info = train_eval(data, model, protocol, sd, sw_fn, loss, model_kw)
        ev = evaluate(data, y_pred)
        preds.append(y_pred)
        res['seeds'].append({'seed': sd, 'info': info, 'eval': ev})
        pp = '  '.join(f"{k}:{v['target']['r2']:+.3f}" for k, v in ev['per_position'].items())
        print(f"  [{subject}/{name}] seed {sd}  pooled R² {ev['overall']['r2']:.4f}  "
              f"({info['epochs']} ep, {info['train_s']:.0f}s)  {pp}", flush=True)
    ens = np.mean(preds, axis=0)
    res['ensemble'] = evaluate(data, ens)
    res['summary'] = summarize(res)
    tag = f'{subject}__{name}__{protocol}'
    with open(os.path.join(out_dir, tag + '.json'), 'w') as f:
        json.dump(res, f, indent=1, default=float)
    if save_preds:
        np.savez_compressed(os.path.join(out_dir, tag + '.npz'),
                            preds=np.array(preds), y_test=data['y_test'],
                            test_pos=data['test_pos'], test_y_ref=data['test_y_ref'],
                            test_offset=data['test_offset'],
                            test_trial=data['test_trial'],
                            operating_positions=np.array(data['operating_positions']))
    print(format_summary(res), flush=True)
    return res


def format_summary(res):
    s = res['summary']
    lines = [f"== {res['subject']} / {res['name']} ({res['model']}, {res['protocol']}, "
             f"{len(res['seeds'])} seeds)"]
    lines.append('   pos      R²(mean±sd)   R²_ref   RMSE   | ensemble R²')
    for k, v in s.items():
        e = res['ensemble']['per_position'][k]['target']['r2']
        lines.append(f"  {k}  {v['r2_mean']:+.3f}±{v['r2_std']:.3f}  {v['r2_ref_mean']:+.3f}  "
                     f"{v['rmse_mean']:.3f}  | {e:+.3f}")
    pooled = np.mean([r['eval']['overall']['r2'] for r in res['seeds']])
    lines.append(f"  pooled R² (mean over seeds): {pooled:.4f}   "
                 f"ensemble pooled: {res['ensemble']['overall']['r2']:.4f}")
    return '\n'.join(lines)
