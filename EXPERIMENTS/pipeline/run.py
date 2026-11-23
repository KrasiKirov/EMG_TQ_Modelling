"""
Run ONE experiment configuration for one subject and one seed, trained on session 1 and tested on session 2.

    python -m EXPERIMENTS.pipeline.run --exp A0_active_trainmax --subject JM \
        --seed 0 --threads 1

The configuration is read from EXPERIMENTS/configs/<exp>.json.  Every
configuration starts from the LSTM-8 baseline and lists only what it changes
(see configs/X00f_baseline_framework.json for the full baseline definition).

Data building, passive-torque models, EMG normalisation and the training loop come from ML/research, which
builds the same data as the frozen original pipeline. As in the original split, the last validation window of
every training trial is dropped. Training settings default to the original ones (Nadam 0.003, batch 8,
200 epochs, ReduceLROnPlateau 0.7/10, EarlyStopping 25).

Outputs: EXPERIMENTS/results/<exp>/<subject>/<direction>/seed<k>.{npz,json}
Study v2 runs (ids TS2_ / FE_) write to EXPERIMENTS/results_v2/ instead, refuse to start on uncommitted code, and
record the hash of EXPERIMENTS/protocol/study_v2.json. Every run records its provenance and config hash.
"""

import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP)
sys.path.insert(0, ROOT)

import numpy as np                                     # noqa: E402

from EXPERIMENTS.pipeline.settings import BASE, DIRECTION, load_config  # noqa: E402,F401  (re-exported)

RESULTS = os.path.join(EXP, 'results')
RESULTS_V2 = os.path.join(EXP, 'results_v2')
PROTOCOL = os.path.join(EXP, 'protocol', 'study_v2.json')
STUDY_PREFIXES = ('TS2_', 'FE_')


def is_study(exp):
    return exp.startswith(STUDY_PREFIXES)


def results_root(exp):
    """Study v2 runs live apart from v1, so v1 scripts never read them."""
    return RESULTS_V2 if is_study(exp) else RESULTS


def protocol_hash():
    from EXPERIMENTS.pipeline.provenance import sha256_file
    h = sha256_file(PROTOCOL)
    if h is None:
        raise SystemExit(f'study runs need the protocol file {PROTOCOL}')
    return h


def make_builder(cfg):
    """Model builder plugged into ML.research.train (same call signature)."""

    def build(name, n_steps, n_features, knots):
        if name != 'lstm':                              # 'lstm8': the original model
            from ML.research.train import build_original
            return build_original(name, n_steps, n_features, knots)
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
        lay = [Input(shape=(n_steps, n_features))]
        for i in range(cfg['layers']):
            lay.append(LSTM(cfg['units'], activation='tanh', return_sequences=i < cfg['layers'] - 1))
            lay.append(Dropout(cfg['dropout']))
        lay.append(Dense(1))
        return Sequential(lay, name='lstm_experiment')
    return build


def drop_last_val_window(data):
    """Remove the last validation window of every trial (original split)."""
    tr = data['meta_val']['trial']
    keep = np.ones(len(tr), bool)
    for i in np.unique(tr):
        keep[np.where(tr == i)[0][-1]] = False
    data = dict(data)
    data['X_val'], data['y_val'] = data['X_val'][keep], data['y_val'][keep]
    data['meta_val'] = {k: v[keep] for k, v in data['meta_val'].items()}
    return data


def add_passive_input(data, subject):
    """Tier-2 option 'extra_input': 'passive_trial_value'.
    Appends one constant input channel per window: the passive torque of that
    trial's angle taken from the PASSIVE TRIALS of the trial's own session (the
    clean static-hold table, the same table the primary answer key uses).  It is
    measured before the active trial, so it is available at prediction time.
    IES has no passive trials -> 0 for every window."""
    from ML.research.data import load_subject
    from ML.research.passive import map_lookup
    maps = load_subject(subject)['maps']['static']
    data = dict(data)
    for part in ('train', 'val', 'test'):
        meta = data[f'meta_{part}']
        p = np.array([map_lookup(maps[s], np.array([a]))[0] if maps.get(s) else 0.0
                      for s, a in zip(meta['session'], meta['angle'])], np.float32)
        X = data[f'X_{part}']
        data[f'X_{part}'] = np.concatenate([X, np.repeat(p[:, None, None], X.shape[1], axis=1)], axis=2)
    return data


def fit_ridge(data):
    """Study v2 A1: ridge on lagged envelopes (ML.models.baselines.LaggedRidge); alpha chosen on the session-1
    validation block from the protocol's grid. Deterministic: the seed is not used."""
    from ML.models.baselines import LaggedRidge
    with open(PROTOCOL, encoding='utf-8') as f:
        rp = json.load(f)['ridge']
    m = LaggedRidge(rp['alphas'], rp['lags_steps']).fit(data['X_train'], data['y_train'], data['X_val'], data['y_val'])
    vp = m.predict(data['X_val'])
    info = {'epochs': 0, 'alpha': m.alpha, 'val_mae_by_alpha': {str(k): v for k, v in m.val_mae.items()}, 'n_params': int(len(m.weights)),
            'val_pred': vp, 'val_mae': float(np.mean(np.abs(vp - data['y_val'])))}
    return m.predict(data['X_test']), info


def fit_carry(data, cfg, seed):
    """state='carry': train ML.research.stateful on whole trials; return test predictions on the same samples
    (and in the same order) as the windowed model's data['y_test'] / data['meta_test']."""
    from ML.research import config as RC, stateful as ST
    from ML.research.train import angle_groups, snap
    knots = angle_groups(data['train_angles'])
    k_of = lambda r: int(snap(np.array([r['angle']]), knots)[0])
    st = data['streams']
    train = [dict(s, k=k_of(s['rec'])) for s in st['train']]
    test = [dict(s, k=k_of(s['rec'])) for s in st['test']]
    p = RC.PROTOCOLS[cfg['train_protocol']]
    preds, info = ST.fit_predict_carry(train, test, n_knots=len(knots), chunk=cfg['chunk'] or cfg['window'] * cfg['input_step'],
                                       seed=seed, epochs=p['epochs'], early=p['early'], lr=cfg['lr'] or p['lr'],
                                       lr_factor=p['lr_factor'], lr_patience=p['lr_patience'], balance=cfg['balance'],
                                       units=cfg['units'], layers=cfg['layers'], dropout=cfg['dropout'],
                                       clipnorm=cfg['clipnorm'])
    vs = {s['rec']['idx']: v for s, v in zip(train, info.pop('val_streams'))}
    mv = data['meta_val']
    val_pred = np.array([vs[int(i)][j] for i, j in zip(mv['trial'], mv['last'])], np.float64)
    info.update(val_pred=val_pred, val_mae=float(np.mean(np.abs(val_pred - data['y_val']))), clipnorm=cfg['clipnorm'])
    by_trial = {s['rec']['idx']: (s, pr) for s, pr in zip(test, preds)}
    meta = data['meta_test']
    pred = np.empty(len(meta['trial']))
    for i in np.unique(meta['trial']):
        rows = np.where(meta['trial'] == i)[0]
        s, pr = by_trial[int(i)]
        pred[rows] = pr[meta['last'][rows]]
        assert np.allclose(np.asarray(s['y'])[meta['last'][rows]], data['y_test'][rows])
    info.update(knots=knots.tolist())
    return pred, info, data


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True)
    ap.add_argument('--subject', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--save-weights', action='store_true', help='also save the trained weights next to the results')
    a = ap.parse_args(argv)
    a.direction = DIRECTION          # always session 1 → session 2; not an option

    from EXPERIMENTS.pipeline import provenance as PV
    from EXPERIMENTS.pipeline.spec import CONFIGS
    spec, cfg = load_config(a.exp)
    study = is_study(a.exp)
    prov = PV.collect(threads=a.threads, protocol_hash=protocol_hash() if study else None, subject=a.subject,
                      spec_path=os.path.join(str(CONFIGS), f'{a.exp}.json'))
    if study and prov['git_dirty']:
        raise SystemExit(f'{a.exp}: study runs refuse uncommitted code (ML/, EXPERIMENTS/pipeline|protocol|tuning, '
                         'or the spec itself); commit first')

    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(a.threads)
        tf.config.threading.set_inter_op_parallelism_threads(max(1, a.threads))
    except RuntimeError:          # TensorFlow already running in this process (tests): keep its settings
        pass

    from ML.research.run import prepare
    from ML.research import train as RT

    RT.build_model = make_builder(cfg)                  # adds units / layers / dropout knobs
    t0 = time.time()
    rcfg = {k: cfg[k] for k in ('passive', 'emg', 'window', 'envelope', 'trim_end_s', 'input_step', 'train_stride',
                                'state', 'lp_order', 'val_gap', 'metadata_passive', 'scale_part')}
    data, details = prepare(a.subject, a.direction, rcfg)
    if study:
        from ML.research.splits import check_common_targets
        try:
            check_common_targets(data['train_recs'])
        except ValueError as e:
            raise SystemExit(f'{a.subject}: {e}')
    data = drop_last_val_window(data)
    ew, ew_mode = cfg['epoch_windows'], 'passes'
    if ew == 'full':                                   # equal updates across overlap (study v2)
        ew, ew_mode = data['n_train_full'], 'exact'
    if cfg.get('extra_input') == 'passive_trial_value':
        data = add_passive_input(data, a.subject)
    t_prep = time.time() - t0
    out = os.path.join(results_root(a.exp), a.exp, a.subject, a.direction)
    os.makedirs(out, exist_ok=True)
    if cfg['state'] == 'carry':
        pred, info, data = fit_carry(data, cfg, a.seed)
    elif cfg['model'] == 'ridge_lagged':
        pred, info = fit_ridge(data)
    else:
        pred, info = RT.fit_predict(data, cfg['model'], cfg['balance'], cfg['pf_extra'], cfg['train_protocol'], a.seed,
                                    lr=cfg['lr'], rest_weight=cfg['rest_weight'], shuffle=cfg['shuffle'],
                                    save_weights=os.path.join(out, f'seed{a.seed}.weights.h5') if a.save_weights else None,
                                    epoch_windows=ew, epoch_mode=ew_mode, clipnorm=cfg['clipnorm'], batch=cfg['batch'])
    t_train = time.time() - t0 - t_prep

    y, meta = data['y_test'].astype(np.float64), data['meta_test']
    stem = os.path.join(out, f'seed{a.seed}')
    from ML.research.splits import common_mask
    mv = data['meta_val']
    np.savez_compressed(stem + '_val.npz', y=data['y_val'].astype(np.float32),
                        y_pred=info.pop('val_pred').astype(np.float32), trial=mv['trial'].astype(np.int16),
                        last=mv['last'].astype(np.int32), angle=mv['angle'].astype(np.float32),
                        session=(mv['session'] == 'S2').astype(np.int8), t=mv['t'].astype(np.float32),
                        common=common_mask(mv, data['trial_len']))
    np.savez_compressed(stem + '.npz', y_own_target=y.astype(np.float32), y_pred=pred.astype(np.float32),
                        trial=meta['trial'].astype(np.int16), angle=meta['angle'].astype(np.float32),
                        session=(meta['session'] == 'S2').astype(np.int8), t=meta['t'].astype(np.float32))
    res = {'experiment': a.exp, 'subject': a.subject, 'direction': a.direction, 'seed': a.seed,
           'config': cfg, 'hypothesis': spec.get('hypothesis'), 'details': details,
           'train_info': {k: v for k, v in info.items() if k != 'init_weights'},
           'n_windows': dict({p: int(len(data[f'y_{p}'])) for p in ('train', 'val', 'test')},
                             train_full=int(data['n_train_full'])),
           'prep_s': t_prep, 'train_s': t_train, 'config_hash': PV.config_hash(cfg), 'provenance': prov,
           'versions': {'tensorflow': tf.__version__, 'numpy': np.__version__, 'python': sys.version.split()[0]}}
    with open(stem + '.json.tmp', 'w') as f:
        json.dump(res, f, indent=1, default=float)
    os.replace(stem + '.json.tmp', stem + '.json')
    print(f'[{a.exp} {a.subject} {a.direction} s{a.seed}] {info["epochs"]} ep (best {info.get("best_epoch", "-")}) {t_train:.0f}s',
          flush=True)


if __name__ == '__main__':
    main()
