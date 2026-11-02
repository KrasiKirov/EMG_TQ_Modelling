"""
Run ONE job: (study, subject, protocol, configuration, seed).

    python -m ML.research.run --study screen --subject YES --protocol S1toS2 \
        --config E03_final --seed 0 --train-protocol fast --threads 2 --plot

Steps
-----
1. load the subject bundle (cached)                         data.load_subject
2. normalise EMG with statistics of the training data       emg_norm.normalise
3. detect rest periods (EMG only) in the training data      passive.rest_segments
4. fit the passive model, build targets (measured − P_eff)  passive.PassiveModel
5. cut windows for the protocol                             splits.build
6. train with seed (optionally starting from weights pre-trained
   on the other subjects, transfer.py), predict the test windows  train.fit_predict
7. metrics per angle, recalibration, save                   evaluate

Outputs (RESULTS_DIR/<study>/<subject>/<protocol>/<config>/)
------------------------------------------------------------
<train-protocol>_seed<k>.json   config, passive/EMG details, training info,
                                metrics per angle and overall, recalibration
<train-protocol>_seed<k>.npz    y_true, y_pred and window metadata (float32)
full_trial_inspection_<session>_<train-protocol>_seed<k>.png   (--plot)
"""

import os
import sys
import json
import time
import argparse

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np

from ML.research import config as C
from ML.research import experiments
from ML.research.data import load_subject
from ML.research.emg_norm import normalise, push_directions
from ML.research.passive import PassiveModel, rest_segments
from ML.research import splits
from ML.research import truth
from ML.research.train import fit_predict
from ML.research.evaluate import metrics, per_group, recalibrated, plot_trials


def out_dir(study, subject, protocol, config):
    return os.path.join(C.RESULTS_DIR, study, subject, protocol, config)


def prepare(subject, protocol, cfg):
    """Steps 1-5: everything up to the windowed arrays."""
    bundle = load_subject(subject)
    recs = bundle['active']
    spec = splits.PROTOCOLS[protocol]
    train_recs = [r for r in recs if r['session'] in spec['train']]
    tf_ = 0.6 if protocol == 'pooled' else C.TRAIN_FRAC
    val_end = 0.8 if protocol == 'pooled' else 1.0
    window = cfg.get('window', C.WINDOW)

    feats, emg_info = normalise(cfg['emg'], bundle, train_recs, recs, train_frac=tf_, val_end_frac=val_end,
                                window=window)
    base_env, _ = normalise('trainmax', bundle, train_recs, recs, train_frac=tf_, val_end_frac=val_end,
                            window=window)
    tr_segs, va_segs = rest_segments(train_recs, base_env, train_frac=tf_, val_end_frac=val_end, window=window)
    if cfg['passive'] == 'none':
        # total-torque study: the target is the processed measured torque itself
        # (1 kHz -> trim -> zero-phase FIR decimation to 100 Hz, data._process_active);
        # nothing is subtracted.
        targets = {r['idx']: r['tq'].copy() for r in recs}
        pinfo = {'kind': 'none', 'target': 'measured total torque (no passive subtraction)'}
    elif cfg['passive'] == 'restanchor':
        targets, pinfo = restanchor_targets(bundle, protocol)
    else:
        pm = PassiveModel(cfg['passive'], bundle).fit(train_recs, tr_segs, va_segs)
        targets, pinfo = {r['idx']: r['tq'] - pm.predict(r) for r in recs}, pm.describe()
    rest = rest_flags(bundle, protocol)
    data = splits.build(protocol, recs, feats, targets, window=window, rest=rest)
    details = {'passive': pinfo, 'emg': {k: v for k, v in emg_info.items() if k in (
                   'kind', 'rmin_frac', 'transductive', 'auto_decision', 'adaptive', 'scale', 'session_scale',
                   'scale_source', 'guarded_channels', 'n_samples_scale')},
               'push_directions': push_directions(bundle, train_recs, base_env, train_frac=tf_, window=window),
               'n_windows': {p: int(len(data[f'y_{p}'])) for p in ('train', 'val', 'test')},
               'rest_frac_train': float(np.mean(data['meta_train']['rest']))}
    return data, details


def restanchor_targets(bundle, protocol):
    """passive='restanchor': target = measured - in-trial passive baseline B(t)
    through the EMG-detected rest segments of the SAME trial (truth.py).

    Training sessions: rest segments are searched only in the part of each
    trial that is training/validation data (pooled: first 80 %), so no
    test-part torque enters a training label.  Test sessions: the full trial
    is used, so the test target equals the common reference truth."""
    spec = splits.PROTOCOLS[protocol]
    targets, n_segs = {}, {}
    for sess in ('S1', 'S2'):
        limit = 0.8 if (protocol == 'pooled') else None
        segs = truth.session_rest(bundle, sess, max_frac=limit if sess in spec['train'] else None)
        for r in bundle['active']:
            if r['session'] == sess:
                targets[r['idx']] = r['tq'] - truth.baseline(r, segs[r['idx']])
                n_segs[r['idx']] = len(segs[r['idx']])
    return targets, {'kind': 'restanchor', 'sigma_s': truth.SIGMA_S, 'n_rest_segments_per_trial': n_segs}


def rest_flags(bundle, protocol):
    """{trial idx: (T,) bool} EMG-detected rest per sample, used to (a) down-
    weight / drop rest windows in training (rest_weight option) and (b)
    report rest-window metrics.  Same restriction as restanchor_targets for
    the training sessions of the pooled protocol."""
    spec = splits.PROTOCOLS[protocol]
    out = {}
    for sess in ('S1', 'S2'):
        limit = 0.8 if (protocol == 'pooled' and sess in spec['train']) else None
        segs = truth.session_rest(bundle, sess, max_frac=limit)
        for r in bundle['active']:
            if r['session'] == sess:
                out[r['idx']] = truth.rest_mask(len(r['tq']), segs[r['idx']])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--study', required=True)
    ap.add_argument('--subject', required=True)
    ap.add_argument('--protocol', required=True, choices=list(splits.PROTOCOLS))
    ap.add_argument('--config', required=True, choices=list(experiments.CONFIGS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train-protocol', default='fast', choices=list(C.PROTOCOLS))
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--plot', action='store_true', help='save full-trial inspection plot(s)')
    a = ap.parse_args()

    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(a.threads)
    tf.config.threading.set_inter_op_parallelism_threads(max(1, a.threads // 2))

    cfg = experiments.get(a.config)
    t0 = time.time()
    data, details = prepare(a.subject, a.protocol, cfg)
    t_prep = time.time() - t0
    init, lr = None, None
    if cfg.get('pretrain'):                      # cross-subject transfer (transfer.py)
        from ML.research import transfer
        init, lr = transfer.pretrained_model_path(a.subject, cfg, a.seed), transfer.FINETUNE_LR
        t_prep = time.time() - t0                # pre-training time is counted as preparation
    pred, info = fit_predict(data, cfg['model'], cfg['balance'], cfg['pf_extra'], a.train_protocol, a.seed,
                             init_weights=init, lr=lr, rest_weight=cfg.get('rest_weight'))
    info['train_s'] = time.time() - t0 - t_prep
    info['prep_s'] = t_prep

    y, meta = data['y_test'].astype(np.float64), data['meta_test']
    res = {'study': a.study, 'subject': a.subject, 'protocol': a.protocol, 'config': a.config,
           'config_spec': {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
           'seed': a.seed, 'train_protocol': a.train_protocol, 'details': details, 'train_info': info,
           'overall': metrics(y, pred), 'per_angle': per_group(y, pred, meta)}
    if a.protocol != 'pooled':
        res['recalibration_15s'] = recalibrated(y, pred, meta, calib_s=15.0)

    d = out_dir(a.study, a.subject, a.protocol, a.config)
    os.makedirs(d, exist_ok=True)
    stem = os.path.join(d, f'{a.train_protocol}_seed{a.seed}')
    np.savez_compressed(stem + '.npz', y_true=y.astype(np.float32), y_pred=pred.astype(np.float32),
                        trial=meta['trial'].astype(np.int16), angle=meta['angle'].astype(np.float32),
                        session=(meta['session'] == 'S2').astype(np.int8), t=meta['t'].astype(np.float32))
    if a.plot:
        for sess in sorted(set(meta['session'])):
            m = meta['session'] == sess
            sub = {k: v[m] for k, v in meta.items()}
            plot_trials(sub, y[m], pred[m], f"{d}/full_trial_inspection_{sess}_{a.train_protocol}_seed{a.seed}.png",
                        f"{a.subject} | {a.config} | train {'+'.join(splits.PROTOCOLS[a.protocol]['train'])} -> "
                        f"test {sess} | {a.train_protocol} seed {a.seed}")
    with open(stem + '.json.tmp', 'w') as f:
        json.dump(res, f, indent=1, default=float)
    os.replace(stem + '.json.tmp', stem + '.json')   # JSON last = job complete
    pa = '  '.join(f"{k}:{v['r2']:+.2f}" for k, v in res['per_angle'].items())
    print(f"[{a.subject}/{a.protocol}/{a.config}/{a.train_protocol} s{a.seed}] {info['epochs']} ep "
          f"{info['train_s']:.0f}s  {pa}", flush=True)


if __name__ == '__main__':
    main()
