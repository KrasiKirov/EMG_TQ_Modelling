"""
Run ONE experiment configuration for one subject, one direction and one seed.

    python EXPERIMENTS/code/run_experiment.py --exp X01_clean_passive --subject JM \
        --direction S1toS2 --seed 0 --threads 1

The configuration is read from EXPERIMENTS/configs/<exp>.json.  Every
configuration starts from the LSTM-8 baseline and lists only what it changes
(see configs/X00f_baseline_framework.json for the full baseline definition).

Data building, passive-torque models, EMG normalisation and the training loop
come from ML/research (shown by check_equivalence.py to build the same data as
the frozen original pipeline).  To match the original exactly, the last
validation window of every training trial is dropped (the original split
leaves it out).  Training settings default to the original ones
(Nadam 0.003, batch 8, 200 epochs, ReduceLROnPlateau 0.7/10, EarlyStopping 25).

Outputs: EXPERIMENTS/results/<exp>/<subject>/<direction>/seed<k>.{npz,json}
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

BASE = dict(passive='legacy', emg='trainmax', units=8, layers=2, dropout=0.3, model='lstm',
            balance='none', pf_extra=None, window=20, lr=None, train_protocol='orig', rest_weight=None,
            extra_input=None)


def load_config(exp):
    with open(os.path.join(EXP, 'configs', f'{exp}.json'), encoding='utf-8') as f:
        spec = json.load(f)
    cfg = dict(BASE)
    cfg.update(spec.get('changes', {}))
    return spec, cfg


def make_builder(cfg):
    """Model builder plugged into ML.research.train (same call signature)."""
    from ML.research import models as RM

    def build(name, n_steps, n_features, knots):
        if name != 'lstm':
            return RM.build(name, n_steps, n_features, knots)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True)
    ap.add_argument('--subject', required=True)
    ap.add_argument('--direction', required=True, choices=['S1toS2', 'S2toS1'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=1)
    a = ap.parse_args()

    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(a.threads)
    tf.config.threading.set_inter_op_parallelism_threads(max(1, a.threads))

    from ML.research.run import prepare
    from ML.research import train as RT

    spec, cfg = load_config(a.exp)
    RT.build_model = make_builder(cfg)                  # adds units / layers / dropout knobs
    t0 = time.time()
    rcfg = {k: cfg[k] for k in ('passive', 'emg', 'window')}
    data, details = prepare(a.subject, a.direction, rcfg)
    data = drop_last_val_window(data)
    if cfg.get('extra_input') == 'passive_trial_value':
        data = add_passive_input(data, a.subject)
    t_prep = time.time() - t0
    pred, info = RT.fit_predict(data, cfg['model'], cfg['balance'], cfg['pf_extra'], cfg['train_protocol'], a.seed,
                                lr=cfg['lr'], rest_weight=cfg['rest_weight'])
    t_train = time.time() - t0 - t_prep

    y, meta = data['y_test'].astype(np.float64), data['meta_test']
    out = os.path.join(EXP, 'results', a.exp, a.subject, a.direction)
    os.makedirs(out, exist_ok=True)
    stem = os.path.join(out, f'seed{a.seed}')
    np.savez_compressed(stem + '.npz', y_own_target=y.astype(np.float32), y_pred=pred.astype(np.float32),
                        trial=meta['trial'].astype(np.int16), angle=meta['angle'].astype(np.float32),
                        session=(meta['session'] == 'S2').astype(np.int8), t=meta['t'].astype(np.float32))
    res = {'experiment': a.exp, 'subject': a.subject, 'direction': a.direction, 'seed': a.seed,
           'config': cfg, 'hypothesis': spec.get('hypothesis'), 'details': details,
           'train_info': {k: v for k, v in info.items() if k != 'init_weights'},
           'n_windows': {p: int(len(data[f'y_{p}'])) for p in ('train', 'val', 'test')},
           'prep_s': t_prep, 'train_s': t_train,
           'versions': {'tensorflow': tf.__version__, 'numpy': np.__version__, 'python': sys.version.split()[0]}}
    with open(stem + '.json.tmp', 'w') as f:
        json.dump(res, f, indent=1, default=float)
    os.replace(stem + '.json.tmp', stem + '.json')
    print(f'[{a.exp} {a.subject} {a.direction} s{a.seed}] {info["epochs"]} ep (best {info.get("best_epoch", "-")}) {t_train:.0f}s',
          flush=True)


if __name__ == '__main__':
    main()
