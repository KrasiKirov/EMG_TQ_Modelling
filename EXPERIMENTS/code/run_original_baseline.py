"""
Run the ORIGINAL LSTM-8 pipeline (frozen copy in EXPERIMENTS/baseline/original_code)
for one subject and one seed, and save every prediction.

Nothing in the original method is changed.  This wrapper only:
  * points the original code at the data folder (the frozen copy has no data),
  * fixes the random seed (the original code did not set one, so its results
    varied from run to run),
  * saves predictions and metrics to EXPERIMENTS/results/X00_baseline/.

Protocol = the original one: train on the 'test' session (S1: first 80 % of
each trial for fitting, last 20 % for early stopping), evaluate on the whole
'retest' session (S2).  Original training settings from the frozen config:
Nadam lr 0.003, batch 8, max 200 epochs, ReduceLROnPlateau(0.7, patience 10),
EarlyStopping(patience 25, restore best weights).

Usage:
    python EXPERIMENTS/code/run_original_baseline.py --subject YES --seed 0 --threads 1
"""

import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
SNAP = os.path.join(EXP, 'baseline', 'original_code')
MAIN_DATA = os.environ.get('EMG_DATA_DIR')            # set by ML/experiments/env.sh

sys.path.insert(0, SNAP)                              # frozen original code first
# make sure nothing from the modified worktree package shadows the frozen copy
for k in [k for k in sys.modules if k == 'ML' or k.startswith('ML.')]:
    del sys.modules[k]

import numpy as np                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subject', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=1)
    a = ap.parse_args()

    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(a.threads)
    tf.config.threading.set_inter_op_parallelism_threads(max(1, a.threads))

    import ML.config as cfg
    assert os.path.abspath(cfg.__file__).startswith(os.path.abspath(SNAP)), cfg.__file__
    if MAIN_DATA:
        cfg.DATA_DIR = MAIN_DATA
    from ML.training import load_subject, train_model, evaluate_model
    from ML.preprocessing import dataset_builder as db
    from ML.preprocessing.flb_reader import read_flb
    from ML.evaluation.metrics import snap_to_operating_points

    tf.keras.utils.set_random_seed(a.seed)
    t0 = time.time()
    data = load_subject(a.subject)
    t_prep = time.time() - t0
    model, hist = train_model(data['X_train'], data['y_train'], data['X_val'], data['y_val'],
                              label=f'LSTM8 ({a.subject})', save_path=None, verbose=0)
    res = evaluate_model(model, data['X_test'], data['y_test'], data['operating_positions'],
                         mvc_tq_test=data['mvc_tq_test'])
    t_train = time.time() - t0 - t_prep

    # passive maps of the retest (tested) session exactly as the original code builds them
    fname, _sid = cfg.SUBJECTS[a.subject]
    trials = read_flb(os.path.join(cfg.DATA_DIR, fname))
    _mvc, passive, _act = db.classify_trials(trials)
    ov = cfg.PASSIVE_TORQUE_OVERRIDES.get(a.subject)
    if ov and 'retest' in ov:
        legacy_retest = sorted(ov['retest'], key=lambda e: e[0])
    else:
        rp = [d for d in passive if d.attrs.get('session') == 'retest']
        legacy_retest = db.get_passive_torque_map(rp) if rp else []

    pos = data['X_test'][:, -1, cfg.POSITION_FEATURE_INDEX].astype(np.float64)
    snapped, valid = snap_to_operating_points(pos, data['operating_positions'])
    out = os.path.join(EXP, 'results', 'X00_baseline', a.subject, 'S1toS2')
    os.makedirs(out, exist_ok=True)
    stem = os.path.join(out, f'seed{a.seed}')
    np.savez_compressed(stem + '.npz', y_true_original=np.asarray(res['y_test'], np.float32),
                        y_pred=np.asarray(res['y_pred'], np.float32), position=pos.astype(np.float32),
                        position_snapped=np.asarray(snapped, np.float32))
    h = hist.history
    best = int(np.argmin(h['val_loss']))
    info = {'experiment': 'X00_baseline', 'subject': a.subject, 'direction': 'learned S1 (test) -> tested S2 (retest)',
            'seed': a.seed, 'code': 'EXPERIMENTS/baseline/original_code (frozen copy of the main checkout ML/)',
            'settings': {k: getattr(cfg, k) for k in ('LEARNING_RATE', 'LR_FACTOR', 'LR_PATIENCE', 'LR_MIN',
                                                     'EARLY_PATIENCE', 'BATCH_SIZE', 'EPOCHS', 'N_STEPS')},
            'n_windows': {'train': int(len(data['y_train'])), 'val': int(len(data['y_val'])), 'test': int(len(data['y_test']))},
            'operating_positions': [float(p) for p in data['operating_positions']],
            'legacy_passive_map_retest': [[float(p), float(t)] for p, t in legacy_retest],
            'epochs_run': len(h['loss']), 'best_epoch': best + 1,
            'train_loss': [float(v) for v in h['loss']], 'val_loss': [float(v) for v in h['val_loss']],
            'n_params': int(model.count_params()), 'prep_s': t_prep, 'train_s': t_train,
            'per_position_original': {f'{float(k):+.4f}': {kk: float(vv) for kk, vv in v.items()}
                                      for k, v in res['per_position'].items()},
            'overall_original': {k: float(v) for k, v in res['overall'].items()}}
    with open(stem + '.json.tmp', 'w') as f:
        json.dump(info, f, indent=1)
    os.replace(stem + '.json.tmp', stem + '.json')
    print(f'[X00 {a.subject} s{a.seed}] {len(h["loss"])} ep, best {best + 1}, {t_train:.0f}s  ' +
          '  '.join(f'{float(k):+.2f}:{v["r2"]:+.2f}' for k, v in sorted(res['per_position'].items())), flush=True)


if __name__ == '__main__':
    main()
