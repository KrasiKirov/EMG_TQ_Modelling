"""Confirmation runs through the REAL pipeline (load_subject → train_model →
evaluate_model), not the experiment harness.

python -m ML.experiments.run_pipeline --subject YES --name final --config final \
    --seeds 0 1 2 --protocol orig --threads 2

Configs
-------
legacy : exactly the pre-investigation pipeline (old passive map, no gain,
         no label re-zeroing, raw position, plain MSE)
final  : the configuration chosen from the screening experiments
         (see FINAL_CONFIG below)
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np

LEGACY_CONFIG = {
    'dataset': dict(passive_method='legacy', passive_gain=False,
                    train_label_rezero=False, emg_clip=None, emg_floor=None,
                    position_feature='raw'),
    'train': dict(position_balance=None, anchored=False),
}

# Chosen from the screening results (ML/experiments/results) — same as
# `train.py --recommended`
FINAL_CONFIG = {
    'dataset': dict(passive_method='static', passive_gain=True,
                    train_label_rezero=False, emg_clip=None, emg_floor=None,
                    position_feature='raw', emg_agc_norm='auto'),
    'train': dict(position_balance='scale', anchored=False, moe=True,
                  moe_expert_bias=False),
}

CONFIGS = {'legacy': LEGACY_CONFIG, 'final': FINAL_CONFIG}

PROTOCOLS = {
    'orig': dict(batch_size=8, lr=0.003, epochs=200),     # ML.config defaults
    'fast': dict(batch_size=64, lr=0.003, epochs=150),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subject', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--config', default='final',
                   help="'legacy' | 'final' | JSON {'dataset':{...}, 'train':{...}}")
    p.add_argument('--protocol', default='orig', choices=list(PROTOCOLS))
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--no-slot', action='store_true')
    p.add_argument('--epochs', type=int, default=None, help='override (smoke tests)')
    a = p.parse_args()

    from ML.experiments.run import acquire_slot
    _slot = None if a.no_slot else acquire_slot()

    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(a.threads)
    tf.config.threading.set_inter_op_parallelism_threads(max(1, a.threads // 2))
    import ML.config as cfg
    from ML.training import load_subject, train_model, evaluate_model
    from ML.experiments.harness import RESULTS_DIR

    conf = CONFIGS[a.config] if a.config in CONFIGS else json.loads(a.config)
    prot = dict(PROTOCOLS[a.protocol])
    if a.epochs:
        prot['epochs'] = a.epochs
    if a.protocol == 'fast':      # same schedule as the harness 'fast' protocol
        cfg.LR_FACTOR, cfg.LR_PATIENCE, cfg.EARLY_PATIENCE = 0.5, 6, 15
        import ML.training as tr
        tr.LR_FACTOR, tr.LR_PATIENCE, tr.EARLY_PATIENCE = 0.5, 6, 15

    data = load_subject(a.subject, **conf['dataset'])
    out = {'subject': a.subject, 'name': a.name, 'config': conf,
           'protocol': a.protocol, 'seeds': []}
    preds = []
    for sd in a.seeds:
        tf.keras.utils.set_random_seed(sd)
        t0 = time.time()
        model, hist = train_model(data['X_train'], data['y_train'],
                                  data['X_val'], data['y_val'],
                                  label=f'{a.subject}/{a.name}/seed{sd}',
                                  lr=prot['lr'], epochs=prot['epochs'],
                                  batch_size=prot['batch_size'], verbose=0,
                                  **conf['train'])
        res = evaluate_model(model, data['X_test'], data['y_test'],
                             data['operating_positions'],
                             mvc_tq_test=data['mvc_tq_test'])
        preds.append(res['y_pred'])
        pp = {f'{k:+.3f}': {m: float(v) for m, v in d.items()}
              for k, d in res['per_position'].items()}
        out['seeds'].append({'seed': sd, 'epochs': len(hist.history['loss']),
                             'train_s': time.time() - t0,
                             'overall': {m: float(v) for m, v in res['overall'].items()},
                             'per_position': pp})
        print(f"[{a.subject}/{a.name}] seed {sd}: pooled R² {res['overall']['r2']:.4f}  "
              + '  '.join(f"{k}:{v['r2']:+.3f}" for k, v in pp.items()), flush=True)
    keys = list(out['seeds'][0]['per_position'])
    out['summary'] = {k: {m: float(np.mean([s['per_position'][k][m] for s in out['seeds']]))
                          for m in ('r2', 'rmse', 'mae', 'vaf')} for k in keys}
    for k in keys:
        out['summary'][k]['mse'] = float(np.mean([s['per_position'][k]['rmse'] ** 2
                                                   for s in out['seeds']]))
        out['summary'][k]['r2_std'] = float(np.std([s['per_position'][k]['r2']
                                                     for s in out['seeds']]))
    out['summary']['pooled'] = {
        'r2': float(np.mean([s['overall']['r2'] for s in out['seeds']])),
        'rmse': float(np.mean([s['overall']['rmse'] for s in out['seeds']])),
        'mse': float(np.mean([s['overall']['rmse'] ** 2 for s in out['seeds']])),
    }
    os.makedirs(os.path.join(RESULTS_DIR, 'pipeline'), exist_ok=True)
    path = os.path.join(RESULTS_DIR, 'pipeline', f'{a.subject}__{a.name}__{a.protocol}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print('pooled R² (mean over seeds):', out['summary']['pooled']['r2'], '->', path, flush=True)


if __name__ == '__main__':
    main()
