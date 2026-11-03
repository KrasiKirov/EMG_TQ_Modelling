"""One training batch up close: 8 fit windows dealt by a shuffle, run through the saved final model (V0, EG seed 0).

    python extract4/w04_batch.py         # .venv311, PYTHONPATH = repository root, EMG_DATA_DIR set

The 8 windows are the first 8 of a random permutation of EG's session-1 fit windows (what a shuffled epoch deals
first). For each: its angle, its time in the trial, the 20 x 5 inputs, the measured torque at its last sample and
the model's prediction (both in Nm). Writes data4/w04.npz and data4/facts_w04.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

VIDEO = Path(__file__).resolve().parents[1]
ROOT = VIDEO.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'EXPERIMENTS' / 'code'))
RES = ROOT / 'EXPERIMENTS' / 'results'


def main():
    from run_experiment import load_config, make_builder
    from ML.research.run import prepare
    from ML.research import train as RT

    _, cfg = load_config('V0_saved_weights')
    data, _ = prepare('EG', 'S1toS2', {k: cfg[k] for k in ('passive', 'emg', 'window')})
    info = json.loads((RES / 'V0_saved_weights' / 'EG' / 'S1toS2' / 'seed0.json').read_text())
    knots, sd = np.array(info['train_info']['knots']), np.array(info['train_info']['sd_per_knot'])
    model = make_builder(cfg)('lstm', 20, 5, knots)
    model.load_weights(str(RES / 'V0_saved_weights' / 'EG' / 'S1toS2' / 'seed0.weights.h5'))

    X, y, meta = data['X_train'], data['y_train'], data['meta_train']
    ang = meta['angle']
    t0 = {a: meta['t'][ang == a].min() for a in np.unique(ang)}
    pick = np.random.default_rng(0).permutation(len(X))[:8]
    scale = sd[RT.snap(ang[pick], knots)]
    pred = model.predict(X[pick], verbose=0).ravel() * scale
    order = sorted(np.unique(ang))
    np.savez_compressed(VIDEO / 'data4' / 'w04.npz', win=X[pick], y=y[pick], pred=pred, angle=ang[pick],
                        t=np.array([meta['t'][i] - t0[ang[i]] for i in pick]),
                        angle_rank=np.array([order.index(a) for a in ang[pick]]))
    facts = dict(n_fit=int(len(X)), batch=8, batches_per_epoch=int(np.ceil(len(X) / 8)),
                 n_params=info['train_info']['n_params'],
                 windows=[dict(angle_rad=round(float(ang[i]), 3), t_s=round(float(meta['t'][i] - t0[ang[i]]), 2),
                               torque_Nm=round(float(y[i]), 2), pred_Nm=round(float(p), 2))
                          for i, p in zip(pick, pred)])
    (VIDEO / 'data4' / 'facts_w04.json').write_text(json.dumps(facts, indent=1), encoding='utf-8')
    print(json.dumps(facts, indent=1))


if __name__ == '__main__':
    main()
