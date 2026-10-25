"""
Check that the experiment framework (ML/research, config E01_old) builds EXACTLY the
same training / validation / test data as the frozen original pipeline.

    python EXPERIMENTS/code/check_equivalence.py dump-original YES out.npz
    python EXPERIMENTS/code/check_equivalence.py dump-research YES out.npz
    python EXPERIMENTS/code/check_equivalence.py compare a.npz b.npz

If the arrays match, every experiment run through ML/research differs from the
baseline ONLY in the one thing the experiment changes.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
mode = sys.argv[1]

if mode == 'dump-original':
    sys.path.insert(0, os.path.join(EXP, 'baseline', 'original_code'))
    import ML.config as cfg
    if os.environ.get('EMG_DATA_DIR'):
        cfg.DATA_DIR = os.environ['EMG_DATA_DIR']
    from ML.training import load_subject
    d = load_subject(sys.argv[2])
    np.savez(sys.argv[3], **{k: d[k] for k in ('X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test')})
elif mode == 'dump-research':
    sys.path.insert(0, os.path.dirname(EXP))
    from ML.research.run import prepare
    from ML.research import experiments
    data, _ = prepare(sys.argv[2], 'S1toS2', experiments.get('E01_old'))
    np.savez(sys.argv[3], **{k: data[k] for k in ('X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test')})
elif mode == 'compare':
    a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
    ok = True
    for k in ('X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'):
        same_shape = a[k].shape == b[k].shape
        diff = float(np.max(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)))) if same_shape else float('nan')
        print(f'{k:8s} original {a[k].shape} research {b[k].shape} max|diff| {diff:.3g}')
        ok &= same_shape and diff < 1e-4
    print('EQUIVALENT' if ok else 'NOT EQUIVALENT')
