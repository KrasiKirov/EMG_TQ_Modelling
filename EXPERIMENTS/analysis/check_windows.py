"""Check that the window study's window lengths, steps and overlaps are built as their specs say, on real data.

    python EXPERIMENTS/analysis/check_windows.py [--subject EG]

For every window-study spec it builds the arrays exactly as EXPERIMENTS/pipeline/run.py does and checks:
  1. each window has W steps spaced `input_step` samples apart (shifting the start by one step moves every slot by one);
  2. the first test prediction is at 1 s (start cut) + the window span, i.e. the history length is right;
  3. training windows start `train_stride` samples apart; validation and test windows 1 sample apart (scoring identical);
  4. when the stride is a multiple of the step: consecutive training windows share exactly W - stride/step steps,
     so the overlap is (span - stride) / span.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from EXPERIMENTS.pipeline import run as R  # noqa: E402
from EXPERIMENTS.pipeline.spec import load_all  # noqa: E402
from ML.research.run import prepare  # noqa: E402

KEYS = ('passive', 'emg', 'window', 'envelope', 'trim_end_s', 'input_step', 'train_stride', 'state', 'lp_order', 'val_gap', 'metadata_passive')


def check(sid, subject):
    _, c = R.load_config(sid)
    d, _ = prepare(subject, 'S1toS2', {k: c[k] for k in KEYS})
    d = R.drop_last_val_window(d)
    X, m, T, mt = d['X_train'], d['meta_train'], d['X_test'], d['meta_test']
    W, step, k = X.shape[1], c['input_step'], c['train_stride']
    res = {}
    i = np.where(mt['trial'][:-step] == mt['trial'][step:])[0][:2000]
    res['step spacing'] = np.array_equal(T[i + step, :-1], T[i, 1:])
    res['history length'] = abs(mt['t'].min() - (1.0 + (W - 1) * step / 100)) < 1e-6
    same = m['trial'][1:] == m['trial'][:-1]
    same_t = mt['trial'][1:] == mt['trial'][:-1]
    res['training stride'] = set(np.diff(m['last'])[same]) == {k} and set(np.diff(mt['last'])[same_t]) == {1}
    span = W * step
    if k % step == 0 and k < span:              # steps line up: the next window repeats the last W - k/step steps
        j = np.where(same)[0][:2000]
        res['shared steps'] = np.array_equal(X[j + 1, :W - k // step], X[j, k // step:])
    if k == 1 and step > 1:
        overlap = f'starts 10 ms apart on a {step * 10} ms grid'
    else:
        overlap = f'{(span - k) / span:.0%} of the {span * 10} ms span'
    return W, step, k, overlap, res, len(X)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subject', default='EG')
    a = ap.parse_args()
    bad = 0
    for s in load_all():
        if not s.id.startswith('W') or s.raw['changes'].get('state') == 'carry' or s.id.startswith('W00b8'):
            continue
        W, step, k, ov, res, n = check(s.id, a.subject)
        fails = [name for name, ok in res.items() if not ok]
        bad += bool(fails)
        print(f'{s.id:16s} {W:4d} steps x {step * 10:3d} ms = {W * step / 100:5.2f} s, stride {k:2d}, overlap {ov:30s}'
              f' train windows {n:6d}  {"OK" if not fails else "FAIL: " + ", ".join(fails)}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
