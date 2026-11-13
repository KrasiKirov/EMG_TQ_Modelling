"""Shuffling on (N0) vs off (S0_noshuffle): same settings, 6 people x 3 seeds, session 1 -> session 2.

    python extract4/w03_shuffle.py       # .venv311, PYTHONPATH = repository root

Scores both ways the study uses: the whole trial (dashboard rule) and, after the 10 s calibration, from 30 s on.
Medians over seeds per (person, angle), then over cells; R2 below 0 shown as 0. Also epochs run / kept.
Writes data4/facts_w03.json.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

VIDEO = Path(__file__).resolve().parents[1]
ROOT = VIDEO.parents[1]
sys.path.insert(0, str(ROOT / 'EXPERIMENTS' / 'normalization'))
from calibration_test import correct            # noqa: E402

RES = ROOT / 'EXPERIMENTS' / 'results'


def r2(y, p):
    return 1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)


def scores(exp):
    cells = {}
    for f in sorted(glob.glob(str(RES / exp / '*' / 'S1toS2' / 'seed*.npz'))):
        subj = Path(f).parts[-3]
        z = np.load(f)
        y, p, ang, t = (z[k].astype(float) for k in ('y_own_target', 'y_pred', 'angle', 't'))
        angles = sorted(np.unique(ang))
        for a in angles:
            m = ang == a
            yy, pp, tt = y[m], p[m], t[m] - t[m][0]
            cal, sc = tt < 10, tt >= 30
            pc = correct(pp[cal], yy[cal], pp, 'gain+offset')
            key = (subj, round(float(a), 3), int(a in angles[:2]))
            cells.setdefault(key, []).append((np.mean(abs(pp - yy)), max(r2(yy, pp), 0),
                                              np.mean(abs(pc - yy)[sc]), max(r2(yy[sc], pc[sc]), 0)))
    out = {}
    for toes, name in ((1, 'toes'), (0, 'other')):
        med = np.array([np.median(v, axis=0) for k, v in cells.items() if k[2] == toes])
        out[name] = dict(mae=round(float(np.median(med[:, 0])), 3), r2=round(float(np.median(med[:, 1])), 3),
                         mae_cal=round(float(np.median(med[:, 2])), 3), r2_cal=round(float(np.median(med[:, 3])), 3))
    runs = [json.loads(Path(f).read_text()) for f in glob.glob(str(RES / exp / '*' / 'S1toS2' / 'seed*.json'))]
    ep = [r['train_info']['epochs'] for r in runs]
    best = [r['train_info']['best_epoch'] for r in runs]
    out['runs'] = len(runs)
    out['epochs'] = [min(ep), int(np.median(ep)), max(ep)]
    out['best_epoch'] = [min(best), int(np.median(best)), max(best)]
    return out, cells


def main():
    a, ca = scores('N0_total_trainmax')
    b, cb = scores('S0_noshuffle')
    per_person = {}
    for subj in sorted({k[0] for k in ca}):
        for toes, name in ((1, 'toes'), (0, 'other')):
            va = np.median([np.median(v, axis=0)[0] for k, v in ca.items() if k[0] == subj and k[2] == toes])
            vb = np.median([np.median(v, axis=0)[0] for k, v in cb.items() if k[0] == subj and k[2] == toes])
            per_person.setdefault(subj, {})[name] = [round(float(va), 2), round(float(vb), 2)]
    facts = dict(shuffle_on=a, shuffle_off=b, per_person_mae_on_off=per_person,
                 note='shuffle_on = N0_total_trainmax; shuffle_off = S0_noshuffle (identical settings otherwise)')
    (VIDEO / 'data4' / 'facts_w03.json').write_text(json.dumps(facts, indent=1), encoding='utf-8')
    print(json.dumps(facts, indent=1))


if __name__ == '__main__':
    main()
