"""Where in a push does the final model (N0, before calibration) make its errors?

    python extract4/w01_phases.py        # .venv311, PYTHONPATH = repository root

Each session-2 trial's measured torque is split into phases: rest, onset (first 0.3 s of a push), plateau, and
release (the 0.5 s after a push ends). A push = |torque - rest level| above 25 % of the trial's swing, where the
rest level is the most common torque value of the trial. The error is measured after removing each trial's mean
offset (so the phases are compared on timing and shape, not on the session offset), plus the raw error.
Also: the time lag that best lines the prediction up with the measured torque (cross-correlation, +-0.5 s).
Writes data4/w01.npz (one example trial) and data4/facts_w01.json.
"""
import glob
import json
from pathlib import Path

import numpy as np

VIDEO = Path(__file__).resolve().parents[1]
RES = VIDEO.parents[0] / 'results' / 'N0_total_trainmax'
FS = 100


def phases(y):
    """0 rest, 1 onset, 2 plateau, 3 release, -1 unassigned (push gaps shorter than the margins)."""
    h, e = np.histogram(y, bins=60)
    rest = (e[np.argmax(h)] + e[np.argmax(h) + 1]) / 2
    act = np.abs(y - rest) > 0.25 * np.ptp(y)
    lab = np.zeros(len(y), int)
    d = np.diff(act.astype(int), prepend=0, append=0)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    for s, e_ in zip(starts, ends):
        lab[s:e_] = 2
        lab[s:min(e_, s + int(0.3 * FS))] = 1
        lab[e_:min(len(y), e_ + int(0.5 * FS))] = 3
    return lab, rest


def best_lag(y, p, max_s=0.5):
    y, p = y - y.mean(), p - p.mean()
    lags = np.arange(-int(max_s * FS), int(max_s * FS) + 1)
    c = [np.dot(y[max(0, -k):len(y) - max(0, k)], p[max(0, k):len(p) - max(0, -k)]) for k in lags]
    return lags[int(np.argmax(c))] / FS          # > 0: the prediction comes later than the torque


def main():
    names = ['rest', 'onset', 'plateau', 'release']
    cells = {n: [] for n in names}
    cells_raw = {n: [] for n in names}
    lags, example = [], None
    for f in sorted(glob.glob(str(RES / '*' / 'S1toS2' / 'seed*.npz'))):
        z = np.load(f)
        y, p, ang = (z[k].astype(float) for k in ('y_own_target', 'y_pred', 'angle'))
        subj = Path(f).parts[-3]
        for a in np.unique(ang):
            m = ang == a
            yy, pp = y[m], p[m]
            if np.ptp(yy) < 2:                                     # nearly flat trial: no clear pushes
                continue
            lab, rest = phases(yy)
            e_shape = np.abs((pp - pp.mean()) - (yy - yy.mean()))
            e_raw = np.abs(pp - yy)
            for k, n in enumerate(names):
                if (lab == k).sum() > 50:
                    cells[n].append(float(np.mean(e_shape[lab == k])))
                    cells_raw[n].append(float(np.mean(e_raw[lab == k])))
            lags.append(best_lag(yy, pp))
            if subj == 'EG' and 'seed0' in f and abs(a - (-0.038)) < 0.02:
                example = dict(y=yy, p=pp, lab=lab, t=np.arange(len(yy)) / FS)
    facts = dict(
        trials=len(lags), source='N0_total_trainmax, session 2, 6 people x 3 seeds (trials with a torque swing >= 2 Nm)',
        shape_mae_median={n: round(float(np.median(v)), 3) for n, v in cells.items()},
        raw_mae_median={n: round(float(np.median(v)), 3) for n, v in cells_raw.items()},
        lag_s=dict(median=round(float(np.median(lags)), 3), p5=round(float(np.percentile(lags, 5)), 3),
                   p95=round(float(np.percentile(lags, 95)), 3),
                   share_within_50ms=round(float(np.mean(np.abs(lags) <= 0.05)), 3)))
    if example is not None:
        np.savez_compressed(VIDEO / 'data4' / 'w01.npz', **{f'ex_{k}': v for k, v in example.items()})
    (VIDEO / 'data4' / 'facts_w01.json').write_text(json.dumps(facts, indent=1), encoding='utf-8')
    print(json.dumps(facts, indent=1))


if __name__ == '__main__':
    main()
