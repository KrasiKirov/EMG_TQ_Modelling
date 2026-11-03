"""Where in a push does the final model (N0, before calibration) make its errors?

    python extract4/w01_phases.py        # .venv311, PYTHONPATH = repository root

Each session-2 trial is split into phases from the MUSCLE ACTIVITY (the EMG at the end of each window, as the
model sees it): a push = the summed envelope of the four muscles above 30 % of the way from its 5th to its 95th
percentile in that trial, kept only if it lasts at least 0.3 s. Phases: rest, onset (first 0.3 s of a push),
plateau, release (the 0.5 s after a push ends). The error is measured after removing each trial's mean
offset (so the phases are compared on timing and shape, not on the session offset), plus the raw error.
Also: the time lag that best lines the prediction up with the measured torque (cross-correlation, +-0.5 s).
Writes data4/w01.npz (one example trial) and data4/facts_w01.json.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

VIDEO = Path(__file__).resolve().parents[1]
RES = VIDEO.parents[0] / 'results' / 'N0_total_trainmax'
FS = 100


def phases(emg):
    """0 rest, 1 onset, 2 plateau, 3 release, from the summed EMG envelope at each window's last sample."""
    lo, hi = np.percentile(emg, [5, 95])
    act = emg > lo + 0.3 * (hi - lo)
    d0 = np.diff(act.astype(int), prepend=0, append=0)
    for s, e_ in zip(np.where(d0 == 1)[0], np.where(d0 == -1)[0]):
        if e_ - s < int(0.3 * FS):
            act[s:e_] = False
    lab = np.zeros(len(emg), int)
    d = np.diff(act.astype(int), prepend=0, append=0)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    for s, e_ in zip(starts, ends):
        lab[s:e_] = 2
        lab[s:min(e_, s + int(0.3 * FS))] = 1
        lab[e_:min(len(emg), e_ + int(0.5 * FS))] = 3
    return lab


def best_lag(y, p, max_s=0.5):
    y, p = y - y.mean(), p - p.mean()
    lags = np.arange(-int(max_s * FS), int(max_s * FS) + 1)
    c = [np.dot(y[max(0, -k):len(y) - max(0, k)], p[max(0, k):len(p) - max(0, -k)]) for k in lags]
    return lags[int(np.argmax(c))] / FS          # > 0: the prediction comes later than the torque


def main():
    sys.path.insert(0, str(VIDEO.parents[1]))
    sys.path.insert(0, str(VIDEO.parents[0] / 'code'))
    from run_experiment import load_config
    from ML.research.run import prepare
    _, cfg = load_config('N0_total_trainmax')
    emg_cache = {}
    names = ['rest', 'onset', 'plateau', 'release']
    cells = {n: [] for n in names}
    cells_raw = {n: [] for n in names}
    lags, example = [], None
    for f in sorted(glob.glob(str(RES / '*' / 'S1toS2' / 'seed*.npz'))):
        z = np.load(f)
        y, p, ang = (z[k].astype(float) for k in ('y_own_target', 'y_pred', 'angle'))
        subj = Path(f).parts[-3]
        if subj not in emg_cache:
            data, _ = prepare(subj, 'S1toS2', {k: cfg[k] for k in ('passive', 'emg', 'window')})
            emg_cache[subj] = data['X_test'][:, -1, :4].sum(1)
        emg = emg_cache[subj]
        assert len(emg) == len(y), 'saved predictions and windows must line up'
        for a in np.unique(ang):
            m = ang == a
            yy, pp = y[m], p[m]
            if np.ptp(yy) < 2:                                     # nearly flat trial: no clear pushes
                continue
            lab = phases(emg[m])
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
        trials=len(lags), phases_from='summed EMG envelope of the 4 muscles (window end)', source='N0_total_trainmax, session 2, 6 people x 3 seeds (trials with a torque swing >= 2 Nm)',
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
