"""Measured torque against the best and the worst run at each person's worst positions, for the Results page.

    python -m EXPERIMENTS.analysis.study_waves [--out EXPERIMENTS/dashboard_react/src/data/study-waves.json]

For every window-study setting in study.json and each person with saved session 1 -> 2 predictions, every
position of every seed is scored with pipeline.scoring.score_trial (the page's own numbers). The person's worst
positions are the 2 with the lowest median R² over seeds. At each of them the best and the worst run are the seeds
with the highest and the lowest R² there.

Each curve keeps real 10 ms samples only: the trial is cut into 0.2 s slots (about one pixel of the chart) and the
lowest and the highest sample of each slot are kept at their own times, in time order. So every drawn point is a
saved value, and every peak and dip of the full curve is drawn. Values: little-endian int16 in 0.01 Nm
(-32768 = empty slot); times: uint8 offset in 10 ms steps from the slot's start; both base64.

The predictions are not in git, so settings without local predictions keep what the existing file has.
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np

from EXPERIMENTS.pipeline.scoring import score_trial

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'EXPERIMENTS/dashboard_react/src/data'
RESULTS = ROOT / 'EXPERIMENTS/results'
DIRECTION = 'S1toS2'
SLOTS = (1.0, 0.2, 441)         # first slot start (s), slot width (s), slots: covers each trial's 1.2-89.1 s
STEP = 0.01                     # sample period of the saved predictions (s)
MISSING = -32768
WORST = 2                       # positions shown per person


def envelope(ts, v):
    """per slot, the lowest and the highest sample in time order: (values, offsets in 10 ms steps)"""
    t0, width, n = SLOTS
    if np.any(np.diff(ts) > 1.5 * STEP) or np.any(np.diff(ts) <= 0):
        raise ValueError('trial has gaps or unsorted times; the slot format assumes one sample every 10 ms')
    per = int(round(width / STEP))
    tick = np.round((ts - t0) / STEP).astype(int)          # samples since the first slot start (times are float32)
    if np.any(np.abs(tick * STEP + t0 - ts) > STEP / 4):
        raise ValueError('sample times are not on the 10 ms grid')
    slot, off = tick // per, tick % per
    if slot.min() < 0 or slot.max() >= n:
        raise ValueError('trial runs outside the slot range')
    vals = np.full(2 * n, np.nan)
    offs = np.zeros(2 * n, dtype=np.uint8)
    for s in np.unique(slot):
        idx = np.flatnonzero(slot == s)
        lo, hi = idx[np.argmin(v[idx])], idx[np.argmax(v[idx])]
        for j, i in enumerate(sorted((lo, hi))):
            vals[2 * s + j] = v[i]
            offs[2 * s + j] = off[i]
    return vals, offs


def packed(a):
    q = np.where(np.isfinite(a), np.round(np.nan_to_num(a) * 100), MISSING)
    if np.any(q[q != MISSING] <= MISSING) or np.any(q > 32767):
        raise ValueError('torque outside the int16 range of 0.01 Nm')
    return base64.b64encode(q.astype('<i2').tobytes()).decode()


def trials(f, positions):
    """per study position: (measured, predicted, time, score row) of one saved run, or None when it has no trial there"""
    z = np.load(f, allow_pickle=False)
    y, p, ang, ts = (z[k].astype(float) for k in ('y_own_target', 'y_pred', 'angle', 't'))
    out = []
    for pos in positions:
        m = np.isclose(np.round(ang, 3), pos, atol=1e-3)
        out.append((y[m], p[m], ts[m], score_trial(y[m], p[m], ts[m])) if m.any() else None)
    return out


def person(files, positions):
    seeds = {int(f.stem[4:]): trials(f, positions) for f in files}
    r2 = lambda sd, k: seeds[sd][k][3]['r2'] if seeds[sd][k] else np.nan
    med = [float(np.nanmedian([r2(sd, k) for sd in seeds])) if any(seeds[sd][k] for sd in seeds) else np.nan
           for k in range(len(positions))]
    worst_k = sorted((k for k in range(len(positions)) if np.isfinite(med[k])), key=lambda k: med[k])[:WORST]
    curves = {name: ([], []) for name in ('measured', 'best', 'worst')}
    info = []
    for k in worst_k:
        have = sorted((sd for sd in seeds if seeds[sd][k]), key=lambda sd: (r2(sd, k), sd))
        lo, hi = have[0], have[-1]
        entry = dict(k=k, pos=positions[k], median_r2=round(med[k], 4))
        for name, sd, use_y in (('measured', hi, True), ('best', hi, False), ('worst', lo, False)):
            yy, pp, ts, row = seeds[sd][k]
            v, o = envelope(ts, yy if use_y else pp)
            curves[name][0].append(v), curves[name][1].append(o)
            if name != 'measured':
                entry[name] = dict(seed=sd, mae=round(row['mae'], 4), r2=round(row['r2'], 4))
        info.append(entry)
    out = dict(seeds=len(seeds), positions=info)       # not 'worst': that key holds the worst run's curve
    for name, (vs, os) in curves.items():
        out[name] = packed(np.concatenate(vs))
        out[name + '_t'] = base64.b64encode(np.concatenate(os).tobytes()).decode()
    return out


def build(old):
    study = json.loads((DATA / 'study.json').read_text(encoding='utf-8'))
    out = dict(slots=list(SLOTS), step=STEP, scale=0.01, missing=MISSING, direction=DIRECTION,
               rule=f'the {WORST} positions with the lowest median R² over seeds; best and worst = highest and lowest R² there',
               runs={})
    if old and old.get('slots') == list(SLOTS):
        out['runs'] = {k: v for k, v in old.get('runs', {}).items() if k in study['settings']}
    for ident in study['settings']:
        found = {}
        for subj, positions in study['positions'].items():
            files = sorted((RESULTS / ident / subj / DIRECTION).glob('seed*.npz'))
            if files:
                found[subj] = person(files, positions)
        if found:
            out['runs'][ident] = found
            print(f'{ident}: {len(found)} people')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=DATA / 'study-waves.json')
    a = ap.parse_args()
    old = json.loads(a.out.read_text(encoding='utf-8')) if a.out.exists() else None
    data = build(old)
    a.out.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    print(f'{a.out}: {len(data["runs"])} settings, {a.out.stat().st_size / 2**20:.1f} MiB')


if __name__ == '__main__':
    main()
