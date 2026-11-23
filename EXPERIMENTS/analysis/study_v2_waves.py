"""Measured torque against the best and the worst seed at each person's worst positions, for study v2's settings.

    python -m EXPERIMENTS.analysis.study_v2_waves [--out EXPERIMENTS/dashboard_react/src/data/study-v2-waves.json]

Same file format as the retired v1 page's waveforms (helpers in common.py), for the page's 18 study v2 settings. The scores come
from study_v2_export.final_scores, i.e. on confirm's own samples (from the shared scoring start), so the waveform
headers agree with the page's R² boxes: the person's worst positions are the WORST positions with the lowest median R²
over seeds; at each, the best and the worst seed have the highest and the lowest R² there (one seed: both are it).
The curves cover the whole trial. Session-2 predictions are read through confirm (protocol/lock.py) only.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from EXPERIMENTS.analysis import confirm as CF, study_v2_export as EX
from EXPERIMENTS.analysis.common import MISSING, SLOTS, STEP, WORST, envelope, packed
from EXPERIMENTS.paths import RESULTS_V2
from EXPERIMENTS.protocol import protocol as PT

OUT = EX.OUT.parent / 'study-v2-waves.json'


def _person(f: dict, runs: dict, e: str, p: str, d: str) -> dict:
    r2 = np.array([[np.nan if v is None else v for v in row] for row in f['r2']], float)
    med = np.nanmedian(r2, axis=1)
    worst_k = sorted((k for k in range(len(med)) if np.isfinite(med[k])), key=lambda k: (med[k], k))[:WORST]
    curves = {name: ([], []) for name in ('measured', 'best', 'worst')}
    info = []
    for k in worst_k:
        have = sorted((j for j in range(len(f['seeds'])) if np.isfinite(r2[k, j])), key=lambda j: (r2[k, j], j))
        lo, hi = have[0], have[-1]
        entry = dict(k=k, pos=f['positions'][k], median_r2=round(float(med[k]), 4))
        for name, j, use_y in (('measured', hi, True), ('best', hi, False), ('worst', lo, False)):
            z = runs[e, p, d, f['seeds'][j]][1]
            trial = EX.trial_order(z)[k][1]
            m = np.asarray(z['trial']) == trial
            ts = np.asarray(z['t'], float)[m]
            v, o = envelope(ts, np.asarray(z['y_own_target' if use_y else 'y_pred'], float)[m])
            curves[name][0].append(v), curves[name][1].append(o)
            if name != 'measured':
                entry[name] = dict(seed=f['seeds'][j], mae=round(f['mae'][k][j], 4), r2=round(float(r2[k, j]), 4))
        info.append(entry)
    out = dict(seeds=len(f['seeds']), positions=info)       # not 'worst': that key holds the worst seed's curve
    for name, (vs, os) in curves.items():
        out[name] = packed(np.concatenate(vs))
        out[name + '_t'] = base64.b64encode(np.concatenate(os).tobytes()).decode()
    return out


def build(rd=None) -> dict:
    rd = Path(rd or RESULTS_V2)
    loaded = CF.final_runs(rd)
    final, _ = EX.final_scores(rd, loaded)
    d = PT.load()['primary_direction']
    out = dict(slots=list(SLOTS), step=STEP, scale=0.01, missing=MISSING, direction=d,
               rule=f'the {WORST} positions with the lowest median R² over seeds; best and worst = highest and '
                    'lowest R² there', runs={})
    for e, per in final.items():
        out['runs'][e] = {p: _person(f, loaded[0], e, p, d) for p, f in per.items()}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args(argv)
    data = build()
    a.out.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    print(f'{a.out}: {len(data["runs"])} settings, {a.out.stat().st_size / 2 ** 20:.1f} MiB')


if __name__ == '__main__':
    main()
