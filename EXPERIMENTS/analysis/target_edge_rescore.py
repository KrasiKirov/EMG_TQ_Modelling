"""Deviation 2026-11-23: how much the target-edge fix changes the study v2 session-2 scores.

The torque target used to be down-sampled (zero-phase FIR) after the trims, so its first and last ~50 ms were wrong.
ML/research/data.py now down-samples first (CACHE_VERSION 2). This script re-scores the saved session-2 predictions of
every final (FE_*) experiment against the corrected target and prints MAE and R² before and after, per experiment and
person (median over seeds). The predictions themselves are unchanged: models were trained on the old targets.

    PYTHONPATH=. .venv/bin/python EXPERIMENTS/analysis/target_edge_rescore.py
"""
import glob
import os
import pickle

import numpy as np

from ML.research import config as C
from ML.research import data as D

PEOPLE = ['CH', 'EG', 'HM', 'IES', 'JM', 'YES']
OLD = os.path.join(C.CACHE_DIR, '{}_bundle_v1_causal_end1_lp2.pkl')


def deltas(person):
    old = pickle.load(open(OLD.format(person), 'rb'))
    new = D.load_subject(person, envelope='causal', trim_end_s=1.0, lp_order=2)
    out = {}
    for a, b in zip(old['active'], new['active']):
        assert a['idx'] == b['idx'] and np.array_equal(a['t'], b['t']) and np.array_equal(a['env_raw'], b['env_raw'])
        for t, d, q, ps in zip(a['t'], b['tq'] - a['tq'], a['tq'], a['pos']):
            out[(a['idx'], round(float(t), 2))] = (float(d), float(q), float(ps))
    return out


def scores(y, p):
    return np.mean(np.abs(y - p)), 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def main():
    rows = []
    for person in PEOPLE:
        dl = deltas(person)
        for exp in sorted(glob.glob('EXPERIMENTS/results_v2/FE_*')):
            per_seed = []
            for f in sorted(glob.glob(f'{exp}/{person}/S1toS2/seed*[0-9].npz')):
                z = np.load(f)
                keys = [(int(i), round(float(t), 2)) for i, t in zip(z['trial'], z['t'])]
                y = z['y_own_target'].astype(float); p = z['y_pred'].astype(float)
                d = np.array([dl[k][0] for k in keys]); q = np.array([dl[k][1] for k in keys]); ps = np.array([dl[k][2] for k in keys])
                off = y - q   # target = measured torque - passive(angle); the angle is unchanged, so the fix adds d
                for tr in np.unique(z['trial']):
                    m = z['trial'] == tr
                    res = off[m] - np.polyval(np.polyfit(ps[m], off[m], 1), ps[m])   # passive(angle): close to linear over one trial
                    assert np.ptp(res) < 0.1, (f, tr, np.ptp(res))
                per_seed.append((*scores(y, p), *scores(y + d, p), int((np.abs(d) > 1e-6).sum()), len(y)))
            if per_seed:
                rows.append((os.path.basename(exp), person, *np.median(np.array(per_seed), axis=0)))
    print(f"{'experiment':22s} {'who':4s} {'MAE old':>8s} {'MAE new':>8s} {'change':>7s} {'R2 old':>7s} {'R2 new':>7s} {'changed samples':>16s}")
    for e, who, m0, r0, m1, r1, nc, n in rows:
        print(f'{e:22s} {who:4s} {m0:8.4f} {m1:8.4f} {100 * (m1 / m0 - 1):+6.2f}% {r0:7.4f} {r1:7.4f} {int(nc):7d} / {int(n)}')
    ch = np.array([100 * (r[4] / r[2] - 1) for r in rows]); dr = np.array([r[5] - r[3] for r in rows])
    print(f'MAE change over all experiment x person rows: median {np.median(ch):+.3f} %, range {ch.min():+.3f} to {ch.max():+.3f} %')
    print(f'R2 change: median {np.median(dr):+.5f}, range {dr.min():+.5f} to {dr.max():+.5f}')


if __name__ == '__main__':
    main()
