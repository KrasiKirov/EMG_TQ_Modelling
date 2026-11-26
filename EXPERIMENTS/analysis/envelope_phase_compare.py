"""Deviation 2026-11-23: what the causal (real-time) envelope costs against a zero-phase one, for A0.

Compares A0 (causal envelope; seeds 10-12 are the FE_cell_carry_2s runs, 13-19 FE_A0) with FE_A0_zerophase (the same
setting with a zero-phase envelope), seed by seed, on session 2. Descriptive only, not a pre-registered comparison.

Both arms are scored against the corrected target (CACHE_VERSION 2: A0 was saved with the old target, whose first and
last ~50 ms per trial differ; see target_edge_rescore.py), on the samples both arms predict, from 1 s into the trial.
nMAE = MAE / the 99th percentile of |target| over the scored samples (one scale per person, shared by both arms).
Lag = the shift (10 ms steps, 0-500 ms) of the prediction that best lines it up with the target, over all positions.

    PYTHONPATH=. .venv/bin/python EXPERIMENTS/analysis/envelope_phase_compare.py
"""
import os

import numpy as np

from ML.research import data as D

RD = 'EXPERIMENTS/results_v2'
PEOPLE = ['CH', 'EG', 'HM', 'IES', 'JM', 'YES']
SEEDS = range(10, 20)
SCORE_FROM_S = 1.0


def causal_path(person, seed):
    exp = 'FE_cell_carry_2s' if seed < 13 else 'FE_A0'
    return f'{RD}/{exp}/{person}/S1toS2/seed{seed}.npz'


def zero_path(person, seed):
    return f'{RD}/FE_A0_zerophase/{person}/S1toS2/seed{seed}.npz'


def keyed(z):
    return {(int(i), round(float(t), 2)): k for k, (i, t) in enumerate(zip(z['trial'], z['t']))}


def lag_ms(y, p, trial):
    """Best shift of p behind y (10 ms steps), summed over trials so no shift crosses a trial boundary."""
    best = []
    for L in range(0, 51):
        s = 0.0
        for tr in np.unique(trial):
            a, b = y[trial == tr], p[trial == tr]
            a, b = a - a.mean(), b - b.mean()
            s += np.dot(a[:len(a) - L], b[L:]) if L else np.dot(a, b)
        best.append(s)
    return 10 * int(np.argmax(best))


def main():
    rows = []
    for person in PEOPLE:
        b = D.load_subject(person, envelope='causal')
        t0 = {r['idx']: float(r['t'][0]) for r in b['active']}
        for seed in SEEDS:
            if not (os.path.exists(causal_path(person, seed)) and os.path.exists(zero_path(person, seed))):
                continue
            zc, zz = np.load(causal_path(person, seed)), np.load(zero_path(person, seed))
            kc, kz = keyed(zc), keyed(zz)
            keys = sorted(k for k in kc.keys() & kz.keys() if k[1] - t0[k[0]] >= SCORE_FROM_S - 1e-9)
            ic, iz = np.array([kc[k] for k in keys]), np.array([kz[k] for k in keys])
            trial = np.array([k[0] for k in keys])
            yz = zz['y_own_target'][iz].astype(float)
            # the zero-phase run was saved with the corrected target. The causal run's differs by up to ~8 Nm at trial
            # edges (old target) and by ~0.0002 Nm elsewhere (its passive torque uses the plain-sampled, not the
            # FIR-decimated, angle). Both arms are scored against the zero-phase run's target.
            d = yz - zc['y_own_target'][ic].astype(float)
            assert np.percentile(np.abs(d), 99) < 0.05, (person, seed, 'targets differ beyond the edge samples')
            y = yz
            scale = float(np.percentile(np.abs(y), 99))
            for arm, z, idx in (('causal', zc, ic), ('zerophase', zz, iz)):
                p = z['y_pred'][idx].astype(float)
                mae = float(np.mean(np.abs(p - y)))
                r2 = float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2))
                rows.append((person, seed, arm, mae, mae / scale, r2, lag_ms(y, p, trial), len(y)))
    if not rows:
        raise SystemExit('no seed has both arms yet')
    arr = {(p, s, a): (m, n, r, l) for p, s, a, m, n, r, l, _ in rows}
    print(f"{'who':4s} {'seeds':>5s} | {'nMAE causal':>11s} {'zero-ph':>8s} {'change':>7s} | {'R2 causal':>9s} {'zero-ph':>8s} | "
          f"{'lag causal':>10s} {'zero-ph':>8s}")
    allc = []
    for person in PEOPLE:
        ss = sorted({s for p, s, a in arr if p == person})
        if not ss:
            continue
        c = np.array([arr[(person, s, 'causal')] for s in ss]); z = np.array([arr[(person, s, 'zerophase')] for s in ss])
        ch = 100 * (z[:, 1] / c[:, 1] - 1)
        allc.append(ch)
        print(f'{person:4s} {len(ss):5d} | {np.median(c[:, 1]):11.4f} {np.median(z[:, 1]):8.4f} {np.median(ch):+6.1f}% | '
              f'{np.median(c[:, 2]):9.3f} {np.median(z[:, 2]):8.3f} | {np.median(c[:, 3]):8.0f}ms {np.median(z[:, 3]):6.0f}ms')
    allc = np.concatenate(allc)
    print(f'nMAE change zero-phase vs causal, every person x seed: median {np.median(allc):+.1f} %, '
          f'range {allc.min():+.1f} to {allc.max():+.1f} %, zero-phase better in {int((allc < 0).sum())} of {len(allc)}')


if __name__ == '__main__':
    main()
