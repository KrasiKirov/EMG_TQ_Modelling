"""
Baseline -> balanced positions -> + 10 s calibration, all scored the same way (10 Nov 2026 deck).

Last semester's LSTM (X00f, framework re-run of the original settings, and X00, original code) has no saved
prediction arrays any more, only the dashboard waveforms: every prediction and the answer key sampled every
0.2 s (0.01 Nm steps). This script rescoress every model from those waveforms so all rows share:
  * the answer key (active torque = measured - clean passive table, the dashboard key),
  * session 1 -> session 2, 6 people, seeds 0-2 (X00: 0-9 shown separately),
  * scoring from 30 s to the end of each trial (the calibration rule of calibration_test.py),
  * the 10 s gain+offset calibration of calibration_test.py (first 10 s of each trial).
A0 / A0+cal (balanced positions, active torque) are scored from their npz on the same 0.2 s grid against
their own key (checked against the dashboard key below).
Summary rules: median over seeds per person x position, R2 < 0 -> 0, median over positions of a group.
"""
import base64, json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXP, 'normalization'))
from calibration_test import correct, metrics  # noqa: E402

D = json.load(open(os.path.join(EXP, 'dashboard_react', 'src', 'data', 'results.json'), encoding='utf-8'))
G0, DT, N = D['grid']
GRID = G0 + DT * np.arange(int(N))
CAL_S, SCORE_S = 10.0, 30.0


def dec(s):
    return np.frombuffer(base64.b64decode(s), '<i2').astype(float) / 100


def score(rows, model, subj, seed, pos, key, pred, t):
    for a, y, p in zip(pos, key, pred):
        sc, cal = t >= SCORE_S, t < CAL_S
        td = int(a in sorted(pos)[:2])
        base = dict(model=model, subject=subj, seed=seed, position_rad=a, toes_down=int(td), peak=np.percentile(np.abs(y[sc]), 99))
        rows.append(dict(base, cal=0, **metrics(y[sc], p[sc])))
        rows.append(dict(base, cal=1, **metrics(y[sc], correct(p[cal], y[cal], p, 'gain+offset')[sc])))


rows, keydiff = [], []
for k, w in D['wave'].items():
    e, s, d, seed = k.split('|')
    if d != 'S1toS2' or e not in ('X00_baseline', 'X00f_baseline_framework', 'X04_equal_angle_weight'):
        continue
    pos = D['positions'][f'{s}|{d}']
    key = dec(D['key'][f'{s}|{d}']).reshape(len(pos), -1)
    pred = dec(w).reshape(len(pos), -1)
    score(rows, e, s, int(seed), pos, key, pred, GRID)

for s in D['subjects']:
    pos = D['positions'][f'{s}|S1toS2']
    key = dec(D['key'][f'{s}|S1toS2']).reshape(len(pos), -1)
    for seed in range(3):
        z = np.load(os.path.join(EXP, 'results', 'A0_active_trainmax', s, 'S1toS2', f'seed{seed}.npz'))
        y, p, ang, t = (z[k].astype(float) for k in ('y_own_target', 'y_pred', 'angle', 't'))
        angs = sorted(np.unique(ang))
        Y, P = [], []
        for a in angs:
            m = ang == a
            tt = t[m] - t[m][0]
            i = np.clip(np.searchsorted(tt, GRID), 0, m.sum() - 1)
            Y.append(y[m][i]); P.append(p[m][i])
        Y, P = np.array(Y), np.array(P)
        if seed == 0:
            keydiff.append((s, np.round(np.median(Y - key, axis=1), 2).tolist()))
        score(rows, 'A0_active_trainmax', s, seed, [round(a, 3) for a in angs], Y, P, GRID)

R = pd.DataFrame(rows)
R.to_csv(os.path.join(HERE, 'ladder_runs.csv'), index=False, float_format='%.5f')
cells = R.groupby(['model', 'cal', 'subject', 'position_rad', 'toes_down'])[['mae', 'r2', 'vaf', 'peak']].median().reset_index()
cells['r2'] = cells.r2.clip(lower=0)
g = cells.groupby(['model', 'cal', 'toes_down'])[['mae', 'r2', 'vaf', 'peak']].median().unstack('toes_down').round(3)
per = cells.groupby(['model', 'cal', 'subject', 'toes_down']).mae.median().unstack(['toes_down', 'subject']).round(2)
print(g.to_string()); print(per.to_string())
print('A0 key minus dashboard key, median per position (Nm):')
for s, v in keydiff:
    print(' ', s, v)
g.to_csv(os.path.join(HERE, 'ladder_groups.csv'))
