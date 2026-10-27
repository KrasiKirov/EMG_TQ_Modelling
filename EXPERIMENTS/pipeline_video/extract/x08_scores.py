"""Chapters 10 and 11 data: every score, rebuilt from the saved runs, and the dashboard's aggregation.

Scores: the code of EXPERIMENTS/code/presentation_metrics.py (dorsiflexion worktree): answer key =
measured torque - the tested session's static passive map, read at every window's last sample;
R2, MAE, RMSE, bias, peak = max |key|. Every value is checked against the dashboard's results.json.
Aggregation: ExperimentTab.tsx score() (median over seeds per person-position, R2 < 0 -> 0 first,
then median over cells), the box plots (percentiles over seeds) and aggregate.ts groupSummary.
Writes data/x08.npz, data/facts_x08.json.
"""
import glob
import json

import numpy as np

from common import DASHBOARD, SUBJECTS, run_dir, save_facts, save_npz, use_d01_code

use_d01_code()
from ML.research.data import load_subject                    # noqa: E402
from ML.research.passive import PassiveModel                 # noqa: E402

FS = 100.0
facts, arrays = {}, {}


def metrics(y, p):                                            # presentation_metrics.py:81-88
    e = p - y
    ss = np.sum((y - y.mean()) ** 2)
    return dict(r2=float(1 - np.sum(e ** 2) / ss), mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e ** 2))),
                bias=float(np.mean(e)), peak=float(np.max(np.abs(y))), ss_res=float(np.sum(e ** 2)), ss_tot=float(ss),
                mean=float(y.mean()), n=int(len(y)))


rows, waves = [], {}
for s in SUBJECTS:
    b = load_subject(s)
    st = PassiveModel('static', b)                            # presentation_metrics.py:70-78
    recs = {r['idx']: dict(session=r['session'], angle=float(r['angle']), t0=float(r['t'][0]),
                           key=(r['tq'] - st.predict(r)).astype(np.float64)) for r in b['active']}
    for d in ('S1toS2', 'S2toS1'):
        for f in sorted(glob.glob(str(run_dir(s) / d / 'seed*.npz'))):
            seed = int(f.replace('\\', '/').split('/')[-1][4:-4])
            z = np.load(f)
            trial, t, p = z['trial'].astype(int), z['t'].astype(np.float64), z['y_pred'].astype(np.float64)
            for i in np.unique(trial):
                m = trial == i
                R = recs[i]
                k = np.rint((t[m] - R['t0']) * FS).astype(int)    # presentation_metrics.py:107-112
                y = R['key'][k]
                rows.append(dict(subject=s, direction=d, seed=seed, trial=int(i), position=round(R['angle'], 3),
                                 **metrics(y, p[m])))
                if s == 'EG' and d == 'S1toS2' and seed == 0:
                    waves[int(i)] = (t[m], y, p[m], R['angle'])
for d in ('S1toS2', 'S2toS1'):                                # toes-down = 2 most negative positions
    for s in SUBJECTS:
        pos = sorted({r['position'] for r in rows if r['subject'] == s and r['direction'] == d})
        for r in rows:
            if r['subject'] == s and r['direction'] == d:
                r['toes'] = int(r['position'] in pos[:2])

# ── check every value against the dashboard's results.json ──────────────────
res = json.loads((DASHBOARD / 'src' / 'data' / 'results.json').read_text(encoding='utf-8'))
ix = {c: i for i, c in enumerate(res['cols'])}
dash = {}
for x in res['rows']:
    exp = x[ix['experiment']]
    if exp in ('D01_cleanpassive+autogain+equalweight', 'C01_autogain+equalweight') and \
            (exp.startswith('C01') == (x[ix['subject']] in ('EG', 'YES'))):
        dash[(x[ix['subject']], x[ix['direction']], x[ix['seed']], round(x[ix['position_rad']], 3))] = x
worst = {'r2': 0.0, 'mae': 0.0, 'peak_torque': 0.0}
for r in rows:
    x = dash[(r['subject'], r['direction'], r['seed'], r['position'])]
    for k, mine in (('r2', r['r2']), ('mae', r['mae']), ('peak_torque', r['peak'])):
        worst[k] = max(worst[k], abs(mine - x[ix[k]]))
facts['check_vs_results_json'] = dict(rows=len(rows), dashboard_rows=len(dash), max_abs_diff=worst)


def score(rs, key, floor=True):                               # ExperimentTab.tsx:29-34
    cells = {}
    for r in rs:
        v = max(r[key], 0.0) if (key == 'r2' and floor) else r[key]
        cells.setdefault((r['subject'], r['position']), []).append(v)
    return float(np.median([np.median(v) for v in cells.values()])), len(cells)


def group_summary(rs, key):                                  # aggregate.ts:46-57 (run key = seed)
    by_seed = {}
    for r in rs:
        by_seed.setdefault(r['seed'], []).append(max(r[key], 0.0) if key == 'r2' else r[key])
    per_seed = np.array([np.median(v) for v in by_seed.values()])
    return dict(p5=float(np.percentile(per_seed, 5)), p95=float(np.percentile(per_seed, 95)),
                p1=float(np.percentile(per_seed, 1)), p99=float(np.percentile(per_seed, 99)),
                centre=score(rs, key)[0], n_seeds=len(per_seed))


agg = {}
for d in ('S1toS2', 'S2toS1'):
    rd = [r for r in rows if r['direction'] == d]
    agg[d] = {}
    for scope, sel in (('all', rd), ('toes', [r for r in rd if r['toes']]), ('other', [r for r in rd if not r['toes']])):
        agg[d][scope] = dict(r2=score(sel, 'r2')[0], mae=score(sel, 'mae')[0], cells=score(sel, 'r2')[1],
                             kpi_r2=group_summary(sel, 'r2'), kpi_mae=group_summary(sel, 'mae'))
    agg[d]['per_subject'] = {s: dict(r2=score([r for r in rd if r['subject'] == s], 'r2')[0],
                                     mae=score([r for r in rd if r['subject'] == s], 'mae')[0])
                             for s in SUBJECTS}
    cells = {}
    for r in rd:
        cells.setdefault(f"{r['subject']}|{r['position']:+.3f}", []).append(max(r['r2'], 0.0))
    agg[d]['cells_r2'] = {k: float(np.median(v)) for k, v in cells.items()}
facts['aggregate'] = agg

# ── EG, session 1 -> 2: the 8 x 10 table, box-plot numbers, the live example ──
eg = [r for r in rows if r['subject'] == 'EG' and r['direction'] == 'S1toS2']
pos = sorted({r['position'] for r in eg})
seeds = sorted({r['seed'] for r in eg})
tab = np.array([[next(r['r2'] for r in eg if r['position'] == p and r['seed'] == sd) for sd in seeds] for p in pos])
tab_mae = np.array([[next(r['mae'] for r in eg if r['position'] == p and r['seed'] == sd) for sd in seeds] for p in pos])
box = {f'{p:+.3f}': {q: float(np.percentile(np.maximum(tab[i], 0), q)) for q in (1, 5, 25, 50, 75, 95, 99)}
       for i, p in enumerate(pos)}
facts['eg_table'] = dict(positions=pos, seeds=seeds, r2=tab, mae=tab_mae, median_r2=np.median(np.maximum(tab, 0), axis=1),
                         median_mae=np.median(tab_mae, axis=1), box=box,
                         peaks=[next(r['peak'] for r in eg if r['position'] == p) for p in pos])
live = next(r for r in eg if r['seed'] == 0 and r['position'] == -0.038)
facts['eg_live'] = live
for i, (t, y, p, ang) in waves.items():
    arrays[f'w{i}_t'], arrays[f'w{i}_y'], arrays[f'w{i}_p'] = t, y, p
facts['eg_wave_trials'] = sorted(((round(a, 3), i) for i, (_, _, _, a) in waves.items()))
facts['eg_seed0'] = {f"{r['position']:+.3f}": dict(r2=r['r2'], mae=r['mae'], peak=r['peak'], trial=r['trial'])
                     for r in eg if r['seed'] == 0}
save_npz('x08', **arrays)
save_facts('x08', {'x08': facts})
print('check vs results.json', facts['check_vs_results_json'])
print('S1toS2', {k: (round(v['r2'], 4), round(v['mae'], 4), v['cells']) for k, v in agg['S1toS2'].items() if k in ('all', 'toes', 'other')})
print('live', {k: round(v, 4) if isinstance(v, float) else v for k, v in live.items()})
