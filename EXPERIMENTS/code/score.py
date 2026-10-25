"""
Score every saved run in EXPERIMENTS/results/ -> EXPERIMENTS/results/all_evaluations.csv
(one row per experiment x subject x direction x seed x position x answer key).

Answer keys
-----------
passive_trial  (PRIMARY) measured torque - passive torque from the clean static holds
               of the passive trials of the TESTED session (YES: lab worksheet values;
               IES has no passive trials -> nothing subtracted, flagged).
original       the original target: measured - the original passive table of the tested
               session (contains JM's bad entry).  For framework experiments that change
               the passive model, 'own_target' is also stored.
in_trial       (supplementary, framework runs only) measured - smooth line through the
               EMG-detected relaxed moments of the same trial.

Metrics (per position = one test trial)
---------------------------------------
R2    = 1 - sum((pred-key)^2) / sum((key-mean(key))^2)
MSE   = mean((pred-key)^2)                [Nm^2]
RMSE  = sqrt(MSE)                         [Nm]
NRMSE = RMSE / (max(key) - min(key))      original project definition (ML/evaluation/metrics.py)

Also writes a reference predictor ('train_mean'): for each position, the mean target of
the TRAINING session at that position (built with the same key definition), no test data.
"""
import os
import sys
import glob
import json

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP)
sys.path.insert(0, ROOT)
from ML.research.data import load_subject            # noqa: E402
from ML.research.passive import map_lookup, PassiveModel   # noqa: E402
from ML.research import truth as T                   # noqa: E402

FS = 100.0
RES = os.path.join(EXP, 'results')
_cache = {}


def subj(s):
    if s not in _cache:
        b = load_subject(s)
        st, lg = PassiveModel('static', b), PassiveModel('legacy', b)
        tr = T.subject_truth(s)
        recs = {}
        for r in b['active']:
            recs[r['idx']] = dict(session=r['session'], angle=float(r['angle']), t0=float(r['t'][0]),
                                  passive_trial=(r['tq'] - st.predict(r)).astype(np.float64),
                                  original=(r['tq'] - lg.predict(r)).astype(np.float64),
                                  in_trial=tr[r['idx']]['truth'].astype(np.float64))
        _cache[s] = dict(bundle=b, recs=recs)
    return _cache[s]


def metrics(y, p):
    e = p - y
    ss = np.sum((y - y.mean()) ** 2)
    rng = y.max() - y.min()
    mse = float(np.mean(e ** 2))
    return dict(r2=float(1 - np.sum(e ** 2) / ss) if ss > 0 else np.nan, mse=mse, rmse=float(np.sqrt(mse)),
                nrmse=float(np.sqrt(mse) / rng) if rng > 0 else np.nan, bias=float(np.mean(e)),
                std_key=float(np.std(y)), n=int(len(y)))


def tested_session(direction):
    return 'S2' if direction == 'S1toS2' else 'S1'


rows = []
# ── framework runs (have trial / time for every window) ─────────────────────
for f in sorted(glob.glob(os.path.join(RES, '*', '*', 'S*to*', 'seed*.npz'))):
    p = f.replace('\\', '/').split('/')
    exp, s, d, seed = p[-4], p[-3], p[-2], int(p[-1][4:-4])
    if exp == 'X00_baseline':
        continue
    z = np.load(f)
    S = subj(s)
    trial, t, pred, own = z['trial'].astype(int), z['t'].astype(np.float64), z['y_pred'].astype(np.float64), z['y_own_target'].astype(np.float64)
    for i in np.unique(trial):
        m = trial == i
        R = S['recs'][i]
        k = np.rint((t[m] - R['t0']) * FS).astype(int)
        keys = {'passive_trial': R['passive_trial'][k], 'original': R['original'][k], 'in_trial': R['in_trial'][k], 'own_target': own[m]}
        for kn, y in keys.items():
            rows.append(dict(experiment=exp, subject=s, direction=d, seed=seed, tested_session=R['session'],
                             position_rad=round(R['angle'], 3), key=kn, **metrics(y, pred[m])))

# ── original-code baseline runs (windows identified by their position) ──────
for f in sorted(glob.glob(os.path.join(RES, 'X00_baseline', '*', 'S1toS2', 'seed*.npz'))):
    p = f.replace('\\', '/').split('/')
    s, seed = p[-3], int(p[-1][4:-4])
    z = np.load(f)
    info = json.load(open(f[:-4] + '.json'))
    S = subj(s)
    maps = S['bundle']['maps']
    pos, snapped = z['position'].astype(np.float64), z['position_snapped'].astype(np.float64)
    y_orig, pred = z['y_true_original'].astype(np.float64), z['y_pred'].astype(np.float64)
    # passive_trial key = original target + (original table - clean table) at the window's position
    corr = map_lookup(maps['legacy']['S2'], pos) - map_lookup(maps['static']['S2'], pos)
    y_pt = y_orig + corr
    s2_angles = [R['angle'] for R in S['recs'].values() if R['session'] == 'S2']
    for a in np.unique(snapped):
        m = snapped == a
        a_r = min(s2_angles, key=lambda x: abs(x - a))      # same position label as the framework runs
        for kn, y in (('passive_trial', y_pt[m]), ('original', y_orig[m])):
            rows.append(dict(experiment='X00_baseline', subject=s, direction='S1toS2', seed=seed, tested_session='S2',
                             position_rad=round(float(a_r), 3), key=kn, **metrics(y, pred[m])))

# ── reference predictor: training-session mean target at each position ─────
for s in sorted({r['subject'] for r in rows}):
    S = subj(s)
    for d in ('S1toS2', 'S2toS1'):
        trs, tes = ('S1', 'S2') if d == 'S1toS2' else ('S2', 'S1')
        train = {round(R['angle'], 2): R for R in S['recs'].values() if R['session'] == trs}
        for R in S['recs'].values():
            if R['session'] != tes:
                continue
            near = min(train, key=lambda a: abs(a - R['angle']))
            for kn in ('passive_trial', 'original', 'in_trial'):
                y = R[kn][int(FS * 0.19):]           # first window ends after 20 samples
                ref = float(np.mean(train[near][kn][: int(0.8 * len(train[near][kn]))]))
                rows.append(dict(experiment='REF_train_mean', subject=s, direction=d, seed=0, tested_session=tes,
                                 position_rad=round(R['angle'], 3), key=kn, **metrics(y, np.full(len(y), ref))))

df = pd.DataFrame(rows)
df['problematic'] = 0
for (s, ses), x in df.groupby(['subject', 'tested_session']):
    pf = sorted(x['position_rad'].unique())[:2]
    df.loc[x.index[x['position_rad'].isin(pf)], 'problematic'] = 1
df['no_passive_trials'] = (df['subject'] == 'IES') & df['key'].isin(['passive_trial', 'original'])
df.to_csv(os.path.join(RES, 'all_evaluations.csv'), index=False, encoding='utf-8-sig')
print('rows', len(df), 'experiments', sorted(df.experiment.unique()))
