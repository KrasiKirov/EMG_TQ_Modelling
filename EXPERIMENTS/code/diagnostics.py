"""
Phase 2 diagnostics of the baseline (X00, original code, learned S1 -> tested S2).

For every subject x position x seed (primary answer key = passive trials of the tested
session) it computes:
  * the target's spread (std) and range
  * R2, RMSE of LSTM-8 and of the reference predictor (training-session mean at that position)
  * the MSE split into three parts (Murphy 1988 decomposition):
        MSE = (mean_pred - mean_key)^2            -> OFFSET  (wrong level)
            + (std_pred - std_key)^2              -> SCALE   (wrong push size)
            + 2 std_pred std_key (1 - corr)       -> SHAPE   (wrong timing / random)
  * the time lag that maximises the prediction-target correlation (alignment check)
  * run-to-run spread of R2 over seeds
and training behaviour (epochs, best epoch, train vs validation loss).

Writes results/diagnostics_X00.csv, reports/02_diagnostics_tables.md and plots/X00_*.png.
Selection rule for Stage A: the 2 subjects with the LOWEST mean (over the 2 problematic
positions) of the seed-averaged R2 (each floored at -1), primary key, S1->S2.
"""
import os
import sys
import json
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(EXP))
from ML.research.data import load_subject           # noqa: E402
from ML.research.passive import map_lookup           # noqa: E402

RES, PLOTS, REP = (os.path.join(EXP, d) for d in ('results', 'plots', 'reports'))
os.makedirs(PLOTS, exist_ok=True)
ev = pd.read_csv(os.path.join(RES, 'all_evaluations.csv'))


def lagcorr(y, p, maxlag=50):
    best = (0, -2)
    for L in range(-maxlag, maxlag + 1):
        a, b = (y[L:], p[:len(p) - L]) if L >= 0 else (y[:L], p[-L:])
        if len(a) > 100 and np.std(a) > 0 and np.std(b) > 0:
            c = float(np.corrcoef(a, b)[0, 1])
            if c > best[1]:
                best = (L, c)
    return best


rows, curves = [], []
for f in sorted(glob.glob(os.path.join(RES, 'X00_baseline', '*', 'S1toS2', 'seed*.npz'))):
    p = f.replace('\\', '/').split('/')
    s, seed = p[-3], int(p[-1][4:-4])
    z = np.load(f)
    info = json.load(open(f[:-4] + '.json'))
    b = load_subject(s)
    maps = b['maps']
    s2 = sorted([r['angle'] for r in b['active'] if r['session'] == 'S2'])
    pos, snapped = z['position'].astype(np.float64), z['position_snapped'].astype(np.float64)
    y = z['y_true_original'].astype(np.float64) + map_lookup(maps['legacy']['S2'], pos) - map_lookup(maps['static']['S2'], pos)
    pr = z['y_pred'].astype(np.float64)
    curves.append(dict(subject=s, seed=seed, epochs=info['epochs_run'], best_epoch=info['best_epoch'],
                       train_loss_best=info['train_loss'][info['best_epoch'] - 1], val_loss_best=info['val_loss'][info['best_epoch'] - 1],
                       train_loss_last=info['train_loss'][-1], val_loss_last=info['val_loss'][-1], train_s=info['train_s']))
    for a in np.unique(snapped):
        m = snapped == a
        yy, pp = y[m], pr[m]
        e = pp - yy
        corr = float(np.corrcoef(yy, pp)[0, 1]) if np.std(pp) > 0 else np.nan
        lag, cmax = lagcorr(yy, pp)
        a_r = min(s2, key=lambda x: abs(x - a))
        rows.append(dict(subject=s, seed=seed, position_rad=round(a_r, 3), std_key=np.std(yy), range_key=np.ptp(yy),
                         r2=1 - np.sum(e ** 2) / np.sum((yy - yy.mean()) ** 2), rmse=np.sqrt(np.mean(e ** 2)),
                         offset=(pp.mean() - yy.mean()) ** 2, scale=(pp.std() - yy.std()) ** 2,
                         shape=2 * pp.std() * yy.std() * (1 - corr), corr=corr, best_lag_samples=lag, corr_at_best_lag=cmax,
                         mean_key=yy.mean(), mean_pred=pp.mean(), std_pred=pp.std()))
        if seed == 0:
            pass

d = pd.DataFrame(rows)
d['problematic'] = 0
for s, x in d.groupby('subject'):
    pf = sorted(x.position_rad.unique())[:2]
    d.loc[x.index[x.position_rad.isin(pf)], 'problematic'] = 1
ref = ev[(ev.experiment == 'REF_train_mean') & (ev.key == 'passive_trial') & (ev.direction == 'S1toS2')][['subject', 'position_rad', 'r2', 'rmse']]
ref = ref.rename(columns={'r2': 'ref_r2', 'rmse': 'ref_rmse'})
d.to_csv(os.path.join(RES, 'diagnostics_X00.csv'), index=False)
cv = pd.DataFrame(curves)
cv.to_csv(os.path.join(RES, 'diagnostics_X00_training.csv'), index=False)

g = d.groupby(['subject', 'position_rad', 'problematic']).agg(
    std_key=('std_key', 'mean'), r2_mean=('r2', 'mean'), r2_sd=('r2', 'std'), rmse=('rmse', 'mean'),
    offset=('offset', 'mean'), scale=('scale', 'mean'), shape=('shape', 'mean'), corr=('corr', 'mean'),
    lag=('best_lag_samples', 'median')).reset_index()
g = g.merge(ref, on=['subject', 'position_rad'], how='left')
g['mse'] = g['rmse'] ** 2
for c in ('offset', 'scale', 'shape'):
    g[c + '_pct'] = 100 * g[c] / (g['offset'] + g['scale'] + g['shape'])

pfm = g[g.problematic == 1].assign(r2f=lambda x: x.r2_mean.clip(lower=-1)).groupby('subject')['r2f'].mean().sort_values()
worst2 = list(pfm.index[:2])

L = ['# Phase 2 diagnostics of the baseline (X00, original code, learned S1 → tested S2, 3 seeds)', '',
     'Primary answer key: passive trials of the tested session. MSE split (Murphy 1988): **offset** = wrong level, **scale** = wrong push size, **shape** = wrong timing / unrelated.', '',
     '## Selection of the 2 worst subjects (rule fixed in code/diagnostics.py)', '',
     'Mean over the 2 problematic positions of the seed-averaged R² (each floored at −1):', '',
     '| subject | problematic-position R² |', '|---|---|'] + [f'| {s} | {v:.3f} |' for s, v in pfm.items()] + \
    ['', f'**Selected for Stage A: {worst2[0]} and {worst2[1]}.**', '', '## Every position', '',
     '| subject | position (rad) | problematic | target std (Nm) | R² (mean ± sd over seeds) | RMSE (Nm) | reference R² | reference RMSE | offset % | scale % | shape % | corr | lag (samples) |',
     '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
for r in g.sort_values(['subject', 'position_rad']).itertuples():
    L.append(f'| {r.subject} | {r.position_rad:+.3f} | {"yes" if r.problematic else ""} | {r.std_key:.2f} | {r.r2_mean:.2f} ± {r.r2_sd:.2f} | {r.rmse:.2f} | '
             f'{r.ref_r2:.2f} | {r.ref_rmse:.2f} | {r.offset_pct:.0f} | {r.scale_pct:.0f} | {r.shape_pct:.0f} | {r.corr:.2f} | {r.lag:.0f} |')
L += ['', '## Training behaviour', '', '| subject | seed | epochs | best epoch | train loss (best) | val loss (best) | train time (s) |', '|---|---|---|---|---|---|---|']
for r in cv.sort_values(['subject', 'seed']).itertuples():
    L.append(f'| {r.subject} | {r.seed} | {r.epochs} | {r.best_epoch} | {r.train_loss_best:.3f} | {r.val_loss_best:.3f} | {r.train_s:.0f} |')
open(os.path.join(REP, '02_diagnostics_tables.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')

# plots: problematic positions of every subject, seed 0: prediction vs key, and residuals
for s in sorted(d.subject.unique()):
    f = os.path.join(RES, 'X00_baseline', s, 'S1toS2', 'seed0.npz')
    if not os.path.exists(f):
        continue
    z = np.load(f)
    b = load_subject(s)
    s2 = sorted([r['angle'] for r in b['active'] if r['session'] == 'S2'])
    pos, snapped = z['position'].astype(np.float64), z['position_snapped'].astype(np.float64)
    y = z['y_true_original'] + map_lookup(b['maps']['legacy']['S2'], pos) - map_lookup(b['maps']['static']['S2'], pos)
    angs = sorted(np.unique(snapped))
    fig, axs = plt.subplots(len(angs), 2, figsize=(13, 2.1 * len(angs)), gridspec_kw={'width_ratios': [3, 1]})
    for i, a in enumerate(angs):
        m = snapped == a
        tt = np.arange(m.sum()) / 100.0
        axs[i, 0].plot(tt, y[m], color='k', lw=0.8, label='answer key')
        axs[i, 0].plot(tt, z['y_pred'][m], color='#2a78d6', lw=0.8, label='LSTM-8')
        r2 = 1 - np.sum((z['y_pred'][m] - y[m]) ** 2) / np.sum((y[m] - y[m].mean()) ** 2)
        axs[i, 0].set_title(f'{s} · {a:+.2f} rad · R² {r2:.2f} · RMSE {np.sqrt(np.mean((z["y_pred"][m] - y[m]) ** 2)):.2f} Nm', fontsize=10, loc='left')
        axs[i, 1].scatter(y[m][::10], (z['y_pred'][m] - y[m])[::10], s=2, color='#c0392b')
        axs[i, 1].axhline(0, color='k', lw=0.6)
        axs[i, 1].set_title('residual vs key', fontsize=9)
    axs[0, 0].legend(fontsize=8, loc='upper right')
    axs[-1, 0].set_xlabel('time (s)')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, f'X00_{s}_S1toS2_seed0_all_positions.png'), dpi=110)
    plt.close(fig)
print('worst2', worst2)
