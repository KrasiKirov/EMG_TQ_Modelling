"""Seed spread of the 0.5 s + retest map model: 60-epoch cap vs 150-epoch cap (same seeds).

Usage: python compare_budget.py <60-epoch runs> <150-epoch runs> <dashboard dir>
Cells are subject x test trial (one position each); spread = SD across seeds 0-2.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

short, long_, dash = sys.argv[1:4]
SEEDS = [0, 1, 2]


def load(root):
    rows, runs = [], []
    for metrics in glob.glob(os.path.join(root, '*', '*_lstm_seed*', 'metrics.json')):
        folder = os.path.dirname(metrics)
        m = json.load(open(metrics))
        h = json.load(open(os.path.join(folder, 'history.json')))
        subject = os.path.basename(folder).split('_')[0]
        manifest = json.load(open(os.path.join(os.path.dirname(folder), f'{subject}_manifest.json')))
        angle = {r['trial']: r['mean_position_rad'] for r in manifest['trials']}
        tested = sorted((angle[int(k.split('/')[1])], k) for k in m['splits']['test']['trials'])
        for rank, (pos, key) in enumerate(tested):
            t = m['splits']['test']['trials'][key]
            rows.append(dict(subject=subject, trial=int(key.split('/')[1]), pos=pos, toes=int(rank < 2),
                             seed=m['seed'], mae=t['mae'], bias=t['bias'], r2=t['r2']))
        runs.append(dict(subject=subject, seed=m['seed'], epochs=h['epochs_executed'], kept=h['best_epoch_one_based'],
                         seconds=m['training_seconds']))
    return pd.DataFrame(rows), pd.DataFrame(runs)


def spread(d):
    d = d[d.seed.isin(SEEDS)]
    g = d.groupby(['subject', 'trial'])
    cell = g.agg(mae_sd=('mae', 'std'), bias_sd=('bias', 'std'), mae=('mae', 'median'),
                 r2=('r2', lambda x: np.median(np.clip(x, 0, None))), toes=('toes', 'first'), n=('seed', 'nunique'))
    return cell[cell.n == len(SEEDS)]


a, ra = load(short)
b, rb = load(long_)
ca, cb = spread(a), spread(b)
j = ca.join(cb, lsuffix='_60', rsuffix='_150', how='inner')
print(f'paired cells: {len(j)} (subjects {sorted(j.index.get_level_values(0).unique())}, seeds {SEEDS})\n')
print('                          60 epochs   150 epochs')
for label, col, scope in [('MAE spread, all (Nm)', 'mae_sd', None), ('MAE spread, toes-down', 'mae_sd', 1),
                          ('MAE spread, other', 'mae_sd', 0), ('offset (bias) spread', 'bias_sd', None),
                          ('MAE level (Nm)', 'mae', None), ('R2 (floored), all', 'r2', None),
                          ('R2 (floored), toes-down', 'r2', 1), ('R2 (floored), other', 'r2', 0)]:
    s = j if scope is None else j[j.toes_60 == scope]
    print(f'  {label:24s} {s[col + "_60"].median():9.3f}   {s[col + "_150"].median():9.3f}')
ratio = (j.mae_sd_150 / j.mae_sd_60)
print(f'\n  median per-cell ratio of MAE spread (150 / 60): {ratio.median():.2f}; cells with less spread: {(ratio < 1).sum()} of {len(ratio)}')
print('\nper subject, median MAE spread (Nm), 60 -> 150:')
for s, g in j.groupby(level=0):
    print(f'  {s:4s} {g.mae_sd_60.median():.3f} -> {g.mae_sd_150.median():.3f}')

print('\ntraining (seeds 0-2):')
for name, r, cap in [('60 epochs', ra[ra.seed.isin(SEEDS)], 60), ('150 epochs', rb, 150)]:
    capped = (r.epochs == cap).sum()
    print(f'  {name:10s} epochs run median {r.epochs.median():.0f} [{r.epochs.min()}-{r.epochs.max()}], kept {r.kept.median():.0f} '
          f'[{r.kept.min()}-{r.kept.max()}], hit cap {capped}/{len(r)}, train time median {r.seconds.median() / 60:.1f} min')

# Reference: the new model (D01) on the same seeds, for the four people it covers.
res = json.load(open(f'{dash}/src/data/results.json', encoding='utf-8'))
c = pd.DataFrame(res['rows'], columns=res['cols'])
c = c[(c.experiment == 'D01_cleanpassive+autogain+equalweight') & (c.direction == 'S1toS2') & c.seed.isin(SEEDS)]
d01 = c.groupby(['subject', 'position_rad']).mae.std()
common = sorted(set(c.subject))
print(f'\nreference, same seeds, {",".join(common)} only: MAE spread  D01 {d01.median():.3f}  |  '
      f'60 ep {j.loc[common].mae_sd_60.median():.3f}  |  150 ep {j.loc[common].mae_sd_150.median():.3f}')
