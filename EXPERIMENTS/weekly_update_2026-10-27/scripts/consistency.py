"""Seed-to-seed consistency of the saved models (train session 1 -> test session 2).

For every subject x position cell: spread of a metric across training seeds.
Cells are then summarised by their median (typical cell) and 90th percentile (bad cells),
for all positions, toes-down and the other positions.
"""
import json
import sys

import numpy as np
import pandas as pd

dash = sys.argv[1]
res = json.load(open(f'{dash}/src/data/results.json', encoding='utf-8'))
ctrl = pd.DataFrame(res['rows'], columns=res['cols']).rename(columns={
    'experiment': 'exp', 'subject': 'subj', 'direction': 'dir', 'position_rad': 'pos', 'toes_down': 'toes'})
cat = json.load(open(f'{dash}/src/data/presentation-catalog.json', encoding='utf-8'))
m50p = pd.DataFrame([r for r in cat['rows'] if r['exp'] == 'main/session-passive'])

models = {
    '0.5 s + retest map': m50p,
    'D01 new model (0.2 s)': ctrl[ctrl.exp == 'D01_cleanpassive+autogain+equalweight'],
    'D02 = D01 + 0.5 s': ctrl[ctrl.exp == 'D02_D01+window_0.5s'],
    'X00 original (0.2 s)': ctrl[ctrl.exp == 'X00_baseline'],
}
print('seeds per subject (S1toS2):')
for name, d in models.items():
    d = d[d.dir == 'S1toS2']
    print(f'  {name:24s}', dict(d.groupby('subj').seed.nunique()))

def spread(d, metric, floor=False):
    v = d.assign(val=d[metric].clip(lower=0) if floor else d[metric])
    cells = v.groupby(['subj', 'pos']).agg(sd=('val', 'std'), n=('val', 'size'), toes=('toes', 'first'),
                                            p95=('val', lambda x: np.percentile(x, 95)),
                                            p5=('val', lambda x: np.percentile(x, 5)))
    cells = cells[cells.n >= 3]
    cells['range'] = cells.p95 - cells.p5
    out = {}
    for scope, c in [('all', cells), ('toes', cells[cells.toes == 1]), ('other', cells[cells.toes == 0])]:
        out[scope] = (c.sd.median(), c.sd.quantile(.9), c.range.median())
    return out

print('\nSpread across seeds within each subject x position (median cell SD | 90th-pct cell SD | median 5-95 range):')
for metric, floor, unit in [('mae', False, 'Nm'), ('r2', True, 'R2 floored at 0'), ('bias', False, 'Nm')]:
    print(f'\n  {metric} ({unit})')
    for name, d in models.items():
        s = spread(d[d.dir == 'S1toS2'], metric, floor)
        print(f'    {name:24s} ' + '  '.join(f'{k}: {a:.3f} | {b:.3f} | {r:.3f}' for k, (a, b, r) in s.items()))

print('\nMAE level (median of cell medians, Nm) and relative spread (median cell SD / cell median):')
for name, d in models.items():
    d = d[d.dir == 'S1toS2']
    g = d.groupby(['subj', 'pos']).mae.agg(['median', 'std'])
    print(f'  {name:24s} level {g["median"].median():.3f}  relative spread {(g["std"] / g["median"]).median():.3f}')

print('\nPer-seed summary (median MAE over positions for each seed) -> range across seeds, per subject:')
for name, d in models.items():
    d = d[d.dir == 'S1toS2']
    per_seed = d.groupby(['subj', 'seed']).mae.median().groupby('subj').agg(lambda x: x.max() - x.min())
    print(f'  {name:24s}', {k: round(v, 3) for k, v in per_seed.items()}, f'median {per_seed.median():.3f}')

print('\nPer-subject median cell SD of MAE (Nm):')
for name, d in models.items():
    d = d[d.dir == 'S1toS2']
    g = d.groupby(['subj', 'pos']).mae.std().groupby('subj').median()
    print(f'  {name:24s}', {k: round(v, 3) for k, v in g.items()})
