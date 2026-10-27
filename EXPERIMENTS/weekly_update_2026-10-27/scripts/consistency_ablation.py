"""Which single change moves seed-to-seed spread? Paired against the original model (X00).

Each experiment is compared with X00 on exactly the same subjects, positions and seeds
(train session 1 -> test session 2), so seed count and subject mix cannot drive the difference.
Usage: python consistency_ablation.py <dashboard dir> <experiment results dir> <0.5 s runs dir>
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

dash, results, runs = sys.argv[1:4]
res = json.load(open(f'{dash}/src/data/results.json', encoding='utf-8'))
ctrl = pd.DataFrame(res['rows'], columns=res['cols'])
ctrl = ctrl[ctrl.direction == 'S1toS2']
cat = json.load(open(f'{dash}/src/data/presentation-catalog.json', encoding='utf-8'))
m50p = pd.DataFrame([r for r in cat['rows'] if r['exp'] == 'main/session-passive']).rename(
    columns={'subj': 'subject', 'pos': 'position_rad', 'toes': 'toes_down'})
m50p['experiment'] = '0.5 s + retest map'
X00 = ctrl[ctrl.experiment == 'X00_baseline']


def cells(d, subjects, seeds):
    d = d[d.subject.isin(subjects) & d.seed.isin(seeds)].copy()
    d['p'] = d.position_rad.round(2)  # different pipelines give the same angle to +-0.005 rad
    return d.groupby(['subject', 'p']).agg(sd=('mae', 'std'), toes=('toes_down', 'first'))


def paired(d, ref=X00):
    subjects = sorted(set(d.subject) & set(ref.subject))
    seeds = sorted(set.intersection(*[set(g.seed) for _, g in d[d.subject.isin(subjects)].groupby('subject')]))
    a, b = cells(d, subjects, seeds), cells(ref, subjects, seeds)
    j = a.join(b, rsuffix='_ref', how='inner')
    return j.sd.median(), j.sd_ref.median(), (j.sd / j.sd_ref).median(), subjects, len(seeds), len(j)


print('median within-cell SD of MAE across seeds (Nm): experiment vs X00 on the same people/positions/seeds')
for name, d in [('0.5 s + retest map', m50p)] + [(e, ctrl[ctrl.experiment == e]) for e in
        ['X01_clean_passive', 'X04_equal_angle_weight', 'X05_autogain', 'C01_autogain+equalweight',
         'D01_cleanpassive+autogain+equalweight', 'D02_D01+window_0.5s', 'X06_window_0.4s',
         'X02a_dropout_0.0', 'X02c_dropout_0.5', 'X03_lr_0.001']]:
    if d.empty:
        continue
    sd, ref, ratio, subjects, nseeds, ncells = paired(d)
    print(f'  {name:40s} {sd:.3f} vs X00 {ref:.3f}  (median ratio {ratio:.2f})  {",".join(subjects)}; {nseeds} seeds; {ncells} cells')

d01 = ctrl[ctrl.experiment == 'D01_cleanpassive+autogain+equalweight']
sd, ref, ratio, subjects, nseeds, ncells = paired(m50p, d01)
print(f'\n  0.5 s + retest map vs D01 directly: {sd:.3f} vs {ref:.3f} (median ratio {ratio:.2f}); {",".join(subjects)}; {nseeds} seeds; {ncells} cells')

print('\nepochs run and kept epoch across seeds (median [10th-90th pct]):')
def describe(v):
    v = np.asarray(v, float)
    return f'{np.median(v):.0f} [{np.percentile(v, 10):.0f}-{np.percentile(v, 90):.0f}]'
hist = [json.load(open(p)) for p in glob.glob(os.path.join(runs, '*', '*_lstm_seed*', 'history.json'))]
print(f'  {"0.5 s + retest map":40s} run {describe([h["epochs_executed"] for h in hist])}  '
      f'kept {describe([h["best_epoch_one_based"] for h in hist])}  (n={len(hist)}, cap 60, patience 12)')
for e in ['X00_baseline', 'D01_cleanpassive+autogain+equalweight']:
    info = [json.load(open(f)).get('train_info', {}) for f in glob.glob(os.path.join(results, e, '*', 'S1toS2', 'seed*.json'))]
    info = [i for i in info if 'best_epoch' in i]
    if info:
        print(f'  {e:40s} run {describe([i["epochs"] for i in info])}  kept {describe([i["best_epoch"] for i in info])}  (n={len(info)}, cap 200, patience 25)')
