"""
Re-score the earlier exploration (round 2, ML/research/results/*) and the report-1 network-type
comparison (ML/experiments/results/*__F_*) with the presentation conventions, so every number
in the deck uses the same definitions.

    python EXPERIMENTS/code/rescore_round2.py

Round 2 (ML/research/results/<study>/<subject>/<direction>/<exp>/fast_seed<k>.npz):
  scored on the PRIMARY answer key (passive trials of the tested session), exactly as
  presentation_metrics.py; R2, MAE per test trial.  These runs used the 'fast' training
  protocol (batch 64) and mostly the 3 subjects YES, JM, EG.
Report-1 network types (ML/experiments/results/<subj>__<name>__fast.npz, 3 seeds inside):
  that older harness stored only its own training target (y_test), so they are scored on
  that; all five network types share the same target and data, so only the comparison
  between them is meaningful.  Learn session 1 -> test session 2 only.
Writes EXPERIMENTS/results/round2_rescored.csv and results/families_rescored.csv
"""
import os
import sys
import glob

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import presentation_metrics as PM                      # noqa: E402

R2 = os.path.join(ROOT, 'ML', 'research', 'results')
FAM = os.path.join(ROOT, 'ML', 'experiments', 'results')


def score(y, p):
    e = p - y
    return dict(r2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), mae=float(np.mean(np.abs(e))))


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(R2, '*', '*', 'S*to*', '*', 'fast_seed*.npz'))):
        parts = f.replace('\\', '/').split('/')
        study, s, d, exp = parts[-5], parts[-4], parts[-3], parts[-2]
        if study.startswith(('superseded', '_')) or s not in PM.SUBJ:
            continue
        seed = int(os.path.basename(f).split('seed')[1][:-4])
        for pos, t, y, p in PM.trials(exp, s, f):
            rows.append(dict(study=study, experiment=exp, subject=s, direction=d, seed=seed, position_rad=round(pos, 3), **score(y, p)))
    R = pd.DataFrame(rows)
    R['toes_down'] = 0
    for (s, d), x in R.groupby(['subject', 'direction']):
        R.loc[x.index[x.position_rad.isin(sorted(x.position_rad.unique())[:2])], 'toes_down'] = 1
    R.to_csv(os.path.join(EXP, 'results', 'round2_rescored.csv'), index=False)

    rows = []
    for f in sorted(glob.glob(os.path.join(FAM, '*__fast.npz'))):
        s, name = os.path.basename(f).split('__')[:2]
        if not (name.startswith('F_') or name == 'H_pgain_posw'):
            continue
        z = np.load(f)
        y, tr, pos = z['y_test'].astype(np.float64), z['test_trial'], z['test_pos']
        for k, p in enumerate(z['preds'].astype(np.float64)):
            for i in np.unique(tr):
                m = tr == i
                rows.append(dict(model=name, subject=s, seed=k, position_rad=round(float(np.median(pos[m])), 3), **score(y[m], p[m])))
    F = pd.DataFrame(rows)
    F['toes_down'] = 0
    for s, x in F.groupby('subject'):
        F.loc[x.index[x.position_rad.isin(sorted(x.position_rad.unique())[:2])], 'toes_down'] = 1
    F.to_csv(os.path.join(EXP, 'results', 'families_rescored.csv'), index=False)
    print(R.groupby(['study', 'experiment']).agg(subjects=('subject', 'nunique'), seeds=('seed', 'nunique')).to_string())
    print(F.groupby('model').agg(subjects=('subject', 'nunique'), seeds=('seed', 'nunique'), r2=('r2', 'median')).to_string())


if __name__ == '__main__':
    main()
