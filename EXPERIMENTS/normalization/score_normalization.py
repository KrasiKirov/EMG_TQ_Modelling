"""
Score the normalization study and apply the selection rule of PLAN.md.

    python EXPERIMENTS/normalization/score_normalization.py [--stage 1|2]

Reads EXPERIMENTS/results/<exp>/<subject>/S1toS2/seed<k>.npz (LSTM arms N0-N3, ridge
references NR0-NR3) and writes, in EXPERIMENTS/normalization/:
  runs_stage<k>.csv     one row per experiment x subject x seed x position
  cells_stage<k>.csv    one row per experiment x subject x position (over seeds)
  decision_stage1.md    the rule of PLAN.md applied mechanically (stage 1 only)

Target = total measured torque (what the runs were trained on, y_own_target).
Metrics per position (one session-2 trial):
  R2     1 - sum(e^2) / sum((y - mean y)^2),   e = prediction - truth
  MAE    mean |e|                               [Nm]
  bias   mean e (offset; > 0 = predicts too high)  [Nm]
  VAF    1 - var(e) / var(y): R2 with the offset removed (shape only)
  peak   max |y|                                [Nm]
Reference 'S1 mean': for every position, the mean torque of the session-1 trial at that
angle over its training part (first 80 % of its windows), predicted as a constant.
Toes-down = the 2 most toes-down positions of each person.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP)
sys.path.insert(0, ROOT)

ARMS = ['N0_total_trainmax', 'N1_total_trainp99', 'N2_total_mvc', 'N3_total_sessionp99']
REFS = ['NR' + a[1:] for a in ARMS]
LABEL = {'N0': 'N0 control (session-1 max)', 'N1': 'N1 session-1 99th percentile',
         'N2': 'N2 MVC', 'N3': 'N3 per-session 99th percentile'}
STAGE_SUBJECTS = {1: ['YES', 'JM', 'EG'], 2: ['IES', 'CH', 'HM']}
SIMPLICITY = ['N1', 'N2', 'N3']


def metrics(y, p):
    e = p - y
    ss = np.sum((y - y.mean()) ** 2)
    return dict(r2=1 - np.sum(e ** 2) / ss, mae=np.mean(np.abs(e)), rmse=np.sqrt(np.mean(e ** 2)),
                bias=np.mean(e), vaf=1 - np.var(e) / np.var(y), peak=np.max(np.abs(y)), sd=np.std(y))


def s1_means(subject):
    """{angle: mean torque of the session-1 trial at that angle, training part}"""
    from ML.research.data import load_subject
    from ML.research import config as C
    out = {}
    for r in load_subject(subject)['active']:
        if r['session'] != 'S1':
            continue
        n_win = len(r['t']) - C.WINDOW + 1
        cut = int(n_win * C.TRAIN_FRAC) + C.WINDOW - 1
        out[float(r['angle'])] = float(np.mean(r['tq'][:cut]))
    return out


def score(subjects):
    rows = []
    for subject in subjects:
        means = s1_means(subject)
        ref_done = False
        for exp in ARMS + REFS:
            for f in sorted(glob.glob(os.path.join(EXP, 'results', exp, subject, 'S1toS2', 'seed*.npz'))):
                z = np.load(f)
                y, p, ang = z['y_own_target'].astype(float), z['y_pred'].astype(float), z['angle'].astype(float)
                seed = int(os.path.basename(f)[4:-4])
                angles = sorted(np.unique(ang))
                for a in angles:
                    m = ang == a
                    toes = int(a in angles[:2])
                    rows.append(dict(exp=exp[:3].rstrip('_'), exp_id=exp, subject=subject, seed=seed,
                                     position_rad=round(a, 3), toes_down=toes, **metrics(y[m], p[m])))
                    if not ref_done:
                        s1a = min(means, key=lambda k: abs(k - a))
                        assert abs(s1a - a) < 0.05, (subject, a, s1a)
                        rows.append(dict(exp='S1mean', exp_id='S1 mean at this position', subject=subject, seed=0,
                                         position_rad=round(a, 3), toes_down=toes,
                                         **metrics(y[m], np.full(m.sum(), means[s1a]))))
                ref_done = True
    return pd.DataFrame(rows)


def cells(runs):
    g = runs.groupby(['exp', 'subject', 'position_rad', 'toes_down'])
    out = g.agg(seeds=('seed', 'nunique'), r2=('r2', 'median'), mae=('mae', 'median'), bias=('bias', 'median'),
                vaf=('vaf', 'median'), peak=('peak', 'median'), mae_worst=('mae', 'max'), mae_best=('mae', 'min'),
                r2_min=('r2', 'min'), r2_max=('r2', 'max')).reset_index()
    return out


def group(c, exp, toes, subject=None):
    x = c[(c.exp == exp) & (c.toes_down == toes)]
    if subject:
        x = x[x.subject == subject]
    return x


def decide(c, subjects):
    L = ['# Stage 1 decision (rule of PLAN.md applied mechanically)', '',
         'Numbers: median over the 3 seeds for each person and position, then median over the cells of the group. '
         'MAE in Nm, total torque.', '']
    base = {k: group(c, 'N0', k) for k in (0, 1)}
    n0_td, n0_ot = base[1].mae.median(), base[0].mae.median()
    n0_worst = base[1].mae_worst.median()
    L += ['| Arm | toes-down MAE | vs N0 | people better at toes-down | other MAE | worst other person vs N0 | '
          'toes-down worst-seed MAE | passes |', '|---|---|---|---|---|---|---|---|']
    passing = {}
    for arm in ['N0', 'N1', 'N2', 'N3']:
        td, ot = group(c, arm, 1).mae.median(), group(c, arm, 0).mae.median()
        worst = group(c, arm, 1).mae_worst.median()
        better = [s for s in subjects if group(c, arm, 1, s).mae.median() < group(c, 'N0', 1, s).mae.median()]
        other_ratio = max(group(c, arm, 0, s).mae.median() / group(c, 'N0', 0, s).mae.median() for s in subjects)
        if arm == 'N0':
            L.append(f'| {LABEL[arm]} | {td:.3f} | - | - | {ot:.3f} | - | {worst:.3f} | reference |')
            continue
        r1 = (n0_td - td) >= 0.05 and td <= 0.9 * n0_td
        r2 = len(better) >= 2
        r3 = (ot - n0_ot) <= 0.05 and other_ratio <= 1.10
        r4 = worst <= n0_worst
        ok = r1 and r2 and r3 and r4
        if ok:
            passing[arm] = td
        fails = [n for n, r in zip(('1 toes-down', '2 people', '3 elsewhere', '4 worst seed'), (r1, r2, r3, r4)) if not r]
        L.append(f'| {LABEL[arm]} | {td:.3f} | {td - n0_td:+.3f} ({(td / n0_td - 1) * 100:+.0f} %) | '
                 f'{len(better)} of 3 ({", ".join(better) or "none"}) | {ot:.3f} | {(other_ratio - 1) * 100:+.0f} % | '
                 f'{worst:.3f} | {"yes" if ok else "no, fails " + ", ".join(fails)} |')
    if passing:
        best = min(passing, key=passing.get)
        close = [a for a in SIMPLICITY if a in passing and passing[a] - passing[best] <= 0.02]
        winner = close[0]
        L += ['', f'**Winner: {LABEL[winner]}.**' + (f' ({LABEL[best]} was lowest, but within 0.02 Nm; the simpler '
                                                      f'arm wins.)' if winner != best else '')]
    else:
        winner = 'N0'
        L += ['', '**No arm passes. N0 (control) stays.**']
    others = {a: group(c, a, 1).mae.median() for a in ['N1', 'N2', 'N3'] if a != winner}
    runner = min(others, key=others.get)
    L += [f'Stage 2 runs: {winner}' + (f' and {runner} (lowest toes-down MAE among the others, for information only).'
                                       if winner == 'N0' else ' and N0.')]
    return winner, runner, '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=int, default=1, choices=[1, 2])
    a = ap.parse_args()
    subjects = STAGE_SUBJECTS[a.stage]
    runs = score(subjects)
    runs.to_csv(os.path.join(HERE, f'runs_stage{a.stage}.csv'), index=False, float_format='%.5f')
    c = cells(runs)
    c.to_csv(os.path.join(HERE, f'cells_stage{a.stage}.csv'), index=False, float_format='%.5f')
    print(runs.groupby(['exp', 'subject']).seed.nunique().unstack().to_string())
    if a.stage == 1:
        winner, runner, text = decide(c, subjects)
        open(os.path.join(HERE, 'decision_stage1.md'), 'w', encoding='utf-8').write(text)
        print(text)


if __name__ == '__main__':
    main()
