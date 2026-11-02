"""
Short-calibration test on top of the N0 runs (rules fixed before any result was seen).

    python EXPERIMENTS/normalization/calibration_test.py

Question: if the first few seconds of each session-2 trial (torque measured) are used to correct the
N0 predictions, how much of the day-to-day drift goes away, and how short can that block be?

Rules
-----
* Model: the existing N0 runs (total torque, 0.2 s window, balanced positions, session-1 maximum
  scaling), all 6 people, seeds 0, 1, 2, session 1 -> session 2. No retraining.
* Each session-2 trial is one angle (~89 s). Calibration block = its first C seconds,
  C in {5, 10, 20, 30} s, counted from the first predicted sample.
* Corrections, fitted on the calibration block only, per trial (= per angle):
    offset       p' = p + b,       b = mean(y - p)
    gain+offset  p' = a p + b,     least squares; a clipped to [0.5, 2] (b refitted with the
                 clipped a); if the block's torque standard deviation is below 0.2 Nm (no push to
                 learn a gain from), offset only.
* Scoring: only samples from 30 s after the first predicted sample to the end of the trial, for
  every C and for the uncorrected N0, so all variants are scored on the same unseen stretch.
* Summary as everywhere in this study: per person and position, median over the 3 seeds; then
  median over the cells of a group (toes-down = each person's 2 most toes-down positions).
  R2 below zero shown as 0.
* Headline, chosen in advance: gain+offset with C = 10 s. The other lengths are shown as a curve.

Writes EXPERIMENTS/normalization/calibration_runs.csv and calibration_summary.md.
"""
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
SUBJECTS = ['YES', 'JM', 'EG', 'IES', 'CH', 'HM']
LENGTHS = [5, 10, 20, 30]
SCORE_FROM_S = 30.0
GAIN_LIMITS = (0.5, 2.0)
MIN_SD = 0.2


def metrics(y, p):
    e = p - y
    ss = np.sum((y - y.mean()) ** 2)
    return dict(mae=np.mean(np.abs(e)), r2=1 - np.sum(e ** 2) / ss, vaf=1 - np.var(e) / np.var(y), bias=np.mean(e))


def correct(p_cal, y_cal, p, mode):
    if mode == 'offset' or np.std(y_cal) < MIN_SD:
        return p + np.mean(y_cal - p_cal)
    a, b = np.polyfit(p_cal, y_cal, 1)
    a = float(np.clip(a, *GAIN_LIMITS))
    b = float(np.mean(y_cal - a * p_cal))
    return a * p + b


def main():
    rows = []
    for subject in SUBJECTS:
        for f in sorted(glob.glob(os.path.join(EXP, 'results', 'N0_total_trainmax', subject, 'S1toS2', 'seed*.npz'))):
            z = np.load(f)
            y, p, ang, t = (z[k].astype(float) for k in ('y_own_target', 'y_pred', 'angle', 't'))
            seed = int(os.path.basename(f)[4:-4])
            angles = sorted(np.unique(ang))
            for a in angles:
                m = ang == a
                yy, pp, tt = y[m], p[m], t[m] - t[m][0]
                score = tt >= SCORE_FROM_S
                base = dict(subject=subject, seed=seed, position_rad=round(a, 3), toes_down=int(a in angles[:2]))
                rows.append(dict(base, variant='N0, no calibration', seconds=0,
                                 **metrics(yy[score], pp[score])))
                for c in LENGTHS:
                    cal = tt < c
                    for mode in ('offset', 'gain+offset'):
                        pc = correct(pp[cal], yy[cal], pp, mode)
                        rows.append(dict(base, variant=mode, seconds=c, **metrics(yy[score], pc[score])))
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(HERE, 'calibration_runs.csv'), index=False, float_format='%.5f')

    cells = runs.groupby(['variant', 'seconds', 'subject', 'position_rad', 'toes_down'])[['mae', 'r2', 'vaf', 'bias']] \
        .median().reset_index()
    cells['r2'] = cells.r2.clip(lower=0)
    group = cells.groupby(['variant', 'seconds', 'toes_down'])[['mae', 'r2', 'vaf']].median().unstack('toes_down')
    person = cells.groupby(['variant', 'seconds', 'subject', 'toes_down']).mae.median().unstack(['toes_down', 'subject'])
    out = ['# Short-calibration test (N0 runs, 6 people, 3 seeds, scored from 30 s to the end of each trial)', '',
           '## Group medians (toes-down = 1)', '', '```', group.round(3).to_string(), '```', '',
           '## MAE per person (Nm)', '', '```', person.round(2).to_string(), '```', '']
    open(os.path.join(HERE, 'calibration_summary.md'), 'w', encoding='utf-8').write('\n'.join(out))
    print('\n'.join(out))


if __name__ == '__main__':
    main()
