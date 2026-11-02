"""
Score the total-torque study (EXPERIMENTS/results/TT*/<subject>/<direction>/seed<k>.npz).

    python EXPERIMENTS/code/score_total_torque.py

Answer key: the processed MEASURED TOTAL TORQUE of the tested session, read
again from the subject bundle at each window's last sample (independent of the
saved target, which must match it exactly; a mismatch stops the script).  No
passive table is involved, and the passive-subtracted keys of score.py are
never used here.

Positions: one test trial = one position.  It is labelled by its own mean
angle (rad, 3 decimals) and matched to the source-session position whose
training-trial angle is nearest (the same nearest-knot rule the balancing uses,
knots = training angles merged within 0.02 rad); the distance is saved as
match_distance_rad.

Metrics per run x position (e = prediction - measurement, y = measurement):
  MAE  = mean|e|                       Nm
  RMSE = sqrt(mean e^2)                Nm
  bias = mean(e)                       Nm  (prediction minus measurement)
  R2   = 1 - sum(e^2) / sum((y - mean y)^2)        raw, can be negative
  VAF  = 100 * (1 - var(y - p) / var(y))  %  (population variances; ignores a
         constant offset, unlike R2)
Reference predictor REF_train_mean_total: for each tested position, the mean
measured total torque of the training-part targets (first 80 % of the windows)
of the matched source-session position, scored on the same test samples.

Outputs
  EXPERIMENTS/results/total_torque_scores.csv     one row per run x position
  EXPERIMENTS/results/total_torque_summary.csv    medians over seeds / positions with run counts
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
from ML.research.data import load_subject      # noqa: E402
from ML.research import config as RC           # noqa: E402
from ML.research.train import angle_groups, snap   # noqa: E402

RES = os.path.join(EXP, 'results')
WINDOW = RC.WINDOW


def metrics(y, p):
    e = p - y
    ss = np.sum((y - y.mean()) ** 2)
    vy = np.var(y)
    return dict(mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e ** 2))), bias=float(np.mean(e)),
                r2=float(1 - np.sum(e ** 2) / ss) if ss > 0 else np.nan,
                vaf=float(100 * (1 - np.var(y - p) / vy)) if vy > 0 else np.nan,
                mean_measured=float(y.mean()), sd_measured=float(np.std(y)),
                peak_abs_measured=float(np.max(np.abs(y))), n=int(len(y)))


def train_part_targets(rec):
    """Targets of the training windows of one source-session trial (first 80 % of its windows)."""
    n_win = len(rec['tq']) - WINDOW + 1
    return rec['tq'][WINDOW - 1: WINDOW - 1 + int(n_win * RC.TRAIN_FRAC)]


def position_match(bundle, direction):
    src, tgt = ('S1', 'S2') if direction == 'S1toS2' else ('S2', 'S1')
    src_recs = [r for r in bundle['active'] if r['session'] == src]
    knots = angle_groups(sorted({float(np.round(r['angle'], 3)) for r in src_recs}))
    k_src = snap([r['angle'] for r in src_recs], knots)
    ref = {j: float(np.mean(np.concatenate([train_part_targets(r) for r, k in zip(src_recs, k_src) if k == j])))
           for j in range(len(knots))}
    out = {}
    for r in bundle['active']:
        if r['session'] == tgt:
            j = int(snap([r['angle']], knots)[0])
            out[r['idx']] = dict(knot=j, source_angle=float(knots[j]), distance=float(abs(knots[j] - r['angle'])),
                                 ref_mean=ref[j])
    return out


def main():
    rows, bundles, matches = [], {}, {}
    files = sorted(glob.glob(os.path.join(RES, 'TT*', '*', 'S*to*', 'seed*.npz')))
    for f in files:
        p = f.replace('\\', '/').split('/')
        exp, s, d, seed = p[-4], p[-3], p[-2], int(p[-1][4:-4])
        if not os.path.exists(f[:-4] + '.json'):
            continue                                   # job still running (JSON is written last)
        if s not in bundles:
            bundles[s] = load_subject(s)
        b = bundles[s]
        if (s, d) not in matches:
            matches[(s, d)] = position_match(b, d)
        recs = {r['idx']: r for r in b['active']}
        with np.load(f) as z:
            trial, last = z['trial'].astype(int), z['last'].astype(int)
            pred, saved = z['y_pred'].astype(np.float64), z['y_measured_total'].astype(np.float64)
        key = np.array([recs[i]['tq'][k] for i, k in zip(trial, last)])
        if not np.array_equal(key.astype(np.float32), saved.astype(np.float32)):
            raise RuntimeError(f'{f}: saved target is not the measured total torque')
        info = json.load(open(f[:-4] + '.json'))
        for i in np.unique(trial):
            m = trial == i
            mt = matches[(s, d)][i]
            rows.append(dict(experiment=exp, subject=s, direction=d, seed=seed, tested_session=recs[i]['session'],
                             trial=int(i), position_rad=round(recs[i]['angle'], 3),
                             matched_source_rad=round(mt['source_angle'], 3), match_distance_rad=mt['distance'],
                             epochs=info['train_info']['epochs'], best_epoch=info['train_info']['best_epoch'],
                             train_s=info['train_s'], **metrics(key[m], pred[m])))
    # reference predictor, on the same test samples (every window's last sample)
    for (s, d), mt in matches.items():
        recs = {r['idx']: r for r in bundles[s]['active']}
        for i, x in mt.items():
            y = recs[i]['tq'][WINDOW - 1:]
            rows.append(dict(experiment='REF_train_mean_total', subject=s, direction=d, seed=-1,
                             tested_session=recs[i]['session'], trial=int(i), position_rad=round(recs[i]['angle'], 3),
                             matched_source_rad=round(x['source_angle'], 3), match_distance_rad=x['distance'],
                             **metrics(y, np.full(len(y), x['ref_mean']))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, 'total_torque_scores.csv'), index=False, encoding='utf-8-sig')
    # summary: per position median over seeds, then subject-level median over positions
    met = ['mae', 'rmse', 'bias', 'r2', 'vaf']
    per_pos = df.groupby(['experiment', 'subject', 'direction', 'position_rad'])
    pos = per_pos[met].median().join(per_pos['seed'].nunique().rename('n_seeds')).reset_index()
    subj = pos.groupby(['experiment', 'subject', 'direction'])
    summ = subj[met].median().join(subj['n_seeds'].min().rename('min_seeds_per_position')).join(
        subj['position_rad'].nunique().rename('n_positions')).reset_index()
    summ.to_csv(os.path.join(RES, 'total_torque_summary.csv'), index=False, encoding='utf-8-sig')
    print(f'{len(files)} run files, {len(df)} rows')
    with pd.option_context('display.width', 200, 'display.max_rows', 200):
        print(summ.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
