"""
Complete per-position results for every saved training run, scored three ways.

Run (Git Bash, repo root):
    source ML/experiments/env.sh; export WT="$(pwd)"
    $PY notes/final_pipeline/results_tables/make_tables.py

Answer keys (what the prediction is compared with, per test trial)
------------------------------------------------------------------
passive_trial : measured torque − passive torque from the PASSIVE TRIALS of the
                TESTED session (clean static holds only, interpolated at the
                trial's angle; YES: the lab worksheet values).  This is the
                supervisor's choice.  IES has no passive trials, so nothing is
                subtracted for IES (flagged in the output).
in_trial      : measured torque − smooth line through the EMG-detected relaxed
                moments of the same trial (ML/research/truth.py), the key used
                in report 2.
own_target    : each configuration's own test target (what it was trained to
                predict, built on the tested session), i.e. how last semester
                scored.

Metrics per test trial (= one angle of one tested session)
----------------------------------------------------------
r2    = 1 − Σ(pred − key)² / Σ(key − mean(key))²
mse   = mean((pred − key)²)            [Nm²]
rmse  = sqrt(mse)                      [Nm]
nrmse = rmse / (98th − 2nd percentile of key)   (same definition as
        ML/research/evaluate.metrics)
bias  = mean(pred − key)               [Nm]

Nothing in the repo is modified; outputs go next to this script.
"""

import os
import sys
import glob
import json

import numpy as np
import pandas as pd

WT = os.environ['WT']
sys.path.insert(0, WT)
from ML.research.data import load_subject              # noqa: E402
from ML.research.passive import PassiveModel           # noqa: E402
from ML.research import truth as T                     # noqa: E402
from ML.research import experiments as E               # noqa: E402

R = os.path.join(WT, 'ML', 'research', 'results')
OUT = os.path.dirname(os.path.abspath(__file__))
FS = 100.0

PLAIN = {
    'E01_old': 'Old pipeline (last semester)',
    'E02_minimal': 'Clean passive table × correction + equal angle weights (old small network)',
    'E03_final': 'First-report pipeline: clean passive table × correction, auto-gain, per-angle corrections, divide by swing',
    'E04_final_holdgain': 'E03 but passive from the relaxed hold just before each trial',
    'E06_final_drift': 'E03 but passive slowly relaxing during the trial (fitted drift)',
    'E10_final_mvcglobal': 'E03 but EMG ÷ max push (MVC) instead of auto-gain',
    'E11_final_mvcangle': 'E03 but EMG ÷ max push at that angle instead of auto-gain',
    'E13_final_pf3': 'E03 + ×3 weight on the 2 most toes-down angles',
    'E14_final_pf10': 'E03 + ×10 weight on the 2 most toes-down angles',
    'E16_final_ga50': 'E03 + random EMG-size changes during training',
    'E18_minimal_ga50': 'E02 + random EMG-size changes during training',
    'E19_minimal_agc': 'E02 + auto-gain (old small network)',
    'E20_xsub_agc': 'E19 + pre-training on the other 5 people',
    'E21_xsub_trainmax': 'E02 + pre-training on the other 5 people',
    'E22_restanchor_final': 'New pipeline: in-trial passive line, auto-gain, per-angle corrections, divide by swing',
    'E23_restanchor_minimal': 'In-trial passive line, old small network, no auto-gain',
    'E24_final_norest': 'E03 with the relaxed periods removed from training',
    'E25_final_rest02': 'E03 with the relaxed periods weighted ×0.2',
    'E26_final_sessp99': 'E03 but each session scaled by its own EMG level (needs the whole test session)',
    'E27_ridge': 'Straight-line model (ridge regression), one per angle',
    'E28_ridge_agc': 'Straight-line model + auto-gain',
    'E29_final_win40': 'E03 with a 0.4 s window instead of 0.2 s',
    'E31_restanchor_agc_lstm8': 'In-trial passive line + auto-gain, old small network',
    'E32_restanchor_noagc_moe': 'New pipeline without auto-gain (used for CH)',
    'E33_final_noscale': 'New pipeline without dividing by the swing',
    'E34_final_oldpassive': 'New pipeline but with the old passive table',
    'E35_final_lstm8': 'New pipeline but with the old small network',
    'FINAL': 'New pipeline (auto-gain switched off automatically for CH)',
}

_B, _KEYS = {}, {}


def keys_for(subj):
    """{trial idx: {'passive_trial': key, 'in_trial': key, 't0': ...}} for a subject."""
    if subj in _KEYS:
        return _KEYS[subj]
    b = load_subject(subj)
    pm = PassiveModel('static', b)
    tr = T.subject_truth(subj)
    out = {}
    for rec in b['active']:
        out[rec['idx']] = {'passive_trial': (rec['tq'] - pm.predict(rec)).astype(np.float64),
                           'in_trial': tr[rec['idx']]['truth'].astype(np.float64),
                           't0': float(rec['t'][0]), 'session': rec['session'], 'angle': rec['angle']}
    _KEYS[subj] = out
    return out


def metrics(y, p):
    e = p - y
    ss = np.sum((y - y.mean()) ** 2)
    rng = np.percentile(y, 98) - np.percentile(y, 2)
    mse = float(np.mean(e ** 2))
    return dict(r2=float(1 - np.sum(e ** 2) / ss) if ss > 0 else np.nan, mse=mse, rmse=float(np.sqrt(mse)),
                nrmse=float(np.sqrt(mse) / rng) if rng > 0 else np.nan, bias=float(np.mean(e)),
                std_key=float(np.std(y)), n=int(len(y)))


def main():
    rows = []
    files = sorted(glob.glob(os.path.join(R, '*', '*', 'S*to*', '*', 'fast_seed*.npz')))
    for f in files:
        parts = f.replace('\\', '/').split('/')
        study, subj, prot, cfg = parts[-5], parts[-4], parts[-3], parts[-2]
        if study not in ('screen', 'confirm', 'ablation', 'final_check', 'passive'):
            continue
        seed = int(parts[-1].split('seed')[1].split('.')[0])
        z = np.load(f)
        K = keys_for(subj)
        trial, t, pred, own = z['trial'].astype(int), z['t'].astype(np.float64), z['y_pred'].astype(np.float64), z['y_true'].astype(np.float64)
        for i in np.unique(trial):
            m = trial == i
            k = np.rint((t[m] - K[i]['t0']) * FS).astype(int)
            keyset = {'passive_trial': K[i]['passive_trial'][k], 'in_trial': K[i]['in_trial'][k], 'own_target': own[m]}
            for kname, y in keyset.items():
                d = metrics(y, pred[m])
                rows.append(dict(study=study, config=cfg, description=PLAIN.get(cfg, cfg), subject=subj,
                                 direction=('learned S1 → tested S2' if prot == 'S1toS2' else 'learned S2 → tested S1'),
                                 protocol=prot, run=seed + 1, tested_session=K[i]['session'],
                                 angle_rad=round(float(K[i]['angle']), 3), angle_deg=round(float(np.degrees(K[i]['angle'])), 1),
                                 key=kname, **d))
        print('done', study, subj, prot, cfg, seed, flush=True)

    df = pd.DataFrame(rows)
    df['no_passive_trials'] = df['subject'].eq('IES') & df['key'].eq('passive_trial')
    df.to_csv(os.path.join(OUT, 'all_runs_per_position.csv'), index=False, encoding='utf-8-sig')
    print('rows', len(df))


if __name__ == '__main__':
    main()
