"""Final summary tables (markdown + Discord chunks) from the pipeline results.

python -m ML.experiments.final_report [--discord-json out.json]

FINAL = `train.py --recommended`  (static passive map + passive gain + AGC auto
+ MoE scale-only).  AGC 'auto' is ON for YES/JM/EG/HM/IES (one push direction)
and OFF for CH (pushes in both directions) → FINAL equals the P11 run for the
first five subjects and the P12 run for CH (identical configurations).
"""

import os
import re
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ML.experiments.harness import RESULTS_DIR

PIPE = os.path.join(RESULTS_DIR, 'pipeline')
SUBJECTS = ['YES', 'JM', 'IES', 'EG', 'CH', 'HM']
FINAL_RUN = {s: 'P11_core_agc_moe_scale' for s in SUBJECTS}
FINAL_RUN['CH'] = 'P12_core_moe_scale'
WEAK = {'YES': ['-0.486', '-0.381'], 'JM': ['-0.787', '-0.644', '-0.360']}

METHODS = [  # (pipeline run, label, honest)
    ('legacy',                 'Baseline (original pipeline)',               'yes'),
    ('P0_core_w',              'Static passive map + passive gain + angle-weighted loss', 'yes'),
    ('P5_core_moe',            'Passive fixes + MoE head (per-angle mean prior)', 'yes'),
    ('P7_core_agc_w',          'Passive fixes + EMG AGC + angle-weighted loss', 'yes*'),
    ('P8_core_agc_moe',        'Passive fixes + AGC + MoE (per-angle mean prior)', 'yes*'),
    ('P9_core_dual_moe',       'Passive fixes + AGC dual view + MoE',          'yes*'),
    ('P10_core_dual_w',        'Passive fixes + AGC dual view + weights',      'yes*'),
    ('P12_core_moe_scale',     'Passive fixes + MoE scale-only (no AGC)',      'yes'),
    ('P11_core_agc_moe_scale', 'FINAL: passive fixes + AGC + MoE scale-only',  'yes*'),
]
HARNESS_EXTRA = [  # screening-only variants (harness, fast protocol)
    ('rezero_drift',      'Re-zero target on in-trial rest (uses TEST torque)', 'NO'),
    ('drift_posw',        'Re-zero (drift) + angle weights (uses TEST torque)', 'NO'),
    ('anc_floor_drift',   'Re-zero + zero-anchored LSTM (uses TEST torque)',    'NO'),
    ('knotpos',           'Set-point angle instead of measured angle',          'yes'),
    ('clip12',            'Clip EMG artefacts only',                            'yes'),
    ('H_anc_pgain_smooth_floortrain_posw', 'Passive gain + zero-anchored LSTM', 'yes'),
]


def load_pipe(subject, name, protocol='fast'):
    p = os.path.join(PIPE, f'{subject}__{name}__{protocol}.json')
    return json.load(open(p)) if os.path.exists(p) else None


def load_harness(subject, name):
    p = os.path.join(RESULTS_DIR, f'{subject}__{name}__fast.json')
    return json.load(open(p)) if os.path.exists(p) else None


def r2(res, k):
    if res is None:
        return None
    s = res['summary']
    if k in s:
        return s[k]['r2'] if 'r2' in s[k] else s[k]['r2_mean']
    return None


def fmt(v, w=6):
    return f'{v:+{w}.2f}' if v is not None else f"{'—':>{w}}"


def methods_table():
    cols = [(s, k) for s in WEAK for k in WEAK[s]]
    hdr = f"{'Method':<50} {'honest':<6} " + ' '.join(f"{s[:3]+k:>9}" for s, k in cols)
    lines = [hdr, '-' * len(hdr)]
    for name, label, honest in METHODS:
        vals = [r2(load_pipe(s, name), k) for s, k in cols]
        if all(v is None for v in vals):
            continue
        lines.append(f"{label[:50]:<50} {honest:<6} " + ' '.join(fmt(v, 9) for v in vals))
    lines.append('— screening-only variants (experiment harness) —')
    for name, label, honest in HARNESS_EXTRA:
        vals = [r2(load_harness(s, name), k) for s, k in cols]
        if all(v is None for v in vals):
            continue
        lines.append(f"{label[:50]:<50} {honest:<6} " + ' '.join(fmt(v, 9) for v in vals))
    return '\n'.join(lines)


def final_table(subject):
    leg, fin = load_pipe(subject, 'legacy'), load_pipe(subject, FINAL_RUN[subject])
    if leg is None or fin is None:
        return f'({subject}: results pending)'
    keys = [k for k in leg['summary'] if k != 'pooled']
    lines = [f"{'pos':>7} | {'R2 base':>7} {'R2 new':>7} | {'MSE base':>8} {'MSE new':>7} | {'RMSE b':>6} {'RMSE n':>6}",
             '-' * 66]
    for k in keys + ['pooled']:
        a, b = leg['summary'][k], fin['summary'][k]
        r2a = a['r2'] if k != 'pooled' else None
        r2b = b['r2'] if k != 'pooled' else None
        lines.append(f"{k:>7} | {fmt(r2a, 7)} {fmt(r2b, 7)} | {a['mse']:8.2f} {b['mse']:7.2f} | "
                     f"{a['rmse']:6.2f} {b['rmse']:6.2f}")
    return '\n'.join(lines)


def orig_table():
    """Original training protocol (batch 8): legacy seed 0 (log) vs FINAL seed 0."""
    lines = []
    for s in ('YES', 'JM'):
        logf = os.path.join(PIPE, f'{s}__legacy__orig_seed0.log')
        fin = load_pipe(s, 'P11_final', 'orig')
        if not os.path.exists(logf) or fin is None:
            lines.append(f'{s}: pending')
            continue
        line = [l for l in open(logf, encoding='utf-8') if 'seed 0:' in l][0]
        leg = dict(re.findall(r'([+-]\d\.\d{3}):([+-]\d+\.\d+)', line))
        keys = list(fin['summary'].keys())
        keys = [k for k in keys if k != 'pooled']
        lines.append(f'{s}  ' + '  '.join(f"{k}: {float(leg.get(k, 'nan')):+.2f}→{fin['summary'][k]['r2']:+.2f}"
                                          for k in keys))
    return '\n'.join(lines)


def regression_table():
    lines = [f"{'subj':<4} | {'weakest angle':>13} {'R2 base':>8} {'R2 new':>7} | {'min R2 base':>11} {'min new':>7} | "
             f"{'pooled MSE':>10} {'→':>1} {'new':>5}", '-' * 84]
    for s in SUBJECTS:
        leg, fin = load_pipe(s, 'legacy'), load_pipe(s, FINAL_RUN[s])
        if leg is None or fin is None:
            lines.append(f'{s:<4} | pending')
            continue
        keys = [k for k in leg['summary'] if k != 'pooled']
        most_pf = min(keys, key=float)
        rb = [leg['summary'][k]['r2'] for k in keys]
        rn = [fin['summary'][k]['r2'] for k in keys]
        lines.append(f"{s:<4} | {most_pf:>13} {leg['summary'][most_pf]['r2']:+8.2f} {fin['summary'][most_pf]['r2']:+7.2f} | "
                     f"{min(rb):+11.2f} {min(rn):+7.2f} | {leg['summary']['pooled']['mse']:10.2f}   "
                     f"{fin['summary']['pooled']['mse']:5.2f}")
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--discord-json', default=None)
    a = p.parse_args()
    sections = {
        'methods': methods_table(),
        'YES': final_table('YES'), 'JM': final_table('JM'),
        'orig': orig_table(), 'regression': regression_table(),
    }
    for k, v in sections.items():
        print(f'\n=== {k}\n{v}')
    if a.discord_json:
        json.dump(sections, open(a.discord_json, 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    main()
