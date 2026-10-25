"""
Markdown tables for the report, generated from analyze.py's summary.json,
so every number in the report comes from the result files (nothing is typed
by hand).

    python -m ML.research.tables --study screen --configs E01_old E03_final ... --out notes/report2/tables.md
"""

import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ML.research import config as C
from ML.research import experiments

FOLDS = ('S1toS2', 'S2toS1')


def load(study):
    with open(os.path.join(C.RESULTS_DIR, study, '_summary', 'summary.json')) as f:
        return json.load(f)


def cell_mean(tab, subs, config, key):
    v = [tab[f'{s}|{p}|{config}'][key] for s in subs for p in FOLDS if f'{s}|{p}|{config}' in tab]
    n = [tab[f'{s}|{p}|{config}']['n_seeds'] for s in subs for p in FOLDS if f'{s}|{p}|{config}' in tab]
    return (float(np.nanmean(v)) if v else None), (min(n) if n else 0), len(v)


def overview(S, configs, labels=None):
    """One row per configuration: means over subjects x folds (x seeds)."""
    t, subs = S['truth'], S['subjects']
    o = S['own']
    lines = ['| # | configuration | R² (truth) | R² weak-2 | RMSE weak-2 (Nm) | RMSE all (Nm) | RMSE (% MVC) | R² active only | phantom at rest (Nm) | R² (own target) | seeds |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for i, c in enumerate(configs):
        m = {k: cell_mean(t, subs, c, k)[0] for k in ('primary', 'weak2', 'weak2_rmse', 'mean_rmse',
                                                      'mean_rmse_pct_mvc', 'r2_active', 'false_rest')}
        own, n, cells = cell_mean(o, subs, c, 'primary')
        if m['primary'] is None:
            continue
        f = lambda v, d=2, sg=True: '—' if v is None else (f'{v:+.{d}f}' if sg else f'{v:.{d}f}')
        name = (labels or {}).get(c, c)
        flag = ' (transductive)' if experiments.CONFIGS.get(c, {}).get('emg') == 'session_p99' else ''
        lines.append(f"| {c[:3]} | {name}{flag} | {f(m['primary'], 3)} | {f(m['weak2'])} | {f(m['weak2_rmse'], 2, False)} | "
                     f"{f(m['mean_rmse'], 2, False)} | {f(m['mean_rmse_pct_mvc'], 1, False)} | {f(m['r2_active'])} | "
                     f"{f(m['false_rest'], 2, False)} | {f(own, 3)} | {n}{'' if cells == 2 * len(subs) else f' ({cells} cells)'} |")
    return '\n'.join(lines)


def per_angle(S, subject, protocol, configs, labels=None, key='r2', d=2):
    """Per-angle values (seed mean) for one subject and fold."""
    t = S['truth']
    rows = [(c, t.get(f'{subject}|{protocol}|{c}')) for c in configs]
    rows = [(c, x) for c, x in rows if x]
    if not rows:
        return ''
    angles = rows[0][1]['angles']
    head = '| configuration | ' + ' | '.join(f"{float(a.split(':')[1]):+.3f}" for a in angles) + ' |'
    lines = [head, '|---' * (len(angles) + 1) + '|']
    for c, x in rows:
        vals = x[key] if key in x else x.get(f'{key}_angles')
        lines.append(f'| {(labels or {}).get(c, c)} | ' + ' | '.join(
            ('—' if v is None or not np.isfinite(v) else (f'{v:+.{d}f}' if key == 'r2' else f'{v:.{d}f}')) for v in vals) + ' |')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--study', default='screen')
    ap.add_argument('--configs', nargs='+', required=True)
    ap.add_argument('--angle-configs', nargs='+', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    S = load(a.study)
    parts = [f'## Overview ({a.study})\n', overview(S, a.configs)]
    for sub in S['subjects']:
        for p in FOLDS:
            for key, title in (('r2', 'R²'), ('rmse', 'RMSE (Nm)')):
                parts.append(f'\n### {sub} {p}: per-angle {title}\n')
                parts.append(per_angle(S, sub, p, a.angle_configs or a.configs, key=key))
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts) + '\n')
    print('\n'.join(parts))


if __name__ == '__main__':
    main()
