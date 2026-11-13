"""Aggregate experiment results into per-position comparison tables.

python -m ML.experiments.compare [--protocol fast] [--subjects YES JM] [--metric r2_mean]
"""

import os
import sys
import glob
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ML.experiments.harness import RESULTS_DIR


def load(protocol, results_dir=RESULTS_DIR):
    out = {}
    for f in glob.glob(os.path.join(results_dir, f'*__{protocol}.json')):
        with open(f) as fh:
            r = json.load(fh)
        out[(r['subject'], r['name'])] = r
    return out


def table(res, subject, metric='r2_mean', baseline='baseline', sort_by='worst2'):
    rows = {n: r for (s, n), r in res.items() if s == subject}
    if not rows:
        return f'(no results for {subject})'
    base = rows.get(baseline)
    keys = list(next(iter(rows.values()))['summary'].keys())
    # the two lowest-R² positions under baseline = the "problem" positions
    if base:
        worst = sorted(keys, key=lambda k: base['summary'][k]['r2_mean'])[:2]
    else:
        worst = keys[:2]
    lines = [f'\n### {subject}  (mean over seeds; problem positions = {worst}; '
             f'"ens" = R² of seed-averaged prediction)']
    hdr = f"{'variant':28s} " + ' '.join(f'{k:>7s}' for k in keys) + \
          f" | {'worst2':>7s} {'others':>7s} {'pooled':>7s} | {'worst2ens':>9s} {'seeds':>5s}"
    lines += [hdr, '-' * len(hdr)]

    def score(r):
        s = r['summary']
        w = np.mean([s[k][metric] for k in worst])
        o = np.mean([s[k][metric] for k in keys if k not in worst])
        pooled = np.mean([x['eval']['overall']['r2'] for x in r['seeds']])
        we = np.mean([r['ensemble']['per_position'][k]['target']['r2'] for k in worst])
        return w, o, pooled, we

    order = sorted(rows, key=lambda n: -score(rows[n])[0])
    for n in order:
        r = rows[n]
        w, o, pooled, we = score(r)
        vals = ' '.join(f"{r['summary'][k][metric]:+7.3f}" for k in keys)
        lines.append(f'{n[:28]:28s} {vals} | {w:+7.3f} {o:+7.3f} {pooled:7.4f} | {we:+9.3f} {len(r["seeds"]):5d}')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', default='fast')
    p.add_argument('--subjects', nargs='+', default=['YES', 'JM'])
    p.add_argument('--metric', default='r2_mean', choices=['r2_mean', 'r2_ref_mean', 'rmse_mean'])
    p.add_argument('--results-dir', default=RESULTS_DIR)
    a = p.parse_args()
    res = load(a.protocol, a.results_dir)
    for s in a.subjects:
        print(table(res, s, a.metric))


if __name__ == '__main__':
    main()
