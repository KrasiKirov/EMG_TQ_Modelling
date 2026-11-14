"""Choose the LSTM hyperparameters from the tuning grid (EXPERIMENTS/configs/HT_*.json) on VALIDATION loss only.

    python EXPERIMENTS/analysis/select_hparams.py [--baseline HT_u8_l2_d30_lr003]

Every tuning run saves its best session-1 validation loss (mean-squared error in Nm², purged validation split).
The session-2 test predictions are never read here.

Rule:
  1. per person, the median validation loss over seeds;
  2. per configuration and person, the log of (that loss / the baseline configuration's loss), so every person counts
     equally whatever the size of their torque;
  3. the configuration with the lowest mean log-ratio across people is the best;
  4. the chosen one is the configuration with the fewest parameters whose mean log-ratio is within one standard error
     (across people) of the best ("one-standard-error rule": prefer the simplest model the data cannot tell apart).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PEOPLE = ['YES', 'JM', 'EG', 'IES', 'CH', 'HM']


def choose(table, baseline):
    """table: {config: (n_params, {person: validation loss})}. Returns (best, chosen, summary)."""
    people = sorted(set.intersection(*(set(v) for _, v in table.values())))
    base = table[baseline][1]
    summary = {}
    for cfg, (n_params, loss) in table.items():
        lr = np.array([np.log(loss[p] / base[p]) for p in people])
        summary[cfg] = {'n_params': n_params, 'mean_log_ratio': float(lr.mean()),
                        'se': float(lr.std(ddof=1) / np.sqrt(len(lr))) if len(lr) > 1 else 0.0,
                        'better_people': int(np.sum(lr < 0)), 'n_people': len(lr)}
    best = min(summary, key=lambda c: summary[c]['mean_log_ratio'])
    limit = summary[best]['mean_log_ratio'] + summary[best]['se']
    ok = [c for c in summary if summary[c]['mean_log_ratio'] <= limit]
    chosen = min(ok, key=lambda c: (summary[c]['n_params'], summary[c]['mean_log_ratio']))
    return best, chosen, summary


def finalists(summary, baseline, k=4, also=()):
    """The k configurations with the lowest mean log-ratio, then `also` (e.g. the screen's one-standard-error pick,
    so the final rule can still prefer a smaller network), then the baseline (kept as the common reference)."""
    out = sorted(summary, key=lambda c: summary[c]['mean_log_ratio'])[:k]
    for c in [*also, baseline]:
        if c not in out:
            out.append(c)
    return out


def epoch_gain(long, short):
    """long / short: {(person, seed): validation loss} for one configuration trained with a higher / lower epoch
    limit. Runs are paired by person and seed; per person the mean log-ratio over its seeds, then across people."""
    pairs = sorted(set(long) & set(short))
    by_person = {}
    for p, s in pairs:
        by_person.setdefault(p, []).append(np.log(long[(p, s)] / short[(p, s)]))
    per = np.array([np.mean(v) for v in by_person.values()])
    return {'mean_log_ratio': float(per.mean()), 'n_pairs': len(pairs), 'n_people': len(per),
            'better_people': int(np.sum(per < 0)), 'per_person': {p: float(np.mean(v)) for p, v in by_person.items()}}


def val_losses(exp, results=ROOT / 'EXPERIMENTS/results'):
    """{(person, seed): best session-1 validation loss} of one experiment (session 1 -> 2)."""
    out = {}
    for p in PEOPLE:
        for f in sorted((Path(results) / exp / p / 'S1toS2').glob('seed*.json')):
            out[(p, int(f.stem[4:]))] = json.loads(f.read_text())['train_info']['best_val_loss']
    return out


def n_params(units, layers, n_in=5):
    total, d = 0, n_in
    for _ in range(layers):
        total += 4 * units * (d + units + 1)
        d = units
    return total + units + 1


def load_table(results=ROOT / 'EXPERIMENTS/results', configs=ROOT / 'EXPERIMENTS/configs', pattern='HT_*.json'):
    table, coverage = {}, {}
    for spec in sorted(Path(configs).glob(pattern)):
        ch = json.loads(spec.read_text(encoding='utf-8'))['changes']
        loss = {}
        for p in PEOPLE:
            vals = [json.loads(f.read_text())['train_info']['best_val_loss']
                    for f in sorted((Path(results) / spec.stem / p / 'S1toS2').glob('seed*.json'))]
            if vals:
                loss[p] = float(np.median(vals))
                coverage[(spec.stem, p)] = len(vals)
        if len(loss) == len(PEOPLE):
            table[spec.stem] = (n_params(ch['units'], ch['layers']), loss)
    return table, coverage


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--baseline', default='HT_u8_l2_d30_lr003')
    ap.add_argument('--json', help='also write the summary here')
    a = ap.parse_args()
    table, coverage = load_table()
    if a.baseline not in table:
        sys.exit(f'baseline {a.baseline} has no complete results yet ({len(table)} complete configurations)')
    best, chosen, summary = choose(table, a.baseline)
    print(f'{len(table)} configurations with all six people; seeds per person: {sorted(set(coverage.values()))}')
    print(f"{'configuration':24s} {'params':>7s} {'val loss vs baseline':>21s} {'± SE':>6s} {'people better':>14s}")
    for c in sorted(summary, key=lambda c: summary[c]['mean_log_ratio']):
        s = summary[c]
        mark = ' <- best' if c == best else ''
        mark += ' <- chosen (1-SE rule)' if c == chosen else ''
        print(f"{c:24s} {s['n_params']:7d} {100 * (np.exp(s['mean_log_ratio']) - 1):+20.1f}% {100 * s['se']:5.1f}% "
              f"{s['better_people']:>8d}/{s['n_people']}{mark}")
    if a.json:
        Path(a.json).write_text(json.dumps({'best': best, 'chosen': chosen, 'baseline': a.baseline, 'summary': summary}, indent=1))


if __name__ == '__main__':
    main()
