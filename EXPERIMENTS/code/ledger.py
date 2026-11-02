"""Update or add one row of EXPERIMENTS/experiment_ledger.csv.

    python EXPERIMENTS/code/ledger.py X04_equal_angle_weight status=done "decision=keep" "reason=..."
"""
import os
import sys
import csv

LEDGER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiment_ledger.csv')


def update(exp_id, **fields):
    with open(LEDGER, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    cols = list(rows[0].keys()) if rows else ['experiment_id']
    for k in fields:
        if k not in cols:
            cols.append(k)
    for r in rows:
        if r['experiment_id'] == exp_id:
            r.update(fields)
            break
    else:
        rows.append({'experiment_id': exp_id, **fields})
    with open(LEDGER, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    update(sys.argv[1], **dict(a.split('=', 1) for a in sys.argv[2:]))
