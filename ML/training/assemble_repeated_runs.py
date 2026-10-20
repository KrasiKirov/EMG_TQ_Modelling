"""Assemble per-subject repeated-run metrics into one complete result file."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ML.config import ML_DIR, SUBJECTS
from ML.training.repeated_runs import _json_safe, _write_summary_csv


def main():
    report_root = os.path.join(ML_DIR, 'reports', 'repeated_runs')
    run_results = []
    missing = []
    for subject in SUBJECTS:
        path = os.path.join(report_root, subject, 'metrics.json')
        if not os.path.exists(path):
            missing.append(path)
            continue
        with open(path) as handle:
            subject_results = json.load(handle)
        if len(subject_results) != 10:
            raise SystemExit(f'{subject}: expected 10 runs, found {len(subject_results)}')
        run_results.extend(subject_results)

    if missing:
        raise SystemExit('Missing subject metrics:\n' + '\n'.join(missing))
    if len(run_results) != 10 * len(SUBJECTS):
        raise SystemExit(f'Expected {10 * len(SUBJECTS)} runs, found {len(run_results)}')

    all_runs_path = os.path.join(report_root, 'all_runs.json')
    with open(all_runs_path, 'w') as handle:
        json.dump(_json_safe(run_results), handle, indent=2)
    _write_summary_csv(run_results, os.path.join(report_root, 'summary.csv'))
    print(f'Assembled {len(SUBJECTS)} subjects × 10 runs -> {all_runs_path}')


if __name__ == '__main__':
    main()
