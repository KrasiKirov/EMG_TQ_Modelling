"""Create reviewer-facing tables and figures from repeated-run results.

This module does not train models. It reads ``all_runs.json`` after the
10-run evaluation has finished and creates the subject-by-position table and
the combined all-subjects variability figure requested by the reviewer.

Example
-------
    python -m ML.training.postprocess_repeated_runs
"""

import csv
import json
import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ML.config import ML_DIR, PLOT_DIR
from ML.evaluation import plot_all_subjects_per_position
from ML.training.repeated_runs import _write_summary_csv


METRICS = ('r2', 'vaf', 'rmse', 'mae')


def _position_key(value):
    """Sort numeric position keys while tolerating malformed legacy values."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _write_subject_position_table(run_results, path):
    """Write one reviewer-facing row per subject and ankle position."""
    fields = [
        'subject', 'position', 'n_samples', 'n_runs',
        'r2_mean', 'r2_sd', 'vaf_mean', 'vaf_sd',
        'rmse_mean', 'rmse_sd', 'mae_mean', 'mae_sd',
    ]
    rows = []
    subjects = sorted({run['subject'] for run in run_results})
    for subject in subjects:
        subject_runs = [run for run in run_results
                        if run['subject'] == subject]
        positions = sorted(subject_runs[0]['per_position'], key=_position_key)
        for position in positions:
            row = {
                'subject': subject,
                'position': position,
                'n_samples': int(round(np.mean([
                    run['per_position'][position]['n']
                    for run in subject_runs
                ]))),
                'n_runs': len(subject_runs),
            }
            for metric in METRICS:
                values = np.asarray([
                    run['per_position'][position][metric]
                    for run in subject_runs
                ], dtype=float)
                row[f'{metric}_mean'] = float(np.nanmean(values))
                row[f'{metric}_sd'] = float(
                    np.nanstd(values, ddof=1) if len(values) > 1 else 0.0
                )
            rows.append(row)

    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_run_timings(run_results, path):
    """Write one timing row for every completed subject/run pair."""
    fields = ['subject', 'run', 'seed', 'epochs_completed',
              'train_s', 'eval_s', 'duration_s']
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in sorted(run_results,
                          key=lambda item: (item['subject'], item['run'])):
            writer.writerow({field: run.get(field, '') for field in fields})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-root', default=None)
    parser.add_argument('--plot-root', default=None)
    args = parser.parse_args()

    report_dir = args.report_root or os.path.join(
        ML_DIR, 'reports', 'repeated_runs')
    results_path = os.path.join(report_dir, 'all_runs.json')
    if not os.path.exists(results_path):
        raise SystemExit(f'Missing completed results: {results_path}')

    with open(results_path) as handle:
        run_results = json.load(handle)
    if not run_results:
        raise SystemExit('all_runs.json contains no run results')

    _write_summary_csv(run_results, os.path.join(report_dir, 'summary.csv'))
    table_path = os.path.join(report_dir, 'subject_by_position_summary.csv')
    _write_subject_position_table(run_results, table_path)
    timing_path = os.path.join(report_dir, 'run_timings.csv')
    _write_run_timings(run_results, timing_path)

    plot_dir = args.plot_root or os.path.join(PLOT_DIR, 'repeated_runs')
    os.makedirs(plot_dir, exist_ok=True)
    combined_path = os.path.join(plot_dir, 'all_subjects_per_position.png')
    plot_all_subjects_per_position(run_results, combined_path)

    print(f'Subject-by-position table saved -> {table_path}')
    print(f'Per-run timings saved -> {timing_path}')
    print(f'Combined all-subjects plot saved -> {combined_path}')


if __name__ == '__main__':
    main()
