"""Repeated-run evaluation for the within-subject LSTM benchmark.

Each run keeps the data split, preprocessing, architecture, and hyperparameters
fixed while changing the random seed used for model initialization and batch
ordering. Results are saved separately from the historical checkpoints and
plots.

Example
-------
    python -m ML.training.repeated_runs --runs 10
    python -m ML.training.repeated_runs --subject CH --runs 10 --epochs 200
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ML.config import (SUBJECTS, ML_DIR, PLOT_DIR, EPOCHS, BATCH_SIZE,
                       LEARNING_RATE)
from ML.training import load_subject, train_model, evaluate_model
from ML.evaluation import (plot_repeated_run_summary,
                           plot_per_position_variability)


DEFAULT_SEEDS = [11, 23, 37, 53, 71, 89, 101, 113, 127, 139]


def _seeds(n_runs):
    """Return a stable seed list for the requested number of runs."""
    if n_runs <= len(DEFAULT_SEEDS):
        return DEFAULT_SEEDS[:n_runs]
    generated = np.random.default_rng(20261012).integers(
        1, 2**31 - 1, size=n_runs, dtype=np.int64)
    return [int(x) for x in generated]


def _json_safe(value):
    """Convert NumPy scalars/arrays and float-keyed mappings for JSON."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _write_run_csv(run_results, path):
    """Write one long-form CSV with pooled and per-position metrics."""
    fields = ['subject', 'run', 'seed', 'scope', 'position', 'n',
              'train_s', 'eval_s', 'duration_s',
              'r2', 'vaf', 'rmse', 'nrmse', 'mae']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run in run_results:
            overall = run['overall']
            writer.writerow({
                'subject': run['subject'], 'run': run['run'],
                'seed': run['seed'], 'scope': 'pooled', 'position': '',
                'n': run['n_test'], 'train_s': run['train_s'],
                'eval_s': run['eval_s'], 'duration_s': run['duration_s'],
                **overall,
            })
            for pos, metrics in sorted(run['per_position'].items()):
                writer.writerow({
                    'subject': run['subject'], 'run': run['run'],
                    'seed': run['seed'], 'scope': 'position',
                    'position': pos, 'train_s': run['train_s'],
                    'eval_s': run['eval_s'], 'duration_s': run['duration_s'],
                    **metrics,
                })


def _write_summary_csv(run_results, path):
    """Write mean ± SD summaries for pooled and per-position metrics."""
    rows = []
    metric_keys = ['r2', 'vaf', 'rmse', 'nrmse', 'mae']
    for subject in sorted({r['subject'] for r in run_results}):
        subject_runs = [r for r in run_results if r['subject'] == subject]
        for scope, position, values_by_metric in [
            ('pooled', '', {
                key: [r['overall'][key] for r in subject_runs]
                for key in metric_keys
            }),
        ]:
            values_by_metric.update({
                key: [r[key] for r in subject_runs]
                for key in ['duration_s', 'train_s', 'eval_s']
            })
            for metric, values in values_by_metric.items():
                values = np.asarray(values, dtype=float)
                rows.append({
                    'subject': subject, 'scope': scope,
                    'position': position, 'metric': metric,
                    'mean': float(np.nanmean(values)),
                    'sd': float(np.nanstd(values, ddof=1)),
                    'n_samples': int(np.mean([r['n_test'] for r in subject_runs])),
                    'n_runs': len(values),
                })

        positions = sorted(subject_runs[0]['per_position'])
        for position in positions:
            for metric in metric_keys:
                values = np.asarray([
                    r['per_position'][position][metric]
                    for r in subject_runs
                ], dtype=float)
                rows.append({
                    'subject': subject, 'scope': 'position',
                    'position': position, 'metric': metric,
                    'mean': float(np.nanmean(values)),
                    'sd': float(np.nanstd(values, ddof=1)),
                    'n_samples': int(round(np.mean([
                        r['per_position'][position]['n']
                        for r in subject_runs
                    ]))),
                    'n_runs': len(values),
                })

    fields = ['subject', 'scope', 'position', 'metric', 'mean', 'sd',
              'n_samples', 'n_runs']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_subject(subject_key, n_runs=10, epochs=EPOCHS,
                batch_size=BATCH_SIZE, lr=LEARNING_RATE,
                report_root=None, plot_root=None, yes_eight_positions=False):
    """Train and evaluate one subject repeatedly, returning all metrics."""
    seeds = _seeds(n_runs)
    data = load_subject(subject_key, yes_eight_positions=yes_eight_positions)
    run_results = []
    report_dir = report_root or os.path.join(ML_DIR, 'reports', 'repeated_runs')
    subject_report_dir = os.path.join(report_dir, subject_key)
    os.makedirs(subject_report_dir, exist_ok=True)
    if data['reconstruction']:
        with open(os.path.join(subject_report_dir, 'reconstruction.json'), 'w') as f:
            json.dump(data['reconstruction'], f, indent=2)

    for run_number, seed in enumerate(seeds, start=1):
        print(f'\n[{subject_key}] run {run_number}/{n_runs} (seed={seed})')
        run_started = time.perf_counter()
        train_started = time.perf_counter()
        model, history = train_model(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            label=f'LSTM ({subject_key}, run {run_number})',
            lr=lr, epochs=epochs, batch_size=batch_size,
            verbose=0, seed=seed,
        )
        train_seconds = time.perf_counter() - train_started
        eval_started = time.perf_counter()
        result = evaluate_model(
            model, data['X_test'], data['y_test'],
            data['operating_positions'],
            mvc_tq_test=data['mvc_tq_test'],
        )
        eval_seconds = time.perf_counter() - eval_started
        duration_seconds = time.perf_counter() - run_started
        run_results.append({
            'subject': subject_key,
            'run': run_number,
            'seed': seed,
            'window_steps': data['window_steps'],
            'window_ms': data['window_steps'] * 10,
            'passive_map_session': data['passive_session'],
            'passive_map_source': data['passive_map_source'],
            'reconstruction': data['reconstruction'],
            'epochs_completed': len(history.history['loss']),
            'n_test': int(len(result['y_test'])),
            'train_s': train_seconds,
            'eval_s': eval_seconds,
            'duration_s': duration_seconds,
            'overall': result['overall'],
            'per_position': result['per_position'],
        })
        # Retain every completed run even if a later run is interrupted.
        partial_path = os.path.join(subject_report_dir, 'metrics.partial.json')
        with open(partial_path + '.tmp', 'w') as f:
            json.dump(_json_safe(run_results), f, indent=2)
        os.replace(partial_path + '.tmp', partial_path)
        overall = result['overall']
        print(f'  pooled R²={overall["r2"]:.4f}  '
              f'RMSE={overall["rmse"]:.4f} Nm  '
              f'MAE={overall["mae"]:.4f} Nm  '
              f'time={duration_seconds / 60:.2f} min '
              f'(train={train_seconds / 60:.2f}, eval={eval_seconds:.2f}s)')
        tf.keras.backend.clear_session()

    report_dir = report_root or os.path.join(ML_DIR, 'reports', 'repeated_runs')
    plot_dir = plot_root or os.path.join(PLOT_DIR, 'repeated_runs')
    subject_report_dir = os.path.join(report_dir, subject_key)
    subject_plot_dir = os.path.join(plot_dir, subject_key)
    os.makedirs(subject_report_dir, exist_ok=True)
    os.makedirs(subject_plot_dir, exist_ok=True)

    with open(os.path.join(subject_report_dir, 'metrics.json'), 'w') as f:
        json.dump(_json_safe(run_results), f, indent=2)
    with open(os.path.join(subject_report_dir, 'metadata.json'), 'w') as f:
        json.dump({
            'subject': subject_key,
            'runs': n_runs,
            'seeds': seeds,
            'window_steps': data['window_steps'],
            'window_ms': data['window_steps'] * 10,
            'passive_map_session': data['passive_session'],
            'passive_map_source': data['passive_map_source'],
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'reconstruction': data['reconstruction'],
        }, f, indent=2)
    _write_run_csv(run_results,
                   os.path.join(subject_report_dir, 'metrics_long.csv'))
    plot_repeated_run_summary(
        run_results,
        os.path.join(subject_plot_dir, 'pooled_metric_variability.png'),
        subject=subject_key,
    )
    plot_per_position_variability(
        run_results,
        os.path.join(subject_plot_dir, 'per_position_variability.png'),
        subject=subject_key,
    )
    return run_results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', choices=list(SUBJECTS), default=None,
                        help='Subject to evaluate (default: all subjects)')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of independent seeded runs (default: 10)')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--report-root', default=None,
                        help='Output directory (default: reports/repeated_runs)')
    parser.add_argument('--plot-root', default=None,
                        help='Plot directory (default: plots/repeated_runs)')
    parser.add_argument('--merge-into', default=None,
                        help='Replace this subject in an existing all_runs.json')
    parser.add_argument('--yes-eight-positions', action='store_true',
                        help='Explicit historical YES reconstruction; use a separate output root')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.runs < 2:
        raise SystemExit('--runs must be at least 2 to estimate variability')
    if args.yes_eight_positions and (args.subject != 'YES' or not args.report_root or args.merge_into):
        raise SystemExit('Reconstruction requires --subject YES, a separate --report-root, and no --merge-into')
    subjects = [args.subject] if args.subject else list(SUBJECTS)
    all_results = []
    for subject_key in subjects:
        all_results.extend(run_subject(
            subject_key, n_runs=args.runs, epochs=args.epochs,
            batch_size=args.batch, lr=args.lr,
            report_root=args.report_root, plot_root=args.plot_root,
            yes_eight_positions=args.yes_eight_positions,
        ))

    report_root = args.report_root or os.path.join(
        ML_DIR, 'reports', 'repeated_runs')
    summary_path = os.path.join(report_root, 'all_runs.json')
    if args.merge_into:
        merge_path = os.path.abspath(args.merge_into)
        if not os.path.exists(merge_path):
            raise SystemExit(f'Cannot merge: missing {merge_path}')
        with open(merge_path) as f:
            existing = json.load(f)
        evaluated_subjects = set(subjects)
        all_results = [r for r in existing
                       if r['subject'] not in evaluated_subjects] + all_results
        summary_path = merge_path
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(_json_safe(all_results), f, indent=2)
    _write_summary_csv(
        all_results,
        os.path.join(os.path.dirname(summary_path), 'summary.csv'),
    )
    print(f'\nSaved repeated-run reports under {os.path.dirname(summary_path)}')


if __name__ == '__main__':
    main()
