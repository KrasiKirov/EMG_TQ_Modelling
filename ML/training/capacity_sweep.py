"""
Capacity sweep: compare per-position R² across different LSTM unit counts.

For each unit count in CAPACITY_SWEEP_UNITS (default [8, 16, 32]):
  1. Train a fresh model on the subject's test-session data
  2. Predict on train, validation, and held-out retest splits
  3. Compute per-position R² for all three splits

Produces a single comparison plot showing how performance changes with
capacity at each of the 8 ankle positions — useful for diagnosing whether
the model is capacity-limited at the extreme positions.

Usage
-----
    python ML/training/capacity_sweep.py --subject HM
    python ML/training/capacity_sweep.py --subject EG --units 8 16 32 64
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ML.config import (SUBJECTS, MODEL_DIR, PLOT_DIR,
                        CAPACITY_SWEEP_UNITS, LEARNING_RATE, BATCH_SIZE, EPOCHS)
from ML.training import load_subject, train_model, evaluate_model
from ML.evaluation import compute_per_position_metrics, plot_capacity_comparison


def parse_args():
    p = argparse.ArgumentParser(description='Sweep LSTM unit counts and compare per-position R²')
    p.add_argument('--subject', choices=list(SUBJECTS.keys()), default=None)
    p.add_argument('--units', nargs='+', type=int, default=None,
                   help='Unit counts to sweep (default: CAPACITY_SWEEP_UNITS from config)')
    p.add_argument('--epochs', default=EPOCHS, type=int)
    p.add_argument('--batch',  default=BATCH_SIZE, type=int)
    p.add_argument('--lr',     default=LEARNING_RATE, type=float)
    return p.parse_args()


def run_sweep(subject_key, unit_counts, lr, epochs, batch_size):
    """Train one model per unit count; collect per-position metrics for all splits.

    Returns
    -------
    sweep_results : list of dict
        Each dict has:
          'n_units'      : int
          'per_position' : {'train': {pos: metrics}, 'val': ..., 'test': ...}
          'overall'      : {'train': metrics, 'val': metrics, 'test': metrics}
    known_positions : list of float
    """
    data = load_subject(subject_key)

    X_train, y_train = data['X_train'], data['y_train']
    X_val,   y_val   = data['X_val'],   data['y_val']
    X_test,  y_test  = data['X_test'],  data['y_test']
    known_pos        = data['operating_positions']

    sweep_results = []

    for n_units in unit_counts:
        print(f'\n{"─"*60}')
        print(f'  {n_units} LSTM units')
        print(f'{"─"*60}')

        ckpt_dir = os.path.join(MODEL_DIR, subject_key)
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f'lstm_{n_units}units.keras')

        model, _ = train_model(
            X_train, y_train, X_val, y_val,
            label=f'LSTM-{n_units} ({subject_key})',
            save_path=ckpt_path,
            lr=lr, epochs=epochs, batch_size=batch_size,
            n_units=n_units,
            verbose=0,
        )

        # ── Test split (denormalized via evaluate_model) ───────────────
        result_test = evaluate_model(
            model, X_test, y_test, known_pos,
            mvc_tq_test=data['mvc_tq_test'])

        # ── Train / Val splits ─────────────────────────────────────────
        # Note: shown in raw units (no MVC denormalization for train/val)
        y_pred_train = model.predict(X_train, verbose=0).flatten()
        y_pred_val   = model.predict(X_val,   verbose=0).flatten()

        from ML.evaluation import compute_metrics
        overall_train = compute_metrics(y_train, y_pred_train)
        overall_val   = compute_metrics(y_val,   y_pred_val)

        per_pos_train = compute_per_position_metrics(
            X_train, y_train, y_pred_train, known_pos)
        per_pos_val   = compute_per_position_metrics(
            X_val, y_val, y_pred_val, known_pos)

        sweep_results.append({
            'n_units': n_units,
            'per_position': {
                'train': per_pos_train,
                'val':   per_pos_val,
                'test':  result_test['per_position'],
            },
            'overall': {
                'train': overall_train,
                'val':   overall_val,
                'test':  result_test['overall'],
            },
        })

        print(f'\n  Results:')
        print(f'    Train R²={overall_train["r2"]:.4f}  RMSE={overall_train["rmse"]:.3f} Nm')
        print(f'    Val   R²={overall_val["r2"]:.4f}  RMSE={overall_val["rmse"]:.3f} Nm')
        print(f'    Test  R²={result_test["overall"]["r2"]:.4f}  '
              f'RMSE={result_test["overall"]["rmse"]:.3f} Nm')

    return sweep_results, known_pos


def main():
    args = parse_args()
    subject_key = args.subject or next(iter(SUBJECTS))
    unit_counts = args.units or CAPACITY_SWEEP_UNITS

    print(f'\nCapacity sweep — subject: {subject_key}')
    print(f'Unit counts: {unit_counts}')

    sweep_results, known_pos = run_sweep(
        subject_key, unit_counts,
        lr=args.lr, epochs=args.epochs, batch_size=args.batch)

    # ── Summary table ──────────────────────────────────────────────────
    print(f'\n\n{"="*70}')
    print(f'CAPACITY SWEEP SUMMARY ({subject_key})')
    print(f'{"="*70}')
    print(f'  {"Units":>6}  {"Train R²":>9}  {"Val R²":>9}  {"Test R²":>9}  '
          f'{"Test RMSE":>10}')
    print(f'  {"------":>6}  {"-"*9}  {"-"*9}  {"-"*9}  {"-"*10}')
    for res in sweep_results:
        print(f'  {res["n_units"]:>6}  '
              f'{res["overall"]["train"]["r2"]:9.4f}  '
              f'{res["overall"]["val"]["r2"]:9.4f}  '
              f'{res["overall"]["test"]["r2"]:9.4f}  '
              f'{res["overall"]["test"]["rmse"]:10.4f}')

    # ── Plot ───────────────────────────────────────────────────────────
    plot_dir = os.path.join(PLOT_DIR, subject_key)
    os.makedirs(plot_dir, exist_ok=True)

    plot_capacity_comparison(
        sweep_results, known_pos,
        os.path.join(plot_dir, 'capacity_sweep.png'),
        title=f'Capacity Sweep — Per-Position R² ({subject_key})')

    print(f'\nDone.')


if __name__ == '__main__':
    main()
