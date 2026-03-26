"""Train a stacked LSTM to predict ankle torque (Nm) from sEMG + ankle position."""

import os
import sys
import argparse
# Allow `python ML/training/train.py` from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ML.config import (SUBJECTS, MODEL_DIR, PLOT_DIR,
                        LEARNING_RATE, BATCH_SIZE, EPOCHS)
from ML.training import load_subject, train_model, evaluate_model
from ML.evaluation import (plot_full_trials, plot_per_position_metrics,
                            plot_r2_summary)


def parse_args():
    p = argparse.ArgumentParser(description='Train LSTM for ankle torque prediction')
    p.add_argument('--subject', choices=list(SUBJECTS.keys()),
                   default=None,
                   help='Subject short name (e.g. HM, EG). Sets --flb and --subject-id automatically.')
    p.add_argument('--flb',    default=None, help='Path to .flb file (overrides --subject)')
    p.add_argument('--subject-id', default=None,
                   help="Subject ID for channel ordering (overrides --subject)")
    p.add_argument('--epochs', default=EPOCHS, type=int)
    p.add_argument('--batch',  default=BATCH_SIZE, type=int)
    p.add_argument('--lr',     default=LEARNING_RATE, type=float)
    p.add_argument('--test-trials', nargs='+', type=int, default=None,
                   help='1-indexed test trial numbers (default: auto-detect from comments)')
    p.add_argument('--retest-trials', nargs='+', type=int, default=None,
                   help='1-indexed retest trial numbers (default: auto-detect from comments)')
    p.add_argument('--normalize-position', action='store_true', default=False,
                   help='Min-max normalize ankle position to [0,1] (for cross-subject transfer)')
    p.add_argument('--normalize-mvc', action='store_true', default=False,
                   help='Normalize EMG and torque by per-position MVC (for cross-subject transfer)')
    return p.parse_args()


def main():
    args = parse_args()
    subject_key = args.subject or next(iter(SUBJECTS))

    subj_model_dir = os.path.join(MODEL_DIR, subject_key)
    subj_plot_dir  = os.path.join(PLOT_DIR, subject_key)
    os.makedirs(subj_model_dir, exist_ok=True)
    os.makedirs(subj_plot_dir,  exist_ok=True)

    # ── 1. Load & build dataset ───────────────────────────────────────────────
    data = load_subject(
        subject_key,
        normalize_position=args.normalize_position,
        normalize_mvc=args.normalize_mvc,
        test_trial_indices=args.test_trials,
        retest_trial_indices=args.retest_trials,
        flb_path=args.flb,
        subject_id=args.subject_id,
    )

    X_train, y_train = data['X_train'], data['y_train']
    X_val,   y_val   = data['X_val'],   data['y_val']

    print(f'\nDataset shapes:')
    print(f'  Train : X={X_train.shape}  y={y_train.shape}')
    print(f'  Val   : X={X_val.shape}  y={y_val.shape}')
    print(f'  Test  : X={data["X_test"].shape}  y={data["y_test"].shape}')
    print(f'  Target range — min: {y_train.min():.3f} Nm  max: {y_train.max():.3f} Nm')

    # ── 2. Train ──────────────────────────────────────────────────────────────
    ckpt_path = os.path.join(subj_model_dir, 'best_lstm.keras')
    model, history = train_model(
        X_train, y_train, X_val, y_val,
        label=f'LSTM ({subject_key})',
        save_path=ckpt_path,
        lr=args.lr, epochs=args.epochs, batch_size=args.batch,
        verbose=1,
    )
    model.summary()

    # ── 3. Evaluate on held-out retest set ────────────────────────────────────
    #
    # Pooled R² uses the global mean across all positions, so it gets a boost
    # from separating positions.  Per-position R² uses the local mean at each
    # position and is the stricter, more informative metric.
    #
    result = evaluate_model(
        model, data['X_test'], data['y_test'],
        data['operating_positions'],
        mvc_tq_test=data['mvc_tq_test'],
    )

    y_test  = result['y_test']
    y_pred  = result['y_pred']
    overall = result['overall']
    per_pos = result['per_position']
    known_pos = data['operating_positions']
    r2 = overall['r2']

    n_test = len(y_test)
    n_pos  = len(known_pos)
    print(f'\n{"="*70}')
    print(f'TEST RESULTS')
    print(f'  Data split : held-out retest session ({n_pos} positions, '
          f'{n_test} windows, fully unseen during training)')
    print(f'  Targets    : net active torque (Nm) = measured − passive')
    print(f'{"="*70}')
    print(f'  R²    = 1 − SS_res/SS_tot          : {r2:.4f}')
    print(f'  VAF   = (1 − var(e)/var(y)) × 100  : {overall["vaf"]:.2f}%')
    print(f'  RMSE  = √(mean(e²))                : {overall["rmse"]:.4f} Nm')
    print(f'  NRMSE = RMSE / (y_max − y_min)      : {overall["nrmse"]:.4f}  ({overall["nrmse"] * 100:.2f}%)')
    print(f'  MAE   = mean(|e|)                   : {overall["mae"]:.4f} Nm')
    print(f'{"="*70}')

    # ── Per-position table ────────────────────────────────────────────────────
    print(f'\nPer-position test results (retest trials — unseen during training):')
    print(f'  {"Position":>12}  {"R²":>7}  {"VAF (%)":>8}  {"RMSE (Nm)":>10}  {"NRMSE (%)":>10}  {"MAE (Nm)":>9}  {"N":>8}')
    print(f'  {"-"*12}  {"-"*7}  {"-"*8}  {"-"*10}  {"-"*10}  {"-"*9}  {"-"*8}')
    for pos in sorted(per_pos):
        m = per_pos[pos]
        print(f'  {pos:+12.3f}  {m["r2"]:7.4f}  {m["vaf"]:8.2f}  {m["rmse"]:10.4f}  '
              f'{m["nrmse"]*100:10.2f}  {m["mae"]:9.4f}  {m["n"]:8d}')
    print(f'  {"Pooled":>12}  {r2:7.4f}  {"":>8}  {"":>10}  {"":>10}  {"":>9}  {n_test:8d}')

    # ── 4. Save plots ─────────────────────────────────────────────────────────
    pos_r2 = plot_per_position_metrics(
        data['X_test'], y_test, y_pred,
        os.path.join(subj_plot_dir, 'per_position_metrics.png'),
        known_positions=known_pos, pooled_r2=r2)

    plot_r2_summary(pos_r2, r2,
                    os.path.join(subj_plot_dir, 'r2_vs_position.png'))

    plot_full_trials(data['X_test'], y_test, y_pred,
                     os.path.join(subj_plot_dir, 'full_trial_inspection.png'),
                     known_positions=known_pos)

    print(f'\nBest model saved → {ckpt_path}')


if __name__ == '__main__':
    main()
