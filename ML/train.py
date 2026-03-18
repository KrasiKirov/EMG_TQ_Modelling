"""Train a stacked LSTM to predict ankle torque (Nm) from sEMG + ankle position."""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Allow `python ML/train.py` from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import (ReduceLROnPlateau, EarlyStopping,
                                        ModelCheckpoint)

from ML.utils.flb_reader import read_flb
from ML.utils.dataset_builder import build_dataset
from ML.models.lstm_model import build_model

# ── Paths ─────────────────────────────────────────────────────────────────────
ML_DIR     = os.path.dirname(__file__)
MODEL_DIR  = os.path.join(ML_DIR, 'models')
PLOT_DIR   = os.path.join(ML_DIR, 'plots')

# ── Subject registry ─────────────────────────────────────────────────────────
# Maps a short name to (flb_filename, subject_id_for_channel_ordering)
SUBJECTS = {
    'S3': ('YES_281124.flb', 'default'),
    'HM': ('HM_110425.flb',  'IES01'),
}

# ── Hyperparameters ───────────────────────────────────────────────────────────
LEARNING_RATE  = 0.003
LR_FACTOR      = 0.7
LR_PATIENCE    = 10
LR_MIN         = 1e-6
EARLY_PATIENCE = 25
BATCH_SIZE     = 8
EPOCHS         = 200
N_STEPS        = 20
N_FEATURES     = 5


def parse_args():
    p = argparse.ArgumentParser(description='Train LSTM for ankle torque prediction')
    p.add_argument('--subject', choices=list(SUBJECTS.keys()),
                   default=None,
                   help='Subject short name (e.g. S3, HM). Sets --flb and --subject-id automatically.')
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
    return p.parse_args()


def plot_history(history, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['loss'],     label='Train MSE')
    axes[0].plot(history.history['val_loss'], label='Val MSE')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE (Nm²)')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, linewidth=0.4)

    axes[1].plot(history.history['mae'],     label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Val MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (Nm)')
    axes[1].set_title('Mean Absolute Error')
    axes[1].legend()
    axes[1].grid(True, linewidth=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Training curves saved → {out_path}')


def plot_predictions(y_true, y_pred, out_path: str, n_samples: int = 2000):
    fig, ax = plt.subplots(figsize=(14, 4))
    idx = np.arange(min(n_samples, len(y_true)))
    ax.plot(idx, y_true[idx], label='True torque', linewidth=0.8, color='steelblue')
    ax.plot(idx, y_pred[idx], label='Predicted',   linewidth=0.8, color='red', alpha=0.8)
    ax.set_xlabel('Window index')
    ax.set_ylabel('Net active torque (Nm)')
    ax.set_title(f'Retest session predictions (first {len(idx)} windows)')
    ax.legend()
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Prediction plot saved → {out_path}')


def _snap_to_operating_points(positions, known_positions):
    """Assign each window's position to the nearest known operating point."""
    known = np.array(sorted(known_positions))
    dists = np.abs(positions[:, None] - known[None, :])
    return known[np.argmin(dists, axis=1)], known


def evaluate_per_position(X_test, y_test, y_pred, out_path: str,
                          known_positions=None):
    """Compute and plot metrics grouped by ankle position."""
    # Position is the 5th feature (index 4), take value at last timestep
    positions = X_test[:, -1, 4]

    if known_positions is not None:
        pos_snapped, unique_pos = _snap_to_operating_points(positions, known_positions)
    else:
        pos_snapped = np.round(positions / 0.10) * 0.10
        unique_pos = np.sort(np.unique(pos_snapped))

    print(f'\nPer-position test results (retest trials — unseen during training):')
    print(f'  {"Position (rad)":>15}  {"R²":>7}  {"VAF (%)":>8}  {"RMSE (Nm)":>10}  {"MAE (Nm)":>9}  {"N windows":>10}')
    print(f'  {"-"*15}  {"-"*7}  {"-"*8}  {"-"*10}  {"-"*9}  {"-"*10}')

    pos_labels, r2s, vafs, rmses, maes, counts = [], [], [], [], [], []

    for pos in unique_pos:
        mask = pos_snapped == pos
        n = mask.sum()
        if n == 0:
            continue

        yt = y_test[mask]
        yp = y_pred[mask]

        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        vaf = (1.0 - np.var(yt - yp) / np.var(yt)) * 100.0 if np.var(yt) > 0 else float('nan')
        rmse = np.sqrt(np.mean((yt - yp) ** 2))
        mae = np.mean(np.abs(yt - yp))

        print(f'  {pos:+15.3f}  {r2:7.4f}  {vaf:8.2f}  {rmse:10.4f}  {mae:9.4f}  {n:10d}')

        pos_labels.append(pos)
        r2s.append(r2)
        vafs.append(vaf)
        rmses.append(rmse)
        maes.append(mae)
        counts.append(n)

    # Plot per-position metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(range(len(pos_labels)), r2s, color='steelblue')
    axes[0].set_xticks(range(len(pos_labels)))
    axes[0].set_xticklabels([f'{p:+.2f}' for p in pos_labels], rotation=45, fontsize=8)
    axes[0].set_ylabel('R²')
    axes[0].set_title('R² per Position')
    axes[0].set_xlabel('Ankle Position (rad)')
    axes[0].grid(True, axis='y', linewidth=0.4)
    axes[0].axhline(y=0, color='k', linewidth=0.5)

    axes[1].bar(range(len(pos_labels)), rmses, color='coral')
    axes[1].set_xticks(range(len(pos_labels)))
    axes[1].set_xticklabels([f'{p:+.2f}' for p in pos_labels], rotation=45, fontsize=8)
    axes[1].set_ylabel('RMSE (Nm)')
    axes[1].set_title('RMSE per Position')
    axes[1].set_xlabel('Ankle Position (rad)')
    axes[1].grid(True, axis='y', linewidth=0.4)

    axes[2].bar(range(len(pos_labels)), maes, color='mediumpurple')
    axes[2].set_xticks(range(len(pos_labels)))
    axes[2].set_xticklabels([f'{p:+.2f}' for p in pos_labels], rotation=45, fontsize=8)
    axes[2].set_ylabel('MAE (Nm)')
    axes[2].set_title('MAE per Position')
    axes[2].set_xlabel('Ankle Position (rad)')
    axes[2].grid(True, axis='y', linewidth=0.4)

    fig.suptitle('Per-Position Test Metrics (Retest Trials — Unseen Data)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Per-position plot saved → {out_path}')

    return dict(zip(pos_labels, r2s))


def plot_predictions_per_position(X_test, y_test, y_pred, out_path: str,
                                  known_positions=None):
    """Plot true vs predicted torque for each ankle position in separate subplots."""
    positions = X_test[:, -1, 4]
    if known_positions is not None:
        pos_snapped, valid_pos = _snap_to_operating_points(positions, known_positions)
    else:
        pos_snapped = np.round(positions / 0.10) * 0.10
        valid_pos = np.sort(np.unique(pos_snapped))
    n_pos = len(valid_pos)

    cols = min(n_pos, 4)
    rows = (n_pos + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)

    for i, pos in enumerate(valid_pos):
        ax = axes[i // cols][i % cols]
        mask = pos_snapped == pos
        yt = y_test[mask]
        yp = y_pred[mask]
        n_show = min(1500, len(yt))

        ax.plot(range(n_show), yt[:n_show], linewidth=0.7, color='steelblue', label='True')
        ax.plot(range(n_show), yp[:n_show], linewidth=0.7, color='red', alpha=0.7, label='Predicted')
        ax.set_title(f'Position {pos:+.2f} rad', fontsize=9)
        ax.set_ylabel('Torque (Nm)', fontsize=8)
        ax.set_xlabel('Window index', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linewidth=0.3)
        ax.legend(fontsize=7, loc='upper right')

    for i in range(n_pos, rows * cols):
        axes[i // cols][i % cols].set_visible(False)

    fig.suptitle('Predictions per Ankle Position (Retest Session — Unseen Data)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Per-position predictions plot saved → {out_path}')


def plot_scatter_per_position(X_test, y_test, y_pred, out_path: str,
                              known_positions=None):
    """Scatter plot of predicted vs actual torque per position with identity line."""
    positions = X_test[:, -1, 4]
    if known_positions is not None:
        pos_snapped, valid_pos = _snap_to_operating_points(positions, known_positions)
    else:
        pos_snapped = np.round(positions / 0.10) * 0.10
        valid_pos = np.sort(np.unique(pos_snapped))
    n_pos = len(valid_pos)

    cols = min(n_pos, 4)
    rows = (n_pos + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), squeeze=False)

    for i, pos in enumerate(valid_pos):
        ax = axes[i // cols][i % cols]
        mask = pos_snapped == pos
        yt = y_test[mask]
        yp = y_pred[mask]

        # Subsample for readability if too many points
        if len(yt) > 3000:
            idx = np.random.default_rng(42).choice(len(yt), 3000, replace=False)
            yt_plot, yp_plot = yt[idx], yp[idx]
        else:
            yt_plot, yp_plot = yt, yp

        ax.scatter(yt_plot, yp_plot, s=3, alpha=0.3, color='steelblue', edgecolors='none')

        # Identity line
        all_vals = np.concatenate([yt, yp])
        vmin, vmax = all_vals.min(), all_vals.max()
        margin = (vmax - vmin) * 0.05
        ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
                'k--', linewidth=0.8, label='y = x')
        ax.set_xlim(vmin - margin, vmax + margin)
        ax.set_ylim(vmin - margin, vmax + margin)
        ax.set_aspect('equal')

        # Compute R² for annotation
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

        ax.set_title(f'Position {pos:+.2f} rad  (R²={r2:.3f})', fontsize=9)
        ax.set_xlabel('Actual Torque (Nm)', fontsize=8)
        ax.set_ylabel('Predicted Torque (Nm)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linewidth=0.3)

    for i in range(n_pos, rows * cols):
        axes[i // cols][i % cols].set_visible(False)

    fig.suptitle('Predicted vs Actual Torque per Position (Retest Session — Unseen Data)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Scatter plot saved → {out_path}')


def main():
    args = parse_args()

    # Resolve subject → flb path + subject_id
    if args.subject:
        flb_name, subj_id = SUBJECTS[args.subject]
        flb_path = args.flb or os.path.join(ML_DIR, flb_name)
        subject_id = args.subject_id or subj_id
    else:
        # Fallback to S3 defaults when neither --subject nor --flb is given
        flb_path = args.flb or os.path.join(ML_DIR, SUBJECTS['S3'][0])
        subject_id = args.subject_id or SUBJECTS['S3'][1]

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR,  exist_ok=True)

    # ── 1. Load & build dataset ───────────────────────────────────────────────
    print(f'\nReading: {flb_path}')
    trials = read_flb(flb_path, subject_id=subject_id)

    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     emg_max, passive_entries,
     operating_positions) = build_dataset(
        trials,
        test_trial_indices=args.test_trials,
        retest_trial_indices=args.retest_trials,
        subtract_passive=True,
    )

    print(f'\nDataset shapes:')
    print(f'  Train : X={X_train.shape}  y={y_train.shape}')
    print(f'  Val   : X={X_val.shape}  y={y_val.shape}')
    print(f'  Test  : X={X_test.shape}  y={y_test.shape}')
    print(f'  Target range — min: {y_train.min():.3f} Nm  max: {y_train.max():.3f} Nm')

    # ── 2. Build model ────────────────────────────────────────────────────────
    model = build_model(n_steps=N_STEPS, n_features=N_FEATURES)
    model.compile(
        optimizer=Nadam(learning_rate=args.lr),
        loss='mse',
        metrics=['mae']
    )
    model.summary()

    # ── 3. Callbacks ──────────────────────────────────────────────────────────
    ckpt_path = os.path.join(MODEL_DIR, 'best_lstm.keras')
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=LR_FACTOR,
                          patience=LR_PATIENCE, min_lr=LR_MIN, verbose=1),
        EarlyStopping(monitor='val_loss', patience=EARLY_PATIENCE,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(ckpt_path, monitor='val_loss',
                        save_best_only=True, verbose=1),
    ]

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print(f'\nTraining  (batch={args.batch}, lr={args.lr}, epochs={args.epochs})')
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks,
        shuffle=True,
    )

    # ── 5. Evaluate on test set ───────────────────────────────────────────────
    y_pred = model.predict(X_test, verbose=0).flatten()

    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot
    vaf    = (1.0 - np.var(y_test - y_pred) / np.var(y_test)) * 100.0
    rmse   = np.sqrt(np.mean((y_test - y_pred) ** 2))
    nrmse  = rmse / (y_test.max() - y_test.min())
    mae    = np.mean(np.abs(y_test - y_pred))

    print(f'\nTest results:')
    print(f'  R²         : {r2:.4f}')
    print(f'  VAF        : {vaf:.2f}%')
    print(f'  RMSE       : {rmse:.4f} Nm')
    print(f'  NRMSE      : {nrmse:.4f}  ({nrmse * 100:.2f}%)')
    print(f'  MAE        : {mae:.4f} Nm')

    # ── 6. Per-position evaluation ───────────────────────────────────────────
    # Use the operating positions returned by build_dataset (one mean position
    # per retest trial).  This is exact — no clustering heuristics needed.
    known_pos = operating_positions

    evaluate_per_position(X_test, y_test, y_pred,
                          os.path.join(PLOT_DIR, 'per_position_metrics.png'),
                          known_positions=known_pos)

    # ── 7. Save plots ─────────────────────────────────────────────────────────
    plot_history(history,    os.path.join(PLOT_DIR, 'training_curves.png'))
    plot_predictions(y_test, y_pred, os.path.join(PLOT_DIR, 'test_predictions.png'))

    # Per-position prediction overlays
    plot_predictions_per_position(X_test, y_test, y_pred,
                                  os.path.join(PLOT_DIR, 'predictions_per_position.png'),
                                  known_positions=known_pos)

    # Per-position scatter plots (predicted vs actual)
    plot_scatter_per_position(X_test, y_test, y_pred,
                              os.path.join(PLOT_DIR, 'scatter_per_position.png'),
                              known_positions=known_pos)

    print(f'\nBest model saved → {ckpt_path}')


if __name__ == '__main__':
    main()
