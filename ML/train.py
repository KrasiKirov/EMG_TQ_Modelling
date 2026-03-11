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
FLB_PATH   = os.path.join(os.path.dirname(__file__), 'YES_281124.flb')
MODEL_DIR  = os.path.join(os.path.dirname(__file__), 'models')
PLOT_DIR   = os.path.join(os.path.dirname(__file__), 'plots')

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
    p.add_argument('--flb',    default=FLB_PATH,  help='Path to .flb file')
    p.add_argument('--epochs', default=EPOCHS, type=int)
    p.add_argument('--batch',  default=BATCH_SIZE, type=int)
    p.add_argument('--lr',     default=LEARNING_RATE, type=float)
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
    ax.set_title(f'Test set predictions (first {len(idx)} windows)')
    ax.legend()
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Prediction plot saved → {out_path}')


def main():
    args = parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR,  exist_ok=True)

    # ── 1. Load & build dataset ───────────────────────────────────────────────
    print(f'\nReading: {args.flb}')
    trials = read_flb(args.flb)

    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     mvc_max, passive_entries) = build_dataset(trials, subtract_passive=True)

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
    rmse   = np.sqrt(np.mean((y_test - y_pred) ** 2))
    nrmse  = rmse / (y_test.max() - y_test.min())
    mae    = np.mean(np.abs(y_test - y_pred))

    print(f'\nTest results:')
    print(f'  R²         : {r2:.4f}')
    print(f'  RMSE       : {rmse:.4f} Nm')
    print(f'  NRMSE      : {nrmse:.4f}  ({nrmse * 100:.2f}%)')
    print(f'  MAE        : {mae:.4f} Nm')

    # ── 6. Save plots ─────────────────────────────────────────────────────────
    plot_history(history,    os.path.join(PLOT_DIR, 'training_curves.png'))
    plot_predictions(y_test, y_pred, os.path.join(PLOT_DIR, 'test_predictions.png'))

    print(f'\nBest model saved → {ckpt_path}')


if __name__ == '__main__':
    main()
