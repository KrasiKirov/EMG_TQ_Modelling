"""Shared training utilities for sEMG-to-torque LSTM models.

Provides common functions used by the training entry points:
  - load_subject: load and preprocess one subject's data
  - train_model: train a fresh LSTM with standard callbacks
  - evaluate_model: predict, denormalize, and compute metrics
"""

import os
import numpy as np

from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import (ReduceLROnPlateau, EarlyStopping,
                                        ModelCheckpoint)

from ML.config import (SUBJECTS, DATA_DIR, LEARNING_RATE, LR_FACTOR,
                        LR_PATIENCE, LR_MIN, EARLY_PATIENCE, BATCH_SIZE,
                        EPOCHS, N_STEPS, N_FEATURES, LSTM_UNITS,
                        POSITION_FEATURE_INDEX)
from ML.preprocessing.flb_reader import read_flb
from ML.preprocessing.dataset_builder import build_dataset
from ML.models.lstm import build_model
from ML.evaluation import compute_metrics, compute_per_position_metrics, snap_to_operating_points


def load_subject(subject_key, normalize_position=False, normalize_mvc=False,
                 test_trial_indices=None, retest_trial_indices=None,
                 flb_path=None, subject_id=None, passive_entries_override=None):
    """Load one subject's data with optional normalization.

    Parameters
    ----------
    subject_key : str
        Short name from SUBJECTS registry (e.g. 'HM', 'EG').
    normalize_position : bool
        Min-max normalize ankle position to [0, 1].
    normalize_mvc : bool
        Normalize EMG and torque by per-position MVC.
    test_trial_indices, retest_trial_indices : list of int or None
        Override auto-detected trial splits (1-indexed).
    flb_path : str or None
        Override the FLB file path from SUBJECTS registry.
    subject_id : str or None
        Override the subject ID from SUBJECTS registry.

    Returns
    -------
    dict with keys: X_train, y_train, X_val, y_val, X_test, y_test,
                    emg_max, operating_positions, pos_range, mvc_tq_test
    """
    if flb_path is None or subject_id is None:
        flb_name, subj_id = SUBJECTS[subject_key]
        if flb_path is None:
            flb_path = os.path.join(DATA_DIR, flb_name)
        if subject_id is None:
            subject_id = subj_id

    print(f'\n{"─"*60}')
    print(f'Loading subject {subject_key}: {flb_path}')
    trials = read_flb(flb_path, subject_id=subject_id)

    (X_train, y_train,
     X_val, y_val,
     X_test, y_test,
     emg_max, passive_entries,
     operating_positions, pos_range,
     mvc_tq_test) = build_dataset(
        trials,
        test_trial_indices=test_trial_indices,
        retest_trial_indices=retest_trial_indices,
        subtract_passive=True,
        normalize_position=normalize_position,
        normalize_mvc=normalize_mvc,
        passive_entries_override=passive_entries_override,
    )
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'emg_max': emg_max,
        'operating_positions': operating_positions,
        'pos_range': pos_range,
        'mvc_tq_test': mvc_tq_test,
    }


def compute_position_weights(X_train, y_train, operating_positions,
                              clip_max=5.0):
    """Per-sample weights inversely proportional to per-position torque IQR.

    Compensates for MSE being dominated by high-torque positions by upweighting
    small-signal positions (e.g. max plantarflexion IQR ~1.5 Nm vs ~11 Nm at
    max dorsiflexion).  Weights are normalized so their mean equals 1, preserving
    the effective learning rate.

    Parameters
    ----------
    X_train : np.ndarray, shape (N, n_steps, n_features)
    y_train : np.ndarray, shape (N,)
    operating_positions : list of float
    clip_max : float
        Maximum upweight factor (default 5.0).

    Returns
    -------
    weights : np.ndarray, shape (N,), mean == 1.0
    """
    positions = X_train[:, -1, POSITION_FEATURE_INDEX]
    pos_snapped, valid_pos = snap_to_operating_points(positions, operating_positions)

    iqr_map = {}
    for p in valid_pos:
        mask = pos_snapped == p
        yt = y_train[mask]
        iqr = float(np.percentile(yt, 75) - np.percentile(yt, 25))
        iqr_map[p] = max(iqr, 0.1)   # floor to avoid division by zero

    mean_iqr = float(np.mean(list(iqr_map.values())))
    raw = np.array([mean_iqr / iqr_map[p] for p in pos_snapped], dtype=np.float32)
    raw = np.clip(raw, 1.0 / clip_max, float(clip_max))
    return (raw / raw.mean()).astype(np.float32)


def train_model(X_train, y_train, X_val, y_val, label='model',
                save_path=None, lr=LEARNING_RATE, epochs=EPOCHS,
                batch_size=BATCH_SIZE, n_units=LSTM_UNITS, verbose=0,
                sample_weight=None):
    """Train a fresh LSTM model and return it with training history.

    Parameters
    ----------
    X_train, y_train : training data
    X_val, y_val : validation data
    label : str
        Descriptive label for console output.
    save_path : str or None
        If given, save best checkpoint to this path.
    lr, epochs, batch_size, n_units : training hyperparameters
    verbose : int
        0 = quiet (for batch/sweep runs), 1 = full output.

    Returns
    -------
    model : tensorflow.keras.Model — trained model with best weights restored.
    history : tensorflow.keras.callbacks.History
    """
    model = build_model(n_steps=N_STEPS, n_features=N_FEATURES, n_units=n_units)
    model.compile(optimizer=Nadam(learning_rate=lr),
                  loss='mse', metrics=['mae'])

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=LR_FACTOR,
                          patience=LR_PATIENCE, min_lr=LR_MIN,
                          verbose=verbose),
        EarlyStopping(monitor='val_loss', patience=EARLY_PATIENCE,
                      restore_best_weights=True, verbose=1),
    ]
    if save_path:
        callbacks.append(ModelCheckpoint(save_path, monitor='val_loss',
                                         save_best_only=True,
                                         verbose=verbose))

    print(f'  Training {label}...')
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=verbose,
        sample_weight=sample_weight,
    )
    return model, history


def evaluate_model(model, X_test, y_test, operating_positions,
                   mvc_tq_test=None):
    """Predict, denormalize to Nm if needed, compute metrics.

    Parameters
    ----------
    model : trained Keras model
    X_test, y_test : test data
    operating_positions : list of float
        Known operating points for per-position breakdown.
    mvc_tq_test : np.ndarray or None
        Per-window MVC torque for denormalization.

    Returns
    -------
    dict with keys: y_pred, y_test (denormalized), overall, per_position
    """
    y_pred = model.predict(X_test, verbose=0).flatten()
    y_test = y_test.copy()

    if mvc_tq_test is not None:
        y_pred = y_pred * mvc_tq_test
        y_test = y_test * mvc_tq_test

    overall = compute_metrics(y_test, y_pred)
    per_pos = compute_per_position_metrics(X_test, y_test, y_pred,
                                           operating_positions)
    return {
        'y_pred': y_pred,
        'y_test': y_test,
        'overall': overall,
        'per_position': per_pos,
    }
