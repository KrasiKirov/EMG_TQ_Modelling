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

from ML.config import (SUBJECTS, TRIAL_OVERRIDES, PASSIVE_TORQUE_OVERRIDES,
                        DATA_DIR, LEARNING_RATE,
                        LR_FACTOR, LR_PATIENCE, LR_MIN, EARLY_PATIENCE,
                        BATCH_SIZE, EPOCHS, N_STEPS, N_FEATURES,
                        POSITION_FEATURE_INDEX)
from ML.preprocessing.flb_reader import read_flb
from ML.preprocessing.dataset_builder import build_dataset
from ML.models.lstm import build_model
from ML.evaluation import compute_metrics, compute_per_position_metrics


def load_subject(subject_key, normalize_position=False, normalize_mvc=False,
                 subtract_passive=True,
                 test_trial_indices=None, retest_trial_indices=None,
                 flb_path=None, subject_id=None, **dataset_options):
    """Load one subject's data with optional normalization.

    Extra keyword arguments (``dataset_options``) are passed to
    ``build_dataset`` — e.g. passive_method, passive_gain,
    train_label_rezero, emg_clip, emg_floor, position_feature.

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
    flb_name, subj_id = SUBJECTS[subject_key]
    if flb_path is None:
        flb_path = os.path.join(DATA_DIR, flb_name)
    if subject_id is None:
        subject_id = subj_id

    # Apply per-subject trial index overrides when not provided by the caller
    overrides = TRIAL_OVERRIDES.get(subject_key, {})
    if test_trial_indices is None and 'test' in overrides:
        test_trial_indices = overrides['test']
        print(f'  Using hardcoded test-trial indices for {subject_key}: {test_trial_indices}')
    if retest_trial_indices is None and 'retest' in overrides:
        retest_trial_indices = overrides['retest']
        print(f'  Using hardcoded retest-trial indices for {subject_key}: {retest_trial_indices}')

    print(f'\n{"="*60}')
    print(f'Loading subject {subject_key}: {flb_path}')
    trials = read_flb(flb_path, subject_id=subject_id)

    passive_overrides = PASSIVE_TORQUE_OVERRIDES.get(subject_key)
    if passive_overrides:
        print(f'  Using hardcoded passive torques for {subject_key} (from worksheet)')

    (X_train, y_train,
     X_val, y_val,
     X_test, y_test,
     emg_max, passive_entries,
     operating_positions, pos_range,
     mvc_tq_test) = build_dataset(
        trials,
        test_trial_indices=test_trial_indices,
        retest_trial_indices=retest_trial_indices,
        subtract_passive=subtract_passive,
        normalize_position=normalize_position,
        normalize_mvc=normalize_mvc,
        passive_overrides=passive_overrides,
        **dataset_options,
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


def _position_knots(X, tol=0.02):
    """Operating angles present in X (last-step position, clustered)."""
    p = np.sort(X[:, -1, POSITION_FEATURE_INDEX])
    groups = np.split(p, np.where(np.diff(p) > tol)[0] + 1)
    return np.array([g.mean() for g in groups])


def _snap(X, knots):
    p = X[:, -1, POSITION_FEATURE_INDEX]
    return np.argmin(np.abs(p[:, None] - knots[None, :]), axis=1)


class PositionStandardizer:
    """Per-angle target standardisation  z = (y − μ_k) / σ_k.

    μ_k, σ_k come from the TRAINING windows of each operating angle k; the
    angle of a window is known at prediction time (it is a model input), so
    de-standardising is causal.  Gives every angle the same weight in the
    loss: without it, MSE is dominated by the strong (dorsiflexed) angles and
    the model barely fits the weak plantarflexed ones (YES: σ = 0.4 Nm at
    −0.49 rad vs 15 Nm at +0.14 rad).
    """

    def __init__(self, X_train, y_train, sd_floor=0.1, center=True):
        self.knots = _position_knots(X_train)
        k = _snap(X_train, self.knots)
        # center=False → scale only (μ_k = 0): no per-angle mean prior, so the
        # model cannot carry a training-session torque level/sign into the
        # test session (CH −0.33 rad switches from a dorsiflexion task in the
        # test session to a plantarflexion task in the retest session).
        self.mu = np.array([y_train[k == j].mean() if center else 0.0
                            for j in range(len(self.knots))])
        self.sd = np.array([max(y_train[k == j].std(), sd_floor)
                            for j in range(len(self.knots))])

    def transform(self, X, y):
        k = _snap(X, self.knots)
        return ((y - self.mu[k]) / self.sd[k]).astype(np.float32)

    def inverse(self, X, z):
        k = _snap(X, self.knots)
        return self.mu[k] + self.sd[k] * np.asarray(z).ravel()


def position_weights(X, y_train, X_train):
    """Per-window loss weights 1/var(target of its angle), angle variances
    from the TRAINING windows, normalised to mean 1 over the training set."""
    knots = _position_knots(X_train)
    k_tr = _snap(X_train, knots)
    inv_var = np.array([1.0 / max(np.var(y_train[k_tr == j]), 1e-3)
                        for j in range(len(knots))])
    w = inv_var[_snap(X, knots)]
    return (w / inv_var[k_tr].mean()).astype(np.float32)


def train_model(X_train, y_train, X_val, y_val, label='model',
                save_path=None, lr=LEARNING_RATE, epochs=EPOCHS,
                batch_size=BATCH_SIZE, verbose=0, position_balance=None,
                anchored=False, moe=False, moe_expert_bias=True):
    """Train a fresh LSTM model and return it with training history.

    Parameters
    ----------
    X_train, y_train : training data
    X_val, y_val : validation data
    label : str
        Descriptive label for console output.
    save_path : str or None
        If given, save best checkpoint to this path.
    lr, epochs, batch_size : training hyperparameters
    verbose : int
        0 = quiet (for batch/sweep runs), 1 = full output.
    position_balance : None | 'weights' | 'standardize' | 'scale'
        Make every operating angle count equally in the loss:
        'weights'     — per-window sample weights 1/var(angle), applied to
                        validation too so early stopping follows the same
                        objective;
        'standardize' — train on per-angle standardised targets
                        (``PositionStandardizer``) and de-standardise
                        predictions in ``evaluate_model``;
        'scale'       — like 'standardize' but divide by σ_k only (no
                        per-angle mean prior; robust to a task change).
    anchored : bool
        Use the zero-anchored model (see ``build_model``).
    moe : bool
        Mixture-of-experts head with one expert per training operating angle
        (see ``build_model``); implies position_balance='standardize' unless
        'scale' is given.
    moe_expert_bias : bool
        Per-angle bias in the experts (False = no per-angle offset prior).

    Returns
    -------
    model : tensorflow.keras.Model — trained model with best weights restored.
    history : tensorflow.keras.callbacks.History
    """
    if moe and position_balance != 'scale':
        position_balance = 'standardize'
    model = build_model(n_steps=N_STEPS, n_features=X_train.shape[2],
                        anchored=anchored,
                        moe_knots=_position_knots(X_train) if moe else None,
                        moe_expert_bias=moe_expert_bias)
    model.compile(optimizer=Nadam(learning_rate=lr),
                  loss='mse', metrics=['mae'])

    sample_weight, val_data = None, (X_val, y_val)
    if position_balance == 'weights':
        sample_weight = position_weights(X_train, y_train, X_train)
        val_data = (X_val, y_val, position_weights(X_val, y_train, X_train))
    elif position_balance in ('standardize', 'scale'):
        std = PositionStandardizer(X_train, y_train,
                                   center=(position_balance == 'standardize'))
        y_train = std.transform(X_train, y_train)
        val_data = (X_val, std.transform(X_val, y_val))
        model.position_standardizer = std
        if save_path:
            np.savez(os.path.splitext(save_path)[0] + '_posstd.npz',
                     knots=std.knots, mu=std.mu, sd=std.sd)
    elif position_balance is not None:
        raise ValueError(f'position_balance={position_balance!r}')

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
        sample_weight=sample_weight,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=verbose,
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
    std = getattr(model, 'position_standardizer', None)
    if std is not None:
        y_pred = std.inverse(X_test, y_pred)
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
