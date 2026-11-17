"""Shared training utilities: load_subject, train_model, evaluate_model."""

import os
import numpy as np

from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import (ReduceLROnPlateau, EarlyStopping,
                                        ModelCheckpoint)

from ML.config import (SUBJECTS, TRIAL_OVERRIDES, PASSIVE_TORQUE_OVERRIDES,
                        DATA_DIR, LEARNING_RATE,
                        LR_FACTOR, LR_PATIENCE, LR_MIN, EARLY_PATIENCE,
                        BATCH_SIZE, EPOCHS, N_STEPS, N_FEATURES, LSTM_UNITS,
                        POSITION_FEATURE_INDEX)
from ML.preprocessing.flb_reader import read_flb
from ML.preprocessing.dataset_builder import build_dataset
from ML.models.lstm import build_model
from ML.evaluation import compute_metrics, compute_per_position_metrics, snap_to_operating_points


def load_subject(subject_key, normalize_position=False, normalize_mvc=False,
                 subtract_passive=True,
                 test_trial_indices=None, retest_trial_indices=None,
                 flb_path=None, subject_id=None, passive_entries_override=None,
                 **dataset_options):
    """
    Load one subject through ``build_dataset``; extra keyword arguments are passed to it.

    Trial indices are 1-based overrides; flb_path / subject_id override the SUBJECTS registry.
    Returns a dict: X_train, y_train, X_val, y_val, X_test, y_test, emg_max, operating_positions, pos_range,
    mvc_tq_test.
    """
    if flb_path is None or subject_id is None:
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
        passive_entries_override=passive_entries_override,
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
    """
    Per-angle target standardisation z = (y - mu_k) / sd_k, from the training windows of each angle.

    The angle is a model input, so de-standardising is causal. Without it MSE is dominated by the strong angles
    (YES: sd 0.4 Nm at -0.49 rad vs 15 Nm at +0.14 rad).
    """

    def __init__(self, X_train, y_train, sd_floor=0.1, center=True):
        self.knots = _position_knots(X_train)
        k = _snap(X_train, self.knots)
        # center=False: scale only, so no training-session torque level carries into the test session
        # (CH -0.33 rad switches from a dorsiflexion to a plantarflexion task).
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


def compute_position_weights(X_train, y_train, operating_positions,
                              clip_max=5.0):
    """Per-sample weights inversely proportional to the per-position torque IQR, clipped at ``clip_max``, mean 1."""
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
                position_balance=None, anchored=False, moe=False,
                moe_expert_bias=True, sample_weight=None):
    """
    Train a fresh LSTM; returns (model with best weights restored, history).

    position_balance  None | 'weights' (1/var(angle), also on validation) | 'standardize' (``PositionStandardizer``,
                      undone in ``evaluate_model``) | 'scale' (÷ sd_k only, no per-angle mean prior)
    anchored, moe     model variants (see ``build_model``); moe implies 'standardize' unless 'scale' is given
    moe_expert_bias   per-angle bias in the experts
    sample_weight     per-window training weights, not combined with position_balance
    verbose           0 quiet, 1 full output
    """
    if moe and position_balance != 'scale':
        position_balance = 'standardize'
    model = build_model(n_steps=N_STEPS, n_features=X_train.shape[2],
                        n_units=n_units, anchored=anchored,
                        moe_knots=_position_knots(X_train) if moe else None,
                        moe_expert_bias=moe_expert_bias)
    model.compile(optimizer=Nadam(learning_rate=lr),
                  loss='mse', metrics=['mae'])

    val_data = (X_val, y_val)
    if sample_weight is not None and position_balance is not None:
        raise ValueError('pass either sample_weight or position_balance, not both')
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
    """Predict, convert back to Nm if MVC-normalised, and score; returns {y_pred, y_test, overall, per_position}."""
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
