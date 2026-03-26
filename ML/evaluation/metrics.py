"""
Shared evaluation utilities for sEMG-to-torque models.

Consolidates metric computation and position-snapping logic used
by both train.py and cross_subject.py.
"""

import numpy as np

from ML.config import POSITION_FEATURE_INDEX


def snap_to_operating_points(positions, known_positions):
    """Assign each window's position to the nearest known operating point.

    Parameters
    ----------
    positions : np.ndarray, shape (N,)
        Raw (or normalized) position value per window.
    known_positions : array-like
        Set of known operating points.

    Returns
    -------
    snapped : np.ndarray, shape (N,)
        Each entry replaced by the nearest value in *known_positions*.
    known_sorted : np.ndarray
        Sorted array of known positions.
    """
    known = np.array(sorted(known_positions))
    dists = np.abs(positions[:, None] - known[None, :])
    return known[np.argmin(dists, axis=1)], known


def compute_metrics(y_true, y_pred):
    """Compute standard regression metrics.

    Returns
    -------
    dict with keys: r2, vaf, rmse, nrmse, mae
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2    = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    vaf   = (1.0 - np.var(y_true - y_pred) / np.var(y_true)) * 100.0 if np.var(y_true) > 0 else float('nan')
    rmse  = np.sqrt(np.mean((y_true - y_pred) ** 2))
    y_range = y_true.max() - y_true.min()
    nrmse = rmse / y_range if y_range > 0 else float('nan')
    mae   = np.mean(np.abs(y_true - y_pred))
    return {'r2': r2, 'vaf': vaf, 'rmse': rmse, 'nrmse': nrmse, 'mae': mae}


def compute_per_position_metrics(X_test, y_test, y_pred, known_positions):
    """Compute R², RMSE, and NRMSE for each operating position.

    Parameters
    ----------
    X_test : np.ndarray, shape (N, T, F)
        Test feature windows.
    y_test, y_pred : np.ndarray, shape (N,)
    known_positions : array-like
        Set of known operating points.

    Returns
    -------
    dict  {position_value: {'r2': float, 'rmse': float, 'nrmse': float, 'mae': float, 'n': int}}
    """
    positions = X_test[:, -1, POSITION_FEATURE_INDEX]
    pos_snapped, valid_pos = snap_to_operating_points(positions, known_positions)
    pos_metrics = {}
    for pos in valid_pos:
        mask = pos_snapped == pos
        yt = y_test[mask]
        yp = y_pred[mask]
        pos_metrics[pos] = compute_metrics(yt, yp)
        pos_metrics[pos]['n'] = int(mask.sum())
    return pos_metrics
