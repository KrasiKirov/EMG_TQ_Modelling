"""Scores computed from saved predictions (no model is loaded here).

Predictions are always scored as saved.
"""
import numpy as np


def metrics(y, p):
    """Error of predictions p against measured y: r2, mae, rmse, bias, the same after removing each mean, and peak |y|."""
    e = p - y
    ss = np.sum((y - y.mean()) ** 2)
    ed = (p - p.mean()) - (y - y.mean())
    assert ss > 0
    return dict(r2=float(1 - np.sum(e ** 2) / ss), mae=float(np.mean(abs(e))),
                rmse=float(np.sqrt(np.mean(e ** 2))), bias=float(np.mean(e)),
                r2_demeaned=float(1 - np.sum(ed ** 2) / ss), mae_demeaned=float(np.mean(abs(ed))),
                peak=float(np.max(abs(y))))


def person_scale(y_measured):
    """Normalising scale of one person and target: 99th percentile of |measured target| (never a prediction)."""
    return float(np.percentile(np.abs(np.asarray(y_measured, float)), 99))


def score_trial(y, p, t, *, t0=None, scale=None, score_from_s=0.0):
    """Score one trial (one position). Returns the metrics row.

    t0            the trial's first sample time in the measured data (default: t[0])
    score_from_s  samples with t - t0 < score_from_s are dropped, so settings whose first prediction comes at
                  different times are scored on the same samples (study v2: 1.0 s)
    scale         person_scale of the tested session; adds nmae = mae / scale
    With any of these given, also adds nrmse_pos = rmse / (max - min of the measured target) and n; without them
    the row is metrics(y, p).
    """
    if t0 is None and scale is None and not score_from_s:
        return metrics(y, p)
    y, p, t = np.asarray(y, float), np.asarray(p, float), np.asarray(t, float)
    t0 = float(t[0]) if t0 is None else float(t0)
    keep = (t - t0) >= score_from_s
    y, p, t = y[keep], p[keep], t[keep]
    row = metrics(y, p)
    rng = float(np.ptp(y))
    row['nrmse_pos'] = row['rmse'] / rng if rng > 0 else float('nan')
    row['n'] = int(len(y))
    if scale is not None:
        row['nmae'] = row['mae'] / float(scale)
    return row
