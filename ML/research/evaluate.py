"""
Metrics, recalibration and trial-inspection plots.

Metrics per angle (one test trial = one angle of one session)
-------------------------------------------------------------
r2     1 - SS_res / SS_tot with the angle's own mean (the strict per-angle
       R² used throughout the project)
rmse   root-mean-square error (Nm)            mae   mean |error| (Nm)
bias   mean(prediction - truth) (Nm)          vaf   variance accounted for (%)
nrmse  rmse / (98th - 2nd percentile of the true torque)  (range-normalised)

Recalibration ("session calibration")
-------------------------------------
A common practice for myoelectric interfaces: record a short labelled
calibration at the start of each session.  Here, per test trial, a linear
map  y ≈ a·pred + b  is fitted on the first ``calib_s`` seconds and applied to
the rest of the trial; calibrated and uncalibrated predictions are scored on
the SAME remaining samples (the calibration samples are never scored).
"""

import numpy as np

from ML.evaluation import plot_full_trials


def metrics(y, p):
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    e = p - y
    ss_tot = np.sum((y - y.mean()) ** 2)
    rng = np.percentile(y, 98) - np.percentile(y, 2)
    return {
        'r2': float(1 - np.sum(e ** 2) / ss_tot) if ss_tot > 0 else float('nan'),
        'rmse': float(np.sqrt(np.mean(e ** 2))), 'mae': float(np.mean(np.abs(e))),
        'bias': float(np.mean(e)),
        'vaf': float((1 - np.var(e) / np.var(y)) * 100) if np.var(y) > 0 else float('nan'),
        'nrmse': float(np.sqrt(np.mean(e ** 2)) / rng) if rng > 0 else float('nan'),
        'std_true': float(np.std(y)), 'n': int(len(y)),
    }


def group_keys(meta):
    return np.array([f"{s}:{a:+.3f}" for s, a in zip(meta['session'], meta['angle'])])


def per_group(y, p, meta):
    keys = group_keys(meta)
    return {k: metrics(y[keys == k], p[keys == k]) for k in sorted(set(keys), key=lambda s: (s[:2], float(s[3:])))}


def recalibrated(y, p, meta, calib_s=15.0, gap_s=0.2):
    """Per-trial linear recalibration on the first ``calib_s`` seconds."""
    keys = group_keys(meta)
    out = {}
    for k in sorted(set(keys)):
        m = keys == k
        t = meta['t'][m]
        t0 = t.min()
        cal, ev = t < t0 + calib_s, t >= t0 + calib_s + gap_s
        if cal.sum() < 50 or ev.sum() < 100:
            continue
        yy, pp = y[m], p[m]
        A = np.column_stack([pp[cal], np.ones(cal.sum())])
        a, b = np.linalg.lstsq(A, yy[cal], rcond=None)[0]
        out[k] = {'raw': metrics(yy[ev], pp[ev]), 'calibrated': metrics(yy[ev], a * pp[ev] + b),
                  'a': float(a), 'b': float(b)}
    return out


def plot_trials(meta, y, p, out_path, title):
    """Full-trial inspection plot (one panel per angle) using the pipeline's
    own ML.evaluation.plot_full_trials.  Sessions are plotted separately by
    the caller when a protocol tests on both sessions."""
    X = np.zeros((len(y), 1, 5), np.float32)
    X[:, 0, 4] = meta['angle']
    plot_full_trials(X, np.asarray(y), np.asarray(p), out_path,
                     known_positions=sorted(set(np.round(meta['angle'], 6))), title=title)
