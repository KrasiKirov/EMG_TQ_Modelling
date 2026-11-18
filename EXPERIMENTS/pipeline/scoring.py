"""Scores computed from saved predictions (no model is loaded here).

Predictions are always scored as saved, on the whole trial.

Every row also carries an error breakdown: MAE from `score_from_s` to the end of the trial before and after a
per-trial gain+offset fitted on the first `seconds`. The difference is the part of the error that is drift
(scale/offset); what remains is shape error. This only describes the error; it never changes the model's score.
"""
import numpy as np

GAIN_LIMITS = (0.5, 2.0)
MIN_SD = 0.2
DEFAULT_BREAKDOWN = {'seconds': 10.0, 'score_from_s': 30.0}


def metrics(y, p):
    e = p - y
    ss = np.sum((y - y.mean()) ** 2)
    ed = (p - p.mean()) - (y - y.mean())
    assert ss > 0
    return dict(r2=float(1 - np.sum(e ** 2) / ss), mae=float(np.mean(abs(e))),
                rmse=float(np.sqrt(np.mean(e ** 2))), bias=float(np.mean(e)),
                r2_demeaned=float(1 - np.sum(ed ** 2) / ss), mae_demeaned=float(np.mean(abs(ed))),
                peak=float(np.max(abs(y))))


def gain_offset(p_cal, y_cal, p):
    """p' = a p + b fitted on the first block; a clipped to [0.5, 2]; offset only if the torque barely moves."""
    if np.std(y_cal) < MIN_SD:
        return p + np.mean(y_cal - p_cal)
    a, _ = np.polyfit(p_cal, y_cal, 1)
    a = float(np.clip(a, *GAIN_LIMITS))
    b = float(np.mean(y_cal - a * p_cal))
    return a * p + b


def person_scale(y_measured):
    """Normalising scale of one person and target: 99th percentile of |measured target| (never a prediction)."""
    return float(np.percentile(np.abs(np.asarray(y_measured, float)), 99))


def score_trial(y, p, t, *, breakdown=DEFAULT_BREAKDOWN, t0=None, scale=None, score_from_s=0.0):
    """Score one trial (one position). Returns the metrics row.

    t0            the trial's first sample time in the measured data (default: t[0])
    score_from_s  samples with t - t0 < score_from_s are dropped, so settings whose first prediction comes at
                  different times are scored on the same samples (study v2: 1.0 s)
    scale         person_scale of the tested session; adds nmae = mae / scale
    With any of these given, also adds nrmse_pos = rmse / (max - min of the measured target) and n; without them
    the row is exactly the v1 row.
    """
    if t0 is None and scale is None and not score_from_s:
        return _score(y, p, t, breakdown)
    y, p, t = np.asarray(y, float), np.asarray(p, float), np.asarray(t, float)
    t0 = float(t[0]) if t0 is None else float(t0)
    keep = (t - t0) >= score_from_s
    y, p, t = y[keep], p[keep], t[keep]
    row = _score(y, p, t, breakdown)
    rng = float(np.ptp(y))
    row['nrmse_pos'] = row['rmse'] / rng if rng > 0 else float('nan')
    row['n'] = int(len(y))
    if scale is not None:
        row['nmae'] = row['mae'] / float(scale)
    return row


def _score(y, p, t, breakdown):
    rel = t - t[0]
    block, scored = rel < breakdown['seconds'], rel >= breakdown['score_from_s']
    corrected = gain_offset(p[block], y[block], p)
    row = metrics(y, p)
    if scored.sum() > 1 and np.ptp(y[scored]) > 0:
        row['err_raw_mae'] = float(np.mean(abs(p[scored] - y[scored])))
        row['err_after_gain_offset_mae'] = float(np.mean(abs(corrected[scored] - y[scored])))
    return row
