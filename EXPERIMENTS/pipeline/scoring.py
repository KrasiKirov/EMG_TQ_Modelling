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


def score_trial(y, p, t, *, breakdown=DEFAULT_BREAKDOWN):
    """Score one trial (one position) on the whole trial. Returns (row, predictions to display)."""
    rel = t - t[0]
    block, scored = rel < breakdown['seconds'], rel >= breakdown['score_from_s']
    corrected = gain_offset(p[block], y[block], p)
    row, shown = metrics(y, p), p
    if scored.sum() > 1 and np.ptp(y[scored]) > 0:
        row['err_raw_mae'] = float(np.mean(abs(p[scored] - y[scored])))
        row['err_after_gain_offset_mae'] = float(np.mean(abs(corrected[scored] - y[scored])))
    return row, shown
