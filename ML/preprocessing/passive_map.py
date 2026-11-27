"""Passive torque maps: (angle, torque) tables from the static holds of the passive trials (``is_static_hold``:
position sd <= 5 mrad, torque sd <= 1 Nm, at least 5 s), and the passive torque at any angle from a map.
"""

import numpy as np

from ML.config import FS_RAW

from .trials import POSITION_COL, POSITION_TOLERANCE, TORQUE_COL, split_passive_segments

STATIC_MAX_POS_STD = 0.005   # rad
STATIC_MAX_TQ_STD  = 1.0     # Nm
STATIC_MIN_DUR_S   = 5.0     # s


def is_static_hold(seg):
    """True if a passive segment is a static hold: long enough, with position and torque nearly constant."""
    p, t = seg[POSITION_COL].values, seg[TORQUE_COL].values
    return not (len(t) < STATIC_MIN_DUR_S * FS_RAW or p.std() > STATIC_MAX_POS_STD or t.std() > STATIC_MAX_TQ_STD)


def _merge_nearby(raw, pos_tolerance):
    """Merge sorted (position, torque) pairs within ``pos_tolerance`` rad of the running cluster mean (averaged)."""
    clusters = []
    for pos, tq in raw:
        if clusters and abs(pos - clusters[-1][0]) <= pos_tolerance:
            pp, tt, c = clusters[-1]
            clusters[-1] = ((pp * c + pos) / (c + 1), (tt * c + tq) / (c + 1), c + 1)
        else:
            clusters.append((pos, tq, 1))
    return [(pos, tq) for pos, tq, _ in clusters]


def _line(p0, p1, t0, t1, x):
    """The straight line through (p0, t0) and (p1, t1) at x (t0 if the two positions coincide)."""
    if abs(p1 - p0) < 1e-12:
        return np.full(x.shape, float(t0))
    return t0 + (t1 - t0) / (p1 - p0) * (x - p0)


def lookup_passive_torque(passive_entries, angles):
    """
    Passive torque (Nm) at each angle in ``angles`` (rad): linear interpolation in the map, linear extrapolation
    from the two boundary points outside it; 0 for an empty map, the one torque for a one-point map.
    """
    x = np.asarray(angles, np.float64)
    if not passive_entries:
        return np.zeros(x.shape)
    if len(passive_entries) == 1:
        return np.full(x.shape, float(passive_entries[0][1]))
    p = np.array([e[0] for e in passive_entries], np.float64)
    t = np.array([e[1] for e in passive_entries], np.float64)
    out = np.interp(x, p, t)                   # clamps outside the map, so both ends are extrapolated below
    lo, hi = x < p[0], x > p[-1]
    out[lo] = _line(p[0], p[1], t[0], t[1], x[lo])
    out[hi] = _line(p[-2], p[-1], t[-2], t[-1], x[hi])
    return out


def static_hold_passive_map(passive_trials):
    """Sorted [(position, mean torque)] of the static holds (``is_static_hold``) in the passive trials, each trial
    split at device steps first; positions within POSITION_TOLERANCE rad are averaged."""
    raw = []
    for df in passive_trials:
        for seg in split_passive_segments(df):
            if is_static_hold(seg):
                raw.append((float(seg[POSITION_COL].values.mean()), float(seg[TORQUE_COL].values.mean())))
    raw.sort()                       # by position, then torque
    return _merge_nearby(raw, POSITION_TOLERANCE)
