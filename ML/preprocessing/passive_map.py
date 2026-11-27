"""Passive torque maps: (angle, torque) tables from the static holds of the passive trials (``is_static_hold``:
position sd <= 5 mrad, torque sd <= 1 Nm, at least 5 s), and the lookup between their angles.
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


def _extrapolate(positions, torques, x):
    """Linear extrapolation beyond the measured range of passive positions (at least 2 of them)."""
    if x < positions[0]:
        p0, p1 = positions[0], positions[1]
        t0, t1 = torques[0],   torques[1]
    else:
        p0, p1 = positions[-2], positions[-1]
        t0, t1 = torques[-2],   torques[-1]
    if abs(p1 - p0) < 1e-12:
        return float(t0)
    slope = (t1 - t0) / (p1 - p0)
    return float(t0 + slope * (x - p0))


def lookup_passive_torque(passive_entries, ankle_position_rad):
    """
    Passive torque at ``ankle_position_rad``: linear interpolation in the map, linear extrapolation from the two
    boundary points outside it; 0.0 if the map is empty.
    """
    if not passive_entries:
        return 0.0
    if len(passive_entries) == 1:
        return passive_entries[0][1]

    positions = np.array([p for p, _ in passive_entries])
    torques   = np.array([t for _, t in passive_entries])
    # np.interp clamps at the boundaries, so outside the map the two boundary points are extrapolated instead
    return float(np.interp(ankle_position_rad, positions, torques)
                 if positions[0] <= ankle_position_rad <= positions[-1]
                 else _extrapolate(positions, torques, ankle_position_rad))


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
