"""Passive torque maps: (angle, torque) tables from the passive trials, and the lookup between their angles.

The interpolated map (``get_passive_torque_map``) is the original pipeline's; ``static_hold_passive_map`` keeps only
genuinely static holds (``is_static_hold``: position sd <= 5 mrad, torque sd <= 1 Nm, at least 5 s).
"""

import warnings

import numpy as np

from ML.config import FS_RAW

from .trials import POSITION_COL, POSITION_TOLERANCE, TORQUE_COL, split_passive_segments

_PASSIVE_STD_THRESHOLD = 20.0   # Nm  — segment rejected if torque still noisy after split
_PASSIVE_TQ_BOUND      = 20.0   # Nm  — mean passive torque above this is physiologically implausible

STATIC_MAX_POS_STD = 0.005   # rad
STATIC_MAX_TQ_STD  = 1.0     # Nm
STATIC_MIN_DUR_S   = 5.0     # s


def is_static_hold(seg, fs=FS_RAW, position_col=POSITION_COL, torque_col=TORQUE_COL):
    """True if a passive segment is a static hold: long enough, with position and torque nearly constant."""
    p, t = seg[position_col].values, seg[torque_col].values
    return not (len(t) < STATIC_MIN_DUR_S * fs or p.std() > STATIC_MAX_POS_STD or t.std() > STATIC_MAX_TQ_STD)


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


def get_passive_torque_map(passive_trials, position_col=POSITION_COL,
                           torque_col=TORQUE_COL,
                           std_threshold=_PASSIVE_STD_THRESHOLD,
                           pos_tolerance=POSITION_TOLERANCE):
    """
    Sorted [(mean position rad, mean passive torque Nm)] from the passive trials.

    Trials are split with ``split_passive_segments``; segments with torque std above ``std_threshold`` are dropped and
    positions within ``pos_tolerance`` rad are merged by weighted averaging.
    """
    raw = []
    for df in passive_trials:
        for seg in split_passive_segments(df, position_col):
            tq_std = float(seg[torque_col].std())
            if tq_std > std_threshold:
                continue
            mean_pos = float(seg[position_col].mean())
            mean_tq  = float(seg[torque_col].mean())
            if abs(mean_tq) > _PASSIVE_TQ_BOUND:
                continue
            raw.append((mean_pos, mean_tq))
    raw.sort(key=lambda x: x[0])     # by position only (ties keep their trial order)
    # Merge near-identical holds so they cannot give contradictory offsets.
    entries = _merge_nearby(raw, pos_tolerance)
    if len(entries) < 2:
        warnings.warn(
            f"Only {len(entries)} passive torque entry/entries found. "
            f"Extrapolation will be limited to constant value.",
            stacklevel=2,
        )
    return entries


def _extrapolate(positions, torques, x):
    """Linear extrapolation beyond the measured range of passive positions."""
    if len(positions) < 2:
        return float(torques[0]) if len(torques) > 0 else 0.0
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
    # np.interp clamps at boundaries; use manual linear extrapolation instead
    return float(np.interp(ankle_position_rad, positions, torques,
                           left=None, right=None)
                 if positions[0] <= ankle_position_rad <= positions[-1]
                 else _extrapolate(positions, torques, ankle_position_rad))


def static_hold_passive_map(passive_trials, pos_tolerance=POSITION_TOLERANCE):
    """Sorted [(position, mean torque)] of the static holds (``is_static_hold``) in the passive trials, each trial
    split at device steps first; positions within ``pos_tolerance`` rad are averaged."""
    raw = []
    for df in passive_trials:
        for seg in split_passive_segments(df):
            if is_static_hold(seg):
                raw.append((float(seg[POSITION_COL].values.mean()), float(seg[TORQUE_COL].values.mean())))
    raw.sort()                       # by position, then torque
    return _merge_nearby(raw, pos_tolerance)
