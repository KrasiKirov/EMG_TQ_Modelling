"""
dataset_builder.py
------------------
Builds the LSTM input dataset from FLB trial data.

Pipeline
--------
1.  Classify all trials by comment → MVC / passive / active
2.  Extract EMG envelopes at 1000 Hz and normalize using MVC max per channel
3.  Subtract mean passive torque per ankle position → net active torque
4.  Downsample 1000 Hz → 100 Hz  (FIR anti-alias, zero-phase)
5.  Build sliding windows  [N × 20 × 5]
        Features : MG_env_norm, LG_env_norm, SOL_env_norm, TA_env_norm, position (rad)
        Target   : active_torque (Nm) = measured_torque − mean_passive_torque
6.  Temporal split per trial: 60 % train / 20 % val / 20 % test, then stack.

Design choices
--------------
- Ankle position kept in raw radians (no scaling).
- Torque target is net active torque after passive subtraction.
- EMG normalization uses MVC max (trials with comment starting with 'mvc').
- Passive torque is the mean measured torque over the matching passive trial(s)
  for each ankle position tag (p1 … p8).
"""

import re
import numpy as np
import pandas as pd
from scipy.signal import decimate

try:
    from .emg_envelope import process_trials, normalize_with_max
except ImportError:
    from emg_envelope import process_trials, normalize_with_max


# ── Constants ──────────────────────────────────────────────────────────────

POSITION_COL      = 'position'   # ankle angle column name (rad)
TORQUE_COL        = 'torque'     # measured torque column name (Nm)

DOWNSAMPLE_FACTOR = 10           # 1000 Hz → 100 Hz
WINDOW_STEPS      = 20           # 20 steps × 10 ms = 200 ms history
STRIDE_STEPS      = 1            # 1 step = 10 ms between predictions

TRAIN_FRAC        = 0.60
VAL_FRAC          = 0.20
# TEST_FRAC       = 0.20  (implicit: 1 − TRAIN − VAL)

# Trials (active or passive) whose torque std exceeds this are passive
# movement / ramp trials and are excluded from both the training set and
# the passive torque lookup.
_ACTIVE_STD_THRESHOLD  = 30.0   # Nm


# ── Trial classification ────────────────────────────────────────────────────

def _parse_comment(comment: str):
    """
    Parse a trial comment string.

    Returns
    -------
    trial_type : 'mvc' | 'passive' | 'active'
    position   : 'p1' .. 'p8'  (or None)
    """
    comment = comment.lower().strip()
    m = re.search(r'p(\d+)', comment)
    pos = m.group(0) if m else None

    if comment.startswith('mvc'):
        return 'mvc', pos
    elif 'passive' in comment:
        return 'passive', pos
    else:
        return 'active', pos


def classify_trials(trials):
    """
    Split trial DataFrames into MVC, passive, and active groups.

    Parameters
    ----------
    trials : list of pd.DataFrame
        All trials from ``read_flb``.

    Returns
    -------
    mvc, passive, active : list of pd.DataFrame
    """
    mvc, passive, active = [], [], []
    for df in trials:
        t_type, _ = _parse_comment(df.attrs.get('comment', ''))
        if t_type == 'mvc':
            mvc.append(df)
        elif t_type == 'passive':
            passive.append(df)
        else:
            active.append(df)
    return mvc, passive, active


# ── Passive torque map ──────────────────────────────────────────────────────

_PASSIVE_STD_THRESHOLD = 20.0   # Nm  — segment rejected if torque still noisy after split
_PASSIVE_TQ_BOUND      = 30.0   # Nm  — mean passive torque above this is physiologically impossible
_POSITION_TOLERANCE    = 0.05   # rad — positions within this band are merged into one entry
_STEP_VEL_THRESHOLD    = 5.0    # rad/s — single-sample velocity spike flags a device step
_SETTLE_SAMPLES        = 1000   # samples to skip after a step (1 s at 1000 Hz)


def _split_passive_segments(df, position_col=POSITION_COL):
    """
    Split a passive trial at single-sample position steps made by the device.

    Some passive trials contain two back-to-back static holds: the ankle sits
    still at position A, the device instantly repositions to position B, then
    holds there.  The whole-trial torque std is high (two different passive
    levels averaged together), but each half is clean.

    This function detects position jumps larger than ``_STEP_VEL_THRESHOLD``
    rad/s (in a single sample), splits there, and discards ``_SETTLE_SAMPLES``
    after each jump so the transient dies away before we measure torque.

    Returns a list of DataFrame slices — one per static-hold segment.
    """
    pos   = df[position_col].values
    dt    = df.attrs.get('domainIncr', 0.001)
    speed = np.abs(np.diff(pos)) / dt          # instantaneous speed, length T-1
    jumps = np.where(speed > _STEP_VEL_THRESHOLD)[0]   # sample just before jump

    if len(jumps) == 0:
        return [df]

    segments, start = [], 0
    for j in jumps:
        seg = df.iloc[start: j + 1]
        if len(seg) > _SETTLE_SAMPLES:
            segments.append(seg)
        start = j + 1 + _SETTLE_SAMPLES       # skip post-step transient

    if start < len(df):
        seg = df.iloc[start:]
        if len(seg) > _SETTLE_SAMPLES:
            segments.append(seg)

    return segments


def get_passive_torque_map(passive_trials, position_col=POSITION_COL,
                           torque_col=TORQUE_COL,
                           std_threshold=_PASSIVE_STD_THRESHOLD,
                           pos_tolerance=_POSITION_TOLERANCE):
    """
    Build a lookup: ankle position → mean passive torque (Nm).

    Each passive trial is first split at any single-sample position steps so
    that trials containing two concatenated static holds contribute two data
    points rather than zero.  Segments whose torque std still exceeds
    ``std_threshold`` after splitting are discarded.

    Nearby positions (within ``pos_tolerance`` rad) are merged into one
    cluster by weighted averaging to avoid contradictory entries.

    Returns
    -------
    list of (mean_position_rad, mean_passive_torque_Nm), sorted by position.
    """
    raw = []
    for df in passive_trials:
        for seg in _split_passive_segments(df, position_col):
            tq_std = float(seg[torque_col].std())
            if tq_std > std_threshold:
                continue
            mean_pos = float(seg[position_col].mean())
            mean_tq  = float(seg[torque_col].mean())
            if abs(mean_tq) > _PASSIVE_TQ_BOUND:
                continue
            raw.append((mean_pos, mean_tq))
    raw.sort(key=lambda x: x[0])

    # Merge entries within pos_tolerance of each other into one cluster.
    # Prevents two nearly-identical passive holds from producing contradictory
    # offsets for active trials at the same ankle angle.
    clusters = []
    for pos, tq in raw:
        if clusters and abs(pos - clusters[-1][0]) <= pos_tolerance:
            prev_pos, prev_tq, count = clusters[-1]
            clusters[-1] = (
                (prev_pos * count + pos) / (count + 1),
                (prev_tq  * count + tq)  / (count + 1),
                count + 1,
            )
        else:
            clusters.append((pos, tq, 1))

    return [(pos, tq) for pos, tq, _ in clusters]


def _extrapolate(positions, torques, x):
    """Linear extrapolation beyond the measured range of passive positions."""
    if x < positions[0]:
        p0, p1 = positions[0], positions[1]
        t0, t1 = torques[0],   torques[1]
    else:
        p0, p1 = positions[-2], positions[-1]
        t0, t1 = torques[-2],   torques[-1]
    slope = (t1 - t0) / (p1 - p0)
    return float(t0 + slope * (x - p0))


def lookup_passive_torque(passive_entries, ankle_position_rad):
    """
    Interpolate (or extrapolate) passive torque at ``ankle_position_rad``
    from the known passive-torque map.

    Uses linear interpolation between the two nearest known positions.
    Outside the measured range, extrapolates linearly from the two closest
    boundary points.  This is physiologically appropriate: passive torque
    is a smooth, monotonic function of ankle angle.

    Returns 0.0 only if ``passive_entries`` is empty.
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


# ── Downsampling ────────────────────────────────────────────────────────────

def downsample_trial(df, factor=DOWNSAMPLE_FACTOR):
    """
    Downsample all signal columns by ``factor`` using a FIR anti-alias filter.

    'time' is subsampled directly (every ``factor``-th sample).
    All other columns use ``scipy.signal.decimate`` (zero-phase FIR).

    Returns a new DataFrame with preserved attrs.
    """
    signal_cols = [c for c in df.columns if c != 'time']
    new_data = {'time': df['time'].values[::factor]}
    for col in signal_cols:
        new_data[col] = decimate(df[col].values, factor,
                                 ftype='fir', zero_phase=True)
    # decimate may differ by 1 sample — trim to shortest
    min_len = min(len(v) for v in new_data.values())
    new_data = {k: v[:min_len] for k, v in new_data.items()}

    new_df = pd.DataFrame(new_data)
    new_df.attrs.update(df.attrs)
    new_df.attrs['domainIncr'] = df.attrs.get('domainIncr', 0.001) * factor
    return new_df


# ── Sliding-window builder ──────────────────────────────────────────────────

def build_windows(df, emg_columns, passive_entries=None,
                  window=WINDOW_STEPS, stride=STRIDE_STEPS):
    """
    Build sliding-window samples from a single downsampled trial DataFrame.

    Features (5 columns, fixed order):
        [MG_env_norm, LG_env_norm, SOL_env_norm, TA_env_norm, position (rad)]

    Target:
        active_torque = torque − passive_torque(position[t])  (Nm)

    Passive torque is interpolated per timestep from ``passive_entries`` so
    that trials with ankle movement are corrected correctly.

    The target at each window is the value at the last time-step (t₀).

    Parameters
    ----------
    df : pd.DataFrame
        Downsampled trial with ``*_env_norm`` and position/torque columns.
    emg_columns : list of str
        Original EMG column names (e.g. ['mg_emg', 'lg_emg', 'sol_emg', 'ta_emg']).
    passive_entries : list of (float, float) or None
        Position→passive-torque lookup from ``get_passive_torque_map``.
        Pass None (or empty list) to skip passive subtraction.
    window : int
        Time steps per window (default 20 = 200 ms at 100 Hz).
    stride : int
        Steps between windows (default 1 = 10 ms).

    Returns
    -------
    X : np.ndarray  shape (N_windows, window, 5)
    y : np.ndarray  shape (N_windows,)   — net active torque in Nm
    """
    env_cols  = [f'{c}_env_norm' for c in emg_columns]
    feat_cols = env_cols + [POSITION_COL]
    feat  = df[feat_cols].values.astype(np.float32)       # (T, 5)

    if passive_entries:
        passive_tq = np.array(
            [lookup_passive_torque(passive_entries, p)
             for p in df[POSITION_COL].values],
            dtype=np.float32,
        )
    else:
        passive_tq = np.zeros(len(df), dtype=np.float32)

    y_all = (df[TORQUE_COL].values.astype(np.float32) - passive_tq)

    T         = len(feat)
    n_windows = (T - window) // stride + 1

    X = np.stack([feat[i * stride: i * stride + window]
                  for i in range(n_windows)])
    y = np.array([y_all[i * stride + window - 1]
                  for i in range(n_windows)], dtype=np.float32)
    return X, y


# ── Temporal split ──────────────────────────────────────────────────────────

def _temporal_split(X, y, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    N       = len(X)
    n_train = int(N * train_frac)
    n_val   = int(N * val_frac)
    return (
        (X[:n_train],                  y[:n_train]),
        (X[n_train: n_train + n_val],  y[n_train: n_train + n_val]),
        (X[n_train + n_val:],          y[n_train + n_val:]),
    )


# ── Main entry point ────────────────────────────────────────────────────────

def build_dataset(trials,
                  emg_columns=None,
                  downsample_factor=DOWNSAMPLE_FACTOR,
                  window=WINDOW_STEPS,
                  stride=STRIDE_STEPS,
                  train_frac=TRAIN_FRAC,
                  val_frac=VAL_FRAC,
                  subtract_passive=True):
    """
    Full pipeline: classify → MVC-normalize EMG → passive-subtract torque
                   → downsample → sliding windows → temporal split → stack.

    Parameters
    ----------
    trials : list of pd.DataFrame
        All trials from ``read_flb`` (MVC + passive + active).
    emg_columns : list of str, optional
        EMG column names. Auto-detected from the first MVC trial if None.
    downsample_factor : int
        1000 Hz / target Hz (default 10 → 100 Hz).
    window, stride : int
        Window length and step in samples at target Hz.
    train_frac, val_frac : float
        Temporal split fractions (test = 1 − train − val).

    Returns
    -------
    X_train, y_train : np.ndarray
    X_val,   y_val   : np.ndarray
    X_test,  y_test  : np.ndarray
    mvc_max          : dict  {emg_col: global_max_from_mvc_trials}
    passive_entries  : list of (position_rad, passive_torque_Nm)
    """
    # ── Step 1: Classify ──────────────────────────────────────────────────
    mvc_trials, passive_trials, active_trials = classify_trials(trials)
    print(f"Trials classified:  MVC={len(mvc_trials)}  "
          f"passive={len(passive_trials)}  active={len(active_trials)}")

    # Filter out dynamic/ramp active trials (large torque variance → not isometric)
    active_trials = [df for df in active_trials
                     if df[TORQUE_COL].std() <= _ACTIVE_STD_THRESHOLD]
    print(f"Active trials after std filter (≤ {_ACTIVE_STD_THRESHOLD} Nm): "
          f"{len(active_trials)}")

    if not active_trials:
        raise ValueError("No active trials found.")
    if not mvc_trials:
        raise ValueError("No MVC trials found — cannot normalize EMG.")

    # ── Step 2: EMG envelope extraction + MVC normalization ───────────────
    _, mvc_max = process_trials(mvc_trials, emg_columns=emg_columns)
    if emg_columns is None:
        emg_columns = list(mvc_max.keys())

    active_trials = normalize_with_max(active_trials, mvc_max,
                                       emg_columns=emg_columns)

    # ── Step 3: Passive torque lookup (position-based) ────────────────────
    passive_entries = get_passive_torque_map(passive_trials)
    print(f"\nPassive trials used (std < {_PASSIVE_STD_THRESHOLD} Nm):")
    if passive_entries:
        for pos, tq in passive_entries:
            print(f"  pos ≈ {pos:+.3f} rad → passive torque = {tq:+.3f} Nm")
    else:
        print("  (none found — proceeding without passive subtraction)")

    # ── Steps 4–6: Per-trial downsample → window → split ─────────────────
    fs_target = 1000.0 / downsample_factor
    trains, vals, tests = [], [], []

    for df in active_trials:
        df_ds = downsample_trial(df, factor=downsample_factor)
        X, y  = build_windows(df_ds, emg_columns,
                              passive_entries if subtract_passive else None,
                              window=window, stride=stride)
        (Xtr, ytr), (Xv, yv), (Xte, yte) = _temporal_split(X, y,
                                                              train_frac,
                                                              val_frac)
        trains.append((Xtr, ytr))
        vals.append((Xv,   yv))
        tests.append((Xte, yte))

    X_train = np.concatenate([x for x, _ in trains])
    y_train = np.concatenate([y for _, y in trains])
    X_val   = np.concatenate([x for x, _ in vals])
    y_val   = np.concatenate([y for _, y in vals])
    X_test  = np.concatenate([x for x, _ in tests])
    y_test  = np.concatenate([y for _, y in tests])

    print(f"\nDataset built at {fs_target:.0f} Hz  |  "
          f"window={window} steps ({window / fs_target * 1000:.0f} ms)  |  "
          f"stride={stride} step ({stride / fs_target * 1000:.0f} ms)")
    print(f"  Features : {[f'{c}_env_norm' for c in emg_columns] + [POSITION_COL]}")
    print(f"  Target   : net active torque (Nm)")
    print(f"  X_train  : {X_train.shape}   y_train: {y_train.shape}")
    print(f"  X_val    : {X_val.shape}   y_val:   {y_val.shape}")
    print(f"  X_test   : {X_test.shape}   y_test:  {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, mvc_max, passive_entries
