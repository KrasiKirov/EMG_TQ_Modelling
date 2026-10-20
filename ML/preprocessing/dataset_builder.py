"""
dataset_builder.py
------------------
Builds the LSTM input dataset from FLB trial data.

Pipeline
--------
1.  Classify all trials by comment → MVC / passive / active
2.  Extract EMG envelopes at 1000 Hz and normalize using global max per channel
3.  Subtract mean passive torque per ankle position → net active torque
4.  Downsample 1000 Hz → 100 Hz
5.  Build sliding windows  [N × 50 × 5] by default
        Features : MG_env_norm, LG_env_norm, SOL_env_norm, TA_env_norm, position (rad)
        Target   : active_torque (Nm) = measured_torque − passive_torque
6.  Clean temporal split by default: raw session-1 segments are separated before
    preprocessing; retest trials remain fully held out. Pass ``validation_gap_s=None``
    only when reproducing the historical window-first benchmark.

Design choices
--------------
- Ankle position kept in raw radians (no scaling).
- Torque target is net active torque after passive subtraction.
- Clean validation uses training-only EMG maxima. The legacy ``None`` mode uses
  train/validation maxima for exact historical reproduction.
- Passive torque is the mean measured torque over the matching passive trial(s)
  for each ankle position tag (p1 … p8). By default, session-1 passive
  calibration is applied to training/validation and session-2 calibration to
  retest; ``passive_session='retest'`` preserves the historical shared map.
"""

import re
import warnings
import numpy as np
import pandas as pd
from scipy.signal import decimate

try:
    from ML.config import PASSIVE_MAP_SESSION, N_STEPS
except ImportError:
    from config import PASSIVE_MAP_SESSION, N_STEPS

try:
    from .emg_envelope import process_trials, extract_envelope
except ImportError:
    from emg_envelope import process_trials, extract_envelope

# Feature layout — position is the last column in the 5-feature window
_POS_FEAT_IDX = 4


# ── Constants ──────────────────────────────────────────────────────────────

POSITION_COL      = 'position'   # ankle angle column name (rad)
TORQUE_COL        = 'torque'     # measured torque column name (Nm)

DOWNSAMPLE_FACTOR = 10           # 1000 Hz → 100 Hz
WINDOW_STEPS      = N_STEPS      # default: 50 steps × 10 ms = 500 ms history
STRIDE_STEPS      = 1            # 1 step = 10 ms between predictions
TRANSIENT_SAMPLES = 1000         # samples to trim from start of each trial at 1000 Hz
                                 # (1 s — discards EMG envelope filter startup artifact)

TRAIN_FRAC        = 0.80
VAL_FRAC          = 0.20

# Active trials whose torque std exceeds this are likely ramp / dynamic
# movements (not isometric holds) and are excluded from training.
_ACTIVE_STD_THRESHOLD  = 30.0   # Nm


# ── Trial classification ────────────────────────────────────────────────────

def _parse_comment(comment: str):
    """
    Parse a trial comment string.

    Returns
    -------
    trial_type : 'mvc' | 'passive' | 'active'
    session    : 'test' | 'retest' | None
    position   : 'p1' .. 'p8'  (or None)
    """
    comment = comment.lower().strip()

    # Position: match "p3" style OR "pos indx 3" / "indx 3" style
    m = re.search(r'p(\d+)', comment)
    if m:
        pos = m.group(0)
    else:
        m2 = re.search(r'(?:pos\s*)?indx\s*(\d+)', comment)
        pos = f"p{m2.group(1)}" if m2 else None

    # Determine session (retest must be checked before test)
    if 'retest' in comment:
        session = 'retest'
    elif 'test' in comment or 'repeat' in comment:
        session = 'test'
    else:
        session = None

    if comment.startswith('mvc'):
        return 'mvc', session, pos
    elif 'passive' in comment:
        return 'passive', session, pos
    else:
        return 'active', session, pos


def classify_trials(trials):
    """
    Split trial DataFrames into MVC, passive, and active groups.

    Also tags each DataFrame with ``attrs['session']`` ('test', 'retest',
    or None) parsed from the trial comment.

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
        t_type, session, _ = _parse_comment(df.attrs.get('comment', ''))
        df.attrs['session'] = session
        if t_type == 'mvc':
            mvc.append(df)
        elif t_type == 'passive':
            passive.append(df)
        else:
            active.append(df)
    return mvc, passive, active


# ── Passive torque map ──────────────────────────────────────────────────────

_PASSIVE_STD_THRESHOLD = 20.0   # Nm  — segment rejected if torque still noisy after split
_PASSIVE_TQ_BOUND      = 10.0   # Nm — legacy automated cutoff; documented overrides can exceed it
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
                           pos_tolerance=_POSITION_TOLERANCE,
                           session=None):
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
    if session is None:
        selected_trials = passive_trials
    else:
        selected_trials = [df for df in passive_trials
                           if df.attrs.get('session') == session]
        if not selected_trials:
            warnings.warn(
                f'No passive trials labelled session={session!r}; '
                'the passive map will be empty.',
                stacklevel=2)

    raw = []
    for df in selected_trials:
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

    entries = [(pos, tq) for pos, tq, _ in clusters]
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
    Interpolate (or extrapolate) passive torque at ``ankle_position_rad``
    from the known passive-torque map.

    Uses linear interpolation between the two nearest known positions.
    Outside the measured range, extrapolates linearly from the two closest
    boundary points. This is the historical numerical policy; extrapolated
    values still require calibration review and are not independently measured.

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


# ── MVC normalization map ──────────────────────────────────────────────────

def get_mvc_envelope_map(mvc_trials, emg_columns=None, passive_entries=None,
                         pos_tolerance=_POSITION_TOLERANCE):
    """
    Build per-position MVC lookup: ankle position → peak EMG envelope per
    channel + peak net active torque.

    For each MVC trial:
      1. Extract EMG envelope per channel
      2. Record peak envelope value per channel
      3. Record peak |net active torque| = max(|raw_torque - passive_torque|)
      4. Record mean ankle position

    Nearby positions (within ``pos_tolerance``) are merged by taking the
    element-wise max (not mean) — MVC normalization should use the largest
    observed activation at each position.

    Parameters
    ----------
    mvc_trials : list of pd.DataFrame
        MVC trial DataFrames from ``classify_trials()``.
    emg_columns : list of str, optional
        EMG column names. Auto-detected from first trial if None.
    passive_entries : list of (float, float), optional
        Position→passive-torque lookup for net active torque computation.
    pos_tolerance : float
        Merge positions within this many radians (default 0.05).

    Returns
    -------
    list of (position_rad, emg_peaks_dict, peak_active_torque_Nm),
    sorted by position.  emg_peaks_dict maps channel name → peak envelope.
    """
    if emg_columns is None:
        try:
            from .emg_envelope import detect_emg_columns
        except ImportError:
            from emg_envelope import detect_emg_columns
        emg_columns = detect_emg_columns(mvc_trials[0])

    raw = []
    for df in mvc_trials:
        mean_pos = float(df[POSITION_COL].mean())

        # Extract envelope per channel and record peak
        emg_peaks = {}
        for col in emg_columns:
            env = extract_envelope(df[col].values)
            emg_peaks[col] = float(env.max())

        # Peak net active torque at this position
        raw_tq = df[TORQUE_COL].values.astype(np.float64)
        if passive_entries:
            passive_tq = lookup_passive_torque(passive_entries, mean_pos)
        else:
            passive_tq = 0.0
        net_active = np.abs(raw_tq - passive_tq)
        peak_torque = float(net_active.max())

        raw.append((mean_pos, emg_peaks, peak_torque))

    raw.sort(key=lambda x: x[0])

    # Merge entries within pos_tolerance — take element-wise MAX
    clusters = []
    for pos, emg_pk, tq_pk in raw:
        if clusters and abs(pos - clusters[-1][0]) <= pos_tolerance:
            prev_pos, prev_emg, prev_tq, count = clusters[-1]
            merged_emg = {c: max(prev_emg.get(c, 0), emg_pk.get(c, 0))
                          for c in set(prev_emg) | set(emg_pk)}
            merged_tq = max(prev_tq, tq_pk)
            merged_pos = (prev_pos * count + pos) / (count + 1)
            clusters[-1] = (merged_pos, merged_emg, merged_tq, count + 1)
        else:
            clusters.append((pos, emg_pk, tq_pk, 1))

    entries = [(pos, emg, tq) for pos, emg, tq, _ in clusters]
    if len(entries) < 2:
        warnings.warn(
            f"Only {len(entries)} MVC entry/entries found. "
            f"Interpolation will be limited.", stacklevel=2)
    return entries


def lookup_mvc_max(mvc_entries, ankle_position_rad, channel=None):
    """
    Interpolate MVC max at ``ankle_position_rad`` from the MVC map.

    Parameters
    ----------
    mvc_entries : list of (pos, emg_peaks_dict, peak_torque)
        Output of ``get_mvc_envelope_map()``.
    ankle_position_rad : float
        Ankle position to look up.
    channel : str or None
        If given, return interpolated peak EMG for that channel.
        If None, return interpolated peak torque.

    Returns
    -------
    float — interpolated MVC max value.
    """
    if not mvc_entries:
        return 1.0  # fallback: no normalization

    positions = np.array([p for p, _, _ in mvc_entries])

    if channel is not None:
        values = np.array([emg.get(channel, 1.0) for _, emg, _ in mvc_entries])
    else:
        values = np.array([tq for _, _, tq in mvc_entries])

    # Interpolate within range, extrapolate outside
    if positions[0] <= ankle_position_rad <= positions[-1]:
        return float(np.interp(ankle_position_rad, positions, values))
    else:
        return float(_extrapolate(positions, values, ankle_position_rad))


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

def build_windows(df, emg_columns, passive_entries=None, mvc_entries=None,
                  window=WINDOW_STEPS, stride=STRIDE_STEPS):
    """
    Build sliding-window samples from a single downsampled trial DataFrame.

    Features (5 columns, fixed order):
        [MG_env_norm, LG_env_norm, SOL_env_norm, TA_env_norm, position (rad)]

    Target:
        active_torque = torque − passive_torque(position[t])  (Nm)
        When ``mvc_entries`` is provided, target is divided by position-specific
        MVC torque (fraction of max).  The per-window MVC torque is returned
        as ``mvc_tq`` so callers can denormalize predictions back to Nm.

    Parameters
    ----------
    df : pd.DataFrame
        Downsampled trial with ``*_env_norm`` (or ``*_env`` if mvc_entries)
        and position/torque columns.
    emg_columns : list of str
        Original EMG column names (e.g. ['mg_emg', 'lg_emg', 'sol_emg', 'ta_emg']).
    passive_entries : list of (float, float) or None
        Position→passive-torque lookup from ``get_passive_torque_map``.
    mvc_entries : list of (pos, emg_peaks_dict, peak_torque) or None
        Per-position MVC map from ``get_mvc_envelope_map()``.
        When provided, EMG is normalized by per-position MVC peak instead of
        global max. Torque is divided by the corresponding MVC torque.
    window : int
        Time steps per window (default 50 = 500 ms at 100 Hz).
    stride : int
        Steps between windows (default 1 = 10 ms).

    Returns
    -------
    X : np.ndarray  shape (N_windows, window, 5)
    y : np.ndarray  shape (N_windows,)
    mvc_tq : np.ndarray shape (N_windows,) or None
        Per-window MVC torque used for normalization.  Multiply y * mvc_tq
        to recover Nm.  None when mvc_entries is not provided.
    """
    pos_vals = df[POSITION_COL].values

    if mvc_entries:
        # MVC normalization: normalize EMG per-sample by position-specific MVC peak
        env_data = []
        for col in emg_columns:
            env_raw = df[f'{col}_env'].values.astype(np.float32)
            mvc_maxes = np.array([lookup_mvc_max(mvc_entries, p, channel=col)
                                  for p in pos_vals], dtype=np.float32)
            mvc_maxes = np.maximum(mvc_maxes, 1e-10)  # avoid division by zero
            env_data.append(env_raw / mvc_maxes)
        feat = np.column_stack(env_data + [pos_vals.astype(np.float32)])
    else:
        # Standard normalization: use pre-computed *_env_norm columns
        env_cols  = [f'{c}_env_norm' for c in emg_columns]
        feat_cols = env_cols + [POSITION_COL]
        feat = df[feat_cols].values.astype(np.float32)

    # Passive subtraction
    if passive_entries:
        passive_tq = np.array(
            [lookup_passive_torque(passive_entries, p) for p in pos_vals],
            dtype=np.float32,
        )
    else:
        passive_tq = np.zeros(len(df), dtype=np.float32)

    y_all = df[TORQUE_COL].values.astype(np.float32) - passive_tq

    # MVC torque normalization: convert Nm → fraction of MVC.
    # Store the per-sample MVC torque so callers can denormalize back to Nm.
    if mvc_entries:
        mvc_tq_all = np.array([lookup_mvc_max(mvc_entries, p, channel=None)
                               for p in pos_vals], dtype=np.float32)
        mvc_tq_all = np.maximum(mvc_tq_all, 1e-10)
        y_all = y_all / mvc_tq_all
    else:
        mvc_tq_all = None

    T         = len(feat)
    n_windows = (T - window) // stride + 1

    X = np.stack([feat[i * stride: i * stride + window]
                  for i in range(n_windows)])
    y = np.array([y_all[i * stride + window - 1]
                  for i in range(n_windows)], dtype=np.float32)

    if mvc_tq_all is not None:
        mvc_tq = np.array([mvc_tq_all[i * stride + window - 1]
                           for i in range(n_windows)], dtype=np.float32)
    else:
        mvc_tq = None

    return X, y, mvc_tq


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
                  test_trial_indices=None,
                  retest_trial_indices=None,
                  downsample_factor=DOWNSAMPLE_FACTOR,
                  window=WINDOW_STEPS,
                  stride=STRIDE_STEPS,
                  train_frac=TRAIN_FRAC,
                  subtract_passive=True,
                  normalize_position=False,
                  normalize_mvc=False,
                  passive_entries_override=None,
                  passive_entries_fallback=None,
                  passive_fallback_source=None,
                  passive_session=PASSIVE_MAP_SESSION,
                  validation_gap_s=2.0,
                  audit=None,
                  passive_entries_override_by_session=None):
    """
    Full pipeline: classify → normalize EMG → passive-subtract torque
                   → downsample → sliding windows → temporal split → stack.

    Parameters
    ----------
    trials : list of pd.DataFrame
        All trials from ``read_flb`` (MVC + passive + active).
    emg_columns : list of str, optional
        EMG column names. Auto-detected from the first trial if None.
    test_trial_indices : list of int, optional
        1-indexed trial numbers for train/val split (80/20).
        If None, auto-detected from comments containing 'test' (not 'retest').
    retest_trial_indices : list of int, optional
        1-indexed trial numbers for held-out test set.
        If None, auto-detected from comments containing 'retest'.
    downsample_factor : int
        1000 Hz / target Hz (default 10 → 100 Hz).
    window, stride : int
        Window length and step in samples at target Hz.
    train_frac : float
        Fraction of test trials used for training (rest → validation).
    normalize_position : bool
        If True, min-max normalize the ankle position feature to [0, 1].
    normalize_mvc : bool
        If True, normalize EMG by per-position MVC peak and torque by
        per-position MVC torque.  Converts inputs to %MVC activation and
        targets to fraction of MVC torque — enables cross-subject transfer.
    passive_session : {'session_specific', 'test', 'retest', None}
        Session used to build the passive torque map. The default
        ``'session_specific'`` uses the session-1 map for training/validation
        and the session-2 map for retest. ``'retest'`` reproduces the
        historical shared second-session map; ``None`` uses all passive trials.
    passive_entries_override_by_session : dict or None
        Optional explicit maps keyed by ``'test'`` and ``'retest'``. This is
        used when an independently reviewed calibration source is available
        for each session (for example the reconstructed YES worksheet).
    passive_entries_fallback : list of (position, torque) or None
        Explicit fallback map used only when no map can be built for
        ``passive_session``. This is intended for documented exceptions such
        as IES, which has no separately labelled retest passive trials.
    passive_fallback_source : str or None
        Provenance label printed when the fallback map is used.
    validation_gap_s : float or None
        Raw-segment validation gap in seconds. The default 2.0 splits raw
        session-1 trials before filtering, normalizes from training only, and
        trims 1 s at both ends of each session-1 segment. Pass ``None`` only
        for the historical window-first benchmark protocol. Retest processing
        is unchanged apart from using the training-only scale in clean mode.
    audit : dict or None
        If supplied, populate aligned endpoint traces and trial provenance.

    Returns
    -------
    X_train, y_train : np.ndarray
    X_val,   y_val   : np.ndarray
    X_test,  y_test  : np.ndarray
    emg_max          : dict  {emg_col: max_value_used_for_normalization}
    passive_entries  : list of (position_rad, passive_torque_Nm)
    passive_map_source : str — provenance of the map used for subtraction
    operating_positions : list of float — mean ankle position per retest trial, sorted
    pos_range        : tuple (pos_min, pos_max) or None
    mvc_tq_test      : np.ndarray or None — per-window MVC torque for denormalization
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

    # ── Step 2: Determine trial split BEFORE normalization ──────────────
    # This prevents data leakage: the global max used for EMG normalization
    # must come only from train/val trials, not from the held-out retest set.
    if test_trial_indices is not None and retest_trial_indices is not None:
        test_set   = {i - 1 for i in test_trial_indices}
        retest_set = {i - 1 for i in retest_trial_indices}
    else:
        test_set, retest_set, unlabelled = set(), set(), set()
        for df in active_trials:
            idx = df.attrs.get('trial_index')
            session = df.attrs.get('session')
            if session == 'retest':
                retest_set.add(idx)
            elif session == 'test':
                test_set.add(idx)
            else:
                unlabelled.add(idx)
        # Fallback: if retest trials exist but test-session trials have no
        # explicit "test" keyword (e.g. IES format "PUSH, pos indx 1"),
        # treat all unlabelled active trials as the test session.
        if not test_set and retest_set and unlabelled:
            test_set = unlabelled
            print("  (no explicit 'test' keyword — unlabelled active trials "
                  "assigned to test session)")
        if not test_set:
            raise ValueError("No test-session active trials found in comments.")
        print(f"\nAuto-detected trial split from comments:")
        print(f"  Train/Val (test session):  {sorted(i+1 for i in test_set)}")
        print(f"  Held-out  (retest session): {sorted(i+1 for i in retest_set)}")

    # Separate train/val trials from retest trials
    trainval_trials = [df for df in active_trials
                       if df.attrs.get('trial_index') in test_set]
    retest_trials   = [df for df in active_trials
                       if df.attrs.get('trial_index') in retest_set]

    validation_trials = []
    if validation_gap_s is not None:
        if not np.isfinite(validation_gap_s) or validation_gap_s < 0:
            raise ValueError('validation_gap_s must be finite and nonnegative')
        separated = []
        for df in trainval_trials:
            cut = int(len(df) * train_frac)
            half_gap = int(np.ceil(validation_gap_s / df.attrs['domainIncr'] / 2))
            for split, start, stop, dest in [
                ('train', 0, cut - half_gap, separated),
                ('validation', cut + half_gap, len(df), validation_trials),
            ]:
                if stop - start < 2 * TRANSIENT_SAMPLES + window * downsample_factor:
                    raise ValueError('Trial too short for separated validation and edge guards')
                segment = df.iloc[start:stop].copy(deep=True)
                segment.attrs['_validation_split'] = split
                segment.attrs['_raw_interval'] = [start, stop]
                dest.append(segment)
        trainval_trials = separated

    # ── Step 3: EMG envelope extraction + normalization ──────────────────
    # Extract envelopes for all active trials.  In clean mode (the default),
    # train and validation segments use the training-only global max. In
    # legacy mode, train/validation maxima are pooled for reproduction. When
    # normalize_mvc is True, per-position MVC normalization happens later.
    trainval_trials, emg_max = process_trials(trainval_trials,
                                               emg_columns=emg_columns)
    if emg_columns is None:
        emg_columns = list(emg_max.keys())

    if validation_trials:
        validation_trials, _ = process_trials(
            validation_trials, emg_columns=emg_columns, max_values=emg_max)

    # Extract envelopes for retest trials (always needed)
    for df in retest_trials:
        for col in emg_columns:
            env, stages = extract_envelope(df[col].values,
                                         return_intermediates=True)
            df[f'{col}_rect'] = stages['rectified']
            df[f'{col}_env'] = env
            df[f'{col}_env_norm'] = env / emg_max[col]

    if normalize_mvc:
        print(f"EMG envelopes extracted (MVC normalization will be applied per-sample)")
    else:
        print(f"Retest trials normalized using train/val max (no data leakage)")

    # Recombine for the windowing loop below
    active_trials = trainval_trials + validation_trials + retest_trials

    # ── Step 4: Passive torque lookup (position-based) ────────────────────
    def resolve_passive_map(session):
        """Resolve one session's map while retaining explicit legacy options."""
        if passive_entries_override_by_session is not None:
            entries = passive_entries_override_by_session.get(session)
            if entries is not None:
                return sorted(entries, key=lambda x: x[0]), f'manual override ({session})'
        if passive_entries_override is not None:
            return sorted(passive_entries_override, key=lambda x: x[0]), 'manual override'

        session_arg = session if passive_session == 'session_specific' else passive_session
        entries = get_passive_torque_map(passive_trials, session=session_arg)
        source = f'{session_arg or "all sessions"} passive trials'
        if not entries and passive_entries_fallback:
            entries = sorted(passive_entries_fallback, key=lambda x: x[0])
            source = (passive_fallback_source or 'documented fallback passive map')
        if not entries:
            source = 'none (zero passive subtraction)'
        return entries, source

    if passive_session == 'session_specific':
        passive_entries_by_session = {
            session: resolve_passive_map(session) for session in ('test', 'retest')
        }
        passive_entries = passive_entries_by_session['retest'][0]
        passive_map_source = '; '.join(
            f'{session}={source}' for session, (_, source)
            in passive_entries_by_session.items())
        print("\nPassive torque maps (session-matched):")
        for session, (entries, source) in passive_entries_by_session.items():
            print(f"  {session}: {source} ({len(entries)} entries)")
            for pos, tq in entries:
                print(f"    pos ≈ {pos:+.3f} rad → passive torque = {tq:+.3f} Nm")
    else:
        shared_entries, source = resolve_passive_map('retest')
        passive_entries_by_session = {'test': (shared_entries, source),
                                      'retest': (shared_entries, source)}
        passive_entries = shared_entries
        passive_map_source = source
        print(f"\nPassive torque map ({source}; {len(passive_entries)} entries):")
        for pos, tq in passive_entries:
            print(f"  pos ≈ {pos:+.3f} rad → passive torque = {tq:+.3f} Nm")

    if audit is not None:
        audit['passive_maps'] = {
            session: {'entries': entries, 'source': source}
            for session, (entries, source) in passive_entries_by_session.items()
        }

    # ── Step 4b: MVC normalization map (optional) ──────────────────────────
    mvc_map = None
    if normalize_mvc and mvc_trials:
        mvc_map = get_mvc_envelope_map(mvc_trials, emg_columns,
                                       passive_entries_by_session['test'][0])
        print(f"\nMVC normalization map ({len(mvc_map)} positions):")
        for pos, emg_pk, tq_pk in mvc_map:
            emg_str = '  '.join(f'{c}={emg_pk[c]:.4f}' for c in emg_columns)
            print(f"  pos ≈ {pos:+.3f} rad → MVC torque = {tq_pk:.2f} Nm  |  {emg_str}")
        print(f"  → EMG normalized to %MVC activation, torque to fraction of MVC")
    elif normalize_mvc and not mvc_trials:
        warnings.warn("normalize_mvc=True but no MVC trials found. "
                       "Falling back to global-max normalization.", stacklevel=2)

    # ── Steps 5–7: Per-trial downsample → window → split ─────────────────
    fs_target = 1000.0 / downsample_factor
    trains, vals, tests = [], [], []
    mvc_tq_tests = []
    operating_positions = []  # mean position per retest trial
    if audit is not None:
        audit.update(trials=[], traces={'train': [], 'validation': [], 'retest': []})

    for df in active_trials:
        idx = df.attrs.get('trial_index')

        # Skip trials not in either set
        if idx not in test_set and idx not in retest_set:
            continue

        # Trim filter transient from start of trial (1 s at 1000 Hz)
        end = (-TRANSIENT_SAMPLES if validation_gap_s is not None and idx in test_set else None)
        df = df.iloc[TRANSIENT_SAMPLES:end].reset_index(drop=True)

        df_ds = downsample_trial(df, factor=downsample_factor)
        map_session = 'retest' if idx in retest_set else 'test'
        trial_passive_entries = passive_entries_by_session[map_session][0]
        X, y, mtq = build_windows(df_ds, emg_columns,
                                   trial_passive_entries if subtract_passive else None,
                                   mvc_entries=mvc_map,
                                   window=window, stride=stride)

        comment = df.attrs.get('comment', '')
        if idx in test_set and validation_gap_s is not None:
            split = df.attrs['_validation_split']
            (trains if split == 'train' else vals).append((X, y))
            selections = [(split, slice(None))]
        elif idx in test_set:
            # Test-session trial → 80 % train / 20 % val
            (Xtr, ytr), (Xv, yv), _ = _temporal_split(
                X, y, train_frac, 1.0 - train_frac)
            trains.append((Xtr, ytr))
            vals.append((Xv, yv))
            selections = [('train', slice(0, len(Xtr))),
                          ('validation', slice(len(Xtr), len(Xtr) + len(Xv)))]
            print(f"  Trial {idx+1:2d} ({comment:30s}) → TRAIN {len(Xtr):5d} | VAL {len(Xv):5d}")
        else:
            # Retest trial → 100 % held-out test
            tests.append((X, y))
            if mtq is not None:
                mvc_tq_tests.append(mtq)
            operating_positions.append(float(df_ds[POSITION_COL].mean()))
            print(f"  Trial {idx+1:2d} ({comment:30s}) → TEST  {len(X):5d}  (retest — fully held out)")
            selections = [('retest', slice(None))]

        if audit is not None:
            endpoint = df_ds.iloc[window - 1::stride]
            for split, selection in selections:
                ep = endpoint.iloc[selection]
                trace = dict(time=ep.time.to_numpy(),
                             measured=ep.torque.to_numpy(),
                             position=ep.position.to_numpy(),
                             target=y[selection],
                             trial=np.full(len(ep), idx + 1),
                             emg=ep[[f'{c}_env_norm' for c in emg_columns]].to_numpy())
                trace['passive'] = np.array([lookup_passive_torque(trial_passive_entries, p)
                                            if subtract_passive else 0.0 for p in trace['position']])
                audit['traces'][split].append(trace)
                audit['trials'].append(dict(trial=idx + 1, split=split, comment=comment,
                    original_comment=df.attrs.get('original_comment', comment),
                    raw_interval=df.attrs.get('_raw_interval'), n=len(ep),
                    first_time=float(ep.time.iloc[0]), last_time=float(ep.time.iloc[-1]),
                    mean_position=float(df_ds.position.mean())))

    X_train = np.concatenate([x for x, _ in trains])
    y_train = np.concatenate([y for _, y in trains])
    X_val   = np.concatenate([x for x, _ in vals])
    y_val   = np.concatenate([y for _, y in vals])
    X_test  = np.concatenate([x for x, _ in tests])
    y_test  = np.concatenate([y for _, y in tests])

    # MVC torque arrays for denormalization (None if not using MVC)
    mvc_tq_test = (np.concatenate(mvc_tq_tests) if mvc_tq_tests
                   else None)

    # ── Optional: min-max normalize position feature to [0, 1] ────────
    # Computed from train/val data only (no leakage).  Applied AFTER
    # windowing so passive torque subtraction uses raw position values.
    pos_range = None
    if normalize_position:
        pos_min = float(X_train[:, :, _POS_FEAT_IDX].min())
        pos_max = float(X_train[:, :, _POS_FEAT_IDX].max())
        denom = pos_max - pos_min if pos_max > pos_min else 1.0
        for X in (X_train, X_val, X_test):
            X[:, :, _POS_FEAT_IDX] = (X[:, :, _POS_FEAT_IDX] - pos_min) / denom
        pos_range = (pos_min, pos_max)
        print(f"\nPosition normalized to [0, 1]:  "
              f"raw range [{pos_min:+.4f}, {pos_max:+.4f}] rad")

    print(f"\nTrial-based split:")
    print(f"  Train/Val trials (1-indexed): {sorted(i+1 for i in test_set)}")
    if retest_set:
        print(f"  Test trials (retest, 1-indexed): {sorted(i+1 for i in retest_set)}")
    print(f"\nDataset built at {fs_target:.0f} Hz  |  "
          f"window={window} steps ({window / fs_target * 1000:.0f} ms)  |  "
          f"stride={stride} step ({stride / fs_target * 1000:.0f} ms)")
    if normalize_mvc and mvc_map:
        print(f"  Features : {[f'{c}_env/%MVC' for c in emg_columns] + [POSITION_COL]}")
    else:
        print(f"  Features : {[f'{c}_env_norm' for c in emg_columns] + [POSITION_COL]}")
    if normalize_mvc and mvc_map:
        print(f"  Target   : net active torque (Nm) — MVC-normalized during training, denormalized at inference")
    else:
        print(f"  Target   : net active torque (Nm)")
    print(f"  X_train  : {X_train.shape}   y_train: {y_train.shape}")
    print(f"  X_val    : {X_val.shape}   y_val:   {y_val.shape}")
    print(f"  X_test   : {X_test.shape}   y_test:  {y_test.shape}")

    # Normalize operating positions to match the feature space
    if normalize_position and pos_range is not None:
        pos_min, pos_max = pos_range
        denom = pos_max - pos_min if pos_max > pos_min else 1.0
        operating_positions = [(p - pos_min) / denom for p in operating_positions]
    operating_positions = sorted(set(operating_positions))

    if audit is not None:
        for split, chunks in audit['traces'].items():
            audit['traces'][split] = {key: np.concatenate([c[key] for c in chunks])
                                      for key in chunks[0]}
        audit['validation_gap_s'] = validation_gap_s

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            emg_max, passive_entries, operating_positions, pos_range,
            mvc_tq_test, passive_map_source)
