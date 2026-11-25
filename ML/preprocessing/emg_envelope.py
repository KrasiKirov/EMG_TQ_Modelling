"""EMG linear envelope: band-pass, full-wave rectify, low-pass; optional division by the per-channel global maximum."""

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt


# Default parameters

# Canonical muscle order; detect_emg_columns enforces it so the features are [MG, LG, SOL, TA, position].
CANONICAL_EMG_ORDER = ['gm', 'gl', 'sol', 'ta']
DEFAULT_FS = 1000.0        # Hz — sampling frequency used in REKLAB experiments
DEFAULT_BP_LOW = 30.0      # Hz — bandpass lower cutoff
DEFAULT_BP_HIGH = 300.0    # Hz — bandpass upper cutoff
DEFAULT_LP_CUTOFF = 2.0    # Hz — lowpass cutoff for envelope (Can also use 1.5 Hz)
DEFAULT_ORDER = 4          # Butterworth filter order

_EMG_KEYWORDS = {'emg', 'gm', 'gl', 'sol', 'ta', 'mg', 'lg'}
_NON_EMG_NAMES = {'time', 'pos', 'tq', 'position', 'torque'}


def detect_emg_columns(df):
    """
    EMG column names of a trial DataFrame (by keyword, excluding time / position / torque), sorted to
    CANONICAL_EMG_ORDER so the features are always [MG, LG, SOL, TA].
    """
    emg_cols = set()
    for col in df.columns:
        col_lower = col.lower()
        if col_lower.endswith(('_env', '_env_norm', '_rect')):
            continue
        if col_lower in _NON_EMG_NAMES:
            continue
        if any(kw in col_lower for kw in _EMG_KEYWORDS):
            emg_cols.add(col)

    # Sort to canonical order: [gm, gl, sol, ta]
    ordered = [c for c in CANONICAL_EMG_ORDER if c in emg_cols]
    # Append any unexpected columns that weren't in the canonical list
    ordered += sorted(emg_cols - set(ordered))
    return ordered


# Single-signal envelope extraction

def extract_envelope_causal(signal, fs=DEFAULT_FS,
                            bp_low=DEFAULT_BP_LOW, bp_high=DEFAULT_BP_HIGH,
                            lp_cutoff=DEFAULT_LP_CUTOFF, order=DEFAULT_ORDER,
                            settle_s=1.0, lp_order=None):
    """
    Forward-only version of ``extract_envelope``: each output sample uses only current and earlier input.

    Lags the zero-phase envelope by about 0.23 s (4th-order 2 Hz low-pass) or 0.12 s (``lp_order=2``). The offset is
    the mean of the first ``settle_s`` seconds and the low-pass starts from the mean rectified value of the first
    200 ms; the pipeline drops that start (filter start-up), so no kept sample depends on later data.
    """
    x = np.asarray(signal, dtype=np.float64)
    head = max(1, int(settle_s * fs))
    x = x - np.mean(x[:head])
    nyq = 0.5 * fs
    sos_bp = butter(order, [bp_low / nyq, bp_high / nyq], btype='band', output='sos')
    rectified = np.abs(sosfilt(sos_bp, x))
    sos_lp = butter(lp_order or order, lp_cutoff / nyq, btype='low', output='sos')
    zi = sosfilt_zi(sos_lp) * np.mean(rectified[:max(1, int(0.2 * fs))])
    return sosfilt(sos_lp, rectified, zi=zi)[0]


def extract_envelope(signal, fs=DEFAULT_FS,
                     bp_low=DEFAULT_BP_LOW, bp_high=DEFAULT_BP_HIGH,
                     lp_cutoff=DEFAULT_LP_CUTOFF, order=DEFAULT_ORDER,
                     demean=True,
                     return_intermediates=False, lp_order=None):
    """
    Linear envelope of one raw EMG signal: demean (as MATLAB ``demean.m``), zero-phase Butterworth band-pass,
    full-wave rectify, zero-phase Butterworth low-pass (order ``lp_order``, default ``order``).

    Returns the un-normalised envelope (same length), plus a dict of the stages ``demeaned``, ``filtered``,
    ``rectified`` if ``return_intermediates``.
    """
    signal = np.asarray(signal, dtype=np.float64)

    # Step 0: Remove DC offset (matches MATLAB demean.m: y = x - mean(x))
    if demean:
        demeaned = signal - np.mean(signal)
    else:
        demeaned = signal

    nyq = 0.5 * fs

    # Step 1: Bandpass filter
    sos_bp = butter(order, [bp_low / nyq, bp_high / nyq], btype='band', output='sos')
    filtered = sosfiltfilt(sos_bp, demeaned)

    # Step 2: Full-wave rectification
    rectified = np.abs(filtered)

    # Step 3: Lowpass filter → linear envelope
    sos_lp = butter(lp_order or order, lp_cutoff / nyq, btype='low', output='sos')
    envelope = sosfiltfilt(sos_lp, rectified)

    # Clip: sosfiltfilt can produce tiny negative values at edges after rectification
    envelope = np.clip(envelope, 0, None)

    if return_intermediates:
        return envelope, {
            'demeaned': demeaned,
            'filtered': filtered,
            'rectified': rectified,
        }
    return envelope


# Batch processing + normalization


def process_trials(trials, emg_columns=None, max_values=None, fs=DEFAULT_FS,
                   bp_low=DEFAULT_BP_LOW, bp_high=DEFAULT_BP_HIGH,
                   lp_cutoff=DEFAULT_LP_CUTOFF, order=DEFAULT_ORDER,
                   demean=True):
    """
    Add ``{col}_env`` and ``{col}_env_norm`` (÷ per-channel max) columns to every trial.

    The maximum is taken over ``trials`` unless ``max_values`` is given. Returns (trials, {column: max used}).
    """
    trials = [df.copy(deep=True) for df in trials]
    if not trials:
        raise ValueError('At least one trial is required.')
    if emg_columns is None:
        emg_columns = detect_emg_columns(trials[0])

    # Step 1: Extract envelopes
    for df in trials:
        for col in emg_columns:
            env, stages = extract_envelope(df[col].values, fs=fs,
                                         bp_low=bp_low, bp_high=bp_high,
                                         lp_cutoff=lp_cutoff, order=order,
                                         demean=demean,
                                         return_intermediates=True)
            df[f'{col}_rect'] = stages['rectified']
            df[f'{col}_env'] = env

    # Step 2: Determine max per channel
    if max_values is None:
        # Fall back to global max across the provided trials
        max_values = {}
        for col in emg_columns:
            global_max = max(df[f'{col}_env'].max() for df in trials)
            max_values[col] = global_max if global_max > 0 else 1.0

    # Step 3: Normalize
    for df in trials:
        for col in emg_columns:
            df[f'{col}_env_norm'] = df[f'{col}_env'] / max_values[col]

    print(f"Processed {len(trials)} trials × {len(emg_columns)} EMG channels")
    print(f"  Bandpass: {bp_low}–{bp_high} Hz | Lowpass: {lp_cutoff} Hz | "
          f"Order: {order}")
    for col in emg_columns:
        print(f"  {col} max used: {max_values[col]:.6f}")

    return trials, max_values

