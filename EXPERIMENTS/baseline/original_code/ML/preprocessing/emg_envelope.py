"""
emg_envelope.py
---------------
EMG envelope extraction pipeline for sEMG-to-torque modelling.

Implements the standard processing pipeline:
    1. Bandpass filter  — remove motion artefacts and high-frequency noise
    2. Full-wave rectify — |x(t)|
    3. Low-pass filter  — extract the linear envelope
    4. Normalize ÷ global max  — scale to [0, 1] using the global max across active trials

"""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


# Default parameters 

# Canonical EMG muscle order used across all subjects.
# flb_reader normalises column names (e.g. mg_emg → gm) so that every
# subject's DataFrame uses these four names, but detect_emg_columns()
# also enforces this ordering so the feature array is always
#     [MG, LG, SOL, TA, position].
CANONICAL_EMG_ORDER = ['gm', 'gl', 'sol', 'ta']
DEFAULT_EMG_COLUMNS = CANONICAL_EMG_ORDER
DEFAULT_FS = 1000.0        # Hz — sampling frequency used in REKLAB experiments
DEFAULT_BP_LOW = 30.0      # Hz — bandpass lower cutoff
DEFAULT_BP_HIGH = 300.0    # Hz — bandpass upper cutoff
DEFAULT_LP_CUTOFF = 2.0    # Hz — lowpass cutoff for envelope (Can also use 1.5 Hz)
DEFAULT_ORDER = 4          # Butterworth filter order

_EMG_KEYWORDS = {'emg', 'gm', 'gl', 'sol', 'ta', 'mg', 'lg'}
_NON_EMG_NAMES = {'time', 'pos', 'tq', 'position', 'torque'}


def detect_emg_columns(df):
    """
    Auto-detect EMG column names in a trial DataFrame.

    Identifies columns by checking for known EMG-related keywords
    (e.g. 'emg', 'gm', 'sol') while excluding non-EMG columns
    (e.g. 'time', 'pos', 'torque').

    Returns columns sorted to CANONICAL_EMG_ORDER so that the feature
    array is always [MG, LG, SOL, TA] regardless of file header order.

    Parameters
    ----------
    df : pd.DataFrame
        A trial DataFrame as returned by ``read_flb``.

    Returns
    -------
    list of str
        Column names identified as EMG channels, in canonical order.
    """
    emg_cols = set()
    for col in df.columns:
        col_lower = col.lower()
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

def extract_envelope(signal, fs=DEFAULT_FS,
                     bp_low=DEFAULT_BP_LOW, bp_high=DEFAULT_BP_HIGH,
                     lp_cutoff=DEFAULT_LP_CUTOFF, order=DEFAULT_ORDER,
                     demean=True,
                     return_intermediates=False):
    """
    Extract the linear envelope from a single raw EMG signal.

    Steps:
        0. Demean (remove DC offset) — matches MATLAB ``demean.m``
        1. Bandpass filter (zero-phase Butterworth)
        2. Full-wave rectification  |x(t)|
        3. Lowpass filter (zero-phase Butterworth)

    Parameters
    ----------
    signal : array-like
        1-D array of raw EMG samples.
    fs : float
        Sampling frequency in Hz (default 1000).
    bp_low : float
        Bandpass lower cutoff in Hz (default 30).
    bp_high : float
        Bandpass upper cutoff in Hz (default 300).
    lp_cutoff : float
        Lowpass cutoff in Hz for envelope extraction (default 2).
    order : int
        Butterworth filter order (default 4).
    demean : bool
        If True, subtract mean before filtering (default True).
        Required for wireless EMG sensors with DC offset.
    return_intermediates : bool
        If True, also return the bandpass-filtered+rectified signal
        (``rectified``) alongside the envelope.

    Returns
    -------
    envelope : np.ndarray
        Un-normalized linear envelope (same length as input).
    rectified : np.ndarray
        Bandpass-filtered and rectified signal — only returned when
        ``return_intermediates=True``.
    """
    signal = np.asarray(signal, dtype=np.float64)

    # Step 0: Remove DC offset (matches MATLAB demean.m: y = x - mean(x))
    if demean:
        signal = signal - np.mean(signal)

    nyq = 0.5 * fs

    # Step 1: Bandpass filter
    sos_bp = butter(order, [bp_low / nyq, bp_high / nyq], btype='band', output='sos')
    filtered = sosfiltfilt(sos_bp, signal)

    # Step 2: Full-wave rectification
    rectified = np.abs(filtered)

    # Step 3: Lowpass filter → linear envelope
    sos_lp = butter(order, lp_cutoff / nyq, btype='low', output='sos')
    envelope = sosfiltfilt(sos_lp, rectified)

    # Clip: sosfiltfilt can produce tiny negative values at edges after rectification
    envelope = np.clip(envelope, 0, None)

    if return_intermediates:
        return envelope, rectified
    return envelope


# Batch processing + normalization


def process_trials(trials, emg_columns=None, max_values=None, fs=DEFAULT_FS,
                   bp_low=DEFAULT_BP_LOW, bp_high=DEFAULT_BP_HIGH,
                   lp_cutoff=DEFAULT_LP_CUTOFF, order=DEFAULT_ORDER,
                   demean=True):
    """
    Process all trials for one subject: extract envelopes and normalize.

    For each EMG column in each trial DataFrame, this function:
        1. Extracts the linear envelope via ``extract_envelope``
        2. Determines the normalization max per channel from the global max
           across the provided trials
        3. Normalizes each envelope to [0, 1]

    New columns are added to each DataFrame:
        - ``{col}_env``      — un-normalized envelope
        - ``{col}_env_norm`` — normalized envelope [0, 1]

    Parameters
    ----------
    trials : list of pd.DataFrame
        Trial DataFrames as returned by ``read_flb``.  Each must contain the
        columns listed in *emg_columns*.
    emg_columns : list of str, optional
        EMG column names to process (default: auto-detected).
    max_values : dict, optional
        Pre-computed per-channel max values.  If provided, these are used
        for normalization instead of the global max computed across *trials*.
    fs : float
        Sampling frequency in Hz.
    bp_low, bp_high : float
        Bandpass cutoffs in Hz.
    lp_cutoff : float
        Lowpass cutoff in Hz for envelope extraction.
    order : int
        Butterworth filter order.

    Returns
    -------
    trials : list of pd.DataFrame
        Same DataFrames with ``*_env`` and ``*_env_norm`` columns added.
    max_values : dict
        ``{column_name: max_used}`` — the global max values used for
        normalization.
    """
    if emg_columns is None:
        emg_columns = detect_emg_columns(trials[0])

    # Step 1: Extract envelopes
    for df in trials:
        for col in emg_columns:
            env, rect = extract_envelope(df[col].values, fs=fs,
                                         bp_low=bp_low, bp_high=bp_high,
                                         lp_cutoff=lp_cutoff, order=order,
                                         demean=demean,
                                         return_intermediates=True)
            df[f'{col}_rect'] = rect
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


