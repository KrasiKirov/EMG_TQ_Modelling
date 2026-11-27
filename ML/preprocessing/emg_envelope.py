"""EMG linear envelope of one raw signal: band-pass, full-wave rectify, low-pass, all forward-only (causal)."""

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from ML.config import FS_RAW

# Canonical muscle order; detect_emg_columns enforces it so the features are [MG, LG, SOL, TA, position].
CANONICAL_EMG_ORDER = ['gm', 'gl', 'sol', 'ta']

BP_LOW = 30.0       # Hz, band-pass lower cutoff
BP_HIGH = 300.0     # Hz, band-pass upper cutoff
BP_ORDER = 4        # Butterworth order of the band-pass
LP_CUTOFF = 2.0     # Hz, envelope low-pass cutoff
LP_ORDER = 2        # Butterworth order of the low-pass
SETTLE_S = 1.0      # s, the offset is the mean of the first SETTLE_S seconds
LP_START_S = 0.2    # s, the low-pass starts from the mean rectified value of this first stretch

_EMG_KEYWORDS = {'emg', 'gm', 'gl', 'sol', 'ta', 'mg', 'lg'}
_NON_EMG_NAMES = {'time', 'pos', 'tq', 'position', 'torque'}


def detect_emg_columns(df):
    """
    EMG column names of a trial DataFrame (by keyword, excluding time / position / torque), in CANONICAL_EMG_ORDER
    so the features are always [MG, LG, SOL, TA]. ValueError for an EMG-like column outside that order.
    """
    emg_cols = {c for c in df.columns
                if c.lower() not in _NON_EMG_NAMES and any(kw in c.lower() for kw in _EMG_KEYWORDS)}
    extra = sorted(emg_cols - set(CANONICAL_EMG_ORDER))
    if extra:
        raise ValueError(f'unexpected EMG column(s) {extra}; expected {CANONICAL_EMG_ORDER}')
    return [c for c in CANONICAL_EMG_ORDER if c in emg_cols]


def _filters():
    """(band-pass, low-pass) Butterworth filters as second-order sections."""
    nyq = 0.5 * FS_RAW
    return (butter(BP_ORDER, [BP_LOW / nyq, BP_HIGH / nyq], btype='band', output='sos'),
            butter(LP_ORDER, LP_CUTOFF / nyq, btype='low', output='sos'))


def extract_envelope(signal):
    """
    Linear envelope of one raw EMG signal (un-normalised, same length): remove the offset, Butterworth band-pass,
    full-wave rectify, Butterworth low-pass. Each output sample uses only current and earlier input.

    The offset is the mean of the first SETTLE_S seconds and the low-pass starts from the mean rectified value of the
    first LP_START_S; the pipeline drops that start (filter start-up), so no kept sample depends on later data.
    """
    x = np.asarray(signal, dtype=np.float64)
    x = x - np.mean(x[:max(1, int(SETTLE_S * FS_RAW))])
    sos_bp, sos_lp = _filters()
    rectified = np.abs(sosfilt(sos_bp, x))
    zi = sosfilt_zi(sos_lp) * np.mean(rectified[:max(1, int(LP_START_S * FS_RAW))])
    return sosfilt(sos_lp, rectified, zi=zi)[0]
