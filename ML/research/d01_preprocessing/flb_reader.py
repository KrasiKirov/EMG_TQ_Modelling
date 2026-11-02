"""
flb_reader.py
-------------
Reads REKLAB .flb binary files and converts each trial to a pandas DataFrame.
Optionally exports all trials to CSV files.

FLB binary format (reverse-engineered from flbio.m / flb2mat.m by R. Kearney, McGill):
  Each trial block:
    [int32]  flb_version  (2, 3, or 4)
    [int32]  nDim
    [int32]  nReal        (number of realizations, usually 1)
    [int32]  nChan        (number of channels, usually 6)
    [int32]  chanLen      (number of samples per channel)
    [int32]  chanFormat   (2 = int16, 4 = float32)
    [int32]  stringLen
    [char×stringLen] domainName
    [float32] domainIncr  (sampling interval in seconds)
    [float32] domainStart
    [int32]  commentLen
    [char×commentLen] comment   (or 'DEFAULT' if commentLen == 0)
    for each channel:
        [int32]  nameLen
        [char×nameLen] chanName  (or 'DEFAULT' if nameLen == 0)
    [float64 × nChan] chanMin
    [float64 × nChan] chanMax
    [float32 or int16 × (chanLen × nChan × nReal)] raw data
        → reshaped to (chanLen, nChan, nReal), column-major order

Channel layout (all subjects except IES01):
    Ch1 = POS  — ankle angle (rad)
    Ch2 = TQ   — torque (Nm)
    Ch3 = GM   — Gastrocnemius Medial EMG (mV)
    Ch4 = GL   — Gastrocnemius Lateral EMG (mV)
    Ch5 = SOL  — Soleus EMG (mV)
    Ch6 = TA   — Tibialis Anterior EMG (mV)
"""

import struct
import os
import numpy as np
import pandas as pd


# ── Channel name map (MATLAB is 1-indexed, Python 0-indexed) ─────────────
DEFAULT_CHANNEL_NAMES = ['position', 'torque', 'gm', 'gl', 'sol', 'ta']

# IES01 subject has a different channel order in the raw file
IES01_CHANNEL_ORDER = [0, 1, 3, 5, 4, 2]   # reorder to: pos, tq, gm, gl, sol, ta

# Map common short names to canonical names used throughout the pipeline.
# EMG columns from different FLB files use different naming conventions;
# this ensures a consistent set of column names across all subjects.
_CANONICAL_NAMES = {
    'pos': 'position',
    'tq': 'torque',
    'mg_emg': 'gm',       # medial gastrocnemius
    'lg_emg': 'gl',       # lateral gastrocnemius
    'sol_emg': 'sol',      # soleus
    'ta_emg': 'ta',        # tibialis anterior
}


# ── Low-level binary helpers ──────────────────────────────────────────────

def _read_int32(f):
    raw = f.read(4)
    if len(raw) < 4:
        return None
    return struct.unpack('<i', raw)[0]


def _read_float32(f):
    return struct.unpack('<f', f.read(4))[0]


def _read_float64(f):
    return struct.unpack('<d', f.read(8))[0]


def _read_string(f, length):
    if length <= 0:
        return 'DEFAULT'
    return f.read(length).decode('latin-1', errors='replace')


# ── Read one trial header (flbrdet equivalent) ────────────────────────────

def _read_header(f):
    """Read the header of the next trial. Returns None at EOF."""
    flb_version = _read_int32(f)
    if flb_version is None:
        return None  # clean EOF

    if flb_version < 2:
        raise ValueError(f"Unsupported FLB version: {flb_version}")

    h = {'version': flb_version}

    if flb_version == 2:
        # Older format
        h['nDim']        = 2
        h['nReal']       = 1
        h['nChan']       = _read_int32(f)
        h['chanLen']     = _read_int32(f)
        h['chanFormat']  = _read_int32(f)
        str_len          = _read_int32(f)
        h['domainName']  = _read_string(f, str_len)
        h['domainIncr']  = _read_float32(f)
        h['domainStart'] = _read_float32(f)
        comment_len      = _read_int32(f)
        h['comment']     = _read_string(f, comment_len)
        h['chanName']    = []
        for _ in range(h['nChan']):
            name_len = _read_int32(f)
            h['chanName'].append(_read_string(f, name_len))
        h['chanMin'] = [_read_float64(f) for _ in range(h['nChan'])]
        h['chanMax'] = [_read_float64(f) for _ in range(h['nChan'])]

    else:
        # Version 3 / 4 format (written by flbwdet)
        h['nDim']        = _read_int32(f)
        h['nReal']       = _read_int32(f)
        h['nChan']       = _read_int32(f)
        h['chanLen']     = _read_int32(f)
        h['chanFormat']  = _read_int32(f)
        str_len          = _read_int32(f)
        h['domainName']  = _read_string(f, str_len)
        h['domainIncr']  = _read_float32(f)
        h['domainStart'] = _read_float32(f)
        comment_len      = _read_int32(f)
        h['comment']     = _read_string(f, comment_len)
        h['chanName']    = []
        for _ in range(h['nChan']):
            name_len = _read_int32(f)
            h['chanName'].append(_read_string(f, name_len))
        h['chanMin'] = [_read_float64(f) for _ in range(h['nChan'])]
        h['chanMax'] = [_read_float64(f) for _ in range(h['nChan'])]

    return h


# ── Read one trial's data block (flbrdata equivalent) ────────────────────

def _read_data(f, h):
    """Read the raw data block for a trial given its header dict."""
    n_samples = h['nChan'] * h['chanLen'] * h['nReal']

    if h['chanFormat'] == 4:
        dtype = np.float32
        itemsize = 4
    elif h['chanFormat'] == 2:
        dtype = np.int16
        itemsize = 2
    else:
        raise ValueError(f"Unknown chanFormat: {h['chanFormat']}")

    raw = f.read(n_samples * itemsize)
    if len(raw) < n_samples * itemsize:
        raise IOError("Unexpected end of file while reading data block.")

    arr = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    # MATLAB reshape: (chanLen, nChan, nReal), column-major (Fortran order)
    arr = arr.reshape((h['chanLen'], h['nChan'], h['nReal']), order='F')
    # Take first realization (nReal is almost always 1)
    return arr[:, :, 0]   # shape: (chanLen, nChan)


# ── Public API ────────────────────────────────────────────────────────────

def read_flb(filepath, subject_id='default', channel_names=None):
    """
    Read all trials from an FLB file.

    Parameters
    ----------
    filepath : str
        Path to the .flb file.
    subject_id : str
        Subject identifier. Use 'IES01' to apply the alternate channel order
        for that subject. All others use the default order.
    channel_names : list of str, optional
        Custom column names for the DataFrame. Defaults to
        ['pos', 'tq', 'gm', 'gl', 'sol', 'ta'].

    Returns
    -------
    list of pd.DataFrame
        One DataFrame per trial. Each has columns:
        ['time', 'pos', 'tq', 'gm', 'gl', 'sol', 'ta']
        and metadata attributes: trial_index, domainIncr, comment, chanName.
    """
    if channel_names is None:
        channel_names = DEFAULT_CHANNEL_NAMES

    trials = []

    with open(filepath, 'rb') as f:
        trial_idx = 0
        while True:
            h = _read_header(f)
            if h is None:
                break   # clean EOF

            data = _read_data(f, h)   # (chanLen, nChan)

            # Enforce 1 ms sampling rate (as done in read_preprocess_data.m)
            Ts = 0.001
            n_samples = data.shape[0]
            time = h['domainStart'] + np.arange(n_samples) * Ts

            # Reorder channels for IES01
            if subject_id.upper() == 'IES01' and data.shape[1] >= 6:
                data = data[:, IES01_CHANNEL_ORDER]
                # After reorder, data matches DEFAULT_CHANNEL_NAMES order
                # (pos, tq, gm, gl, sol, ta) — ignore file header names
                col_names = channel_names
            else:
                # Use channel names from the file header if available, else defaults
                file_chan_names = h.get('chanName', [])
                if file_chan_names and all(c not in ('DEFAULT', '') for c in file_chan_names):
                    col_names = [c.lower().replace(' ', '_') for c in file_chan_names]
                else:
                    col_names = channel_names

            # Normalize to canonical names (e.g. 'pos' -> 'position')
            col_names = [_CANONICAL_NAMES.get(c, c) for c in col_names]

            n_ch = min(data.shape[1], len(col_names))
            df = pd.DataFrame(data[:, :n_ch], columns=col_names[:n_ch])
            df.insert(0, 'time', time)

            # Attach metadata as DataFrame attributes
            df.attrs['trial_index'] = trial_idx
            df.attrs['domainIncr']  = Ts
            df.attrs['domainStart'] = h['domainStart']
            df.attrs['comment']     = h['comment']
            df.attrs['chanName']    = h['chanName']
            df.attrs['source_file'] = os.path.basename(filepath)

            trials.append(df)
            trial_idx += 1

    print(f"Read {len(trials)} trials from '{os.path.basename(filepath)}'")
    return trials


