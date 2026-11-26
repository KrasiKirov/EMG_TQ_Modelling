"""Rest flags: for every sample of every active trial, whether the EMG shows rest (never decided from torque).

The detection itself is ML.preprocessing.rest; this module applies it to a subject's bundle, one session at a time.
"""

import numpy as np

from ML.preprocessing.rest import detect_rest, emg_rest_stats


def session_rest(bundle, session):
    """{trial idx: rest segments [(s, e)]} for every active trial of a session; the EMG is divided by the session's
    per-channel 1 kHz maximum first."""
    recs = [r for r in bundle['active'] if r['session'] == session]
    emg_max = np.max(np.stack([r['env_max1k'] for r in recs]), axis=0)
    emg_max = np.where(emg_max > 0, emg_max, 1.0)
    env = {r['idx']: r['env_raw'] / emg_max for r in recs}
    stats = emg_rest_stats(list(env.values()))
    return {i: detect_rest(e, stats) for i, e in env.items()}


def rest_mask(T, segs):
    """(T,) bool: True inside the segments [(s, e)]."""
    m = np.zeros(T, bool)
    for s, e in segs:
        m[s:e] = True
    return m


def rest_flags(bundle):
    """{trial idx: (T,) bool} EMG-detected rest per sample, for every active trial of both sessions."""
    out = {}
    for sess in ('S1', 'S2'):
        segs = session_rest(bundle, sess)
        for r in bundle['active']:
            if r['session'] == sess:
                out[r['idx']] = rest_mask(len(r['tq']), segs[r['idx']])
    return out
