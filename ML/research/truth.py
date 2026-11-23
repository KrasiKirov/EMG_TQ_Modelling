"""EMG-detected rest periods of every active trial (rest is found from the EMG only, never from torque).

Used for the rest flags of every window. (The 'restanchor' target, a baseline through these periods, was removed in
2026-11; see tag pre-cleanup-2026-11.)
"""

import numpy as np

from ML.preprocessing import rest as RS


def session_rest(bundle, session, max_frac=None):
    """{trial idx: rest segments [(s, e)]} for every active trial of a session.
    ``max_frac`` limits both the statistics and the detection to the first part of each trial."""
    recs = [r for r in bundle['active'] if r['session'] == session]
    emg_max = np.max(np.stack([r['env_max1k'] for r in recs]), axis=0)
    emg_max = np.where(emg_max > 0, emg_max, 1.0)
    env = {}
    for r in recs:
        e = r['env_raw'] / emg_max
        env[r['idx']] = e[:int(len(e) * max_frac)] if max_frac else e
    stats = RS.emg_rest_stats(list(env.values()))
    return {i: RS.detect_rest(e, stats) for i, e in env.items()}


def rest_mask(T, segs):
    m = np.zeros(T, bool)
    for s, e in segs:
        m[s:e] = True
    return m

