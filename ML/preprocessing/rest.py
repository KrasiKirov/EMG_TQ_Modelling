"""EMG-only rest detection at 100 Hz (rest is never detected from torque)."""

import numpy as np

from ML.config import FS   # Hz, after downsampling

SCALE_Q = (5.0, 99.0)   # percentiles whose difference is a channel's scale
THR_LO = 0.03           # rest starts when the activity falls below this
THR_HI = 0.06           # rest ends when the activity rises above this
MIN_RUN_S = 0.5         # s, shorter rest runs are dropped
SETTLE_S = 0.3          # s cut from the start of a run (torque relaxes after the EMG falls)
PRE_S = 0.1             # s cut from the end of a run (the envelope rises early)
MIN_KEEP_S = 0.2        # s, runs shorter than this after the cuts are dropped
FLOOR_Q = 5.0           # percentile of a trial's envelope taken as its floor


def emg_rest_stats(env_list):
    """Per-channel scale from a session's envelopes (normalised)."""
    E = np.concatenate(env_list)
    lo, hi = np.percentile(E, SCALE_Q[0], axis=0), np.percentile(E, SCALE_Q[1], axis=0)
    return {'scale': np.maximum(hi - lo, 1e-3)}


def detect_rest(env, stats):
    """EMG-only rest segments [(s, e), ...] of one trial (100 Hz).

    Activity = mean over channels of (env - trial floor) / session scale. A Schmitt trigger enters rest below THR_LO
    and leaves above THR_HI; each run is then cut by SETTLE_S at the start and PRE_S at the end.
    """
    floor = np.percentile(env, FLOOR_Q, axis=0)
    score = ((env - floor) / stats['scale']).mean(axis=1)
    T = len(score)
    runs, state, a = [], False, 0
    for i in range(T):
        if state and score[i] > THR_HI:
            runs.append((a, i)); state = False
        elif not state and score[i] < THR_LO:
            state, a = True, i
    if state:
        runs.append((a, T))
    settle, pre = int(SETTLE_S * FS), int(PRE_S * FS)
    out = []
    for a, b in runs:
        if b - a < int(MIN_RUN_S * FS):
            continue
        s, e = a + settle, (b - pre if b < T else b)
        if e - s >= int(MIN_KEEP_S * FS):
            out.append((s, e))
    return out
