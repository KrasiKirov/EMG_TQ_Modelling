"""EMG-only rest detection at 100 Hz (rest is never detected from torque).

Moved unchanged from active_torque.py.
"""

import numpy as np

FS_DS = 100.0   # Hz after downsampling


def emg_rest_stats(env_list, q_lo=5.0, q_hi=99.0):
    """Per-channel scale from training-session envelopes (normalised)."""
    E = np.concatenate(env_list)
    lo, hi = np.percentile(E, q_lo, axis=0), np.percentile(E, q_hi, axis=0)
    return {'scale': np.maximum(hi - lo, 1e-3)}


def detect_rest(env, stats, thr_lo=0.03, thr_hi=0.06, min_run_s=0.5,
                settle_s=0.3, pre_s=0.1, min_keep_s=0.2, floor_q=5.0):
    """EMG-only rest segments [(s, e), ...] of one trial (100 Hz).

    Activity = mean over channels of (env − trial floor) / training scale;
    Schmitt trigger enters rest below ``thr_lo`` and leaves above ``thr_hi``.
    Runs are eroded by ``settle_s`` at the start (torque relaxes after EMG
    falls) and ``pre_s`` at the end (zero-phase envelope rises early).
    """
    floor = np.percentile(env, floor_q, axis=0)
    score = ((env - floor) / stats['scale']).mean(axis=1)
    T = len(score)
    runs, state, a = [], False, 0
    for i in range(T):
        if state and score[i] > thr_hi:
            runs.append((a, i)); state = False
        elif not state and score[i] < thr_lo:
            state, a = True, i
    if state:
        runs.append((a, T))
    settle, pre = int(settle_s * FS_DS), int(pre_s * FS_DS)
    out = []
    for a, b in runs:
        if b - a < int(min_run_s * FS_DS):
            continue
        s, e = a + settle, (b - pre if b < T else b)
        if e - s >= int(min_keep_s * FS_DS):
            out.append((s, e))
    return out
