"""EMG-only rest detection at 100 Hz (rest is never detected from torque), and the in-trial rest baseline.

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


# Within-trial relaxation baseline through the rest segments

def _seg_levels(y, segs):
    tc = np.array([(s + e - 1) / 2.0 for s, e in segs])
    med = np.array([np.median(y[s:e]) for s, e in segs])
    n = np.array([e - s for s, e in segs], float)
    return tc, med, n



def _kernel(t, tc, med, w, sigma):
    d = (t[:, None] - tc[None, :]) / sigma
    lw = -0.5 * d * d + np.log(w)[None, :]
    lw -= lw.max(axis=1, keepdims=True)
    k = np.exp(lw)
    return (k * med[None, :]).sum(1) / k.sum(1)


def _reject_outliers(tc, med, n, sigma, k=3.5, min_dev=0.25, iters=2):
    keep = np.ones(len(tc), bool)
    if len(tc) < 4:
        return keep
    w = np.sqrt(np.minimum(n, 300.0))
    for _ in range(iters):
        idx = np.where(keep)[0]
        if len(idx) < 4:
            break
        pred = np.array([_kernel(np.array([tc[i]]), tc[idx[idx != i]],
                                 med[idx[idx != i]], w[idx[idx != i]], sigma)[0]
                         for i in idx])
        res = med[idx] - pred
        mad = 1.4826 * np.median(np.abs(res - np.median(res)))
        bad = np.abs(res) > max(k * mad, min_dev)
        if not bad.any():
            break
        keep[idx[bad]] = False
    return keep


def rest_baseline(y, segs, T, sigma_s=8.0):
    """Gaussian-kernel baseline (σ = sigma_s) through outlier-filtered
    medians of the rest segments; flat extrapolation at the ends.
    Returns zeros if no rest segment was found."""
    if not segs:
        return np.zeros(T)
    tc, med, n = _seg_levels(y, segs)
    sigma = sigma_s * FS_DS
    keep = _reject_outliers(tc, med, n, sigma)
    w = np.sqrt(np.minimum(n[keep], 300.0))
    return _kernel(np.arange(T, dtype=float), tc[keep], med[keep], w, sigma)
