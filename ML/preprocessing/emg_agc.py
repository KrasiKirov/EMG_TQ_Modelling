"""
emg_agc.py
----------
Causal adaptive gain control (AGC) of EMG envelopes.

Why: between the test and retest sessions the EMG amplitude for the same
torque changes, differently per channel and per angle (JM −0.64 / −0.36 rad:
retest plantarflexor envelopes ≈ 0.4–0.5× the test-session size for the same
torque; YES −0.38 rad: ≈ 1.3× larger).  A model with a fixed EMG→torque gain
then under/over-predicts the amplitude, which is what limits JM's per-position
R² at the plantarflexed angles.

What: per channel and per trial,
    a_k(t) = clip((env_k(t) − lo_k(t)) / max(hi_k(t) − lo_k(t), rmin_k), 0, 2)
with lo = running 10th percentile of the past 20 s (noise floor) and hi =
running 95th percentile of the past 45 s (active level), refreshed every 0.5 s
from PAST samples only and blended at the start of a trial with priors from the
TRAINING part of the test session.  Electrode pops / drop-outs are removed
first (sample-and-hold).  Everything is causal — a real-time decoder can
compute exactly the same features — and no torque is used.
"""

import numpy as np

FS = 100            # Hz after downsampling
LO_Q, HI_Q = 10.0, 95.0
LO_WIN_S, HI_WIN_S = 20.0, 45.0
UPDATE_S = 0.5
PRIOR_S = 5.0       # prior weight, in seconds of data
RMIN_FRAC = 0.35    # minimum range = 0.35 × prior range (no noise blow-up)
CLIP_HI = 2.0
SPIKE_THR = 4.0     # artefact: > 4 running ranges above the floor
SPIKE_HOLD_S = 0.3


def causal_quantiles(x, qs, win, upd, n0, prior):
    """Running percentiles of x from PAST samples only, blended with prior
    (weight n/(n+n0)).  Returns (T, len(qs))."""
    x = np.asarray(x, np.float64)
    qs = np.atleast_1d(qs)
    prior = np.asarray(prior, np.float64)
    out = np.empty((len(x), len(qs)))
    for u in range(0, len(x), upd):
        seg = x[max(0, u - win):u]
        if len(seg) == 0:
            v = prior
        else:
            w = len(seg) / (len(seg) + n0)
            v = w * np.percentile(seg, qs) + (1.0 - w) * prior
        out[u:u + upd] = v
    return out


def _running_lo_hi(env, prior_lo, prior_hi):
    upd, n0 = int(UPDATE_S * FS), int(PRIOR_S * FS)
    lo = np.empty_like(env)
    hi = np.empty_like(env)
    for k in range(env.shape[1]):
        lo[:, k] = causal_quantiles(env[:, k], [LO_Q], int(LO_WIN_S * FS), upd, n0,
                                    [prior_lo[k]])[:, 0]
        hi[:, k] = causal_quantiles(env[:, k], [HI_Q], int(HI_WIN_S * FS), upd, n0,
                                    [prior_hi[k]])[:, 0]
    return lo, hi


def _despike(x, lo, rng):
    """Sample-and-hold over artefacts: x far above the running range, or a
    drop-out far below a clearly non-zero floor.  Causal."""
    bad = ((x - lo) / rng > SPIKE_THR) | ((x < 0.25 * lo) & (lo > 0.1))
    if not bad.any():
        return x, 0
    y, last, hold, cnt = x.copy(), lo[0], 0, 0
    for t in range(len(x)):
        if bad[t]:
            hold = int(SPIKE_HOLD_S * FS)
        if bad[t] or hold > 0:
            y[t] = last
            cnt += 1
            if not bad[t]:
                hold -= 1
        else:
            last = x[t]
    return y, cnt


def agc_priors(train_envs):
    """Per-channel (lo, hi) priors: median over training trials of the
    10th / 95th percentiles of the training part of each trial."""
    lo = np.median([np.percentile(e, LO_Q, axis=0) for e in train_envs], axis=0)
    hi = np.median([np.percentile(e, HI_Q, axis=0) for e in train_envs], axis=0)
    return lo, hi


def apply_agc(env, prior_lo, prior_hi):
    """AGC-normalised envelopes (T, K) of one trial.  Returns (env_agc, n_despiked)."""
    rmin = RMIN_FRAC * np.maximum(prior_hi - prior_lo, 1e-3)
    lo, hi = _running_lo_hi(env, prior_lo, prior_hi)
    clean, n_bad = env.copy(), 0
    for k in range(env.shape[1]):
        clean[:, k], n = _despike(env[:, k], lo[:, k], np.maximum(hi[:, k] - lo[:, k], rmin[k]))
        n_bad += n
    if n_bad:   # artefact out of the running percentiles
        lo, hi = _running_lo_hi(clean, prior_lo, prior_hi)
    rng = np.maximum(hi - lo, rmin[None, :])
    return np.clip((clean - lo) / rng, 0.0, CLIP_HI), n_bad
