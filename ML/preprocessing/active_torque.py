"""
active_torque.py
----------------
Better estimates of the passive component of measured torque, so the model
target (net active torque) stays consistent between sessions.

Why this exists — the plantarflexed-angle R² drop
-------------------------------------------------
At the most plantarflexed angles (YES −0.49 rad, JM −0.79/−0.64 rad) the
active torque is tiny (retest std 0.4–0.8 Nm), so a 0.5 Nm error in the
subtracted passive torque alone drives per-position R² negative.  Three
measured problems with plain "measured − passive-hold map" subtraction:

1.  Movement segments leak into the passive map.  The 20 Nm std / |20 Nm|
    bounds admitted JM trial 21 (a movement recording) as +11.7 Nm at
    −0.07 rad.  Every passive trial starts with a clean 20–30 s static hold
    (torque std 0.02–0.3 Nm); everything after the device step is junk.
    → ``static_hold_passive_map``: keep only segments with position std
      < 5 mrad, torque std < 1 Nm and ≥ 5 s.  Identical maps for EG/CH/HM,
      only removes the bad JM entry.

2.  Torque at rest *during active trials* is systematically smaller than the
    passive-hold torque (YES: ≈ 0.77·P at every angle, both sessions —
    tissue conditioning / thixotropy after contractions).  Because P differs
    between sessions (YES −0.49: 7.9 vs 10.6 Nm), plain subtraction leaves a
    session-dependent offset (1−α)·ΔP that the model cannot learn.
    → ``fit_passive_gain``: robust fit  tq_rest ≈ α·P + β  on TRAINING-
      session rest segments only, then P_eff = α·P + β for both sessions.
      Uses no retest torque — the test target is still defined from the
      retest passive map alone.

3.  Passive torque relaxes within a trial (YES −0.49: the relaxed plateau
    sinks ~0.6–1 Nm over 90 s).  EMG cannot explain this, so it corrupts the
    training labels at weak angles.
    → ``rest_baseline``: smooth baseline through EMG-detected rest segments,
      subtracted from TRAINING/VALIDATION labels only (never from the test
      target).

Rest is always detected from EMG only (never from torque).
"""

import numpy as np

FS_DS = 100.0   # Hz after downsampling


# ── 1. Static-hold passive map ──────────────────────────────────────────────

STATIC_MAX_POS_STD = 0.005   # rad
STATIC_MAX_TQ_STD  = 1.0     # Nm
STATIC_MIN_DUR_S   = 5.0     # s


def static_hold_passive_map(passive_trials, split_fn, position_col, torque_col,
                            pos_tolerance, fs=1000.0,
                            max_pos_std=STATIC_MAX_POS_STD,
                            max_tq_std=STATIC_MAX_TQ_STD,
                            min_dur_s=STATIC_MIN_DUR_S):
    """(position, mean torque) for genuinely static passive holds.

    ``split_fn`` splits a trial at device steps (dataset_builder's
    ``_split_passive_segments``).  Nearby positions are averaged.
    """
    raw = []
    for df in passive_trials:
        for seg in split_fn(df, position_col):
            p = seg[position_col].values
            t = seg[torque_col].values
            if len(t) < min_dur_s * fs or p.std() > max_pos_std or t.std() > max_tq_std:
                continue
            raw.append((float(p.mean()), float(t.mean())))
    raw.sort()
    clusters = []
    for pos, tq in raw:
        if clusters and abs(pos - clusters[-1][0]) <= pos_tolerance:
            pp, tt, c = clusters[-1]
            clusters[-1] = ((pp * c + pos) / (c + 1), (tt * c + tq) / (c + 1), c + 1)
        else:
            clusters.append((pos, tq, 1))
    return [(p, t) for p, t, _ in clusters]


# ── EMG-only rest detection ─────────────────────────────────────────────────

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


def _seg_levels(y, segs):
    tc = np.array([(s + e - 1) / 2.0 for s, e in segs])
    med = np.array([np.median(y[s:e]) for s, e in segs])
    n = np.array([e - s for s, e in segs], float)
    return tc, med, n


# ── 2. Passive gain correction ──────────────────────────────────────────────

def fit_passive_gain(records, passive_lookup, huber=3.0, iters=10):
    """Robust fit  tq_rest ≈ α·P(pos) + β  over rest segments of TRAINING
    records.  ``records``: dicts with 'env', 'tq', 'pos' (100 Hz) and
    'segs' (rest segments).  ``passive_lookup(pos_array) -> P``.
    One observation per rest segment (its median), weight √n.
    Returns None when there is no rest segment at all."""
    P_all, y_all, w_all = [], [], []
    for r in records:
        if not r['segs']:
            continue
        tc, med, n = _seg_levels(r['tq'], r['segs'])
        P_all.append(passive_lookup(r['pos'][tc.astype(int)]))
        y_all.append(med)
        w_all.append(np.sqrt(np.minimum(n, 300.0)))
    if not P_all:
        return None
    P, y, w = np.concatenate(P_all), np.concatenate(y_all), np.concatenate(w_all)
    A = np.column_stack([P, np.ones_like(P)])
    ww = w.copy()
    for _ in range(iters):
        coef = np.linalg.lstsq(A * ww[:, None], y * ww, rcond=None)[0]
        res = y - A @ coef
        s = max(1.4826 * np.median(np.abs(res)), 1e-3)
        u = np.abs(res) / (huber * s)
        ww = w * np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-9))
    return float(coef[0]), float(coef[1])


# ── 3. Within-trial relaxation baseline (training labels only) ─────────────

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
