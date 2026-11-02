"""
Baseline / drift / passive-map variants  (designer: baseline_drift)
===================================================================

Attacks three measured mechanisms of the dorsiflexion R² collapse:

  M1  rest-offset shift between sessions (active-trial rest torque minus the
      session passive map differs test vs retest by 0.2-1.7 Nm)
  M2  within-trial baseline drift (passive stress relaxation; YES -0.486 rest
      plateau sinks ~0.6-0.8 Nm over 90 s while the active swing is ~1.2 Nm)
  M7  JM passive map bug (+11.7 Nm at -0.071 rad from movement trial 21)

Findings that shaped the variants (see the dry-run numbers in the report):

  * Every passive trial starts with a clean 20-30 s static hold (seg0, torque
    std 0.02-0.3 Nm); everything after the device step is garbage (torques of
    -700..+160 Nm).  The YES worksheet overrides equal these hold means.
  * Active-trial rest torque is systematically SMALLER in magnitude than the
    passive-hold torque: for YES  tq_rest ≈ 0.78 · P_hold + c  at every angle
    and in both sessions (thixotropy / conditioning by the contractions).
    Because P_hold differs between sessions (YES -0.486: 7.9 vs 10.6 Nm), the
    plain subtraction leaves a session-dependent offset  (1-α)·ΔP  -> M1.
    A gain α fitted on the TRAINING session alone removes most of it.

Variant plugins (all EMG-only rest detection; torque is never used to decide
what "rest" is):

  passive_static        [passive trials only — no active-trial torque]
        static-hold passive map (seg0 holds, torque std < 1 Nm, position std
        < 5 mrad).  Fixes M7 (JM -0.071 rad: +11.7 -> ≈ -2.0 Nm).  Identical
        to the baseline for YES (worksheet overrides = static-hold means).
  passive_static_gain   [passive trials + training-session labels only]
        passive_static, then P_eff = α·P + β (optional γ·P·g(t) relaxation
        term, off by default — it over-corrected in the dry run) fitted on
        test-session rest segments.  No retest torque, no retest EMG
        statistics, causal.  Changes the target via the passive map, but the
        harness 'ref' metric (pred + P_eff − P_ref) is an honest score on the
        ORIGINAL target because P_eff only needs the retest passive map.
        Fitted: YES α=0.771 β=+0.32;  JM α=1.038 β=−0.37.
  tare_causal           [online tare from past rest segments]
        b(t) from rest samples strictly in the past (hysteresis detector with
        causal running noise floors, confirmation + lag), exponential
        forgetting across segments.  A real-time controller with a torque
        sensor that is tared whenever EMG is silent can do exactly this, so
        the 'ref' metric is again an honest score on the original target.
  tare_start            [single tare at the first rest segment]
        constant b = median of the first ≥1 s rest segment (available only
        once it is complete); b0 before that.  Removes M1, leaves M2 (dry run:
        WORSE than passive_static_gain for YES — the first rest of a trial is
        not representative, passive torque drops after the first pushes).
  rezero_smooth         [offline, non-causal]
        two-sided Gaussian-kernel (or log-time) baseline through outlier-
        filtered rest-segment medians from the robust detector.  Upper bound
        / improved core.rezero_drift (core.rest_mask flags 8.8 % of JM "rest"
        samples > 0.3 Nm away from their segment level; this detector 0 %).
        bd_apply='trainval' re-zeroes the training labels only.

Parameters are read from opts['params'] with the prefix ``bd_`` so they never
collide with other plugins.  All plugins store their baseline in
r['bd_baseline'] (and P_eff in r['passive']) for diagnostics.
"""

import numpy as np

from ML.preprocessing import dataset_builder as db
from ML.experiments.harness import passive_curve

FS = 100.0  # Hz after decimation


def _p(ctx, key, default):
    return ctx['opts'].get('params', {}).get(key, default)


# ═════════════════════════════════════════════════════════════════════════════
#  EMG-only rest detection
# ═════════════════════════════════════════════════════════════════════════════

def train_emg_stats(trainval, q_lo=5.0, q_hi=99.0):
    """Per-channel scale / floor statistics from the TRAINING session only."""
    E = np.concatenate([r['env'] for r in trainval])
    lo = np.percentile(E, q_lo, axis=0)
    hi = np.percentile(E, q_hi, axis=0)
    floors = np.array([np.percentile(r['env'], q_lo, axis=0) for r in trainval])
    return {'floor': lo, 'scale': np.maximum(hi - lo, 1e-3),
            'floor_max': floors.max(axis=0)}


def offline_floor(env, q=5.0):
    """Per-channel noise floor of one trial (uses the whole trial's EMG)."""
    return np.broadcast_to(np.percentile(env, q, axis=0), env.shape)


def causal_floor(env, cap, q=5.0, win_s=30.0, every_s=0.25, min_hist_s=1.0):
    """Running per-channel noise floor: q-th percentile of the PAST ``win_s``
    seconds (samples strictly before the refresh instant), refreshed every
    ``every_s`` s and capped at ``cap`` (training-session prior) so a long
    contraction can never be mistaken for the floor."""
    T, C = env.shape
    W, step, h0 = int(win_s * FS), max(1, int(every_s * FS)), int(min_hist_s * FS)
    out = np.empty_like(env)
    cur = np.asarray(cap, float).copy()
    for t0 in range(0, T, step):
        if t0 >= h0:
            cur = np.minimum(np.percentile(env[max(0, t0 - W):t0], q, axis=0), cap)
        out[t0:t0 + step] = cur
    return out


def activity_score(env, floor, scale, how='mean'):
    z = (env - floor) / scale
    return z.mean(axis=1) if how == 'mean' else z.max(axis=1)


def hysteresis_runs(score, lo, hi, min_len):
    """Causal Schmitt trigger: enter rest when score < lo, leave when > hi.
    Returns runs (a, b) of length ≥ min_len."""
    T = len(score)
    below, above = score < lo, score > hi
    runs, state, a = [], False, 0
    for i in range(T):
        if state:
            if above[i]:
                runs.append((a, i))
                state = False
        elif below[i]:
            state, a = True, i
    if state:
        runs.append((a, T))
    return [(a, b) for a, b in runs if b - a >= min_len]


def detect_rest(env, stats, mode='offline', ctx=None):
    """Robust EMG-only rest detector.

    Returns a list of (s, e, a, b): eroded usable range [s, e) inside the raw
    hysteresis run [a, b).  Erosion drops ``settle`` s after the run start
    (torque relaxes after EMG falls) and ``pre`` s before the run end (the
    zero-phase envelope rises before the torque does)."""
    g = (lambda k, d: _p(ctx, k, d)) if ctx is not None else (lambda k, d: d)
    q = g('bd_floor_q', 5.0)
    if mode == 'causal' and g('bd_floor_mode', 'running') == 'train':
        # fixed training-session floor: no statistics of the test EMG at all
        floor = np.broadcast_to(stats['floor_max'], env.shape)
    elif mode == 'causal':
        cap = stats['floor_max'] + g('bd_floor_cap', 0.05) * stats['scale']
        floor = causal_floor(env, cap, q=q, win_s=g('bd_floor_win_s', 30.0))
    else:
        floor = offline_floor(env, q)
    score = activity_score(env, floor, stats['scale'], g('bd_score', 'mean'))
    lo, hi = g('bd_thr_lo', 0.03), g('bd_thr_hi', 0.06)
    runs = hysteresis_runs(score, lo, hi, int(g('bd_min_run_s', 0.5) * FS))
    settle, pre = int(g('bd_settle_s', 0.3) * FS), int(g('bd_pre_s', 0.1) * FS)
    keep = int(g('bd_min_keep_s', 0.2) * FS)
    out = []
    for a, b in runs:
        s, e = a + settle, (b - pre if b < len(score) else b)
        if e - s >= keep:
            out.append((s, e, a, b))
    return out


def runs_to_mask(runs, T):
    m = np.zeros(T, bool)
    for s, e, *_ in runs:
        m[s:e] = True
    return m


# ═════════════════════════════════════════════════════════════════════════════
#  Baseline estimators
# ═════════════════════════════════════════════════════════════════════════════

def seg_levels(y, runs):
    tc = np.array([(s + e - 1) / 2.0 for s, e, *_ in runs])
    med = np.array([np.median(y[s:e]) for s, e, *_ in runs])
    n = np.array([e - s for s, e, *_ in runs], float)
    return tc, med, n


def _kernel_eval(t, tc, med, w, sigma):
    if len(tc) == 0:
        return np.zeros(len(t))
    d = (t[:, None] - tc[None, :]) / sigma
    lw = -0.5 * d * d + np.log(w)[None, :]
    lw -= lw.max(axis=1, keepdims=True)          # log-sum-exp guard
    k = np.exp(lw)
    return (k * med[None, :]).sum(1) / k.sum(1)


def reject_segment_outliers(tc, med, n, sigma, k=3.5, min_dev=0.25, iters=2):
    """Leave-one-out kernel prediction; drop segments whose level deviates by
    more than max(k·MAD, min_dev) Nm from their neighbours (e.g. a sustained
    low-level contraction the EMG detector mistook for rest)."""
    keep = np.ones(len(tc), bool)
    if len(tc) < 4:
        return keep
    w = np.sqrt(np.minimum(n, 300.0))
    for _ in range(iters):
        idx = np.where(keep)[0]
        if len(idx) < 4:
            break
        pred = np.empty(len(idx))
        for j, i in enumerate(idx):
            o = idx[idx != i]
            pred[j] = _kernel_eval(np.array([tc[i]]), tc[o], med[o], w[o], sigma)[0]
        res = med[idx] - pred
        mad = 1.4826 * np.median(np.abs(res - np.median(res)))
        bad = np.abs(res) > max(k * mad, min_dev)
        if not bad.any():
            break
        keep[idx[bad]] = False
    return keep


def kernel_baseline(T, tc, med, n, sigma_s=8.0, reject=True):
    sigma = sigma_s * FS
    keep = reject_segment_outliers(tc, med, n, sigma) if reject else np.ones(len(tc), bool)
    w = np.sqrt(np.minimum(n[keep], 300.0))
    return _kernel_eval(np.arange(T, dtype=float), tc[keep], med[keep], w, sigma), keep


def loglin_baseline(T, tc, med, n, tau_s=5.0, t0_s=1.0, huber=1.5, iters=10):
    """Robust (Huber IRLS) fit  b(t) = a + c·ln(1 + (t + t0)/tau)  through the
    rest-segment levels (stress-relaxation shape; extrapolates sensibly)."""
    t = np.arange(T, dtype=float)
    f = lambda x: np.log1p((x / FS + t0_s) / tau_s)
    if len(tc) < 3:
        return np.full(T, np.median(med) if len(med) else 0.0)
    A = np.column_stack([np.ones_like(tc), f(tc)])
    w = np.sqrt(np.minimum(n, 300.0))
    ww = w.copy()
    for _ in range(iters):
        coef = np.linalg.lstsq(A * ww[:, None], med * ww, rcond=None)[0]
        r = med - A @ coef
        s = max(1.4826 * np.median(np.abs(r)), 1e-3)
        u = np.abs(r) / (huber * s)
        ww = w * np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-9))
    return coef[0] + coef[1] * f(t)


def causal_baseline(y, runs, T, b0=0.0, lag_s=0.3, min_run_s=0.5, pre_s=0.1,
                    tau_forget_s=10.0, every=5, n_cap=300):
    """Online tare.  A rest sample i (inside eroded run (s, e) of raw run
    (a, b)) becomes AVAILABLE at time  max(i + pre, a + min_run) + lag :
    erosion of the run end needs ``pre`` samples of future, confirmation needs
    the run to have lasted ``min_run``, and ``lag`` covers the look-ahead of
    the zero-phase envelope filter.  b(t) = Σ_k w_k m_k / Σ_k w_k over the
    available parts of past segments, m_k = median of the available samples,
    w_k = min(n_k, n_cap) · exp(−(t − t_last_k)/tau_forget).  b(t) = b0 until
    the first sample is available."""
    lag, pre, mr = int(lag_s * FS), int(pre_s * FS), int(min_run_s * FS)
    tau = max(tau_forget_s * FS, 1e-6)
    segs = []
    for s, e, a, b in runs:
        i = np.arange(s, e)
        avail = np.maximum(i + pre, a + mr) + lag     # non-decreasing in i
        segs.append((s, e, avail))
    K = len(segs)
    done_m = np.zeros(K); done_last = np.zeros(K); done_n = np.zeros(K)
    n_done, j = 0, 0          # segments fully available / first not-yet-complete
    out = np.empty(T)
    cur = b0
    for t0 in range(0, T, every):
        while j < K and segs[j][2][-1] <= t0:       # segment j now complete
            s, e, _ = segs[j]
            done_m[j], done_last[j], done_n[j] = np.median(y[s:e]), e - 1, e - s
            j += 1
            n_done = j
        num = den = 0.0
        if n_done:
            w = np.minimum(done_n[:n_done], n_cap) * np.exp(-(t0 - done_last[:n_done]) / tau)
            num, den = float(w @ done_m[:n_done]), float(w.sum())
        if j < K and segs[j][2][0] <= t0:           # partially available segment
            s, e, avail = segs[j]
            k = int(np.searchsorted(avail, t0, side='right'))
            wk = min(k, n_cap) * np.exp(-(t0 - (s + k - 1)) / tau)
            num += wk * float(np.median(y[s:s + k]))
            den += wk
        if den > 0:
            cur = num / den
        out[t0:t0 + every] = cur
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  Passive map from static holds only (passive trials, no active torque)
# ═════════════════════════════════════════════════════════════════════════════

def static_hold_entries(passive_trials, session, max_pos_std=0.005,
                        max_tq_std=1.0, min_dur_s=5.0, pos_tol=db._POSITION_TOLERANCE):
    """(position, torque) pairs from STATIC holds only: device-step split,
    then keep segments whose position std < max_pos_std rad and torque std <
    max_tq_std Nm (instead of the 20 Nm / |20 Nm| bounds, which admit the JM
    movement trial 21).  Clusters within pos_tol are averaged."""
    raw = []
    for df in passive_trials:
        sess = df.attrs.get('session')
        if (session == 'retest') != (sess == 'retest'):
            continue
        for seg in db._split_passive_segments(df):
            p = seg[db.POSITION_COL].values
            t = seg[db.TORQUE_COL].values
            if len(t) < min_dur_s * 1000 or p.std() > max_pos_std or t.std() > max_tq_std:
                continue
            raw.append((float(p.mean()), float(t.mean())))   # mean, as the pipeline
    raw.sort()
    cl = []
    for pos, tq in raw:
        if cl and abs(pos - cl[-1][0]) <= pos_tol:
            pp, tt, c = cl[-1]
            cl[-1] = ((pp * c + pos) / (c + 1), (tt * c + tq) / (c + 1), c + 1)
        else:
            cl.append((pos, tq, 1))
    return [(p, t) for p, t, _ in cl]


def static_maps(ctx):
    """Session passive maps: worksheet overrides when present (they equal the
    static-hold means), else static-hold entries; falls back to ctx maps."""
    from ML.config import PASSIVE_TORQUE_OVERRIDES
    ov = PASSIVE_TORQUE_OVERRIDES.get(ctx['subject'])
    if ov:
        return sorted(ov['test']), sorted(ov.get('retest', ov['test']))
    tm = static_hold_entries(ctx['passive_trials'], 'test')
    rm = static_hold_entries(ctx['passive_trials'], 'retest')
    if len(tm) < 2:
        tm = ctx['test_map']
    if len(rm) < 2:
        rm = tm
    return tm, rm


def _relax_mode(v):
    if v is True:
        return 'log'
    return v if v in ('log', 'lin') else None


def _gain_design(P, tsec, relax, tau_s, t0_s):
    """Columns [P, 1 (, relaxation basis × P)].  relax: None | 'log' (P·ln(1 +
    (t + t0)/τ)) | 'lin' (P·(t + t0)/90 s)."""
    cols = [P, np.ones_like(P)]
    if relax == 'log':
        cols.append(P * np.log1p((tsec + t0_s) / tau_s))
    elif relax == 'lin':
        cols.append(P * (tsec + t0_s) / 90.0)
    return np.column_stack(cols)


def fit_passive_gain(trainval, stats, ctx, relax=None, tau_s=5.0, t0_s=1.0,
                     huber=3.0, iters=10):
    """Fit  tq_rest ≈ α·P + β (+ γ·P·g(t))  on TRAINING-session rest
    segments (robust IRLS, one observation per rest segment, weight √n).
    Uses only training labels and the training passive map."""
    rows, ys, ws = [], [], []
    for r in trainval:
        runs = detect_rest(r["env"], stats, "offline", None)   # strict defaults
        if not runs:
            continue
        tc, med, n = seg_levels(r['tq'], runs)
        P = passive_curve(ctx['test_map'], r['pos'][tc.astype(int)])
        rows.append(_gain_design(P, tc / FS, relax, tau_s, t0_s))
        ys.append(med)
        ws.append(np.sqrt(np.minimum(n, 300.0)))
    A, y, w = np.vstack(rows), np.concatenate(ys), np.concatenate(ws)
    ww = w.copy()
    for _ in range(iters):
        coef = np.linalg.lstsq(A * ww[:, None], y * ww, rcond=None)[0]
        res = y - A @ coef
        s = max(1.4826 * np.median(np.abs(res)), 1e-3)
        u = np.abs(res) / (huber * s)
        ww = w * np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-9))
    return coef


# ═════════════════════════════════════════════════════════════════════════════
#  Plugins
# ═════════════════════════════════════════════════════════════════════════════

def _stats(trainval, ctx):
    if 'bd_stats' not in ctx:
        ctx['bd_stats'] = train_emg_stats(trainval)
    return ctx['bd_stats']


def _b0_fn(trainval, ctx, stats):
    """Initial tare before the first rest segment of a trial.
    bd_b0 = number (default 0 = trust the passive map) or 'train_level':
    the TRAINING-session rest level of the current target at the nearest
    operating angle (a per-angle calibration table built from training labels
    only, computed on the target as it is when this plugin runs)."""
    mode = _p(ctx, 'bd_b0', 0.0)
    if mode != 'train_level':
        return lambda r: float(mode)
    tab = {}
    for r in trainval:
        runs = detect_rest(r["env"], stats, "offline", None)   # strict defaults
        if runs:
            tab[float(np.mean(r['pos']))] = float(np.median(seg_levels(r['y'], runs)[1]))
    ctx['bd_b0_table'] = tab
    if not tab:
        return lambda r: 0.0
    keys = np.array(sorted(tab))
    return lambda r: tab[float(keys[np.argmin(np.abs(keys - np.mean(r['pos'])))])]


def passive_static(trainval, retest, ctx):
    """[TARGET via passive trials only] Static-hold passive maps (fixes JM M7:
    -0.071 rad entry +11.7 -> interpolated ≈ -2 Nm).  No active torque used."""
    tm, rm = static_maps(ctx)
    ctx['test_map'], ctx['retest_map'] = tm, rm
    for r in trainval + retest:
        pm = rm if r['session'] == 'retest' else tm
        r['passive'] = passive_curve(pm, r['pos'])
        r['y'] = r['tq'] - r['passive']


def passive_static_gain(trainval, retest, ctx):
    """[TARGET via passive map; NO retest torque; causal]  Static-hold maps,
    then P_eff = α·P + β (+ γ·P·g(t) if bd_gain_relax) with (α, β, γ) fitted
    on training-session rest segments.  Applied identically to both sessions
    with each session's own passive map.

    params: bd_gain_relax (False | 'lin' | 'log'), bd_gain_tau_s (5, for
    'log'), bd_gain_alpha (None = fit; a number forces α with β = γ = 0)."""
    passive_static(trainval, retest, ctx)
    stats = _stats(trainval, ctx)
    relax = _relax_mode(_p(ctx, 'bd_gain_relax', False))
    tau, t0 = _p(ctx, 'bd_gain_tau_s', 5.0), 1.0   # +1 s: harness trims 1 s
    fixed = _p(ctx, 'bd_gain_alpha', None)
    if fixed is None:
        coef = fit_passive_gain(trainval, stats, ctx, relax, tau, t0)
    else:
        coef = np.array([float(fixed), 0.0] + ([0.0] if relax else []))
    ctx['bd_gain_coef'] = [float(c) for c in coef]
    for r in trainval + retest:
        pm = ctx['retest_map'] if r['session'] == 'retest' else ctx['test_map']
        P = passive_curve(pm, r['pos'])
        tsec = np.arange(len(P)) / FS
        r['passive'] = _gain_design(P, tsec, relax, tau, t0) @ coef
        r['y'] = r['tq'] - r['passive']


def tare_causal(trainval, retest, ctx):
    """[TARGET; causal online tare from PAST rest torque]  y ← y − b(t), b
    from causally-detected rest segments (running noise floor, Schmitt
    trigger, confirmation + lag), exponential forgetting across segments.
    Compose after passive_static_gain for a better b before the first rest
    (b0 = 0 means 'trust the passive map').

    params: bd_tau_forget_s (10), bd_lag_s (0.3), bd_b0 (0 | 'train_level'),
    plus detector params bd_thr_lo/hi, bd_min_run_s, bd_settle_s, bd_pre_s,
    bd_floor_*."""
    stats = _stats(trainval, ctx)
    b0f = _b0_fn(trainval, ctx, stats)
    for r in trainval + retest:
        runs = detect_rest(r['env'], stats, 'causal', ctx)
        b = causal_baseline(r['y'], runs, len(r['y']), b0=b0f(r),
                            lag_s=_p(ctx, 'bd_lag_s', 0.3),
                            min_run_s=_p(ctx, 'bd_min_run_s', 0.5),
                            pre_s=_p(ctx, 'bd_pre_s', 0.1),
                            tau_forget_s=_p(ctx, 'bd_tau_forget_s', 10.0))
        r['bd_baseline'] = b
        r['y'] = r['y'] - b


def tare_start(trainval, retest, ctx):
    """[TARGET; causal single tare]  b = median of the first causally detected
    rest segment lasting ≥ bd_tare_min_s (1 s), applied from the moment that
    segment is complete (+lag); b0 before.  Constant afterwards (no drift
    tracking) — what 'zero the sensor at the start of the trial' gives."""
    stats = _stats(trainval, ctx)
    need = int(_p(ctx, 'bd_tare_min_s', 1.0) * FS)
    lag = int(_p(ctx, 'bd_lag_s', 0.3) * FS)
    b0f = _b0_fn(trainval, ctx, stats)
    for r in trainval + retest:
        T = len(r['y'])
        runs = detect_rest(r['env'], stats, 'causal', ctx)
        b = np.full(T, b0f(r), float)
        for s, e, a, bb in runs:
            if e - s >= need:
                # available once `need` samples (+pre erosion) have been seen
                t_av = min(s + need + int(_p(ctx, 'bd_pre_s', 0.1) * FS) + lag, T)
                b[t_av:] = float(np.median(r['y'][s:s + need]))
                break
        r['bd_baseline'] = b
        r['y'] = r['y'] - b


def rezero_smooth(trainval, retest, ctx):
    """[TARGET; offline, non-causal; uses the whole trial's EMG for floors]
    y ← y − b(t), b = Gaussian-kernel (bd_smooth='kernel', σ = bd_sigma_s,
    default 8 s) or robust log-time fit (bd_smooth='loglin') through the
    outlier-filtered medians of robustly detected rest segments.

    bd_apply='trainval' re-zeroes ONLY the training/validation trials (test
    target untouched): cleaner labels, no test-time label use."""
    stats = _stats(trainval, ctx)
    how = _p(ctx, 'bd_smooth', 'kernel')
    recs = trainval if _p(ctx, 'bd_apply', 'all') == 'trainval' else trainval + retest
    for r in recs:
        T = len(r['y'])
        runs = detect_rest(r['env'], stats, 'offline', ctx)
        tc, med, n = seg_levels(r['y'], runs)
        if len(tc) == 0:
            b = np.zeros(T)
        elif how == 'loglin':
            b = loglin_baseline(T, tc, med, n, tau_s=_p(ctx, 'bd_log_tau_s', 5.0))
        else:
            b, _ = kernel_baseline(T, tc, med, n, sigma_s=_p(ctx, 'bd_sigma_s', 8.0))
        r['bd_baseline'] = b
        r['y'] = r['y'] - b
