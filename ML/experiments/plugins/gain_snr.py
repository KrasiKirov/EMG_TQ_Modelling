"""
gain_snr — EMG-side robustness to session-to-session gain / muscle-mix change,
low SNR and electrode artefacts (mechanisms M3, M5, M4 of the dorsiflexion
investigation).  Target (y / y_ref) is NEVER touched by anything in here.

Running-statistic transforms (despike, snr_soft, agc*, agc_dual) compute
everything CAUSALLY from the current trial's own past (running percentiles,
updated every ``upd`` samples; the value at time t uses only samples strictly
before the last update <= t), blended at the start of a trial with a prior from
the TRAINING portion (first 80 %) of the test-session trials.  A real-time
decoder could compute exactly the same features (verified: corrupting a retest
trial's future leaves all earlier features bit-identical).  They DO use the
unlabeled retest EMG of the current trial's past beyond the 20-step window.
``despike_static`` uses fixed training thresholds only (not transductive).

Why this attacks the problem (JM numbers, train-max-normalised envelopes):
  * At JM -0.787 / -0.645 / -0.360 the retest plantarflexor envelope swing
    (p95 - p10 of the PF sum) is 0.69 / 0.36 / 0.43 x the test-session swing
    although the torque swing (fixed-%MVC step targets) is about the same, so
    the slope of target on summed PF envelope is 1.33 / 2.73 / 2.02 x larger
    in retest -> a model with an absolute EMG->torque gain under-predicts.
    YES goes the other way (retest EMG larger: slope ratio 0.63 at -0.486 and
    -0.382).
  * At -0.787 the push moves from GM (test, swing 0.49) to GL (retest, 0.42).
  * JM noise floor 0.15-0.25 of max (a test-session artefact sets the global
    max), active swing at plantarflexed angles only 3-10 sigma_noise.
  * JM -0.360 retest: GL signal drop-out + 5.9x spike at t = 53 s.

Plugins (preprocessing, fn(trainval, retest, ctx)):
  despike_static   fixed-threshold artefact removal (x > 2x train max or
                   drop-out below 0.25 x noise floor -> sample-and-hold).
  despike          same with running thresholds.
  snr_soft         despike + causal noise-floor soft threshold (absolute scale
                   kept) -- attacks M5/M4 but not the gain shift (proxy: no gain).
  agc              despike + per-channel causal adaptive gain control
                   clip((x - floor_t) / max(top_t - floor_t, rmin), 0, 2),
                   floor = running p10 (20 s), top = running p95 (45 s).
  agc_common       despike + ONE common gain for the three plantarflexors
                   (keeps the muscle mix) + per-channel AGC for TA.
  agc_dual         env = despiked absolute envelopes, extra = AGC envelopes
                   (9 features, position stays at column 4).
  gainaug          per-TRIAL gain augmentation of env columns 0-3 (copies of
                   the trainval records; extra/AGC view untouched).
  mixaug           per-TRIAL muscle-mix augmentation: near-identity
                   row-stochastic re-mix of the PF channels x per-channel gain.
                   Apply AFTER agc*.  (proxy: no measurable gain, 3x cost)

Model (reference as 'ML.experiments.plugins.gain_snr:lstm8_pfsum', also
registered as 'gain_snr_lstm8_pfsum'): baseline 2xLSTM(8) + a position-gated
skip path g(pos) * sum_PF x_t -- with AGC inputs the absolute torque scale
must come from position, this gives that multiplicative path explicitly.

Sample weights: none here; agc pairs well with core.pos_inv_var_weights
(AGC removes the absolute EMG scale, so the small-swing positions must be
fitted through position; without the weighting they are ignored by MSE).
"""

import numpy as np

from ML.preprocessing import dataset_builder as db
from ML.experiments.harness import register_model

PF = (0, 1, 2)   # gm, gl, sol
TA = 3


def _p(ctx, key, default):
    return ctx['opts'].get('params', {}).get(key, default)


# ── Causal running statistics ───────────────────────────────────────────────

def causal_quantiles(x, qs, win=None, upd=50, n0=500, prior=None):
    """Running percentiles of a 1-D signal, strictly causal.

    For update points u = 0, upd, 2·upd, … the percentiles ``qs`` of
    x[max(0, u−win):u] (past samples only; ``win=None`` → expanding) are
    computed and held for samples [u, u+upd).  The empirical value is blended
    with ``prior`` with weight n/(n+n0), n = number of past samples used, so the
    first seconds of a trial fall back on the training prior.
    Returns (T, len(qs))."""
    x = np.asarray(x, np.float64)
    qs = np.atleast_1d(qs)
    T = len(x)
    prior = np.zeros(len(qs)) if prior is None else np.asarray(prior, np.float64)
    out = np.empty((T, len(qs)))
    for u in range(0, T, upd):
        a = 0 if win is None else max(0, u - win)
        seg = x[a:u]
        n = len(seg)
        if n == 0:
            v = prior
        else:
            w = n / (n + n0)
            v = w * np.percentile(seg, qs) + (1.0 - w) * prior
        out[u:u + upd] = v
    return out


def _train_part(r):
    """Samples of a trainval record that end up in the TRAINING windows."""
    return r['env'][: int(len(r['env']) * db.TRAIN_FRAC)]


def _priors(trainval, fn, qs):
    """Median over training trials of percentiles of fn(train part of env)."""
    vals = []
    for r in trainval:
        if r.get('aug_copy'):
            continue
        sig = fn(_train_part(r))                       # (n,) or (n, k)
        vals.append(np.percentile(sig, qs, axis=0))    # (len(qs),) or (len(qs), k)
    return np.median(np.stack(vals), axis=0)


def _win(sec, fs):
    return None if sec is None else int(sec * fs)


def _cfg(ctx):
    fs = 100
    return dict(
        lo_q=_p(ctx, 'lo_q', 10.0),             # rest / noise-floor level
        hi_q=_p(ctx, 'hi_q', 95.0),             # active plateau level
        sig_q=_p(ctx, 'sig_q', 2.0),            # for noise σ: (p_lo − p_sig)
        lo_win=_win(_p(ctx, 'lo_win_s', 20.0), fs),   # None → expanding
        hi_win=_win(_p(ctx, 'hi_win_s', 45.0), fs),
        upd=int(_p(ctx, 'upd_s', 0.5) * fs),
        n0=int(_p(ctx, 'prior_s', 5.0) * fs),
        rmin_frac=_p(ctx, 'rmin_frac', 0.35),   # min range = frac × prior range
        clip_hi=_p(ctx, 'clip_hi', 2.0),
        spike=_p(ctx, 'spike_thr', 4.0),        # in units of running range
        hold=int(_p(ctx, 'spike_hold_s', 0.3) * fs),
        kappa=_p(ctx, 'soft_kappa', 1.0),       # soft threshold = floor + κσ
    )


# ── Artefact suppression (M4) ───────────────────────────────────────────────

def _despike_channel(x, lo, rng, thr, hold):
    """Causal: a sample is an artefact if (x − lo)/rng > thr, or if it is far
    BELOW a clearly non-zero noise floor (signal drop-out / filter ringing of
    an electrode pop: x < 0.25·lo with lo > 0.1 of train max).  Flagged runs,
    extended by ``hold`` samples, are replaced by the last good value
    (sample-and-hold) — no future samples are used."""
    z = (x - lo) / rng
    bad = z > thr
    drop = (x < 0.25 * lo) & (lo > 0.1)
    bad |= drop
    if not bad.any():
        return x, 0
    y = x.copy()
    last = lo[0]
    cnt = 0
    k_hold = 0
    for t in range(len(x)):
        if bad[t]:
            k_hold = hold
        if bad[t] or k_hold > 0:
            y[t] = last
            cnt += 1
            if not bad[t]:
                k_hold -= 1
        else:
            last = x[t]
    return y, cnt


def _running_stats(env, prior_lo, prior_hi, prior_sig, c):
    """Per-channel causal floor (lo), top (hi) and noise σ for a trial."""
    T, K = env.shape
    lo = np.empty((T, K)); hi = np.empty((T, K)); sg = np.empty((T, K))
    for k in range(K):
        q = causal_quantiles(env[:, k], [c['sig_q'], c['lo_q']], c['lo_win'],
                             c['upd'], c['n0'], [prior_sig[k], prior_lo[k]])
        lo[:, k] = q[:, 1]
        sg[:, k] = np.maximum(q[:, 1] - q[:, 0], 1e-4)
        hi[:, k] = causal_quantiles(env[:, k], [c['hi_q']], c['hi_win'], c['upd'],
                                    c['n0'], [prior_hi[k]])[:, 0]
    return lo, hi, sg


def _channel_priors(trainval, c):
    q = _priors(trainval, lambda e: e, [c['sig_q'], c['lo_q'], c['hi_q']])  # (3, K)
    return q[1], q[2], q[0]          # lo, hi, (lo − sig) handled below


def _despike_env(env, lo, hi, rmin, c):
    out = env.copy()
    n_bad = 0
    for k in range(env.shape[1]):
        rng = np.maximum(hi[:, k] - lo[:, k], rmin[k])
        out[:, k], n = _despike_channel(env[:, k], lo[:, k], rng, c['spike'], c['hold'])
        n_bad += n
    return out, n_bad


def _prep(trainval, retest, ctx):
    """Shared: priors, then per-trial running stats + despiked envelopes.
    Stores r['rs'] = dict(lo, hi, sg, env_ds) on every record."""
    c = _cfg(ctx)
    p_lo, p_hi, p_sig_q = _channel_priors(trainval, c)
    rmin = c['rmin_frac'] * np.maximum(p_hi - p_lo, 1e-3)
    ctx['gain_snr'] = {'cfg': c, 'prior_lo': p_lo, 'prior_hi': p_hi, 'rmin': rmin,
                       'n_despiked': {}}
    for r in trainval + retest:
        lo, hi, sg = _running_stats(r['env'], p_lo, p_hi, p_sig_q, c)
        env_ds, n_bad = _despike_env(r['env'], lo, hi, rmin, c)
        if n_bad:
            # recompute stats on the cleaned signal (artefact out of the percentiles)
            lo, hi, sg = _running_stats(env_ds, p_lo, p_hi, p_sig_q, c)
        r['rs'] = {'lo': lo, 'hi': hi, 'sg': sg, 'env_ds': env_ds}
        ctx['gain_snr']['n_despiked'][(r['session'], r['idx'])] = int(n_bad)
    return c, rmin


def _cleanup(trainval, retest):
    for r in trainval + retest:
        r.pop('rs', None)


# ── Plugins ─────────────────────────────────────────────────────────────────

def despike(trainval, retest, ctx):
    """[INPUT, causal] Artefact suppression only (hold-last-good on samples
    > spike_thr running ranges above the running floor, or signal drop-out).
    Uses running statistics of the current trial's past."""
    _prep(trainval, retest, ctx)
    for r in trainval + retest:
        r['env'] = r['rs']['env_ds']
    _cleanup(trainval, retest)


def despike_static(trainval, retest, ctx):
    """[INPUT, causal, NOT transductive] Fixed-threshold artefact suppression:
    env is in train-max units, so any sample above ``static_mult`` (default
    2.0 → twice the largest training envelope; YES retest legitimately reaches
    1.9 because its gain went UP) is an artefact; so is a
    signal drop-out x < 0.25·lo where lo = training-prior noise floor (median
    training p10, only when lo > 0.1).  Flagged runs (+0.3 s) are replaced by
    the last good sample (sample-and-hold).  Unlike core.clip_env the pop is
    removed, not flattened to a 1.2 plateau that still reads as a strong
    contraction; no statistic of other retest samples is used."""
    c = _cfg(ctx)
    p_lo, p_hi, _ = _channel_priors(trainval, c)
    thr = _p(ctx, 'static_mult', 2.0)
    ctx['gain_snr'] = {'cfg': c, 'prior_lo': p_lo, 'prior_hi': p_hi, 'n_despiked': {}}
    for r in trainval + retest:
        out = r['env'].copy()
        n_bad = 0
        for k in range(out.shape[1]):
            out[:, k], n = _despike_abs(out[:, k], p_lo[k], thr, c['hold'])
            n_bad += n
        r['env'] = out
        ctx['gain_snr']['n_despiked'][(r['session'], r['idx'])] = int(n_bad)


def _despike_abs(x, lo, thr, hold):
    """Fixed-threshold version of _despike_channel (x > thr or drop-out)."""
    bad = (x > thr) | ((x < 0.25 * lo) & (lo > 0.1))
    if not bad.any():
        return x, 0
    y = x.copy()
    last, cnt, k_hold = lo, 0, 0
    for t in range(len(x)):
        if bad[t]:
            k_hold = hold
        if bad[t] or k_hold > 0:
            y[t] = last
            cnt += 1
            if not bad[t]:
                k_hold -= 1
        else:
            last = x[t]
    return y, cnt


def snr_soft(trainval, retest, ctx):
    """[INPUT, causal] despike + noise-floor-aware soft threshold:
        env' = max(env − floor_t − κ·σ_t, 0) · s
    floor_t = running lo_q-percentile (20 s window), σ_t = running
    (p_lo − p_2) spread, s = 1 / (median training (p95 − floor)) so the
    feature is O(1).  Absolute EMG scale is kept (no gain normalisation): a
    clean test of how much of the JM loss is noise / floor drift alone."""
    c, rmin = _prep(trainval, retest, ctx)
    vals = []
    for r in trainval + retest:
        rs = r['rs']
        r['env'] = np.maximum(rs['env_ds'] - rs['lo'] - c['kappa'] * rs['sg'], 0.0)
        if r['split'] == 'trainval':
            vals.append(np.percentile(r['env'][: int(len(r['env']) * db.TRAIN_FRAC)], 95, axis=0))
    s = 1.0 / np.maximum(np.median(np.stack(vals), axis=0), 1e-3)
    for r in trainval + retest:
        r['env'] = np.clip(r['env'] * s, 0.0, 3.0)
    ctx['gain_snr']['soft_scale'] = s
    _cleanup(trainval, retest)


def agc(trainval, retest, ctx):
    """[INPUT, causal, uses past EMG of the current trial] Per-channel adaptive
    gain control:  a_k = clip((env_k − lo_k,t) / max(hi_k,t − lo_k,t, rmin_k),
    0, clip_hi), lo = running p10 (20 s), hi = running p95 (45 s).  Removes any
    multiplicative per-channel gain change between sessions."""
    c, rmin = _prep(trainval, retest, ctx)
    for r in trainval + retest:
        rs = r['rs']
        rng = np.maximum(rs['hi'] - rs['lo'], rmin[None, :])
        r['env'] = np.clip((rs['env_ds'] - rs['lo']) / rng, 0.0, c['clip_hi'])
    _cleanup(trainval, retest)


def agc_common(trainval, retest, ctx):
    """[INPUT, causal, uses past EMG of the current trial] Common-gain AGC for
    the plantarflexors:  drive_t = Σ_PF max(env_k − lo_k,t, 0);  each PF channel
    becomes max(env_k − lo_k, 0) / max(hi_drive,t, rmin_drive) (so the three
    still sum to the normalised drive and the muscle mix is preserved, but no
    single silent channel gets its noise blown up); TA gets its own AGC."""
    c, rmin = _prep(trainval, retest, ctx)
    # prior for the drive top (from training parts, floor = per-trial p10)
    d_hi = []
    for r in trainval:
        n = int(len(r['env']) * db.TRAIN_FRAC)
        rs = r['rs']
        d = np.maximum(rs['env_ds'][:n, PF] - rs['lo'][:n, PF], 0).sum(1)
        d_hi.append(np.percentile(d, c['hi_q']))
    p_dhi = float(np.median(d_hi))
    rmin_d = c['rmin_frac'] * p_dhi
    ctx['gain_snr']['prior_drive_hi'] = p_dhi
    for r in trainval + retest:
        rs = r['rs']
        act = np.maximum(rs['env_ds'][:, PF] - rs['lo'][:, PF], 0.0)
        drive = act.sum(1)
        dh = causal_quantiles(drive, [c['hi_q']], c['hi_win'], c['upd'], c['n0'],
                              [p_dhi])[:, 0]
        den = np.maximum(dh, rmin_d)
        out = np.empty_like(r['env'])
        out[:, list(PF)] = np.clip(act / den[:, None], 0.0, c['clip_hi'])
        rng_ta = np.maximum(rs['hi'][:, TA] - rs['lo'][:, TA], rmin[TA])
        out[:, TA] = np.clip((rs['env_ds'][:, TA] - rs['lo'][:, TA]) / rng_ta, 0.0, c['clip_hi'])
        r['env'] = out
    _cleanup(trainval, retest)


def _append_extra(r, ex):
    r['extra'] = ex if r.get('extra') is None else np.column_stack([r['extra'], ex])


def agc_dual(trainval, retest, ctx):
    """[INPUT, causal, uses past EMG of the current trial] Both views:
    env   = despiked absolute envelopes (train-max units, clipped to [0, 1.5])
    extra = per-channel AGC envelopes (as in ``agc``)  → 9 features, position
    stays at column 4.  Alone the model may use either view; follow with
    ``gainaug`` so the absolute view becomes gain-unreliable and the model
    leans on the gain-invariant one while keeping coarse absolute level
    (needed where the retest torque swing itself is smaller, e.g. YES −0.486)."""
    c, rmin = _prep(trainval, retest, ctx)
    for r in trainval + retest:
        rs = r['rs']
        rng = np.maximum(rs['hi'] - rs['lo'], rmin[None, :])
        a = np.clip((rs['env_ds'] - rs['lo']) / rng, 0.0, c['clip_hi'])
        r['env'] = np.clip(rs['env_ds'], 0.0, _p(ctx, 'abs_clip', 1.5))
        _append_extra(r, a)
    _cleanup(trainval, retest)


def gainaug(trainval, retest, ctx):
    """[TRAIN INPUT augmentation, per TRIAL] Append ``gain_copies`` copies of
    every trainval record with the 4 env channels (columns 0-3 only, i.e. the
    absolute view in ``agc_dual``; ``extra`` is copied unchanged — the AGC view
    is exactly invariant to a multiplicative channel gain) multiplied by
    exp(U(−gain_common, gain_common)) · exp(U(−gain_ch, gain_ch)) per channel,
    one draw per copy.  Simulates a new session's electrode gain.  Target and
    retest untouched; copies keep their validation time range (no leakage)."""
    k = int(_p(ctx, 'gain_copies', 2))
    gc, gch = _p(ctx, 'gain_common', 0.4), _p(ctx, 'gain_ch', 0.3)
    rng = np.random.default_rng(_p(ctx, 'gain_seed', 1))
    base = [r for r in trainval if not r.get('aug_copy')]
    new = []
    for j in range(k):
        for r in base:
            cp = {kk: (v.copy() if isinstance(v, np.ndarray) else v) for kk, v in r.items()}
            g = np.exp(rng.uniform(-gc, gc)) * np.exp(rng.uniform(-gch, gch, size=4))
            cp['env'] = cp['env'] * g[None, :]
            cp['aug_copy'] = True
            cp['idx'] = int(r['idx']) + 1000 * (j + 1) + 100
            cp['aug_gain'] = g
            new.append(cp)
    trainval.extend(new)


def mixaug(trainval, retest, ctx):
    """[TRAIN INPUT augmentation, per TRIAL] Append ``mix_copies`` copies of
    every trainval record whose plantarflexor channels are re-mixed by a
    random near-identity row-stochastic matrix M (rows ~ Dirichlet(α·I + 1),
    i.e. part of each muscle's activity is attributed to the others) and then
    scaled by a per-channel gain exp(U(−g, g)).  The target is unchanged.
    One draw per copy = per trial (not per window), so within a copy the
    EMG→torque relation is internally consistent — like a new session.
    Validation windows of the copies come from the validation time range, so
    there is no train→val leakage.  Retest records are never touched."""
    k = int(_p(ctx, 'mix_copies', 2))
    alpha = _p(ctx, 'mix_alpha', 6.0)        # larger → closer to identity
    g = _p(ctx, 'mix_gain', 0.3)             # log-gain half-range
    rng = np.random.default_rng(_p(ctx, 'mix_seed', 0))
    base = [r for r in trainval if not r.get('aug_copy')]
    new = []
    for j in range(k):
        for r in base:
            cp = {kk: (v.copy() if isinstance(v, np.ndarray) else v) for kk, v in r.items()}
            M = np.stack([rng.dirichlet(np.ones(3) + alpha * np.eye(3)[i]) for i in range(3)])
            gains = np.exp(rng.uniform(-g, g, size=3))
            e = cp['env'].copy()
            pf = e[:, list(PF)]
            e[:, list(PF)] = (pf @ M.T) * gains[None, :]
            cp['env'] = e
            cp['aug_copy'] = True
            cp['idx'] = int(r['idx']) + 1000 * (j + 1)
            cp['aug_M'] = M
            cp['aug_gain'] = gains
            new.append(cp)
    trainval.extend(new)


def gainaug_dual(trainval, retest, ctx):
    """[TRAIN INPUT augmentation, per TRIAL] Faithful gain augmentation for
    ``agc_dual`` (must directly follow it).  Like ``gainaug`` (same RNG draws,
    same parameters) but each copy's BOTH views are recomputed from the
    gain-scaled pre-plugin envelopes (env_raw / emg_max) with the SAME training
    priors / rmin: abs view = clip(despike(env·g), 0, abs_clip), AGC view =
    agc(env·g).  That is exactly what this feature pipeline produces for a
    session whose electrode gain changed by g.

    Why not copy the AGC view unchanged (``gainaug``): it is NOT gain-invariant.
    The running floor/top keep a 20 % / 10 % blend with the absolute training
    prior (n0 / (win + n0)), the range is floored at an absolute rmin and the
    drop-out rule uses an absolute lo > 0.1 test.  Measured: under a 0.5x gain
    the top-20 % AGC values keep 0.2-0.9 of their size (JM GM at -0.787 retest
    0.21; YES GM/GL at -0.486 0.51/0.44), i.e. most at the low-activation
    angles we care about.  Copying it unchanged trains on (abs·g, agc(abs))
    pairs that never occur at test time and teaches a false invariance."""
    gs = ctx.get('gain_snr')
    if not gs or 'rmin' not in gs:
        raise RuntimeError('gainaug_dual must follow agc_dual')
    c, rmin = gs['cfg'], gs['rmin']
    k = int(_p(ctx, 'gain_copies', 2))
    gc, gch = _p(ctx, 'gain_common', 0.4), _p(ctx, 'gain_ch', 0.3)
    abs_clip = _p(ctx, 'abs_clip', 1.5)
    rng = np.random.default_rng(_p(ctx, 'gain_seed', 1))
    base = [r for r in trainval if not r.get('aug_copy')]
    env_in = [r['env_raw'] / ctx['emg_max'] for r in base]
    # training priors exactly as agc_dual computed them (train part, unscaled input)
    q = np.median(np.stack([
        np.percentile(e[: int(len(e) * db.TRAIN_FRAC)], [c['sig_q'], c['lo_q'], c['hi_q']], axis=0)
        for e in env_in]), axis=0)
    p_sq, p_lo, p_hi = q[0], q[1], q[2]
    if not (np.allclose(p_lo, gs['prior_lo']) and np.allclose(p_hi, gs['prior_hi'])):
        raise RuntimeError('gainaug_dual: priors differ from agc_dual (plugin order?)')

    def views(e):
        lo, hi, _ = _running_stats(e, p_lo, p_hi, p_sq, c)
        ds, nb = _despike_env(e, lo, hi, rmin, c)
        if nb:
            lo, hi, _ = _running_stats(ds, p_lo, p_hi, p_sq, c)
        a = np.clip((ds - lo) / np.maximum(hi - lo, rmin[None, :]), 0.0, c['clip_hi'])
        return np.clip(ds, 0.0, abs_clip), a

    for r, e in zip(base, env_in):   # unscaled recomputation must reproduce agc_dual
        e0, a0 = views(e)
        if not (np.allclose(e0, r['env']) and np.allclose(a0, np.asarray(r['extra'])[:, -4:])):
            raise RuntimeError('gainaug_dual: agc_dual input was not env_raw/emg_max')
    new = []
    for j in range(k):
        for r, e in zip(base, env_in):
            cp = {kk: (v.copy() if isinstance(v, np.ndarray) else v) for kk, v in r.items()}
            g = np.exp(rng.uniform(-gc, gc)) * np.exp(rng.uniform(-gch, gch, size=4))
            e_abs, a = views(e * g[None, :])
            ex = np.array(cp['extra'], dtype=np.float64, copy=True)
            ex[:, -4:] = a
            cp['env'], cp['extra'] = e_abs, ex
            cp['aug_copy'] = True
            cp['idx'] = int(r['idx']) + 1000 * (j + 1) + 200
            cp['aug_gain'] = g
            new.append(cp)
    trainval.extend(new)


# ── Model ───────────────────────────────────────────────────────────────────

@register_model('gain_snr_lstm8_pfsum')
def lstm8_pfsum(n_steps, n_features, **kw):
    """Baseline 2×LSTM(8) + a position-gated skip path from the summed
    plantarflexor drive at the last step:  ŷ = LSTM(...) + g(pos)·Σ_PF x_t.
    Encourages the model to use the (gain-normalised) combined drive rather
    than session-specific single-muscle weights."""
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    inp = layers.Input(shape=(n_steps, n_features))
    h = layers.LSTM(8, return_sequences=True)(inp)
    h = layers.Dropout(0.3)(h)
    h = layers.LSTM(8)(h)
    h = layers.Dropout(0.3)(h)
    lstm_out = layers.Dense(1)(h)
    pfsum = layers.Lambda(lambda x: tf.reduce_sum(x[:, -1, 0:3], axis=-1, keepdims=True))(inp)
    pos = layers.Lambda(lambda x: x[:, -1, 4:5])(inp)
    gpos = layers.Dense(1)(layers.Dense(8, activation='tanh')(pos))
    skip = layers.Multiply()([gpos, pfsum])
    return Model(inp, layers.Add()([lstm_out, skip]), name='gain_snr_lstm8_pfsum')
