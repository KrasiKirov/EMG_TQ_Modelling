"""
Model / loss variants for the dorsiflexion R² investigation (designer: model_loss).

Target mechanism: M6 — with a plain MSE in Nm the loss is dominated by the
large-torque angles (YES: std 0.4–0.7 Nm at −0.486 rad vs 13–15 Nm at
+0.03…+0.14 → 0.06 % of the training loss, ~1/500 of +0.137's share), so
the small-range angles are fitted last and worst.  The user's metric (mean of per-position R²) is
    R²_p = 1 − MSE_p / Var_p
so the metric-aligned training objective is   Σ_p MSE_p / Var_p,
i.e. an MSE on the *per-position standardised* target.

Design — "position-lookup de-standardisation inside the model"
---------------------------------------------------------------
Plugin ``posstats`` (must be the LAST plugin) computes, from the TRAIN split
of the test (= training) session only, per operating angle:
    mu_p, sd_p   mean / std of the target
    lo_p, hi_p   0.5 / 99.5 percentiles of the target
and appends five constant-per-angle columns to every window
    extra = [pos_k, mu_k, sd_k, lo_k, hi_k]
where k = nearest training angle to the *instantaneous* measured ankle angle
(default ``ml_lookup='nearest'``; ``'linear'`` = piecewise-linear lookup,
log-linear for sd, and pos_k = raw angle).  These are a deterministic function
of an input sensor (ankle angle) and of training labels only → causal,
online-feasible, no test target, not transductive.

Why 'nearest' by default: the within-trial angle excursion is a mechanical
torque read-out (footplate/ankle compliance, 2000–5700 Nm/rad; a linear
y ~ pos fit explains R² 0.93–0.99 of the torque within a trial at YES ≥ −0.28
and JM ≥ −0.22 rad, but only 0.04–0.19 at YES −0.486 and JM −0.79…−0.50).
A linear lookup would leak that torque proxy into the output scale; the
nearest-knot lookup is constant within a trial (except ~0.01 % edge samples).

Every model here strips those five columns from the recurrent input, predicts
a standardised value z and de-standardises in-graph:
    ŷ[Nm] = mu(pos) + sd(pos) · z
so the network output, ``model.predict`` and therefore ``harness.evaluate``
(including the y_ref/test_offset 'ref' metrics) stay in Nm — nothing in the
harness has to know about the standardisation.

The loss is computed in standardised units by overriding
``keras.Model.compute_loss`` (it receives ``x``, so it can read sd from the
window).  It is used for training AND validation, so EarlyStopping /
ReduceLROnPlateau select the epoch on the metric-aligned, position-balanced
loss (the core ``pos_inv_var_weights`` sample weights only re-weight
training; val_loss there is still Nm-MSE and is dominated by large angles).
The compiled ``loss='mse'`` string from the harness is ignored by these
models.  Validation is always the standardised MSE (``ml_gamma_val``), so
val_loss is comparable across the variants below.

Hyper-parameters are read from ``opts['params']`` (keys ``ml_*``; recorded in
the result JSON), overridable by ``model_kw``.  ``posstats`` stashes them and
the training-angle knots in the module-level ``_STATE`` (prepare() runs the
plugin in the same process before ``get_model`` is called).

Variants (builders, use as model 'ML.experiments.plugins.model_loss:<fn>'):
    posnorm_lstm      lstm8 backbone, de-standardised output, standardised MSE
    posnorm_huber     same, standardised Huber (δ = 1 sd) → robust to the
                      un-modelable within-trial baseline drift (M2)
    moe_pos           shared LSTM(16) trunk + shared head + per-angle expert
                      heads gated by a fixed hat basis over training angles
    linres_bounded    per-angle linear EMG model + zero-initialised LSTM
                      residual + soft output bound to the training range of
                      that angle (artefact / gain blow-up containment, M3/M4)
    bnll              heteroscedastic β-NLL (β = 0.5) in standardised units;
                      predict() returns the mean only
    posnorm_knotpos   posnorm_lstm, but the network sees the nearest training
                      angle instead of the raw angle → removes the compliance
                      shortcut that exists only at the large-torque angles, so
                      the EMG pathway must carry every angle (and retest
                      angle-calibration offsets × stiffness cannot leak in)
    lstm8_knotpos     DIAGNOSTIC ablation: the exact baseline lstm8 + Nm MSE
                      (compiled loss), only the angle input replaced by pos_k

None of these changes the target (train or test); none uses retest
statistics; all are causal (window of past EMG + current angle).
"""

import numpy as np

from ML.preprocessing import dataset_builder as db
from ML.experiments.harness import register_model

N_STATS = 5          # appended columns: [pos_k, mu, sd, lo, hi]
I_POS, I_MU, I_SD, I_LO, I_HI = range(N_STATS)
_STATE = {}          # filled by posstats(): knots, per-knot stats, params


def _p(ctx, key, default):
    return ctx['opts'].get('params', {}).get(key, default)


# ── Piecewise-linear lookup over the training angles ────────────────────────

def hat_basis_np(p, knots):
    """(T,) positions → (T, K) hat-function weights (rows sum to 1, clamped
    to the end knots).  Same maths as the in-graph ``HatGate`` layer."""
    p = np.asarray(p, np.float64).reshape(-1, 1)
    k = np.asarray(knots, np.float64)
    if len(k) == 1:
        return np.ones((len(p), 1))
    pc = np.clip(p, k[0], k[-1])
    kl = np.concatenate([[k[0] - 1.0], k[:-1]]); dl = k - kl
    kr = np.concatenate([k[1:], [k[-1] + 1.0]]); dr = kr - k
    left = np.clip((pc - kl) / dl, 0.0, 1.0)
    right = np.clip((kr - pc) / dr, 0.0, 1.0)
    return np.minimum(left, right)


def _train_slice(r, window):
    """Samples whose windows form the TRAIN split (first TRAIN_FRAC of the
    trial's windows) — exactly the y_train targets of this record."""
    n = len(r['y']) - window + 1
    n_tr = int(n * db.TRAIN_FRAC)
    return slice(window - 1, window - 1 + n_tr)


# ── Plugin: per-position statistics → extra columns ─────────────────────────

def posstats(trainval, retest, ctx):
    """Append [pos_k, mu_k, sd_k, lo_k, hi_k] to r['extra'].

    Statistics come from the TRAIN split of the training session only
    (val windows and the whole retest session are never used).  Lookup is at
    the instantaneous measured angle → causal.  Target untouched.
    Params: ml_lookup ('nearest' | 'linear'), ml_sd_floor (0.1 Nm),
            ml_q (0.5 → 0.5/99.5 percentiles),
            ml_merge_rad (0.02: trials closer than this share a knot).
    """
    o = ctx['opts']
    plugins = o.get('plugins', [])
    if plugins and not plugins[-1].endswith(':posstats'):
        raise ValueError('model_loss:posstats must be the LAST plugin '
                         '(models read the last %d feature columns)' % N_STATS)
    W = o['window']
    floor = float(_p(ctx, 'ml_sd_floor', 0.1))
    q = float(_p(ctx, 'ml_q', 0.5))
    merge = float(_p(ctx, 'ml_merge_rad', 0.02))
    lookup = _p(ctx, 'ml_lookup', 'nearest')
    if lookup not in ('nearest', 'linear'):
        raise ValueError(f'ml_lookup={lookup!r}')

    # group training-split samples by operating angle
    items = []
    for r in trainval:
        sl = _train_slice(r, W)
        items.append((float(np.mean(r['pos'][sl])), r['y'][sl]))
    items.sort(key=lambda t: t[0])
    groups = []                       # [[pos list], [y arrays]]
    for p, y in items:
        if groups and p - groups[-1][0][-1] < merge:
            groups[-1][0].append(p); groups[-1][1].append(y)
        else:
            groups.append([[p], [y]])
    knots = np.array([np.mean(g[0]) for g in groups])
    ys = [np.concatenate(g[1]) for g in groups]
    mu = np.array([y.mean() for y in ys])
    sd = np.array([max(y.std(), floor) for y in ys])
    lo = np.array([np.percentile(y, q) for y in ys])
    hi = np.array([np.percentile(y, 100.0 - q) for y in ys])

    for r in trainval + retest:
        if lookup == 'nearest':
            k = np.argmin(np.abs(r['pos'][:, None] - knots[None, :]), axis=1)
            ex = np.column_stack([knots[k], mu[k], sd[k], lo[k], hi[k]])
        else:
            B = hat_basis_np(r['pos'], knots)
            ex = np.column_stack([r['pos'], B @ mu, np.exp(B @ np.log(sd)),
                                  B @ lo, B @ hi])
        r['extra'] = ex if r.get('extra') is None else np.column_stack([r['extra'], ex])

    stats = {'knots': knots, 'mu': mu, 'sd': sd, 'lo': lo, 'hi': hi,
             'lookup': lookup}
    ctx['posstats'] = stats
    _STATE.clear()
    _STATE.update(stats)
    _STATE['subject'] = ctx['subject']
    _STATE['params'] = dict(o.get('params', {}))


# ── Keras pieces (defined lazily so the plugin stays importable w/o TF) ─────

_CLS = {}


def _classes():
    if _CLS:
        return _CLS
    import keras
    from keras import layers, ops

    class SplitStats(layers.Layer):
        """x (B,W,F) → core for the network, stats (B,n) at the last step
        (the step the target belongs to).
        pos_input='raw' : core = x[..., :F−n]  (EMG, raw angle, other extras)
        pos_input='knot': raw angle (col 4) replaced by the nearest training
                          angle pos_k (no within-trial excursion)."""
        def __init__(self, n_stats=N_STATS, pos_input='raw', **kw):
            super().__init__(**kw)
            if pos_input not in ('raw', 'knot'):
                raise ValueError(pos_input)
            self.n_stats, self.pos_input = n_stats, pos_input

        def call(self, x):
            n = self.n_stats
            st = x[:, -1, -n:]
            if self.pos_input == 'raw':
                core = x[:, :, :-n]
            else:
                pk = x[:, :, -n + I_POS:-n + I_POS + 1]
                parts = [x[:, :, :4], pk]
                if x.shape[-1] - n > 5:
                    parts.append(x[:, :, 5:-n])
                core = ops.concatenate(parts, axis=-1)
            return core, st

    class EmgSummary(layers.Layer):
        """Last-step and window-mean of the 4 EMG envelopes → (B, 8)."""
        def call(self, x):
            e = x[:, :, :4]
            return ops.concatenate([e[:, -1, :], ops.mean(e, axis=1)], axis=-1)

    class HatGate(layers.Layer):
        """Fixed hat-basis gate over the training angles (B,n) → (B,K)."""
        def __init__(self, knots, **kw):
            super().__init__(**kw)
            k = np.asarray(knots, np.float32)
            self.k = k
            kl = np.concatenate([[k[0] - 1.0], k[:-1]]).astype(np.float32)
            kr = np.concatenate([k[1:], [k[-1] + 1.0]]).astype(np.float32)
            self.kl, self.dl = kl, (k - kl).astype(np.float32)
            self.kr, self.dr = kr, (kr - k).astype(np.float32)

        def call(self, st):
            p = st[:, I_POS:I_POS + 1]
            if len(self.k) == 1:
                return ops.ones_like(p)
            pc = ops.clip(p, float(self.k[0]), float(self.k[-1]))
            left = ops.clip((pc - self.kl) / self.dl, 0.0, 1.0)
            right = ops.clip((self.kr - pc) / self.dr, 0.0, 1.0)
            return ops.minimum(left, right)

    class GatedSum(layers.Layer):
        def call(self, inputs):
            zk, g = inputs
            return ops.sum(zk * g, axis=-1, keepdims=True)

    class DeStd(layers.Layer):
        """z (B,1) standardised → Nm:  mu(pos) + sd(pos)·z."""
        def call(self, inputs):
            z, st = inputs
            return st[:, I_MU:I_MU + 1] + st[:, I_SD:I_SD + 1] * z

    class SoftBound(layers.Layer):
        """Soft clip of z to the training range of the current angle:
        z_b = c + r·tanh((z − c)/r), c = centre of [zlo, zhi],
        r = half-range·(1 + 2·frac) + abs   (std units).  Monotone, slope 1
        at the centre; |ŷ − mu| can never exceed sd·(|c| + r)."""
        def __init__(self, frac=0.25, abs_margin=0.5, **kw):
            super().__init__(**kw)
            self.frac, self.abs_margin = float(frac), float(abs_margin)

        def call(self, inputs):
            z, st = inputs
            mu, sd = st[:, I_MU:I_MU + 1], st[:, I_SD:I_SD + 1]
            zlo = (st[:, I_LO:I_LO + 1] - mu) / sd
            zhi = (st[:, I_HI:I_HI + 1] - mu) / sd
            c = 0.5 * (zlo + zhi)
            r = 0.5 * (zhi - zlo) * (1.0 + 2.0 * self.frac) + self.abs_margin
            return c + r * ops.tanh((z - c) / r)

    class StdLossModel(keras.Model):
        """Functional model whose loss is computed in per-position
        standardised units (sd read from the input window).

        train: loss_kind ∈ {'mse','huber','bnll'}, weighted by sd^(2−γ)
               (γ = 2: pure standardisation; γ = 1 ≙ 1/std weights)
        val  : standardised MSE with γ_val (default 2) for every variant
        Output column 0 is the Nm prediction; extra columns (bnll log-var)
        are dropped by predict()."""
        def __init__(self, inputs, outputs, loss_kind='mse', gamma=2.0,
                     gamma_val=2.0, delta=1.0, beta=0.5, logv_clip=(-7.0, 5.0),
                     name=None):
            super().__init__(inputs=inputs, outputs=outputs, name=name)
            self.ml_cfg = dict(loss_kind=loss_kind, gamma=float(gamma),
                               gamma_val=float(gamma_val), delta=float(delta),
                               beta=float(beta), logv_clip=tuple(logv_clip))

        def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None,
                         training=True):
            c = self.ml_cfg
            sd = x[:, -1, -N_STATS + I_SD]
            yt = ops.reshape(ops.cast(y, y_pred.dtype), (-1,))
            e = (yt - y_pred[:, 0]) / sd
            kind = c['loss_kind'] if training else 'mse'
            gamma = c['gamma'] if training else c['gamma_val']
            if kind == 'mse':
                l = e * e
            elif kind == 'huber':
                d = c['delta']
                a = ops.abs(e)
                l = 2.0 * ops.where(a <= d, 0.5 * e * e, d * (a - 0.5 * d))
            elif kind == 'bnll':
                lv = ops.clip(y_pred[:, 1], c['logv_clip'][0], c['logv_clip'][1])
                v = ops.exp(lv)
                l = (e * e / v + lv) * ops.stop_gradient(v) ** c['beta']
            else:
                raise ValueError(kind)
            if gamma != 2.0:
                l = l * sd ** (2.0 - gamma)
            if sample_weight is not None:
                l = l * ops.reshape(ops.cast(sample_weight, l.dtype), (-1,))
            loss = ops.mean(l)
            for r in self.losses:
                loss = loss + ops.sum(ops.cast(r, loss.dtype))
            return loss

        def predict_step(self, data):
            return super().predict_step(data)[:, :1]

    _CLS.update(keras=keras, layers=layers, ops=ops, SplitStats=SplitStats,
                EmgSummary=EmgSummary, HatGate=HatGate, GatedSum=GatedSum,
                DeStd=DeStd, SoftBound=SoftBound, StdLossModel=StdLossModel)
    return _CLS


# ── Builder helpers ─────────────────────────────────────────────────────────

def _cfg(kw, defaults):
    """defaults ← opts['params'] (recorded by posstats) ← model_kw."""
    P = dict(defaults)
    params = _STATE.get('params', {})
    for k in defaults:
        if k in params:
            P[k] = params[k]
    for k, v in kw.items():
        if k in defaults:
            P[k] = v
    return P


def _check(n_features, need_knots=False):
    if n_features < 5 + N_STATS:
        raise ValueError(f'n_features={n_features}: add '
                         f'"ML.experiments.plugins.model_loss:posstats" as the '
                         f'last plugin (models need {N_STATS} stats columns)')
    if need_knots and 'knots' not in _STATE:
        raise RuntimeError('posstats plugin has not run in this process')


def _lstm_trunk(core, units, dropout):
    L = _classes()['layers']
    h = L.LSTM(units, activation='tanh', return_sequences=True)(core)
    h = L.Dropout(dropout)(h)
    h = L.LSTM(units, activation='tanh')(h)
    return L.Dropout(dropout)(h)


def _model(inp, out, P, name, loss_kind=None):
    C = _classes()
    return C['StdLossModel'](inp, out, loss_kind=loss_kind or P.get('ml_loss', 'mse'),
                             gamma=P.get('ml_gamma', 2.0),
                             gamma_val=P.get('ml_gamma_val', 2.0),
                             delta=P.get('ml_huber_delta', 1.0),
                             beta=P.get('ml_beta', 0.5), name=name)


# ── Variants 1, 2, 6: posnorm family ────────────────────────────────────────

def _posnorm(n_steps, n_features, kw, name, loss_kind, pos_input):
    _check(n_features)
    P = _cfg(kw, {'ml_units': 8, 'ml_dropout': 0.3, 'ml_gamma': 2.0,
                  'ml_gamma_val': 2.0, 'ml_huber_delta': 1.0,
                  'ml_pos_input': pos_input})
    C = _classes(); L = C['layers']
    inp = L.Input(shape=(n_steps, n_features))
    core, st = C['SplitStats'](pos_input=P['ml_pos_input'])(inp)
    h = _lstm_trunk(core, int(P['ml_units']), float(P['ml_dropout']))
    z = L.Dense(1)(h)
    return _model(inp, C['DeStd']()([z, st]), P, name, loss_kind=loss_kind)


@register_model('model_loss_posnorm')
def posnorm_lstm(n_steps, n_features, **kw):
    """Baseline lstm8 backbone (same inputs: EMG + raw angle);
    ŷ = mu_k + sd_k·z; standardised MSE in train AND val.  Isolates the
    loss / output-parametrisation change (M6)."""
    return _posnorm(n_steps, n_features, kw, 'posnorm_lstm', 'mse', 'raw')


@register_model('model_loss_posnorm_huber')
def posnorm_huber(n_steps, n_features, **kw):
    """posnorm_lstm with a standardised Huber loss (δ = 1 position-sd).
    Standardisation amplifies the un-modelable within-trial baseline drift of
    the small-range angles (M2): at YES −0.486 the EMG-silent level moves
    −0.61 sd between the first and last third of the trial and the rest-period
    spread is 0.33 sd, vs ≤ 0.05 / 0.03 sd at every other YES angle.  Huber
    caps the gradient of those residuals so the network is not pushed to
    explain drift with spurious EMG features."""
    return _posnorm(n_steps, n_features, kw, 'posnorm_huber', 'huber', 'raw')


@register_model('model_loss_posnorm_knotpos')
def posnorm_knotpos(n_steps, n_features, **kw):
    """posnorm_lstm, but the network's angle input is the nearest training
    angle (constant per trial) instead of the raw measured angle.  The raw
    angle's within-trial excursion is a compliance torque read-out at the
    large-torque angles only (R² 0.93–0.99 vs 0.04–0.19 at the dorsiflexed
    angles), so the shared network can fit the angles that dominate it via a
    shortcut that does not exist where we fail.  Removing it forces the EMG
    pathway to carry every angle and removes sensitivity to retest angle
    calibration offsets (offset × 2000–5700 Nm/rad)."""
    return _posnorm(n_steps, n_features, kw, 'posnorm_knotpos', 'mse', 'knot')


# ── Variant 3: moe_pos ──────────────────────────────────────────────────────

@register_model('model_loss_moe')
def moe_pos(n_steps, n_features, **kw):
    """Shared LSTM(16) trunk; z = shared(h) + Σ_k g_k(pos)·expert_k(h) with a
    fixed hat gate over the training angles (one-hot with the default
    nearest lookup): each angle gets its own L2-shrunk linear correction of
    the read-out — the EMG→torque gain and muscle mix change with muscle
    length, and a shared read-out is otherwise fitted to the large angles.
    De-standardised output, standardised MSE."""
    _check(n_features, need_knots=True)
    P = _cfg(kw, {'ml_units': 16, 'ml_dropout': 0.2, 'ml_expert_l2': 1e-3,
                  'ml_gamma': 2.0, 'ml_gamma_val': 2.0, 'ml_pos_input': 'raw'})
    C = _classes(); L = C['layers']
    from keras import regularizers
    knots = _STATE['knots']
    inp = L.Input(shape=(n_steps, n_features))
    core, st = C['SplitStats'](pos_input=P['ml_pos_input'])(inp)
    h = _lstm_trunk(core, int(P['ml_units']), float(P['ml_dropout']))
    z_sh = L.Dense(1)(h)
    zk = L.Dense(len(knots), kernel_initializer='zeros',
                 kernel_regularizer=regularizers.l2(float(P['ml_expert_l2'])),
                 bias_regularizer=regularizers.l2(float(P['ml_expert_l2'])))(h)
    g = C['HatGate'](knots)(st)
    z = L.Add()([z_sh, C['GatedSum']()([zk, g])])
    return _model(inp, C['DeStd']()([z, st]), P, 'moe_pos', loss_kind='mse')


# ── Variant 4: linres_bounded ───────────────────────────────────────────────

@register_model('model_loss_linres_bounded')
def linres_bounded(n_steps, n_features, **kw):
    """z = Σ_k g_k(pos)·(w_k·[EMG_last, EMG_mean] + b_k)   (per-angle linear)
         + LSTM residual (zero-initialised read-out, L2)
    then a soft bound to the training-session range of the current angle,
    ŷ = mu + sd·SoftBound(z).  The linear path gives stable, proportional
    extrapolation (lower seed variance — JM baseline per-angle R² swings by
    up to 0.47 between seeds); the bound caps artefact-driven excursions
    (M4: JM −0.36 GL spike ≈ 6× max → −11 Nm) at the angle's own range
    (JM −0.36: never below −6.5 Nm) instead of the −18…−37 Nm range a shared
    unbounded network can emit."""
    _check(n_features, need_knots=True)
    P = _cfg(kw, {'ml_units': 8, 'ml_dropout': 0.3, 'ml_res_l2': 1e-3,
                  'ml_lin_l2': 1e-4, 'ml_bound_frac': 0.25,
                  'ml_bound_abs': 0.5, 'ml_gamma': 2.0, 'ml_gamma_val': 2.0,
                  'ml_pos_input': 'raw'})
    C = _classes(); L = C['layers']
    from keras import regularizers
    knots = _STATE['knots']
    inp = L.Input(shape=(n_steps, n_features))
    core, st = C['SplitStats'](pos_input=P['ml_pos_input'])(inp)
    f = C['EmgSummary']()(core)
    zk = L.Dense(len(knots), kernel_regularizer=regularizers.l2(float(P['ml_lin_l2'])))(f)
    g = C['HatGate'](knots)(st)
    z_lin = C['GatedSum']()([zk, g])
    h = _lstm_trunk(core, int(P['ml_units']), float(P['ml_dropout']))
    z_res = L.Dense(1, kernel_initializer='zeros',
                    kernel_regularizer=regularizers.l2(float(P['ml_res_l2'])))(h)
    z = L.Add()([z_lin, z_res])
    zb = C['SoftBound'](P['ml_bound_frac'], P['ml_bound_abs'])([z, st])
    return _model(inp, C['DeStd']()([zb, st]), P, 'linres_bounded', loss_kind='mse')


# ── Variant 5: bnll ─────────────────────────────────────────────────────────

@register_model('model_loss_bnll')
def bnll(n_steps, n_features, **kw):
    """Heteroscedastic β-NLL (Seitzer et al. 2022, β = 0.5) in standardised
    units: outputs [ŷ_Nm, log σ²_std]; σ² depends on angle + EMG history, so
    the mean's effective weight σ^(2β−2) = 1/σ adapts per sample.  Rest
    periods at YES −0.486 carry 0.33 sd of un-modelable drift (≤ 0.03 sd at
    the other YES angles) and JM's rest periods carry 0.2–0.5 sd at several
    angles — the variance head can learn "rest at this angle ⇒ noisy" and
    keep full weight on the active swings.  predict() returns ŷ only; val
    loss = standardised MSE of ŷ (selection on point accuracy)."""
    _check(n_features)
    P = _cfg(kw, {'ml_units': 8, 'ml_dropout': 0.3, 'ml_beta': 0.5,
                  'ml_gamma': 2.0, 'ml_gamma_val': 2.0, 'ml_pos_input': 'raw'})
    C = _classes(); L = C['layers']
    inp = L.Input(shape=(n_steps, n_features))
    core, st = C['SplitStats'](pos_input=P['ml_pos_input'])(inp)
    h = _lstm_trunk(core, int(P['ml_units']), float(P['ml_dropout']))
    z = L.Dense(1)(h)
    y_mu = C['DeStd']()([z, st])
    lv = L.Dense(1, kernel_initializer='zeros', bias_initializer='zeros')(
        L.Dense(8, activation='tanh')(h))
    out = L.Concatenate()([y_mu, lv])
    return _model(inp, out, P, 'bnll', loss_kind='bnll')


# ── Diagnostic ablation: baseline with the angle excursion removed ──────────

@register_model('model_loss_lstm8_knotpos')
def lstm8_knotpos(n_steps, n_features, **kw):
    """Exact baseline lstm8 (same layers, plain keras Model, harness-compiled
    Nm loss), only the angle input is pos_k (nearest training angle) instead
    of the raw angle.  Measures how much of the baseline's per-angle R² comes
    from the compliance torque read-out in the angle signal."""
    _check(n_features)
    C = _classes(); L = C['layers']
    inp = L.Input(shape=(n_steps, n_features))
    core, _ = C['SplitStats'](pos_input='knot')(inp)
    h = _lstm_trunk(core, 8, 0.3)
    return C['keras'].Model(inp, L.Dense(1)(h), name='lstm8_knotpos')
