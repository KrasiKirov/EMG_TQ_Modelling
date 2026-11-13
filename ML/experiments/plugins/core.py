"""
Core variant plugins for the dorsiflexion investigation.

Plugin signature (preprocessing):  fn(trainval, retest, ctx) -> None
    Edits per-trial records in place (see harness docstring).  Parameters are
    read from ctx['opts']['params'] so a run is fully described by its opts.

Sample-weight signature:           fn(data) -> np.ndarray (len X_train)
Model-builder signature:           fn(n_steps, n_features, **kw) -> keras.Model

Every plugin documents whether it touches the TEST TARGET.  Plugins that do
(re-zeroing) change the definition of ground truth and must be reported as
such; plugins that only touch inputs / loss / model are pure modelling fixes.
"""

import numpy as np

from ML.preprocessing import dataset_builder as db
from ML.experiments.harness import passive_curve, register_model


def _p(ctx, key, default):
    return ctx['opts'].get('params', {}).get(key, default)


# ── Rest detection (EMG only — never uses torque) ───────────────────────────

def rest_mask(env, pct=20.0, channels=None, min_run=30):
    """Samples where the summed normalised envelope is in the bottom ``pct``
    percent of the trial, keeping only runs ≥ ``min_run`` samples (0.3 s)."""
    e = env if channels is None else env[:, channels]
    tot = e.sum(axis=1)
    m = tot < np.percentile(tot, pct)
    if min_run > 1:
        out = np.zeros_like(m)
        i, n = 0, len(m)
        while i < n:
            if m[i]:
                j = i
                while j < n and m[j]:
                    j += 1
                if j - i >= min_run:
                    out[i:j] = True
                i = j
            else:
                i += 1
        if out.sum() >= 50:
            m = out
    return m


def _rest_segments(mask):
    segs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1
    return segs


# ── TARGET plugins (change ground truth — flag in report) ───────────────────

def rezero_const(trainval, retest, ctx):
    """[TARGET] Subtract, per trial, the median target over EMG-silent samples.
    Removes the per-session passive-subtraction error (constant offset)."""
    pct = _p(ctx, 'rest_pct', 20.0)
    for r in trainval + retest:
        m = rest_mask(r['env'], pct)
        c = float(np.median(r['y'][m]))
        r['y'] = r['y'] - c
        r['rezero_c'] = c


def rezero_drift(trainval, retest, ctx):
    """[TARGET] Subtract a slowly varying baseline interpolated through the
    medians of EMG-silent segments (captures passive stress-relaxation drift
    within a trial as well as the session offset)."""
    pct = _p(ctx, 'rest_pct', 20.0)
    smooth_s = _p(ctx, 'drift_smooth_s', 0.0)
    for r in trainval + retest:
        m = rest_mask(r['env'], pct)
        segs = _rest_segments(m)
        t = np.arange(len(r['y']))
        if len(segs) < 2:
            base = np.full(len(t), np.median(r['y'][m]))
        else:
            tc = np.array([(a + b - 1) / 2 for a, b in segs])
            yc = np.array([np.median(r['y'][a:b]) for a, b in segs])
            base = np.interp(t, tc, yc)        # constant extrapolation at ends
        if smooth_s > 0:
            k = int(smooth_s * 100)
            ker = np.ones(k) / k
            base = np.convolve(np.pad(base, (k // 2, k - k // 2 - 1), mode='edge'), ker, 'valid')
        r['y'] = r['y'] - base
        r['baseline'] = base


def rezero_linear(trainval, retest, ctx):
    """[TARGET] Like rezero_drift but baseline = least-squares line through
    the rest samples (less flexible, more robust)."""
    pct = _p(ctx, 'rest_pct', 20.0)
    for r in trainval + retest:
        m = rest_mask(r['env'], pct)
        t = np.arange(len(r['y']))
        a, b = np.polyfit(t[m], r['y'][m], 1)
        base = a * t + b
        r['y'] = r['y'] - base
        r['baseline'] = base


def rezero_train_only(trainval, retest, ctx):
    """[TRAIN TARGET ONLY] Re-zero training trials (const) but leave the test
    target untouched.  Tells us how much of the gain needs test-time labels."""
    pct = _p(ctx, 'rest_pct', 20.0)
    for r in trainval:
        m = rest_mask(r['env'], pct)
        r['y'] = r['y'] - float(np.median(r['y'][m]))


# ── Passive-map plugins (use passive trials only; change target) ────────────

def passive_bound10(trainval, retest, ctx):
    """[TARGET via passive trials] Rebuild auto-detected passive maps with the
    original 10 Nm plausibility bound (the 20 Nm bound admitted JM trial 21,
    a movement trial, as +11.7 Nm at −0.07 rad).  Worksheet overrides kept."""
    from ML.config import PASSIVE_TORQUE_OVERRIDES
    subj = ctx['subject']
    if PASSIVE_TORQUE_OVERRIDES.get(subj):
        return
    old = db._PASSIVE_TQ_BOUND
    db._PASSIVE_TQ_BOUND = _p(ctx, 'passive_bound', 10.0)
    try:
        pt = ctx['passive_trials']
        test_map = db.get_passive_torque_map(
            [df for df in pt if df.attrs.get('session') != 'retest'])
        rt = [df for df in pt if df.attrs.get('session') == 'retest']
        retest_map = db.get_passive_torque_map(rt) if rt else test_map
    finally:
        db._PASSIVE_TQ_BOUND = old
    ctx['test_map'], ctx['retest_map'] = test_map, retest_map
    for r in trainval + retest:
        pmap = retest_map if r['session'] == 'retest' else test_map
        r['passive'] = passive_curve(pmap, r['pos'])
        r['y'] = r['tq'] - r['passive']


def passive_monotone(trainval, retest, ctx):
    """[TARGET via passive trials] Replace each session's passive map by a
    robust monotone (decreasing in angle) fit — rejects entries that break
    the physiological monotonic passive torque–angle relation."""
    def fit(entries):
        if len(entries) < 3:
            return entries
        e = sorted(entries)
        keep = list(e)
        changed = True
        while changed and len(keep) > 3:
            changed = False
            pos = np.array([p for p, _ in keep]); tq = np.array([t for _, t in keep])
            c = np.polyfit(pos, tq, 2)
            res = np.abs(tq - np.polyval(c, pos))
            k = int(np.argmax(res))
            if res[k] > 3.0 * max(np.median(res), 0.3):
                keep.pop(k); changed = True
        return keep
    from ML.config import PASSIVE_TORQUE_OVERRIDES
    if PASSIVE_TORQUE_OVERRIDES.get(ctx['subject']):
        return
    tm, rm = fit(ctx['test_map']), fit(ctx['retest_map'])
    ctx['test_map'], ctx['retest_map'] = tm, rm
    for r in trainval + retest:
        pmap = rm if r['session'] == 'retest' else tm
        r['passive'] = passive_curve(pmap, r['pos'])
        r['y'] = r['tq'] - r['passive']


# ── INPUT plugins (EMG features only — target untouched) ────────────────────

def clip_env(trainval, retest, ctx):
    """Clip normalised envelopes to [0, c] (c = params.clip, default 1.2).
    Kills electrode-pop artefacts like JM −0.36 retest (GL spike ≈ 6× max)."""
    c = _p(ctx, 'clip', 1.2)
    for r in trainval + retest:
        r['env'] = np.clip(r['env'], 0.0, c)


def hampel_env(trainval, retest, ctx):
    """Replace envelope samples deviating > k·MAD from a running median."""
    from scipy.ndimage import median_filter
    w, k = int(_p(ctx, 'hampel_w', 51)), _p(ctx, 'hampel_k', 6.0)
    for r in trainval + retest:
        e = r['env'].copy()
        for ch in range(e.shape[1]):
            med = median_filter(e[:, ch], size=w, mode='nearest')
            mad = median_filter(np.abs(e[:, ch] - med), size=w, mode='nearest') * 1.4826 + 1e-6
            bad = np.abs(e[:, ch] - med) > k * mad
            e[bad, ch] = med[bad]
        r['env'] = e


def floor_subtract(trainval, retest, ctx):
    """Subtract each channel's own rest-level (noise floor) per trial, clip ≥ 0.
    Uses EMG only.  Makes 'no activation' ≡ 0 regardless of session noise."""
    pct = _p(ctx, 'floor_pct', 5.0)
    for r in trainval + retest:
        floor = np.percentile(r['env'], pct, axis=0)
        r['env'] = np.clip(r['env'] - floor, 0.0, None)


def trial_norm(trainval, retest, ctx):
    """[TRANSDUCTIVE — uses unlabeled test EMG] Normalise each trial's
    envelopes by that trial's own 99th percentile per channel."""
    q = _p(ctx, 'trial_norm_q', 99.0)
    for r in trainval + retest:
        s = np.percentile(r['env'], q, axis=0)
        r['env'] = r['env'] / np.where(s > 0, s, 1.0)


def session_norm(trainval, retest, ctx):
    """[TRANSDUCTIVE — uses unlabeled test EMG] Normalise the retest session by
    the retest session's own per-channel 99th percentile (rescaled so its
    level matches the test session).  Corrects session-wide EMG gain drift."""
    q = _p(ctx, 'session_norm_q', 99.0)
    tv = np.percentile(np.concatenate([r['env'] for r in trainval]), q, axis=0)
    rt = np.percentile(np.concatenate([r['env'] for r in retest]), q, axis=0)
    g = tv / np.where(rt > 0, rt, 1.0)
    for r in retest:
        r['env'] = r['env'] * g


def sqrt_env(trainval, retest, ctx):
    """Compress envelopes with sqrt (boosts low-activation resolution, which
    is where the plantarflexed positions live)."""
    for r in trainval + retest:
        r['env'] = np.sqrt(np.clip(r['env'], 0, None))


def env_derivative(trainval, retest, ctx):
    """Append the time derivative of the summed plantarflexor envelope."""
    for r in trainval + retest:
        d = np.gradient(r['env'][:, :3].sum(1)) * 100.0
        ex = d[:, None]
        r['extra'] = ex if r.get('extra') is None else np.column_stack([r['extra'], ex])


def pos_onehot(trainval, retest, ctx):
    """Append a one-hot of the nearest training operating position."""
    known = np.array(sorted({round(float(np.mean(r['pos'])), 3) for r in trainval}))
    for r in trainval + retest:
        k = int(np.argmin(np.abs(known - np.mean(r['pos']))))
        oh = np.zeros((len(r['pos']), len(known)), np.float32); oh[:, k] = 1
        r['extra'] = oh if r.get('extra') is None else np.column_stack([r['extra'], oh])


# ── Sample weights ──────────────────────────────────────────────────────────

def pos_inv_var_weights(data):
    """Weight each training window by 1/var(target) of its position, so every
    position contributes equally to the loss (YES: −0.49 has std 0.4 Nm vs
    15 Nm at +0.14 → otherwise ~1400× less weight).  Normalised to mean 1."""
    pos = data['X_train'][:, -1, 4]
    known = np.array(sorted({round(float(np.mean(r['pos'])), 3) for r in data['trainval']}))
    snap = np.argmin(np.abs(pos[:, None] - known[None, :]), axis=1)
    w = np.zeros(len(pos))
    for k in range(len(known)):
        m = snap == k
        w[m] = 1.0 / max(np.var(data['y_train'][m]), 1e-3)
    return (w / w.mean()).astype(np.float32)


def pos_inv_std_weights(data):
    """Softer: 1/std per position (geometric middle ground)."""
    pos = data['X_train'][:, -1, 4]
    known = np.array(sorted({round(float(np.mean(r['pos'])), 3) for r in data['trainval']}))
    snap = np.argmin(np.abs(pos[:, None] - known[None, :]), axis=1)
    w = np.zeros(len(pos))
    for k in range(len(known)):
        m = snap == k
        w[m] = 1.0 / max(np.std(data['y_train'][m]), 1e-2)
    return (w / w.mean()).astype(np.float32)


# ── Models ──────────────────────────────────────────────────────────────────

@register_model('lstm32')
def lstm32(n_steps, n_features, units=32, dropout=0.2, **kw):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
    return Sequential([Input(shape=(n_steps, n_features)),
                       LSTM(units, return_sequences=True), Dropout(dropout),
                       LSTM(units), Dropout(dropout), Dense(1)])


@register_model('gru32')
def gru32(n_steps, n_features, units=32, dropout=0.2, **kw):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, GRU, Dropout, Dense
    return Sequential([Input(shape=(n_steps, n_features)),
                       GRU(units, return_sequences=True), Dropout(dropout),
                       GRU(units), Dropout(dropout), Dense(1)])


@register_model('film')
def film(n_steps, n_features, units=16, dropout=0.2, **kw):
    """Position-conditioned gain/bias:  ŷ = g(pos)·h(EMG) + b(pos).
    Mirrors the physiology: the EMG→torque gain depends on muscle length."""
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    inp = layers.Input(shape=(n_steps, n_features))
    emg = layers.Lambda(lambda x: tf.concat([x[:, :, :4], x[:, :, 5:]], axis=-1))(inp)
    pos = layers.Lambda(lambda x: x[:, -1, 4:5])(inp)
    h = layers.LSTM(units, return_sequences=True)(emg)
    h = layers.Dropout(dropout)(h)
    h = layers.LSTM(units)(h)
    h = layers.Dropout(dropout)(h)
    h = layers.Dense(8)(h)
    pe = layers.Dense(16, activation='tanh')(pos)
    g = layers.Dense(8)(pe)
    b = layers.Dense(1)(pe)
    out = layers.Add()([layers.Dense(1)(layers.Multiply()([h, g])), b])
    return Model(inp, out)


@register_model('lstm8_wide_pos')
def lstm8_wide_pos(n_steps, n_features, **kw):
    """Baseline LSTM, but position also enters the head directly."""
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    inp = layers.Input(shape=(n_steps, n_features))
    h = layers.LSTM(8, return_sequences=True)(inp)
    h = layers.Dropout(0.3)(h)
    h = layers.LSTM(8)(h)
    h = layers.Dropout(0.3)(h)
    pos = layers.Lambda(lambda x: x[:, -1, 4:])(inp)
    z = layers.Concatenate()([h, pos])
    z = layers.Dense(16, activation='tanh')(z)
    return Model(inp, layers.Dense(1)(z))
