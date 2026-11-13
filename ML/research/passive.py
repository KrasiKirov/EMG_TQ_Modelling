"""
Passive-torque models: what gets subtracted from the measured torque to obtain
the target (active / voluntary torque = measured − passive).

Rule for every model: it may use (a) passive recordings (the maps and the
static "before-holds") of any session and (b) the TRAINING session's own
active trials during EMG-silent rest periods.  It never uses the torque of the
trials it is evaluated on.

Models (``kind``)
-----------------
legacy        old pipeline: auto-detected map (20 Nm rules), looked up at each
              sample's angle.  (YES: worksheet values.)
static        map from static holds only (fixes the JM movement-trial bug).
gain          alpha·P_static + beta, fitted on training-session rest periods
              (the current pipeline's --recommended target).
hold_gain     alpha·P_hold + beta, where P_hold is the mean of the static hold
              recorded right before THIS trial (falls back to the map).
holdend_gain  same with the last 5 s of that hold (the tissue has already
              relaxed for ~20-30 s at the hold's end).
drift         alpha·P_hold + beta + gamma·P_hold·ln(1 + t/tau): time-varying
              passive torque (stress relaxation during the trial).  tau is
              chosen on the training session's VALIDATION part.
holdextrap    fit the hold's own relaxation curve  P(t) = a + b·ln(1 + t/tau_h)
              (t from the start of the hold) and continue it through the
              active trial (assumes the trial starts when the hold ends), then
              alpha·P(t) + beta on top.

Fits are robust (Huber IRLS), one observation per rest segment (its median),
weighted by sqrt(segment length).
"""

import numpy as np

from ML.preprocessing import dataset_builder as db
from ML.preprocessing import active_torque as at

from . import config as C

TAU_GRID_S = (2.0, 5.0, 10.0, 20.0, 40.0)


# ── small helpers ────────────────────────────────────────────────────────────

def map_lookup(entries, pos):
    """Pipeline-identical map interpolation/extrapolation at each sample."""
    if not entries:
        return np.zeros(len(pos))
    return np.array([db.lookup_passive_torque(entries, float(p)) for p in pos])


def robust_fit(A, y, w, huber=3.0, iters=10):
    """Huber-IRLS weighted least squares; returns coefficients."""
    ww = w.copy()
    coef = np.zeros(A.shape[1])
    for _ in range(iters):
        coef = np.linalg.lstsq(A * ww[:, None], y * ww, rcond=None)[0]
        res = y - A @ coef
        s = max(1.4826 * np.median(np.abs(res)), 1e-3)
        u = np.abs(res) / (huber * s)
        ww = w * np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-9))
    return coef


def _rest_obs(rec, segs):
    """One observation per rest segment: (time centre s, median torque, n)."""
    out = []
    for s, e in segs:
        out.append((float(rec['t'][(s + e - 1) // 2]), float(np.median(rec['tq'][s:e])), e - s))
    return out


def _hold_curve(hold_series, tau_h):
    """Fit a + b·ln(1 + t/tau_h) to a hold torque series (100 Hz)."""
    t = np.arange(len(hold_series)) / C.FS
    A = np.column_stack([np.ones_like(t), np.log1p(t / tau_h)])
    coef = np.linalg.lstsq(A, hold_series, rcond=None)[0]
    return coef, float(np.sqrt(np.mean((A @ coef - hold_series) ** 2)))


# ── the model ────────────────────────────────────────────────────────────────

class PassiveModel:
    """Fit on the training session, then predict P_eff(t) for any trial.

    Parameters
    ----------
    kind : see module docstring
    bundle : subject bundle from data.load_subject
    """

    def __init__(self, kind, bundle):
        self.kind = kind
        self.bundle = bundle
        self.coef = None
        self.tau = None
        self.info = {}

    # base passive level / curve of one trial before alpha, beta, gamma ------
    def _maps(self, rec):
        m = self.bundle['maps']
        return m['legacy'][rec['session']], m['static'][rec['session']]

    def base(self, rec):
        legacy, static = self._maps(rec)
        if self.kind == 'legacy':
            return map_lookup(legacy, rec['pos'])
        if self.kind in ('static', 'gain'):
            return map_lookup(static, rec['pos'])
        h = rec.get('hold')
        if h is None:                                    # no hold: use the map
            return map_lookup(static, rec['pos'])
        if self.kind in ('hold_gain', 'drift'):
            return np.full(len(rec['t']), h['mean'])
        if self.kind == 'holdend_gain':
            return np.full(len(rec['t']), h['end'])
        if self.kind == 'holdextrap':
            series = self._hold_series(h['idx'])
            if series is None or len(series) < 5 * C.FS:
                return np.full(len(rec['t']), h['end'])
            best = min((_hold_curve(series, th) + (th,) for th in TAU_GRID_S), key=lambda z: z[1])
            (a, b), _, th = best
            dur = len(series) / C.FS
            t_since_hold = dur + (rec['t'] - rec['t'][0])
            return a + b * np.log1p(t_since_hold / th)
        raise ValueError(self.kind)

    def _hold_series(self, idx):
        for h in self.bundle['holds']:
            if h['idx'] == idx:
                return np.asarray(h['series'], np.float64)
        return None

    def _design(self, P, t):
        if self.kind == 'drift':
            # t = seconds since the start of the recording (first kept sample = 1 s)
            return np.column_stack([P, np.ones_like(P), P * np.log1p(t / self.tau)])
        return np.column_stack([P, np.ones_like(P)])

    # fitting ------------------------------------------------------------------
    def fit(self, train_recs, rest_segs_train, rest_segs_val):
        """train_recs: training-session records.  rest_segs_*: dict rec idx ->
        list of (s, e) EMG-detected rest segments in the train / val part."""
        if self.kind in ('legacy', 'static'):
            return self
        rows = []
        for r in train_recs:
            base = self.base(r)
            for tc, med, n in _rest_obs(r, rest_segs_train.get(r['idx'], [])):
                i = int(np.argmin(np.abs(r['t'] - tc)))
                rows.append((base[i], tc, med, n))
        if len(rows) < 3:
            self.coef = np.array([1.0, 0.0, 0.0])[:2 if self.kind != 'drift' else 3]
            self.info['warning'] = 'too few rest segments; identity passive model'
            return self
        P, t, y, n = (np.array(c, float) for c in zip(*rows))
        w = np.sqrt(np.minimum(n, 300.0))
        if self.kind != 'drift':
            self.coef = robust_fit(self._design(P, t), y, w)
        else:
            # choose tau on the validation part of the training session
            vrows = []
            for r in train_recs:
                base = self.base(r)
                for tc, med, nn in _rest_obs(r, rest_segs_val.get(r['idx'], [])):
                    i = int(np.argmin(np.abs(r['t'] - tc)))
                    vrows.append((base[i], tc, med, nn))
            best = None
            for tau in TAU_GRID_S:
                self.tau = tau
                coef = robust_fit(self._design(P, t), y, w)
                if vrows:
                    Pv, tv, yv, nv = (np.array(c, float) for c in zip(*vrows))
                    err = float(np.median(np.abs(self._design(Pv, tv) @ coef - yv)))
                else:
                    err = float(np.median(np.abs(self._design(P, t) @ coef - y)))
                if best is None or err < best[0]:
                    best = (err, tau, coef)
            _, self.tau, self.coef = best
            self.info['val_median_abs_err'] = best[0]
        self.info['n_rest_segments'] = len(rows)
        return self

    def predict(self, rec):
        """P_eff(t) for one trial (any session)."""
        base = self.base(rec)
        if self.coef is None:
            return base
        return self._design(base, rec['t']) @ self.coef

    def describe(self):
        d = {'kind': self.kind}
        if self.coef is not None:
            d['coef'] = [float(c) for c in self.coef]
        if self.tau is not None:
            d['tau_s'] = self.tau
        d.update(self.info)
        return d


def rest_segments(recs, env_norm, train_frac=C.TRAIN_FRAC, val_end_frac=1.0, window=C.WINDOW):
    """EMG-only rest detection on the training-session records.

    Returns (train_segs, val_segs): dict rec idx -> [(s, e)], split at the same
    sample where the train / validation windows split (val segs re-indexed to
    the full trial).  ``val_end_frac`` < 1 stops the validation part early
    (pooled protocol: validation = 60-80 %, the last 20 % is test data and
    must not be touched).  Statistics come from the training part only."""
    cuts = {}
    for r in recs:
        T = len(r['t'])
        n_win = T - window + 1
        cuts[r['idx']] = int(n_win * train_frac) + window - 1
    stats = at.emg_rest_stats([env_norm[r['idx']][:cuts[r['idx']]] for r in recs])
    tr, va = {}, {}
    for r in recs:
        cut, env = cuts[r['idx']], env_norm[r['idx']]
        tr[r['idx']] = at.detect_rest(env[:cut - 10], stats)
        end = int(len(r['t']) * val_end_frac)
        va[r['idx']] = [(s + cut, e + cut) for s, e in at.detect_rest(env[cut:end], stats)] if end > cut + 100 else []
    return tr, va
