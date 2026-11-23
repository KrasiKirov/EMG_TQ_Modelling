"""
Passive-torque models: the target is measured torque minus the model's passive torque.

A model may use passive recordings (maps, static holds) of any session and the training session's own active
trials during EMG-silent rest; never the torque of the trials it is evaluated on.

legacy        old pipeline's map (20 Nm rules), looked up at each sample's angle (YES: worksheet values)
static        map from static holds only
gain          alpha·P_static + beta, fitted on training-session rest periods
hold_gain     alpha·P_hold + beta, P_hold = mean of the static hold just before this trial (else the map)
holdend_gain  the same with the last 5 s of that hold
drift         alpha·P_hold + beta + gamma·P_hold·ln(1 + t/tau), tau chosen on the training session's validation part
holdextrap    the hold's own relaxation curve a + b·ln(1 + t/tau_h) continued through the trial, then alpha·P + beta

Fits are Huber IRLS, one observation per rest segment (its median), weighted by sqrt(segment length).
"""

import numpy as np

from ML.preprocessing import rest as RS
from ML.preprocessing.passive_map import lookup_passive_torque

from . import config as C

TAU_GRID_S = (2.0, 5.0, 10.0, 20.0, 40.0)


def map_lookup(entries, pos):
    """Pipeline-identical map interpolation/extrapolation at each sample."""
    if not entries:
        return np.zeros(len(pos))
    return np.array([lookup_passive_torque(entries, float(p)) for p in pos])


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


class PassiveModel:
    """Fit on the training session, then predict P_eff(t) for any trial (kind: see the module docstring)."""

    def __init__(self, kind, bundle):
        self.kind = kind
        self.bundle = bundle
        self.coef = None
        self.tau = None
        self.info = {}

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

    def fit(self, train_recs, rest_segs_train, rest_segs_val):
        """rest_segs_*: rec idx -> [(s, e)] EMG-detected rest in the train / validation part of train_recs."""
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
    """EMG-only rest detection on the training-session records, statistics from the training part only.

    Returns (train_segs, val_segs): rec idx -> [(s, e)], split where the train / validation windows split (val segs
    indexed in the full trial). ``val_end_frac`` < 1 stops the validation part early (pooled: the rest is test)."""
    cuts = {}
    for r in recs:
        T = len(r['t'])
        n_win = T - window + 1
        cuts[r['idx']] = int(n_win * train_frac) + window - 1
    stats = RS.emg_rest_stats([env_norm[r['idx']][:cuts[r['idx']]] for r in recs])
    tr, va = {}, {}
    for r in recs:
        cut, env = cuts[r['idx']], env_norm[r['idx']]
        tr[r['idx']] = RS.detect_rest(env[:cut - 10], stats)
        end = int(len(r['t']) * val_end_frac)
        va[r['idx']] = [(s + cut, e + cut) for s, e in RS.detect_rest(env[cut:end], stats)] if end > cut + 100 else []
    return tr, va
