"""
A COMMON reference truth for scoring every configuration, plus rest/active
metrics.

Why this is needed
------------------
Each configuration defines its own target ("measured - its passive model"),
for the training data AND for the test data. If configurations are scored
against their own targets, they are scored against DIFFERENT truths, and a
passive model can look good just because it makes its own test target easier.
For a fair comparison every model must be scored against the SAME truth: the
best available estimate of the voluntary torque that does not depend on any
configuration.

Reference truth (per test trial)
--------------------------------
    truth(t) = measured torque(t) - B(t)

B(t) is the passive baseline measured directly inside the trial. Torque
during EMG-detected rest is passive (nobody pushes), so B(t) is a smooth
curve through the median torque of every rest segment (Gaussian kernel,
sigma = 8 s, outlier-robust, flat at the ends;
``ML.preprocessing.active_torque.rest_baseline``). It follows the
stress relaxation during the trial and the session-specific passive level.
Literature basis (see the report): stress relaxation of the ankle during
holds (Duong 2001; McNair 2001), and history- and session-dependence of passive
torque (Kay & Blazevich 2009; Reisman 2009; Hoang 2005), so passive torque must
be measured in-session. Anchoring the baseline on relaxed in-trial windows is
our own correction, not a published method.

* Rest is detected from EMG only (``active_torque.detect_rest``), with
  the per-channel scale computed from all trials of the same session.
* The truth uses the test trial's measured torque.  That is fine for a
  truth (the measured torque IS what is being predicted); it is NEVER an
  input of any model.  Configurations whose TRAINING labels use the same
  construction (passive='restanchor') use only training-session torque
  for them.

Metrics against the truth (per test trial = one angle of one session)
--------------------------------------------------------------------
r2, rmse ...      on all windows (same definitions as evaluate.metrics)
r2_active, rmse_active   only windows outside rest segments
false_rest       mean |prediction| inside rest segments (Nm): the
                 "phantom torque" predicted while the person is relaxed;
                 ideally 0
rest_frac        fraction of the trial's windows detected as rest
"""

import os
import pickle

import numpy as np

from ML.preprocessing import active_torque as at

from . import config as C
from .evaluate import metrics

SIGMA_S = 8.0     # s, smoothing of the in-trial passive baseline


def session_rest(bundle, session, max_frac=None):
    """{trial idx: rest segments [(s, e)]} for every active trial of a session.

    EMG only.  ``max_frac`` (e.g. 0.8) limits BOTH the statistics and the
    detection to the first part of every trial (used for training labels
    in the pooled protocol, whose last 20 % is test data)."""
    recs = [r for r in bundle['active'] if r['session'] == session]
    emg_max = np.max(np.stack([r['env_max1k'] for r in recs]), axis=0)
    emg_max = np.where(emg_max > 0, emg_max, 1.0)
    env = {}
    for r in recs:
        e = r['env_raw'] / emg_max
        env[r['idx']] = e[:int(len(e) * max_frac)] if max_frac else e
    stats = at.emg_rest_stats(list(env.values()))
    return {i: at.detect_rest(e, stats) for i, e in env.items()}


def baseline(rec, segs, sigma_s=SIGMA_S):
    """In-trial passive baseline B(t) (Nm) through the rest segments."""
    return at.rest_baseline(rec['tq'], segs, len(rec['tq']), sigma_s=sigma_s)


def rest_mask(T, segs):
    m = np.zeros(T, bool)
    for s, e in segs:
        m[s:e] = True
    return m


_CACHE = {}


def subject_truth(subject):
    """{trial idx: {'truth', 'B', 'rest'}} for every active trial (cached)."""
    if subject in _CACHE:
        return _CACHE[subject]
    path = os.path.join(C.CACHE_DIR, f'{subject}_truth_v1.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            _CACHE[subject] = pickle.load(f)
        return _CACHE[subject]
    from .data import load_subject
    bundle = load_subject(subject)
    out = {}
    for sess in ('S1', 'S2'):
        segs = session_rest(bundle, sess)
        for r in bundle['active']:
            if r['session'] != sess:
                continue
            B = baseline(r, segs[r['idx']])
            out[r['idx']] = {'truth': (r['tq'] - B).astype(np.float32), 'B': B.astype(np.float32),
                             'rest': rest_mask(len(r['tq']), segs[r['idx']]), 't0': float(r['t'][0]),
                             'angle': r['angle'], 'session': sess, 'n_rest_segs': len(segs[r['idx']])}
    with open(path, 'wb') as f:
        pickle.dump(out, f)
    _CACHE[subject] = out
    return out


def rescore(subject, npz):
    """Score saved predictions (an .npz written by run.py) against the
    reference truth.  Returns {'per_angle': {key: metrics...}, 'overall': ...}."""
    z = np.load(npz) if isinstance(npz, str) else npz
    tr = subject_truth(subject)
    trial, t, pred = z['trial'].astype(int), z['t'].astype(np.float64), z['y_pred'].astype(np.float64)
    y = np.empty(len(pred)); rest = np.zeros(len(pred), bool)
    for i in np.unique(trial):
        m = trial == i
        k = np.rint((t[m] - tr[i]['t0']) * C.FS).astype(int)
        y[m] = tr[i]['truth'][k]
        rest[m] = tr[i]['rest'][k]
    sess = np.where(z['session'] == 1, 'S2', 'S1')
    keys = np.array([f'{s}:{a:+.3f}' for s, a in zip(sess, z['angle'].astype(np.float64))])
    per = {}
    for k in sorted(set(keys), key=lambda s: (s[:2], float(s[3:]))):
        m = keys == k
        d = metrics(y[m], pred[m])
        act = m & ~rest
        if act.sum() > 50:
            ma = metrics(y[act], pred[act])
            d.update(r2_active=ma['r2'], rmse_active=ma['rmse'])
        d['false_rest'] = float(np.mean(np.abs(pred[m & rest]))) if (m & rest).sum() else float('nan')
        d['rest_frac'] = float(rest[m].mean())
        per[k] = d
    return {'per_angle': per, 'overall': metrics(y, pred)}
