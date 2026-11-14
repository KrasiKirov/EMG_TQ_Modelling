"""
Evaluation protocols and sliding windows.

Protocols
---------
S1toS2   train on session 1 (first 80 % of each trial = train, last 20 % =
         validation), test on every window of session 2.  This is the
         original evaluation ("does it work on a new session?").
S2toS1   the same with the sessions swapped (second fold of a 2-fold
         cross-session evaluation).
pooled   both sessions in training: in every trial of both sessions the first
         60 % is train, the next 20 % validation, the last 20 % test, with a
         gap of one window (200 ms) at each boundary so no window spans two
         parts.  Measures within-session accuracy (an easier question).

Windows
-------
Input window = WINDOW steps x [GM, GL, SOL, TA, ankle angle]; the target is
the active torque at the window's last step (same as the pipeline).  Every
window carries metadata (trial, session, angle, time) so results can be
grouped per angle and re-sliced (e.g. calibration on the first seconds).
"""

import numpy as np

from . import config as C

PROTOCOLS = {
    'S1toS2': {'train': ['S1'], 'test': ['S2']},
    'S2toS1': {'train': ['S2'], 'test': ['S1']},
    'pooled': {'train': ['S1', 'S2'], 'test': ['S1', 'S2']},
}


def windows(feat, pos, y, window=C.WINDOW, step=1):
    """All windows of one trial: X (N, W, 5), y_last (N,), last index (N,).
    step = samples between the W time steps of one window (1 = every 10 ms sample; 5 = every 50 ms, so a
    20-step window spans 0.95 s). A new window still starts at every sample."""
    F = np.column_stack([feat, pos[:, None]]).astype(np.float32)
    span = (window - 1) * step
    n = len(F) - span
    idx = np.arange(n)[:, None] + (np.arange(window) * step)[None, :]
    last = np.arange(n) + span
    return F[idx], y[last].astype(np.float32), last


def _meta(rec, last, part, rest=None):
    return {'trial': np.full(len(last), rec['idx']), 'angle': np.full(len(last), rec['angle']),
            'session': np.array([rec['session']] * len(last)), 't': rec['t'][last].astype(np.float32),
            'last': last, 'part': np.array([part] * len(last)),
            'rest': (rest[last] if rest is not None else np.zeros(len(last), bool))}


def streams(protocol, recs, feats, targets, window=C.WINDOW, step=1, val_gap=False):
    """Whole trials for the carried-state model (``stateful.py``), aligned with ``build``.

    train : one stream per training-session trial with masks that select exactly the target samples of the
            windowed model's training and validation windows (the last validation window excluded, as
            ``drop_last_val_window`` does), so both models learn and are checked on the same samples.
            val_gap: as in ``build``.
    test  : one stream per test-session trial; ``first`` = index of the first sample the windowed model scores,
            so the carried-state predictions are scored on the same samples.
    """
    if protocol == 'pooled':
        raise ValueError("state='carry' supports the cross-session protocols only")
    spec = PROTOCOLS[protocol]
    span = (window - 1) * step
    out = {'train': [], 'test': []}
    for r in recs:
        F = np.column_stack([feats[r['idx']], r['pos'][:, None]]).astype(np.float32)
        y = np.asarray(targets[r['idx']], np.float64)
        T = len(F)
        if r['session'] in spec['train']:
            n = T - span
            cut = span + int(n * C.TRAIN_FRAC)
            i = np.arange(T)
            first_val = cut + (span if val_gap else 0)
            out['train'].append({'X': F, 'y': y, 'rec': r, 'train_mask': (i >= span) & (i < cut),
                                 'val_mask': (i >= first_val) & (i < T - 1)})
        elif r['session'] in spec['test']:
            out['test'].append({'X': F, 'y': y, 'rec': r, 'first': span})
    return out


def build(protocol, recs, feats, targets, window=C.WINDOW, rest=None, step=1, train_stride=1, val_gap=False):
    """Assemble train / val / test arrays for ``protocol``.

    recs     : all active records of the subject (both sessions)
    feats    : {trial idx: (T, 4) normalised EMG}
    targets  : {trial idx: (T,) active torque}
    rest     : optional {trial idx: (T,) bool} EMG-detected rest; stored per
               window (at its last sample) as meta['rest']
    step     : samples between time steps inside a window (see ``windows``)
    train_stride : keep every k-th TRAINING window (k = 1: a new window every sample, consecutive
               windows share (W-1)/W of their steps). Validation and test always keep every window, so
               every setting is scored on the same samples.
    val_gap  : start the validation windows one window span after the training cut, so no validation window
               takes inputs from the training part of its trial (a purged split). Default False = as before.
    """
    spec = PROTOCOLS[protocol]
    parts = {'train': [], 'val': [], 'test': []}
    for r in recs:
        X, y, last = windows(feats[r['idx']], r['pos'], targets[r['idx']], window, step)
        rr = rest[r['idx']] if rest is not None else None
        n = len(X)
        if protocol != 'pooled':
            if r['session'] in spec['train']:
                n_tr = int(n * C.TRAIN_FRAC)
                tr = slice(0, n_tr, train_stride)
                parts['train'].append((X[tr], y[tr], _meta(r, last[tr], 'train', rr)))
                v0 = n_tr + ((window - 1) * step if val_gap else 0)
                parts['val'].append((X[v0:], y[v0:], _meta(r, last[v0:], 'val', rr)))
            elif r['session'] in spec['test']:
                parts['test'].append((X, y, _meta(r, last, 'test', rr)))
        else:
            a, b = int(n * 0.6), int(n * 0.8)
            gap = (window - 1) * step + 1           # one window span between parts
            sl = {'train': slice(0, a, train_stride), 'val': slice(a + gap, b), 'test': slice(b + gap, n)}
            for p, s in sl.items():
                parts[p].append((X[s], y[s], _meta(r, last[s], p, rr)))

    out = {}
    for p, items in parts.items():
        out[f'X_{p}'] = np.concatenate([x for x, _, _ in items])
        out[f'y_{p}'] = np.concatenate([y for _, y, _ in items])
        out[f'meta_{p}'] = {k: np.concatenate([m[k] for _, _, m in items]) for k in items[0][2]}
    out['train_angles'] = sorted({float(np.round(a, 3)) for a in out['meta_train']['angle']})
    return out
