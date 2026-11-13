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


def windows(feat, pos, y, window=C.WINDOW):
    """All windows of one trial: X (N, W, 5), y_last (N,), last index (N,)."""
    F = np.column_stack([feat, pos[:, None]]).astype(np.float32)
    n = len(F) - window + 1
    idx = np.arange(n)[:, None] + np.arange(window)[None, :]
    last = np.arange(n) + window - 1
    return F[idx], y[last].astype(np.float32), last


def _meta(rec, last, part, rest=None):
    return {'trial': np.full(len(last), rec['idx']), 'angle': np.full(len(last), rec['angle']),
            'session': np.array([rec['session']] * len(last)), 't': rec['t'][last].astype(np.float32),
            'last': last, 'part': np.array([part] * len(last)),
            'rest': (rest[last] if rest is not None else np.zeros(len(last), bool))}


def build(protocol, recs, feats, targets, window=C.WINDOW, rest=None):
    """Assemble train / val / test arrays for ``protocol``.

    recs     : all active records of the subject (both sessions)
    feats    : {trial idx: (T, 4) normalised EMG}
    targets  : {trial idx: (T,) active torque}
    rest     : optional {trial idx: (T,) bool} EMG-detected rest; stored per
               window (at its last sample) as meta['rest']
    """
    spec = PROTOCOLS[protocol]
    parts = {'train': [], 'val': [], 'test': []}
    for r in recs:
        X, y, last = windows(feats[r['idx']], r['pos'], targets[r['idx']], window)
        rr = rest[r['idx']] if rest is not None else None
        n = len(X)
        if protocol != 'pooled':
            if r['session'] in spec['train']:
                n_tr = int(n * C.TRAIN_FRAC)
                parts['train'].append((X[:n_tr], y[:n_tr], _meta(r, last[:n_tr], 'train', rr)))
                parts['val'].append((X[n_tr:], y[n_tr:], _meta(r, last[n_tr:], 'val', rr)))
            elif r['session'] in spec['test']:
                parts['test'].append((X, y, _meta(r, last, 'test', rr)))
        else:
            a, b = int(n * 0.6), int(n * 0.8)
            sl = {'train': slice(0, a), 'val': slice(a + window, b), 'test': slice(b + window, n)}
            for p, s in sl.items():
                parts[p].append((X[s], y[s], _meta(r, last[s], p, rr)))

    out = {}
    for p, items in parts.items():
        out[f'X_{p}'] = np.concatenate([x for x, _, _ in items])
        out[f'y_{p}'] = np.concatenate([y for _, y, _ in items])
        out[f'meta_{p}'] = {k: np.concatenate([m[k] for _, _, m in items]) for k in items[0][2]}
    out['train_angles'] = sorted({float(np.round(a, 3)) for a in out['meta_train']['angle']})
    return out
