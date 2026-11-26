"""
Evaluation protocols and sliding windows.

S1toS2   train on session 1 (first 80 % of each trial train, last 20 % validation), test on all of session 2
         (the only protocol; S2toS1 and pooled were removed in 2026-11, see tag pre-cleanup-2026-11)

A window is WINDOW steps x [GM, GL, SOL, TA, angle]; the target is the active torque at its last step. Each window
carries metadata (trial, session, angle, time, rest).
"""

import numpy as np

from ML.config import FS

from .config import TRAIN_FRAC, WINDOW

SESSION_SPLITS = {
    'S1toS2': {'train': ['S1'], 'test': ['S2']},
}


def windows(feat, pos, y, window=WINDOW):
    """All windows of one trial: X (N, W, 5) of W consecutive 10 ms samples, y_last (N,), last index (N,).
    A new window starts at every sample."""
    F = np.column_stack([feat, pos[:, None]]).astype(np.float32)
    span = window - 1
    n = len(F) - span
    idx = np.arange(n)[:, None] + np.arange(window)[None, :]
    last = np.arange(n) + span
    return F[idx], y[last].astype(np.float32), last


COMMON_MAX_WINDOW = 1000   # steps: the longest window of study v2 (10 s at 10 ms)


def common_start(T, max_window=COMMON_MAX_WINDOW):
    """First target index every window length up to ``max_window`` validates on, so comparisons across lengths use
    the same targets: the first validation target of the longest window (its training cut plus one window span)."""
    span = max_window - 1
    return 2 * span + int((T - span) * TRAIN_FRAC)


def common_mask(meta, lengths, max_window=COMMON_MAX_WINDOW):
    """Boolean mask over validation rows: True for targets shared by every window length.
    lengths: {trial idx: number of samples in the trial}."""
    start = np.array([common_start(lengths[int(i)], max_window) for i in meta['trial']])
    return np.asarray(meta['last']) >= start


def check_common_targets(recs, min_s=5.0, max_window=COMMON_MAX_WINDOW, fs=FS):
    """Raise ValueError naming the trial if any training trial has < min_s seconds of common validation targets
    (the last validation window of each trial is dropped, so targets end at T - 2)."""
    for r in recs:
        T = len(r['t'])
        avail = (T - 1 - common_start(T, max_window)) / fs
        if avail < min_s:
            raise ValueError(f"trial {r['idx']} ({r['session']}, angle {r['angle']:.3f} rad) has {avail:.1f} s of "
                             f"common validation targets, fewer than {min_s} s")


def _meta(rec, last, part, rest=None):
    return {'trial': np.full(len(last), rec['idx']), 'angle': np.full(len(last), rec['angle']),
            'session': np.array([rec['session']] * len(last)), 't': rec['t'][last].astype(np.float32),
            'last': last, 'part': np.array([part] * len(last)),
            'rest': (rest[last] if rest is not None else np.zeros(len(last), bool))}


def streams(protocol, recs, feats, targets, window=WINDOW, val_gap=False):
    """Whole trials for the carried-state model (``stateful.py``), aligned with ``build``.

    train : one stream per training-session trial with masks that select exactly the target samples of the
            windowed model's training and validation windows (the last validation window excluded, as
            ``drop_last_val_window`` does), so both models learn and are checked on the same samples.
            val_gap: as in ``build``.
    test  : one stream per test-session trial; ``first`` = index of the first sample the windowed model scores,
            so the carried-state predictions are scored on the same samples.
    """
    spec = SESSION_SPLITS[protocol]
    span = window - 1
    out = {'train': [], 'test': []}
    for r in recs:
        F = np.column_stack([feats[r['idx']], r['pos'][:, None]]).astype(np.float32)
        y = np.asarray(targets[r['idx']], np.float64)
        T = len(F)
        if r['session'] in spec['train']:
            n = T - span
            cut = span + int(n * TRAIN_FRAC)
            i = np.arange(T)
            first_val = cut + (span if val_gap else 0)
            out['train'].append({'X': F, 'y': y, 'rec': r, 'train_mask': (i >= span) & (i < cut),
                                 'val_mask': (i >= first_val) & (i < T - 1)})
        elif r['session'] in spec['test']:
            out['test'].append({'X': F, 'y': y, 'rec': r, 'first': span})
    return out


def build(protocol, recs, feats, targets, window=WINDOW, rest=None, val_gap=False):
    """Train / val / test arrays for ``protocol``: every window of every trial (full overlap; the overlap
    setting train_stride was retired 2026-11-24, tag study-v2-final).

    feats, targets: {trial idx: (T, 4) EMG}, {trial idx: (T,) torque}; rest: optional {trial idx: (T,) bool}
    val_gap       start validation one window span after the training cut (purged split)
    """
    spec = SESSION_SPLITS[protocol]
    parts = {'train': [], 'val': [], 'test': []}
    for r in recs:
        X, y, last = windows(feats[r['idx']], r['pos'], targets[r['idx']], window)
        rr = rest[r['idx']] if rest is not None else None
        n = len(X)
        if r['session'] in spec['train']:
            n_tr = int(n * TRAIN_FRAC)
            tr = slice(0, n_tr)
            parts['train'].append((X[tr], y[tr], _meta(r, last[tr], 'train', rr)))
            v0 = n_tr + (window - 1 if val_gap else 0)
            parts['val'].append((X[v0:], y[v0:], _meta(r, last[v0:], 'val', rr)))
        elif r['session'] in spec['test']:
            parts['test'].append((X, y, _meta(r, last, 'test', rr)))

    out = {}
    for p, items in parts.items():
        out[f'X_{p}'] = np.concatenate([x for x, _, _ in items])
        out[f'y_{p}'] = np.concatenate([y for _, y, _ in items])
        out[f'meta_{p}'] = {k: np.concatenate([m[k] for _, _, m in items]) for k in items[0][2]}
    out['train_angles'] = sorted({float(np.round(a, 3)) for a in out['meta_train']['angle']})
    return out
