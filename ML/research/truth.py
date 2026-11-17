"""In-trial passive baseline through the EMG-detected rest periods.

Torque while the EMG says "relaxed" is passive, so B(t) is a smooth curve (Gaussian, sigma 8 s) through the median
torque of each rest segment; measured torque minus B(t) is the voluntary torque. Rest is found from the EMG only.
Used for the 'restanchor' target and for the rest flags of every window.
"""

import os
import pickle

import numpy as np

from ML.preprocessing import active_torque as at

from . import config as C

SIGMA_S = 8.0     # s, smoothing of the in-trial passive baseline


def session_rest(bundle, session, max_frac=None):
    """{trial idx: rest segments [(s, e)]} for every active trial of a session.
    ``max_frac`` limits both the statistics and the detection to the first part of each trial."""
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
    """{trial idx: {'truth', 'B', 'rest', ...}} for every active trial (cached; used by the presentation figures)."""
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
