"""Shared fixtures: a synthetic subject with the same structure as ML.research.data.load_subject.

8 positions from -0.8 to +0.2 rad, 2 sessions, 90 s at 100 Hz per trial; torque = 5 * sum of the 4 envelopes plus
a passive torque that depends on the angle only. No participant data is used.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

FS = 100
T_TRIAL = 9000                    # 90 s
ANGLES = np.linspace(-0.8, 0.2, 8)


def passive_of(angle):
    return 10.0 * np.asarray(angle) + 2.0


def _envelope(rng, T):
    """Smooth, positive bursts with rest between them."""
    raw = rng.standard_normal((T + 400, 4))
    k = np.hanning(201)
    k /= k.sum()
    sm = np.column_stack([np.convolve(raw[:, c], k, mode='same') for c in range(4)])[200:200 + T]
    gate = (np.sin(np.arange(T) / FS * 2 * np.pi / 12.0) > 0).astype(float)[:, None]
    return np.abs(sm) * 4.0 * gate + 0.01


def make_bundle(seed=0, T=T_TRIAL, subject='SYN'):
    rng = np.random.default_rng(seed)
    recs, idx = [], 0
    for sess in ('S1', 'S2'):
        for a in ANGLES:
            env = _envelope(rng, T)
            pos = np.full(T, a) + 0.001 * rng.standard_normal(T)
            tq = 5.0 * env.sum(axis=1) + passive_of(a)
            recs.append({'idx': idx, 'comment': f'{sess} {a:.2f}', 'session': sess, 'angle': float(a),
                         't': 1.0 + np.arange(T) / FS, 'env_raw': env, 'env_max1k': env.max(axis=0) * 1.1,
                         'pos': pos, 'tq': tq, 'hold': None})
            idx += 2
    entries = sorted((float(a), float(passive_of(a))) for a in ANGLES)
    maps = {k: {'S1': list(entries), 'S2': list(entries)} for k in ('legacy', 'static', 'worksheet')}
    mvc = [{'idx': 100 + i, 'angle': float(a), 'env_peak': np.full(4, 5.0), 'tq_peak': 50.0}
           for i, a in enumerate(ANGLES[::3])]
    return {'subject': subject, 'emg_cols': ['GM', 'GL', 'SOL', 'TA'], 'active': recs, 'holds': [], 'mvc': mvc,
            'maps': maps, 'sessions': {'S1': [r['idx'] for r in recs if r['session'] == 'S1'],
                                       'S2': [r['idx'] for r in recs if r['session'] == 'S2']}}


@pytest.fixture
def synthetic_bundle():
    return make_bundle()


@pytest.fixture
def patch_subject(monkeypatch, synthetic_bundle):
    """load_subject('SYN', ...) returns the synthetic bundle (any keyword arguments are ignored)."""
    from ML.research import data as RD, run as RR

    def fake(subject, *args, **kwargs):
        assert subject == 'SYN', subject
        return synthetic_bundle
    monkeypatch.setattr(RD, 'load_subject', fake)
    monkeypatch.setattr(RR, 'load_subject', fake)
    return synthetic_bundle


@pytest.fixture
def tmp_results(tmp_path, monkeypatch):
    """Point run.py's results folders at a temporary directory."""
    from EXPERIMENTS.pipeline import run as PR
    v1, v2 = tmp_path / 'results', tmp_path / 'results_v2'
    monkeypatch.setattr(PR, 'RESULTS', str(v1))
    monkeypatch.setattr(PR, 'RESULTS_V2', str(v2))
    return tmp_path
