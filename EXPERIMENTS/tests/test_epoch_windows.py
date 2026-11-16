"""epoch_windows: an epoch is a fixed number of training windows (whole passes over a small training set), so a
setting with few windows gets as many weight updates per epoch as the reference."""
import numpy as np
import pytest

tf = pytest.importorskip('tensorflow')
from ML.research import train as RT  # noqa: E402


def _data(n_train=64, n_val=32, n_test=16, T=20, F=5, seed=0):
    rng = np.random.default_rng(seed)
    d = {}
    for part, n in (('train', n_train), ('val', n_val), ('test', n_test)):
        X = rng.random((n, T, F)).astype(np.float32)
        d[f'X_{part}'], d[f'y_{part}'] = X, X[:, -1, 0] * 3
        d[f'meta_{part}'] = {'angle': np.full(n, -0.3)}
    d['train_angles'] = np.array([-0.3])
    return d


def _fit(**kw):
    return RT.fit_predict(_data(), 'lstm8', 'none', None, 'fast', seed=0, shuffle=False, **kw)


def test_default_epoch_is_one_pass():
    _, info = _fit()
    assert info['passes_per_epoch'] == 1
    assert info['updates'] == info['epochs'] * 1          # 64 windows, batch 64


def test_epoch_windows_repeats_whole_passes():
    _, info = _fit(epoch_windows=64 * 5 + 20)                # rounds to 5 whole passes
    assert info['passes_per_epoch'] == 5
    assert info['updates'] == info['epochs'] * 5             # counted by the optimizer itself


def test_epoch_windows_is_off_by_default():
    from EXPERIMENTS.pipeline.run import BASE
    assert BASE['epoch_windows'] is None


def test_equal_update_configs_match_the_reference_epoch():
    from EXPERIMENTS.pipeline.run import load_config
    for cid, stride in (('WO2_75_eq', 5), ('WO3_50_eq', 10), ('WO4_00_eq', 20)):
        _, cfg = load_config(cid)
        assert (cfg['train_stride'], cfg['epoch_windows'], cfg['train_protocol']) == (stride, 56256, 'fast')
