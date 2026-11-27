"""ML.training: the gradient clipping, the training info, and epochs whose validation loss is NaN."""
import numpy as np
import pytest

from EXPERIMENTS.pipeline.fitters import drop_last_val_window
from ML import config as RC

UNIT = dict(batch=256, lr=0.003, lr_factor=0.5, lr_patience=6, early=50, epochs=2)


def _short_schedule(monkeypatch):
    """Train for 2 epochs (ML.config.TRAINING replaced for one test)."""
    for k, v in UNIT.items():
        monkeypatch.setitem(RC.TRAINING, k, v)


def _data(window=20):
    from ML.dataset.prepare import prepare
    rcfg = dict(passive='static', window=window, state='reset')
    return drop_last_val_window(prepare('SYN', rcfg)[0])


def test_clipnorm_windowed_applied(patch_subject, monkeypatch):
    from ML.training import windowed as RT
    _short_schedule(monkeypatch)
    _, info = RT.fit_predict(_data(), 8, 2, 0.3)
    assert info['clipnorm'] == 1.0 and info['optimizer_clipnorm'] == 1.0 and info['shuffle'] is False


# --- NaN validation loss ------------------------------------------------------------------------------------------

def _carry_data():
    from ML.dataset.prepare import prepare
    return drop_last_val_window(prepare('SYN', dict(passive='static', window=20, state='carry', chunk=200))[0])


def _carry(data):
    from ML.training import carried as CA
    s = data['streams']
    return CA.fit_predict_carry(s['train'], s['test'], chunk=200, units=4, layers=1, dropout=0.0)


def _nan_validation(monkeypatch, epochs):
    """The validation pass of carried.fit_predict_carry returns NaN in the given epochs (1-based)."""
    from ML.training import carried as CA
    real, calls = CA.run_streams, []

    def fake(model, X, chunk=200):
        out = real(model, X, chunk)
        calls.append(1)
        return np.full_like(out, np.nan) if len(calls) in epochs else out
    monkeypatch.setattr(CA, 'run_streams', fake)


def test_carried_skips_nan_epoch(patch_subject, monkeypatch):
    _short_schedule(monkeypatch)
    _nan_validation(monkeypatch, {1})
    _, info = _carry(_carry_data())
    assert info['best_epoch'] == 2 and np.isfinite(info['best_val_loss'])


def test_carried_all_nan_raises(patch_subject, monkeypatch):
    _short_schedule(monkeypatch)
    _nan_validation(monkeypatch, {1, 2})
    with pytest.raises(FloatingPointError, match='validation loss'):
        _carry(_carry_data())


def test_windowed_skips_nan_epoch(patch_subject, monkeypatch):
    import tensorflow as tf
    from ML.training import windowed as RT

    class NanFirstEpoch(tf.keras.callbacks.EarlyStopping):
        """Early stopping that sees a NaN validation loss in epoch 1."""
        def on_epoch_end(self, epoch, logs=None):
            if epoch == 0:
                logs['val_loss'] = float('nan')
            super().on_epoch_end(epoch, logs)
    monkeypatch.setattr(tf.keras.callbacks, 'EarlyStopping', NanFirstEpoch)
    _short_schedule(monkeypatch)
    _, info = RT.fit_predict(_data(), 8, 1, 0.0)
    assert info['best_epoch'] == 2 and np.isfinite(info['best_val_loss'])


def test_windowed_all_nan_raises(patch_subject, monkeypatch):
    from ML.training import windowed as RT
    _short_schedule(monkeypatch)
    data = _data()
    data['X_val'] = np.full_like(data['X_val'], np.nan)
    with pytest.raises(FloatingPointError, match='validation loss'):
        RT.fit_predict(data, 8, 1, 0.0)
