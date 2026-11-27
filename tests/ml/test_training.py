"""ML.training.windowed: the gradient clipping and the training info."""
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
