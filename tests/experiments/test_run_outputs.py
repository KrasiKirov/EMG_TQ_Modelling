"""pipeline/run.py end to end on the synthetic subject: validation MAE and files, training info, the lagged ridge."""
import json

import numpy as np
import pytest

from EXPERIMENTS.pipeline import run as PR, settings as ST
from ML import config as RC

UNIT = dict(batch=256, lr=0.003, lr_factor=0.5, lr_patience=6, early=50, epochs=2)


def _short_schedule(monkeypatch):
    """Train for 2 epochs (ML.config.TRAINING replaced for one test)."""
    for k, v in UNIT.items():
        monkeypatch.setitem(RC.TRAINING, k, v)


def _cfg(**over):
    cfg = dict(ST.BASE, units=4, layers=1, dropout=0.0)
    cfg.update(over)
    return cfg


@pytest.fixture
def run_unit(patch_subject, tmp_results, monkeypatch):
    from EXPERIMENTS.pipeline import provenance as PV
    _short_schedule(monkeypatch)
    monkeypatch.setattr(PR, 'protocol_hash', lambda *a: 'p' * 64)
    monkeypatch.setattr(PV, 'collect', lambda **kw: dict(git_commit='abc', git_dirty=False, data_sha256={},
                                                         versions={}, threads=1, host='h', protocol_hash=None))

    def go(exp, **over):
        monkeypatch.setattr(PR, 'load_config', lambda e: ({'id': e}, _cfg(**over)))
        PR.main(['--exp', exp, '--subject', 'SYN', '--seed', '0'])
        stem = tmp_results / 'results_v2' / exp / 'SYN' / 'S1toS2' / 'seed0'
        return json.loads(stem.with_suffix('.json').read_text()), np.load(str(stem) + '_val.npz')
    return go


def test_val_mae_recorded(run_unit):
    res, v = run_unit('TS2_unit_a')
    assert np.isclose(res['train_info']['val_mae'], np.mean(np.abs(v['y_pred'] - v['y'])))
    assert v['common'].dtype == bool and v['common'].any()


def test_val_file_has_no_test_predictions(run_unit):
    _, v = run_unit('TS2_unit_b')
    assert set(np.unique(v['session']).tolist()) == {0}       # 0 = S1


def test_info_logs_clipnorm_cap_shuffle(run_unit):
    for exp, over in (('TS2_unit_w', {}), ('TS2_unit_c', dict(state='carry', chunk=200))):
        res, v = run_unit(exp, **over)
        ti = res['train_info']
        assert ti['clipnorm'] == 1.0 and ti['max_epochs'] == UNIT['epochs'] and ti['shuffle'] is False
        assert np.isclose(ti['val_mae'], np.mean(np.abs(v['y_pred'] - v['y'])))


def test_ridge_lagged_run(run_unit):
    res, v = run_unit('FE_unit_ridge', model='ridge_lagged', window=51)
    ti = res['train_info']
    assert str(ti['alpha']) in ti['val_mae_by_alpha'] and ti['n_params'] == 50
    assert np.isclose(ti['val_mae'], np.mean(np.abs(v['y_pred'] - v['y'])))
