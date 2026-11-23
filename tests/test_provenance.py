import json
import os

import numpy as np
import pytest

from EXPERIMENTS.pipeline import provenance as PV
from EXPERIMENTS.pipeline import run as PR

FAST = dict(batch=256, lr=0.003, lr_factor=0.5, lr_patience=6, early=15, epochs=2)


def test_config_hash_stable():
    a = {'units': 8, 'layers': 2, 'lr': 0.01}
    b = {'lr': 0.01, 'layers': 2, 'units': 8}
    assert PV.config_hash(a) == PV.config_hash(b)
    assert len(PV.config_hash(a)) == 16
    assert PV.config_hash(a) != PV.config_hash(dict(a, units=16))


def _data(patch_subject, window=20):
    from ML.research.run import prepare
    rcfg = dict(passive='static', emg='trainmax', window=window, envelope='causal', trim_end_s=1.0, input_step=1,
                train_stride=1, state='reset', lp_order=2, val_gap=True, metadata_passive=True)
    data, _ = prepare('SYN', 'S1toS2', rcfg)
    return PR.drop_last_val_window(data)


def test_same_seed_identical(patch_subject, monkeypatch):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    from ML.research import config as RC, train as RT
    monkeypatch.setitem(RC.PROTOCOLS, 'unit', dict(FAST, epochs=3))
    data = _data(patch_subject)
    p1, _ = RT.fit_predict(data, 'lstm8', protocol='unit', seed=0, shuffle=False)
    p2, _ = RT.fit_predict(data, 'lstm8', protocol='unit', seed=0, shuffle=False)
    assert np.array_equal(p1, p2)


def _cfg(**over):
    cfg = dict(PR.BASE, passive='static', envelope='causal', trim_end_s=1.0, lp_order=2, val_gap=True,
               metadata_passive=True, shuffle=False, train_protocol='unit', units=4, layers=1, dropout=0.0)
    cfg.update(over)
    return cfg


@pytest.fixture
def unit_run(patch_subject, tmp_results, monkeypatch):
    from ML.research import config as RC
    monkeypatch.setitem(RC.PROTOCOLS, 'unit', FAST)
    monkeypatch.setattr(PR, 'load_config', lambda exp: ({'id': exp}, _cfg()))
    monkeypatch.setattr(PR, 'protocol_hash', lambda: 'p' * 64)
    monkeypatch.setattr(PV, 'collect', lambda **kw: dict(
        git_commit='abc', git_dirty=False, data_sha256={}, versions={}, threads=1, host='h',
        protocol_hash=kw.get('protocol_hash')))
    return tmp_results


def test_dirty_tree_refused_for_study_runs(unit_run, monkeypatch):
    monkeypatch.setattr(PV, 'collect', lambda **kw: dict(git_commit='abc', git_dirty=True, data_sha256={},
                                                         versions={}, threads=1, host='h', protocol_hash=None))
    with pytest.raises(SystemExit, match='uncommitted'):
        PR.main(['--exp', 'TS2_R1_x', '--subject', 'SYN', '--seed', '0'])
    PR.main(['--exp', 'W00_unit', '--subject', 'SYN', '--seed', '0'])   # v1 id runs


def test_direction_is_fixed_s1_to_s2(unit_run):
    with pytest.raises(SystemExit):
        PR.main(['--exp', 'W00_unit', '--subject', 'SYN', '--direction', 'S2toS1', '--seed', '0'])
    PR.main(['--exp', 'W00_unit', '--subject', 'SYN', '--seed', '0'])
    res = json.loads((unit_run / 'results' / 'W00_unit' / 'SYN' / 'S1toS2' / 'seed0.json').read_text())
    assert res['direction'] == 'S1toS2'


def test_study_runs_write_results_v2(unit_run):
    PR.main(['--exp', 'TS2_R1_x', '--subject', 'SYN', '--seed', '0'])
    PR.main(['--exp', 'W00_unit', '--subject', 'SYN', '--seed', '0'])
    v2 = unit_run / 'results_v2' / 'TS2_R1_x' / 'SYN' / 'S1toS2' / 'seed0.json'
    v1 = unit_run / 'results' / 'W00_unit' / 'SYN' / 'S1toS2' / 'seed0.json'
    assert v2.exists() and v1.exists()
    assert not (unit_run / 'results' / 'TS2_R1_x').exists()
    res = json.loads(v2.read_text())
    assert res['config_hash'] == PV.config_hash(res['config'])
    assert res['provenance']['protocol_hash'] == 'p' * 64
    assert json.loads(v1.read_text())['provenance']['protocol_hash'] is None


def test_results_root():
    assert PR.results_root('TS2_R1_a') == PR.RESULTS_V2
    assert PR.results_root('FE_A0') == PR.RESULTS_V2
    assert PR.results_root('W00_reference') == PR.RESULTS
