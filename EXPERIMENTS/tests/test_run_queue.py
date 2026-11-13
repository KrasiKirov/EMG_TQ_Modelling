import json

import pytest

from EXPERIMENTS.pipeline.run import BASE
from EXPERIMENTS.pipeline.run_queue import ConfigMismatch, is_done


def _write(path, cfg):
    path.write_text(json.dumps({'config': cfg}), encoding='utf-8')
    return path


def test_missing_result_is_not_done(tmp_path):
    assert is_done(tmp_path / 'seed0.json', dict(BASE)) is False


def test_matching_config_is_done(tmp_path):
    cfg = dict(BASE, passive='static')
    assert is_done(_write(tmp_path / 'seed0.json', cfg), cfg) is True


def test_key_added_to_defaults_later_still_matches(tmp_path):
    cfg = dict(BASE)
    old = {k: v for k, v in cfg.items() if k != 'shuffle'}   # results saved before 'shuffle' existed
    assert is_done(_write(tmp_path / 'seed0.json', old), cfg) is True


def test_changed_config_fails_loudly(tmp_path):
    path = _write(tmp_path / 'seed0.json', dict(BASE, window=20))
    with pytest.raises(ConfigMismatch, match='window'):
        is_done(path, dict(BASE, window=50))
