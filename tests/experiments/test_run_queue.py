"""run_queue.is_done: a stored result counts as done only for the same settings; overlap results are refused."""
import json

import pytest

from EXPERIMENTS.pipeline import run_queue as RQ
from EXPERIMENTS.pipeline.settings import BASE


def test_is_done_ignores_full_overlap_keys(tmp_path):
    f = tmp_path / 'seed0.json'
    f.write_text(json.dumps({'config': dict(BASE, train_stride=1, epoch_windows='full')}))
    assert RQ.is_done(f, dict(BASE))


def test_is_done_refuses_an_overlap_result(tmp_path):
    f = tmp_path / 'seed0.json'
    f.write_text(json.dumps({'config': dict(BASE, train_stride=25, epoch_windows='full')}))
    with pytest.raises(RQ.ConfigMismatch, match='retired'):
        RQ.is_done(f, dict(BASE))
