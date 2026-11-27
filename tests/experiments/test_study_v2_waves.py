import numpy as np
import pytest

from EXPERIMENTS.analysis import study_v2_export as EX, study_v2_waves as WV
from EXPERIMENTS.pipeline import spec as SPEC
from test_export_v2 import PEOPLE, _final


@pytest.fixture
def waves(tmp_path, monkeypatch):
    cfg_dir = tmp_path / 'configs'
    cfg_dir.mkdir()
    monkeypatch.setattr(SPEC, 'CONFIGS', cfg_dir)
    rd = _final(tmp_path / 'results_v2', cfg_dir, bad=2)               # trial 2 = angle 0.1, position 2 of 4
    return rd, WV.build(rd)


def test_worst_positions_follow_final_r2(waves):
    rd, out = waves
    final, _ = EX.final_scores(rd)
    assert set(out['runs']) == set(final) and len(out['runs']) == 18
    w, f = out['runs']['FE_cell_w20_full']['CH'], final['FE_cell_w20_full']['CH']
    assert w['seeds'] == 3 and len(w['positions']) == WV.WORST
    first = w['positions'][0]
    assert (first['k'], first['pos']) == (2, 0.1)
    assert first['median_r2'] == round(float(np.median(f['r2'][2])), 4)
    j = int(np.argmax(f['r2'][2]))
    assert first['best'] == {'seed': f['seeds'][j], 'mae': round(f['mae'][2][j], 4), 'r2': round(f['r2'][2][j], 4)}
    assert first['worst']['r2'] == round(min(f['r2'][2]), 4)
    assert all(w[c] and w[c + '_t'] for c in ('measured', 'best', 'worst'))


def test_single_seed_has_best_only_data(waves):
    _, out = waves
    for p in PEOPLE:
        a1 = out['runs']['FE_A1'][p]
        assert a1['seeds'] == 1
        assert all(x['best'] == x['worst'] and x['best']['seed'] == 10 for x in a1['positions'])
