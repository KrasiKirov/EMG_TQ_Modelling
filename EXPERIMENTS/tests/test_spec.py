import json
import pytest
from EXPERIMENTS.pipeline import spec as S

BASE = {'id': 'T1', 'title': 'Test', 'changes': {'window': 20}, 'hypothesis': 'h'}


def test_defaults_and_hypothesis_as_description():
    s = S.parse(dict(BASE), 'T1')
    assert not hasattr(s, 'calibration')
    assert s.description == 'h' and s.tags == [] and s.dashboard is None and s.results_from == 'T1'


def test_calibration_field_is_rejected():
    for cal in ({'type': 'gain_offset', 'seconds': 10, 'score_from_s': 30}, {'type': 'none'}):
        with pytest.raises(S.SpecError, match='calibration was removed'):
            S.parse(dict(BASE, calibration=cal))


def test_bad_id_rejected():
    with pytest.raises(S.SpecError):
        S.parse(dict(BASE, id='bad id'))
    with pytest.raises(S.SpecError):
        S.parse(dict(BASE), 'OtherName')


def test_tag_writes_dashboard_block(tmp_path):
    (tmp_path / 'T1.json').write_text(json.dumps(BASE))
    s = S.tag('T1', label='Short', group='Models', description='What it does', tags=['b', 'a'], configs=tmp_path)
    raw = json.loads((tmp_path / 'T1.json').read_text())
    assert raw['dashboard'] == {'group': 'Models', 'label': 'Short'} and raw['tags'] == ['a', 'b']
    assert s.dashboard.id == 'T1' and s.dashboard.name == 'Short' and s.description == 'What it does'


def test_all_existing_specs_parse():
    assert len(S.load_all()) >= 25
