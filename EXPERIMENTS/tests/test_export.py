import json
import numpy as np
from EXPERIMENTS.pipeline import export as EX, spec as S


def make_run(results, exp, subject, seed, config=None):
    d = results / exp / subject / 'S1toS2'; d.mkdir(parents=True, exist_ok=True)
    t = np.concatenate([np.arange(0, 60, .01), np.arange(0, 60, .01)])
    ang = np.repeat([-0.2, 0.1], 6000).astype(np.float32)
    y = np.sin(t) * 3
    np.savez_compressed(d / f'seed{seed}.npz', y_own_target=y.astype(np.float32), y_pred=(0.8 * y + 1).astype(np.float32),
                        trial=np.repeat([1, 2], 6000).astype(np.int16), angle=ang, session=np.ones(12000, np.int8),
                        t=(t + np.repeat([0, 100], 6000)).astype(np.float32))
    (d / f'seed{seed}.json').write_text(json.dumps({'config': config or {'window': 20}, 'train_s': 1.0, 'prep_s': 0.5}))


def setup(tmp_path):
    results = tmp_path / 'results'; make_run(results, 'T1', 'EG', 0)
    raw = {'id': 'T1', 'title': 'Test', 'changes': {'window': 20}, 'description': 'desc', 'tags': ['demo'],
           'dashboard': {'group': 'Models', 'label': 'Test model', 'blurb': 'One line.'}}
    catalog = tmp_path / 'cat.json'
    catalog.write_text(json.dumps({'experiments': {}, 'rows': [], 'positions': {}, 'targets': {}, 'predictions': {},
                                   'means': {}, 'times': {}, 'grid': [0, 0.01, 6000], 'targetPool': {}}))
    groups = tmp_path / 'g.json'
    groups.write_text(json.dumps({'landing': 'T1', 'groups': [{'label': 'Models', 'caption': '', 'tone': 'teal', 'open': True}],
                                  'extra_items': []}))
    return results, S.parse(raw, 'T1'), catalog, groups


def test_export_adds_tagged_entry_and_groups(tmp_path):
    results, spec, catalog, groups = setup(tmp_path)
    summary = EX.export([spec], catalog, results, groups)
    data = json.loads(catalog.read_text())
    assert summary == {'T1': {'EG': 1}}
    assert {r['exp'] for r in data['rows']} == {'T1'} and len(data['rows']) == 2
    assert all('err_after_gain_offset_mae' in r for r in data['rows'])
    e = data['experiments']['T1']
    assert e['title'] == 'Test model' and e['tags'] == ['demo'] and 'calibration' not in e
    assert not any(k == 'Calibration' for k, _ in e['overview']['settings'])
    assert e['kind'] == 'spec' and e['blurb'] == 'One line.'
    assert data['dashboard'] == {'landing': 'T1', 'groups': [{'label': 'Models', 'caption': '', 'tone': 'teal', 'open': True,
                                                              'items': [{'id': 'T1', 'label': 'Test model'}]}]}


def test_export_is_idempotent(tmp_path):
    results, spec, catalog, groups = setup(tmp_path)
    EX.export([spec], catalog, results, groups)
    EX.export([spec], catalog, results, groups)
    assert len(json.loads(catalog.read_text())['rows']) == 2


def test_unknown_group_is_an_error(tmp_path):
    import pytest
    results, spec, catalog, groups = setup(tmp_path)
    spec.dashboard.group = 'Nope'
    with pytest.raises(ValueError, match='Nope'):
        EX.export([spec], catalog, results, groups)


def test_report_file_is_attached(tmp_path):
    results, spec, catalog, groups = setup(tmp_path)
    report = tmp_path / 'REPORT.md'; report.write_text('# Study report')
    spec.dashboard.text['report'] = str(report)
    EX.export([spec], catalog, results, groups)
    assert json.loads(catalog.read_text())['experiments']['T1']['reportText'] == '# Study report'


def test_runs_with_different_configs_are_an_error(tmp_path):
    import pytest
    results, spec, catalog, groups = setup(tmp_path)
    make_run(results, 'T1', 'JM', 0, config={'window': 50})        # one person re-run after the spec changed
    before = catalog.read_text()
    with pytest.raises(ValueError, match='config'):
        EX.export([spec], catalog, results, groups)
    assert catalog.read_text() == before


def test_key_added_to_defaults_later_is_not_a_config_change(tmp_path):
    from EXPERIMENTS.pipeline.run import BASE
    results, spec, catalog, groups = setup(tmp_path)
    make_run(results, 'T1', 'JM', 0, config={'window': 20, 'shuffle': BASE['shuffle']})
    assert EX.export([spec], catalog, results, groups) == {'T1': {'EG': 1, 'JM': 1}}


def test_duplicate_dashboard_ids_are_an_error(tmp_path):
    import pytest
    results, spec, catalog, groups = setup(tmp_path)
    twin = S.parse({'id': 'T2', 'title': 'Twin', 'changes': {}, 'results_from': 'T1',
                    'dashboard': {'group': 'Models', 'label': 'Twin', 'id': 'T1'}}, 'T2')
    with pytest.raises(ValueError, match='T1'):
        EX.export([spec, twin], catalog, results, groups)


def test_groups_without_entries_are_left_out(tmp_path):
    results, spec, catalog, groups = setup(tmp_path)
    g = json.loads(groups.read_text()); g['groups'].append({'label': 'Empty', 'caption': '', 'tone': 'blue', 'open': True})
    groups.write_text(json.dumps(g))
    EX.export([spec], catalog, results, groups)
    assert [x['label'] for x in json.loads(catalog.read_text())['dashboard']['groups']] == ['Models']
