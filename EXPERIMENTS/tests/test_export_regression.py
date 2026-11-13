import json
import shutil
from pathlib import Path

import pytest
from EXPERIMENTS.pipeline import export as EX, spec as S

ROOT = Path(__file__).resolve().parents[2]
KEYS = ('r2', 'mae', 'rmse', 'bias', 'r2_demeaned', 'mae_demeaned', 'peak')


@pytest.mark.slow
def test_spec_export_reproduces_current_norm_rows(tmp_path):
    baseline = json.loads((ROOT / 'EXPERIMENTS/tests/fixtures/baseline_rows.json').read_text())
    catalog = tmp_path / 'cat.json'
    shutil.copy(ROOT / 'EXPERIMENTS/dashboard_react/src/data/presentation-catalog.json', catalog)
    specs = [s for s in S.load_all() if s.dashboard and s.dashboard.id.startswith('norm/')]
    assert specs, 'run Task 7 (migrate specs) first'
    EX.export(specs, catalog, ROOT / 'EXPERIMENTS/results', ROOT / 'EXPERIMENTS/dashboard_groups.json')
    new = {(r['exp'], r['subj'], r['seed'], r['pos']): r for r in json.loads(catalog.read_text())['rows']}
    assert len(baseline) == sum(1 for k in new if k[0].startswith('norm/') and new[k]['dir'] == 'S1toS2')
    for old in baseline:
        row = new[(old['exp'], old['subj'], old['seed'], old['pos'])]
        for k in KEYS:
            assert row[k] == pytest.approx(old[k], abs=1e-9), (old['exp'], k)
