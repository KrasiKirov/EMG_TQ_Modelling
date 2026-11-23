import json

import pytest

from EXPERIMENTS.pipeline import spec as S


def test_dashboard_block_rejected():
    raw = {'id': 'X', 'title': 'x', 'changes': {}, 'dashboard': {'group': 'g', 'label': 'l'}}
    with pytest.raises(S.SpecError, match='dashboard'):
        S.parse(raw)


def test_no_committed_spec_has_dashboard():
    with_dash = [p.stem for p in sorted(S.CONFIGS.glob('*.json'))
                 if 'dashboard' in json.loads(p.read_text(encoding='utf-8'))]
    assert with_dash == []


def test_every_committed_spec_parses():
    assert len(S.load_all()) == len(list(S.CONFIGS.glob('*.json')))
