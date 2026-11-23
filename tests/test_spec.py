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


def test_template_points_at_current_code():
    """The template is the starting point for new specs: its text names no removed option or old location."""
    t = json.loads((S.CONFIGS / 'TEMPLATE_reference.json').read_text(encoding='utf-8'))
    text = t['description'] + json.dumps(t['settings_help'])
    assert 'dashboard' not in t['description']
    assert 'pipeline/run.py)' not in text and 'EXPERIMENTS/pipeline/settings.py' in text
    for removed in ("'mvc_angle'", "'agc_soft'", "'agc'", "'weights'"):
        assert removed not in text, removed
