import json

from EXPERIMENTS.pipeline import spec as S


def test_every_committed_spec_parses():
    for p in sorted(S.CONFIGS.glob('*.json')):
        assert S.load(p.stem).id == p.stem


def test_template_points_at_current_code():
    """The template is the starting point for new specs: its text names no removed option or old location."""
    t = json.loads((S.CONFIGS / 'TEMPLATE_reference.json').read_text(encoding='utf-8'))
    text = t['description'] + json.dumps(t['settings_help'])
    assert 'dashboard' not in t['description']
    assert 'pipeline/run.py)' not in text and 'EXPERIMENTS/pipeline/settings.py' in text
    for removed in ("'mvc_angle'", "'agc_soft'", "'agc'", "'weights'"):
        assert removed not in text, removed
