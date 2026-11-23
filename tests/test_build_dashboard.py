import pytest

from EXPERIMENTS.analysis import build_dashboard as BD


@pytest.fixture
def calls(monkeypatch):
    seen = []
    for name in ('confirm', 'study_v2_export', 'study_v2_waves'):
        monkeypatch.setattr(BD.STEPS[name], 'main', lambda argv=None, n=name: seen.append(n))

    class Done:
        returncode = 0
    monkeypatch.setattr(BD.subprocess, 'run', lambda cmd, **kw: seen.append('npm') or Done())
    return seen


def test_steps_in_order(calls):
    BD.main([])
    assert calls == ['confirm', 'study_v2_export', 'study_v2_waves', 'npm']


def test_skip_confirm_and_npm(calls):
    BD.main(['--skip-confirm', '--no-npm'])
    assert calls == ['study_v2_export', 'study_v2_waves']


def test_stops_at_first_failure(calls, monkeypatch):
    def boom(argv=None):
        raise RuntimeError('export broke')
    monkeypatch.setattr(BD.STEPS['study_v2_export'], 'main', boom)
    with pytest.raises(SystemExit, match='study_v2_export'):
        BD.main([])
    assert calls == ['confirm']
