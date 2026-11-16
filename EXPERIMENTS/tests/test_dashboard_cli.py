import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(*args):
    return subprocess.run([sys.executable, '-m', 'EXPERIMENTS.pipeline.dashboard', *args], cwd=ROOT,
                          capture_output=True, text=True)


def test_list_shows_specs():
    r = run('list')
    assert r.returncode == 0, r.stderr
    assert 'N0_total_trainmax' in r.stdout and 'cal=' not in r.stdout and 'N0cal' not in r.stdout


def test_tag_rejects_unknown_group():
    r = run('tag', 'X03_lr_0.001', '--label', 'x', '--group', 'Not a group', '--dry-run')
    assert r.returncode != 0 and 'Not a group' in r.stderr


def test_tag_dry_run_shows_the_entry_without_writing():
    path = ROOT / 'EXPERIMENTS/configs/X03_lr_0.001.json'
    before = path.read_text(encoding='utf-8')
    r = run('tag', 'X03_lr_0.001', '--label', 'Zed', '--group', 'Exploration', '--tag', 'lr', '--dry-run')
    assert r.returncode == 0, r.stderr
    assert '"label": "Zed"' in r.stdout and '"lr"' in r.stdout
    assert path.read_text(encoding='utf-8') == before


def test_export_all_skips_tagged_specs_without_results(tmp_path, monkeypatch, capsys):
    import json
    from EXPERIMENTS.pipeline import dashboard as DB, spec as S
    cfg = tmp_path / 'configs'; cfg.mkdir()
    (cfg / 'NORES.json').write_text(json.dumps({'id': 'NORES', 'title': 't', 'changes': {},
                                                'dashboard': {'group': 'Models', 'label': 'x'}}))
    monkeypatch.setattr(S, 'CONFIGS', cfg)
    seen = {}
    def fake(chosen, *a, all_specs=None, **k):
        seen['ids'], seen['sidebar'] = [s.id for s in chosen], [s.id for s in all_specs]
        return {}
    monkeypatch.setattr(DB, 'export', fake)
    DB.main(['export'])
    assert seen['ids'] == [] and 'NORES' not in seen['sidebar'] and 'NORES' in capsys.readouterr().err
