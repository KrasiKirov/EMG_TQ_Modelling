import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(*args):
    return subprocess.run([sys.executable, '-m', 'EXPERIMENTS.pipeline.dashboard', *args], cwd=ROOT,
                          capture_output=True, text=True)


def test_list_shows_specs_and_calibration():
    r = run('list')
    assert r.returncode == 0, r.stderr
    assert 'N0_total_trainmax' in r.stdout and 'gain_offset' in r.stdout


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
