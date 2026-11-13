import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('vh', ROOT / 'EXPERIMENTS/code/verify_handoff.py')
vh = importlib.util.module_from_spec(spec); spec.loader.exec_module(vh)


def test_manifest_order_is_case_insensitive_on_every_platform():
    paths = [ROOT / 'a/stageD.csv', ROOT / 'a/X00/s.npz', ROOT / 'a/all.csv']
    assert [p.name for p in sorted(paths, key=vh.manifest_order)] == ['all.csv', 'stageD.csv', 's.npz']
