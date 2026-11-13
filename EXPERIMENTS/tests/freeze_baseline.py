"""Copy the current dashboard score rows of every norm/* entry into a test fixture.

Run once, before any pipeline change:  python EXPERIMENTS/tests/freeze_baseline.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / 'EXPERIMENTS/dashboard_react/src/data/presentation-catalog.json'
OUT = Path(__file__).resolve().parent / 'fixtures/baseline_rows.json'

rows = [r for r in json.loads(CATALOG.read_text(encoding='utf-8'))['rows'] if r['exp'].startswith('norm/')]
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(rows, separators=(',', ':')), encoding='utf-8')
print(f'{len(rows)} rows -> {OUT}')
