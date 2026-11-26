"""Folders of the repository, in one place (pathlib.Path)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'EXPERIMENTS'
CONFIGS = EXP / 'configs'                 # one spec per experiment
RESULTS = EXP / 'results'                 # runs outside a pre-registered study (not in git)
RESULTS_V2 = EXP / 'results_v2'           # study v2 runs (protocol.STUDIES)
PROTOCOL_DIR = EXP / 'protocol'           # pre-registrations
LOGS = EXP / 'logs'                       # run_queue console output (not tracked)
DASH_DATA = EXP / 'dashboard_react' / 'src' / 'data'
