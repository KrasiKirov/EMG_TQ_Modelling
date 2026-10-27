"""Paths and helpers shared by the extraction scripts.

Run every script with the project interpreter (.venv311). The D01 code lives, uncommitted,
in the dorsiflexion worktree; ``use_d01_code()`` puts that worktree first on sys.path so
``import ML...`` resolves to D01's package (this branch's ``ML`` is last week's pipeline).

Every script writes its arrays to data/<name>.npz and its numbers to data/facts_<name>.json;
narration/fill.py merges the facts files.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VIDEO = HERE.parent                                   # EXPERIMENTS/pipeline_video
REPO = VIDEO.parent.parent                            # this worktree (experimentation)
DATA = VIDEO / 'data'
DASHBOARD = REPO / 'EXPERIMENTS' / 'dashboard_react'

PROJECTS = Path(os.environ.get('KRASI_ML', r'C:/Users/dramo/Desktop/Projects/krasi_ml'))
D01_WT = Path(os.environ.get('D01_WORKTREE',
                             PROJECTS / 'EMG_TQ_Modelling/.claude/worktrees/dorsiflexion-r2-drop-341137'))
D01_EXP = D01_WT / 'EXPERIMENTS'
RESULTS = D01_EXP / 'results'
DATA_DIR = Path(os.environ.get('EMG_DATA_DIR', PROJECTS / 'EMG_TQ_Modelling/ML/subjects'))
LAST_WEEK_RUNS = Path(os.environ.get('LAST_WEEK_RUNS', PROJECTS / 'EMG_TQ_Modelling_runs/window50_retest_map'))

SUBJECTS = ['CH', 'EG', 'HM', 'IES', 'JM', 'YES']
D01 = 'D01_cleanpassive+autogain+equalweight'
C01 = 'C01_autogain+equalweight'
MUSCLES = ['GM', 'GL', 'SOL', 'TA']


def run_dir(subject):
    """Folder holding D01's saved runs for ``subject`` (C01 for EG and YES, presentation_metrics.py:56)."""
    return RESULTS / (C01 if subject in ('EG', 'YES') else D01) / subject


def use_d01_code():
    os.environ['EMG_DATA_DIR'] = str(DATA_DIR)
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    for p in (str(D01_EXP / 'code'), str(D01_WT)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)


def save_npz(name, **arrays):
    DATA.mkdir(exist_ok=True)
    np.savez_compressed(DATA / f'{name}.npz', **arrays)


def _plain(v):
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return _plain(v.tolist())
    if isinstance(v, np.generic):
        return v.item()
    return v


def save_facts(name, facts):
    DATA.mkdir(exist_ok=True)
    with open(DATA / f'facts_{name}.json', 'w', encoding='utf-8') as f:
        json.dump(_plain(facts), f, indent=1, sort_keys=True)


def load_facts():
    out = {}
    for p in sorted(DATA.glob('facts_*.json')):
        out.update(json.loads(p.read_text(encoding='utf-8')))
    return out


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2))
