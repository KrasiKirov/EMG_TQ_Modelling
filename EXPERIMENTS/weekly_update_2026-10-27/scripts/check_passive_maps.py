"""Load every subject through the 500 ms + retest-passive-map pipeline and report coverage.

Usage: python check_passive_maps.py <overrides.json> <subject> [<subject> ...]
"""
import os
import sys

import numpy as np

from ML.preprocessing.pipeline import PipelineConfig, load_bundle

data_dir = os.environ['EMG_DATA_DIR']
overrides = sys.argv[1]
cfg = PipelineConfig(window=50, train_stride=5, session_passive_calibration=True)
for subject in sys.argv[2:]:
    try:
        b = load_bundle(subject, cfg, data_dir=data_dir, overrides_path=overrides)
    except Exception as exc:
        print(f'{subject}: FAILED {type(exc).__name__}: {exc}')
        continue
    counts = {k: len(d.y) for k, d in b.splits.items()}
    print(f'{subject}: windows {counts}; mode {b.preprocessing["passive_calibration_mode"]}')
    for session, cal in b.preprocessing['passive_by_session'].items():
        print(f'  {session:6s} map: {len(cal["positions"])} pts '
              f'[{cal["positions"][0]:+.3f} .. {cal["positions"][-1]:+.3f}]' if cal['positions'] else f'  {session} map: none')
    records = {r['trial']: r for r in b.manifest['trials']}
    for split in ('train', 'test'):
        trials = np.unique(b[split].metadata['trial'])
        angles = sorted(round(records[int(t)]['mean_position_rad'], 3) for t in trials)
        print(f'  {split:5s}: {len(trials)} positions {angles}')
    bad = [c for c in b.coverage if c.get('status') == 'unavailable']
    if bad:
        print(f'  unavailable blocks: {[(c["trial"], c["split"]) for c in bad]}')
