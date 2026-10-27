"""Where does the session-2 passive map match session 2's relaxed level better than the session-1 map?

Relaxed level of a push trial = median torque while all four EMG envelopes are in their lowest 20 %.
"""
import sys

import numpy as np

from ML.preprocessing.calibration import fit_passive, fit_passive_session
from ML.preprocessing.pipeline import PipelineConfig, process_block
from ML.preprocessing.trial_manifest import load_inventory

data_dir, overrides = sys.argv[1:3]
CFG = PipelineConfig(window=50)
for subject in ['HM', 'EG', 'JM', 'YES', 'CH']:
    trials, manifest = load_inventory(subject, data_dir, overrides)
    p1, p2 = fit_passive(trials, manifest, data_dir), fit_passive_session(trials, manifest, 'retest')
    for i, r in enumerate(manifest['trials']):
        if r['kind'] != 'active' or r['status'] != 'included' or r['session'] != 'retest':
            continue
        a, b = r['valid_intervals'][0]
        s = process_block(trials[i], a, b, CFG)[0]
        quiet = np.all(s[:, :4] <= np.percentile(s[:, :4], 20, axis=0), axis=1)
        if quiet.sum() < 100:
            continue
        rest = float(np.median(s[quiet, 5]))
        angle = float(np.median(s[:, 4]))
        m1, m2 = float(p1.predict([angle])[0]), float(p2.predict([angle])[0])
        y = s[:, 5] - m2
        print(f'{subject:3s} {angle:+.3f} rad  rest {rest:+6.2f}  s2 map {m2:+6.2f} (off {rest - m2:+5.2f})  '
              f's1 map {m1:+6.2f} (off {rest - m1:+5.2f})  active SD {np.nanstd(y):5.2f}')
