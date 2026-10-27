"""Do session-1 and session-2 trials follow the same push cue, and how much does auto-gain align EMG size?"""
import importlib.util
import sys

import numpy as np

from ML.preprocessing.calibration import fit_passive, fit_passive_session
from ML.preprocessing.pipeline import PipelineConfig, process_block
from ML.preprocessing.trial_manifest import load_inventory

data_dir, overrides, agc_path = sys.argv[1:4]
spec = importlib.util.spec_from_file_location('emg_agc', agc_path)
agc = importlib.util.module_from_spec(spec); spec.loader.exec_module(agc)
CFG = PipelineConfig(window=50)

for subject in ['JM', 'HM', 'EG']:
    trials, manifest = load_inventory(subject, data_dir, overrides)
    p1, p2 = fit_passive(trials, manifest, data_dir), fit_passive_session(trials, manifest, 'retest')
    act = lambda sess: [(i, r) for i, r in enumerate(manifest['trials'])
                        if r['kind'] == 'active' and r['status'] == 'included' and r['session'] == sess]
    def sig(i, r):
        a, b = r['valid_intervals'][0]
        return process_block(trials[i], a, b, CFG)[0]
    envs, scale = [], np.zeros(4)
    for i, r in act('test'):
        s = sig(i, r); part = s[: int(len(s) * .8), :4]; envs.append(part); scale = np.maximum(scale, part.max(0))
    lo, hi = agc.agc_priors(envs)
    for i1, r1 in act('test'):
        i2, r2 = min(act('retest'), key=lambda ir: abs(ir[1]['mean_position_rad'] - r1['mean_position_rad']))
        s1, s2 = sig(i1, r1), sig(i2, r2)
        y1, y2 = s1[:, 5] - p1.predict(s1[:, 4]), s2[:, 5] - p2.predict(s2[:, 4])
        n = min(len(y1), len(y2))
        best = max(range(-300, 301, 10), key=lambda L: np.corrcoef(y1[max(0, L):n + min(0, L)], y2[max(0, -L):n - max(0, L)])[0, 1])
        corr = np.corrcoef(y1[max(0, best):n + min(0, best)], y2[max(0, -best):n - max(0, best)])[0, 1]
        a1, a2 = agc.apply_agc(s1[:, :4].astype(float), lo, hi)[0], agc.apply_agc(s2[:, :4].astype(float), lo, hi)[0]
        def gain(env, y):  # EMG per Nm of push (least squares through the origin), on push samples
            m = np.abs(y) > 0.5
            return np.array([np.sum(env[m, k] * np.abs(y[m])) / np.sum(y[m] ** 2) for k in range(4)])
        before = gain(s2[:, :4] / scale, y2) / gain(s1[:, :4] / scale, y1)
        after = gain(a2, y2) / gain(a1, y1)
        print(f'{subject} {r1["mean_position_rad"]:+.2f} rad: torque SD {np.std(y1):4.1f}/{np.std(y2):4.1f} Nm, cue corr {corr:.2f} (lag {best / 100:+.1f} s) '
              f'| S2/S1 EMG per Nm before {np.round(before, 2)} after {np.round(after, 2)}')
