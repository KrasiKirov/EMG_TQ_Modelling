"""Across people: does auto-gain bring session-2 calf EMG size (per Nm of push) closer to session 1?

Every subject except CH (auto-gain switches itself off for people who push both ways), every angle,
calf muscles GM/GL/SOL. Ratio = session-2 EMG per Nm / session-1 EMG per Nm at the same angle.
"""
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
before, after, toes = [], [], []
for subject in ['YES', 'EG', 'JM', 'IES', 'HM']:
    trials, manifest = load_inventory(subject, data_dir, overrides)
    p1, p2 = fit_passive(trials, manifest, data_dir), fit_passive_session(trials, manifest, 'retest')
    if len(p2.positions) < 2:
        p2 = p1  # IES: documented table for both sessions
    act = lambda sess: [(i, r) for i, r in enumerate(manifest['trials'])
                        if r['kind'] == 'active' and r['status'] == 'included' and r['session'] == sess]
    def sig(i, r):
        a, b = r['valid_intervals'][0]
        return process_block(trials[i], a, b, CFG)[0]
    envs, scale = [], np.zeros(4)
    for i, r in act('test'):
        s = sig(i, r); part = s[: int(len(s) * .8), :4]; envs.append(part); scale = np.maximum(scale, part.max(0))
    lo, hi = agc.agc_priors(envs)
    tests = sorted(act('test'), key=lambda ir: ir[1]['mean_position_rad'])
    for rank, (i1, r1) in enumerate(tests):
        i2, r2 = min(act('retest'), key=lambda ir: abs(ir[1]['mean_position_rad'] - r1['mean_position_rad']))
        s1, s2 = sig(i1, r1), sig(i2, r2)
        y1, y2 = s1[:, 5] - p1.predict(s1[:, 4]), s2[:, 5] - p2.predict(s2[:, 4])
        a1, a2 = agc.apply_agc(s1[:, :4].astype(float), lo, hi)[0], agc.apply_agc(s2[:, :4].astype(float), lo, hi)[0]
        def gain(env, y):
            m = np.isfinite(y) & (np.abs(y) > 0.5)
            return np.array([np.sum(env[m, k] * np.abs(y[m])) / np.sum(y[m] ** 2) for k in range(3)])
        before += list(gain(s2[:, :4] / scale, y2) / gain(s1[:, :4] / scale, y1))
        after += list(gain(a2, y2) / gain(a1, y1))
        toes += [rank < 2] * 3
b, a, toes = np.abs(np.log2(before)), np.abs(np.log2(after)), np.array(toes)
for label, m in [('all angles', np.ones_like(toes, bool)), ('toes-down angles', toes)]:
    print(f'{label}: {m.sum()} cases; typical size mismatch before {2 ** np.median(b[m]):.2f}x, after {2 ** np.median(a[m]):.2f}x; '
          f'off by >25%: before {np.mean(b[m] > np.log2(1.25)) * 100:.0f}%, after {np.mean(a[m] > np.log2(1.25)) * 100:.0f}%; '
          f'closer to session 1 after auto-gain: {np.mean(a[m] < b[m]) * 100:.0f}%')
