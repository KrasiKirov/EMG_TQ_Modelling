"""
Resting-level shift between sessions under the primary answer key (passive trials).

For each subject's 2 most toes-down positions and each direction:
  * resting level in the TRAINING session = median of the key while the EMG says
    "relaxed" (first 80 % of the trial, the part used for fitting);
  * resting level in the TESTED session (whole trial);
  * shift = tested - training;
  * R2 of a hypothetical predictor that tracks every push perfectly but keeps the
    training session's resting level: 1 - shift^2 / var(key).  This is NOT an upper
    bound for real models (they can land anywhere); it shows how much of the R2 at a
    position is decided by a level shift the EMG cannot reveal.

Writes results/offset_shift.txt.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(EXP))
from ML.research.data import load_subject       # noqa: E402
from ML.research.passive import PassiveModel     # noqa: E402
from ML.research import truth as T               # noqa: E402

lines = ['subject | position (rad) | direction | resting level training (Nm) | resting level tested (Nm) | shift (Nm) | key std tested (Nm) | R2 of "perfect shape, training resting level"']
for s in ['YES', 'EG', 'JM', 'IES', 'CH', 'HM']:
    b = load_subject(s)
    pm = PassiveModel('static', b)
    tr = T.subject_truth(s)
    recs = {ses: sorted([r for r in b['active'] if r['session'] == ses], key=lambda r: r['angle']) for ses in ('S1', 'S2')}
    for d, (a, t) in {'S1toS2': ('S1', 'S2'), 'S2toS1': ('S2', 'S1')}.items():
        for i in range(2):
            ra, rt = recs[a][i], recs[t][i]
            ka, kt = ra['tq'] - pm.predict(ra), rt['tq'] - pm.predict(rt)
            n80 = int(0.8 * len(ka))
            ma, mt = tr[ra['idx']]['rest'][:n80], tr[rt['idx']]['rest']
            la = np.median(ka[:n80][ma]) if ma.any() else np.nan
            lt = np.median(kt[mt]) if mt.any() else np.nan
            sh, sd = lt - la, np.std(kt)
            lines.append(f'{s} | {rt["angle"]:+.3f} | {d} | {la:+.2f} | {lt:+.2f} | {sh:+.2f} | {sd:.2f} | {1 - sh ** 2 / sd ** 2:+.2f}')
open(os.path.join(EXP, 'results', 'offset_shift.txt'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
