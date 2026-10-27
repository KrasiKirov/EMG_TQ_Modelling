"""Chapter 6 data: passive trials, the still holds, the passive maps and the active-torque target.

Uses D01's own functions: dataset_builder._split_passive_segments (cut at device steps),
the stillness test of active_torque.static_hold_passive_map, the cached maps, PassiveModel('static').
Writes data/x04.npz, data/facts_x04.json.
"""
import os

import numpy as np

from common import SUBJECTS, save_facts, save_npz, use_d01_code

use_d01_code()
from ML.config import DATA_DIR, SUBJECTS as REGISTRY          # noqa: E402
from ML.preprocessing import active_torque as at             # noqa: E402
from ML.preprocessing import dataset_builder as db           # noqa: E402
from ML.preprocessing.flb_reader import read_flb             # noqa: E402
from ML.research.data import load_subject                    # noqa: E402
from ML.research.passive import PassiveModel                 # noqa: E402

facts, arrays = {}, {}
fn, sid = REGISTRY['EG']
trials = read_flb(os.path.join(DATA_DIR, fn), subject_id=sid)
db.classify_trials(trials)                                    # sets attrs['session'], as data.py does via _split_trials

pieces = []
for df in trials:
    kind, sess, pos = db._parse_comment(df.attrs.get('comment', ''))
    if kind != 'passive':
        continue
    for seg in db._split_passive_segments(df, db.POSITION_COL):
        p, q = seg[db.POSITION_COL].values, seg[db.TORQUE_COL].values
        keep = not (len(q) < at.STATIC_MIN_DUR_S * 1000 or p.std() > at.STATIC_MAX_POS_STD or q.std() > at.STATIC_MAX_TQ_STD)
        pieces.append(dict(trial=int(df.attrs['trial_index']), session=sess, pos=pos, start=int(seg.index[0]),
                           stop=int(seg.index[-1]) + 1, seconds=len(q) / 1000.0, angle=float(p.mean()),
                           angle_sd_mrad=float(p.std() * 1000), tq_mean=float(q.mean()), tq_sd=float(q.std()), keep=keep))
facts['eg_pieces'] = pieces

# the map from the kept pieces, rebuilt with the code's own function, must equal the cached one
b = load_subject('EG')
for sess, label in (('S1', 'test'), ('S2', 'retest')):
    ptr = [df for df in trials if db._parse_comment(df.attrs.get('comment', ''))[0] == 'passive'
           and (df.attrs.get('session') == 'retest') == (label == 'retest')]
    rebuilt = at.static_hold_passive_map(ptr, db._split_passive_segments, db.POSITION_COL, db.TORQUE_COL,
                                         db._POSITION_TOLERANCE)
    facts[f'eg_map_check_{sess}'] = float(np.max(np.abs(np.asarray(rebuilt) - np.asarray(b['maps']['static'][sess]))))

# the passive trial before EG's p7 (session 1) at 100 Hz for display
TRIAL = 20
df = trials[TRIAL]
arrays.update(p20_t=np.arange(0, len(df), 10) / 1000.0, p20_pos=df['position'].values[::10],
              p20_tq=df['torque'].values[::10])
facts['p20_pieces'] = [x for x in pieces if x['trial'] == TRIAL]

# maps of every person (static = what D01 subtracts; legacy for JM's rejected point)
maps = {}
for s in SUBJECTS:
    bb = load_subject(s)
    maps[s] = {k: {sess: [list(e) for e in bb['maps'][k][sess]] for sess in ('S1', 'S2')} for k in ('static', 'legacy')}
    maps[s]['worksheet'] = bb['maps']['worksheet'] is not None
    maps[s]['n_holds'] = len(bb['holds'])
facts['maps'] = maps

# targets for EG at -0.038 rad, both sessions
pm = PassiveModel('static', b)
for idx, sess in ((21, 'S1'), (37, 'S2')):
    r = next(x for x in b['active'] if x['idx'] == idx)
    P = pm.predict(r)
    arrays.update({f'{sess}_t': r['t'], f'{sess}_tq': r['tq'], f'{sess}_P': P, f'{sess}_target': r['tq'] - P})
    facts[f'eg_{sess}_P_median'] = float(np.median(P))
    facts[f'eg_{sess}_P_range'] = [float(P.min()), float(P.max())]
    facts[f'eg_{sess}_angle'] = float(r['angle'])

save_npz('x04', **arrays)
save_facts('x04', {'x04': facts})
print('pieces kept', sum(p['keep'] for p in pieces), 'of', len(pieces), 'map checks',
      facts['eg_map_check_S1'], facts['eg_map_check_S2'])
print('P median S1 %.3f S2 %.3f' % (facts['eg_S1_P_median'], facts['eg_S2_P_median']))
for p in facts['p20_pieces']:
    print(p)
