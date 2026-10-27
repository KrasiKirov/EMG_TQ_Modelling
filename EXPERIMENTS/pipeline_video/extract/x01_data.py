"""Chapter 1 data: the trials in EG's file, raw 1 kHz channels, one active trial, the eight angles.

Reads EG's recording with D01's reader (ML.preprocessing.flb_reader, dorsiflexion worktree) and the
cached 100 Hz bundles (ML.research.data.load_subject). Writes data/x01.npz, data/facts_x01.json.
"""
import numpy as np

from common import SUBJECTS, save_facts, save_npz, use_d01_code

use_d01_code()
from ML.config import DATA_DIR, SUBJECTS as REGISTRY          # noqa: E402
from ML.preprocessing import dataset_builder as db           # noqa: E402
from ML.preprocessing.flb_reader import read_flb             # noqa: E402
from ML.research.data import load_subject                    # noqa: E402
import os                                                    # noqa: E402

facts, arrays = {}, {}

# ── every person: sessions and angles (cached bundles = what the runs used) ──
people = {}
for s in SUBJECTS:
    b = load_subject(s)
    people[s] = {sess: sorted(round(r['angle'], 3) for r in b['active'] if r['session'] == sess) for sess in ('S1', 'S2')}
    people[s]['n_samples'] = sorted({len(r['t']) for r in b['active']})
facts['people'] = people

# ── EG's file: all 40 trials in recording order ──────────────────────────────
fn, sid = REGISTRY['EG']
trials = read_flb(os.path.join(DATA_DIR, fn), subject_id=sid)
inv = []
for df in trials:
    kind, sess, pos = db._parse_comment(df.attrs.get('comment', ''))
    inv.append(dict(idx=int(df.attrs['trial_index']), kind=kind, session=sess or '', pos=pos or '',
                    seconds=round(len(df) / 1000.0, 1), angle=round(float(df['position'].mean()), 3)))
facts['eg_trials'] = inv
facts['eg_n_trials'] = len(inv)
facts['eg_n_mvc'] = sum(t['kind'] == 'mvc' for t in inv)
facts['eg_mvc_seconds'] = sorted({t['seconds'] for t in inv if t['kind'] == 'mvc'})
facts['eg_active_seconds'] = sorted({t['seconds'] for t in inv if t['kind'] == 'active'})
facts['eg_channel_names'] = list(trials[21].attrs['chanName'])

b = load_subject('EG')
recs = {r['idx']: r for r in b['active']}
S1_IDX, S2_IDX = 21, 37                                       # EG at -0.038 rad, test / retest
r = recs[S1_IDX]
facts['eg_s1_idx'], facts['eg_s2_idx'] = S1_IDX, S2_IDX
facts['eg_angle'] = round(r['angle'], 3)
facts['eg_angle_sd_mrad'] = float(np.std(r['pos']) * 1000)

# pushes: torque below the midpoint of its 5th-95th percentile for at least 0.5 s
tq = r['tq']
mid = 0.5 * (np.percentile(tq, 5) + np.percentile(tq, 95))
below = np.r_[False, tq < mid, False]
edges = np.flatnonzero(np.diff(below.astype(int)))
runs = [(s, e) for s, e in zip(edges[::2], edges[1::2]) if e - s >= 50]
facts['eg_push_count'] = len(runs)
facts['eg_push_mid_Nm'] = float(mid)
facts['eg_tq_min_Nm'] = float(tq.min())
arrays.update(trial_t=r['t'], trial_tq=tq, trial_pos=r['pos'], push_runs=np.array(runs))

# raw 1 kHz around the third push onset (2 s): all six channels
df = trials[S1_IDX]
onset_100 = runs[2][0]                                        # index in the 100 Hz trial (t[0] = 1 s)
onset_raw = 1000 + onset_100 * 10
s0, s1 = onset_raw - 800, onset_raw + 1200
for c in ('gm', 'gl', 'sol', 'ta', 'position', 'torque'):
    arrays[f'raw_{c}'] = df[c].values[s0:s1].astype(np.float32)
arrays['raw_t'] = (np.arange(s0, s1) / 1000.0).astype(np.float32)
# the same 2 s around a push onset in session 2 (trial 37, same angle)
r2 = recs[S2_IDX]
tq2 = r2['tq']
mid2 = 0.5 * (np.percentile(tq2, 5) + np.percentile(tq2, 95))
b2 = np.r_[False, tq2 < mid2, False]
e2 = np.flatnonzero(np.diff(b2.astype(int)))
runs2 = [(s, e) for s, e in zip(e2[::2], e2[1::2]) if e - s >= 50]
o2 = 1000 + runs2[2][0] * 10
for c in ('gm', 'gl', 'sol', 'ta', 'torque'):
    arrays[f'raw2_{c}'] = trials[S2_IDX][c].values[o2 - 800:o2 + 1200].astype(np.float32)
arrays['raw2_t'] = (np.arange(o2 - 800, o2 + 1200) / 1000.0).astype(np.float32)
facts['eg_raw_window_s'] = [s0 / 1000.0, s1 / 1000.0]
facts['eg_raw_ranges'] = {c: [float(df[c].values.min()), float(df[c].values.max())] for c in ('gm', 'gl', 'sol', 'ta')}

# the eight angles and the size of the pushes (largest |measured torque| per trial)
for sess in ('S1', 'S2'):
    rs = sorted((x for x in b['active'] if x['session'] == sess), key=lambda x: x['angle'])
    facts[f'eg_{sess}_angles'] = [round(x['angle'], 3) for x in rs]
    facts[f'eg_{sess}_peak_abs_tq_Nm'] = [float(np.max(np.abs(x['tq']))) for x in rs]
    facts[f'eg_{sess}_idx'] = [x['idx'] for x in rs]

save_npz('x01', **arrays)
save_facts('x01', {'x01': facts})
print('pushes', len(runs), 'angle sd mrad %.2f' % facts['eg_angle_sd_mrad'], 'tq min %.2f' % tq.min())
print('S1 angles', facts['eg_S1_angles'], 'peaks', np.round(facts['eg_S1_peak_abs_tq_Nm'], 2))
