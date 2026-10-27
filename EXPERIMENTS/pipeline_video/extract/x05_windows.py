"""Chapters 7 and 8 data: windows, the session-1 split, knots and the per-angle standard deviations.

Builds EG's arrays exactly as run_experiment.py does (prepare + drop_last_val_window), once with the
C01 config (how EG's D01 runs are stored) and once with D01's config, and checks they are identical.
Writes data/x05.npz, data/facts_x05.json.
"""
import numpy as np

from common import C01, D01, save_facts, save_npz, use_d01_code

use_d01_code()
import run_experiment as RE                                   # noqa: E402
from ML.research.run import prepare                          # noqa: E402
from ML.research import train as RT                          # noqa: E402
from ML.research import config as RC                         # noqa: E402

facts, arrays = {}, {}
built = {}
for exp in (C01, D01):
    _, cfg = RE.load_config(exp)
    data, details = prepare('EG', 'S1toS2', {k: cfg[k] for k in ('passive', 'emg', 'window')})
    built[exp] = (RE.drop_last_val_window(data), details, cfg)
data, details, cfg = built[C01]
d01 = built[D01][0]
facts['c01_equals_d01'] = all(np.array_equal(data[k], d01[k]) for k in
                              ('X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'))
facts['shapes'] = {p: list(data[f'X_{p}'].shape) for p in ('train', 'val', 'test')}
mt, mv, me = data['meta_train'], data['meta_val'], data['meta_test']
facts['per_trial'] = {int(i): dict(train=int(np.sum(mt['trial'] == i)), val=int(np.sum(mv['trial'] == i)))
                      for i in np.unique(mt['trial'])}
facts['test_per_trial'] = {int(i): int(np.sum(me['trial'] == i)) for i in np.unique(me['trial'])}
facts['config'] = {k: cfg[k] for k in ('passive', 'emg', 'balance', 'window', 'train_protocol', 'model', 'dropout')}
facts['protocol'] = RC.PROTOCOLS[cfg['train_protocol']]

# one training trial: EG session 1 at -0.038 rad
TRIAL = 21
m = mt['trial'] == TRIAL
last = mt['last'][m]
X_tr = data['X_train'][m]
F = np.concatenate([X_tr[0], X_tr[1:, -1, :]])               # the trial's (T', 5) features, rebuilt from windows
y_tr = data['y_train'][m]
mval = mv['trial'] == TRIAL
n_win = int(m.sum() + mval.sum() + 1)                         # + the dropped last validation window
n_tr = int(m.sum())
facts['trial'] = dict(idx=TRIAL, n_samples=n_win + RC.WINDOW - 1, n_windows=n_win, n_train=n_tr,
                      n_val=int(mval.sum()), first_val_last=int(mv['last'][mval][0]),
                      last_train_last=int(last[-1]), overlap_windows=RC.WINDOW - 1,
                      t_cut=float(mv['t'][mval][0]))
# the example window: ends 0.15 s after the start of the trial's third push
tq_target = np.concatenate([np.full(RC.WINDOW - 1, np.nan), y_tr])
mid = 0.5 * (np.nanpercentile(tq_target, 5) + np.nanpercentile(tq_target, 95))
below = np.r_[False, np.nan_to_num(tq_target, nan=0) < mid, False]
edges = np.flatnonzero(np.diff(below.astype(int)))
runs = [(a, e) for a, e in zip(edges[::2], edges[1::2]) if e - a >= 50]
wlast = runs[2][0] + 15
w = int(np.flatnonzero(last == wlast)[0])
facts['example'] = dict(window=w, last=wlast, t_last=float(mt['t'][m][w]), y=float(y_tr[w]))
arrays.update(trial_F=F, trial_target=y_tr, trial_last=last, trial_t=mt['t'][m], window=X_tr[w],
              val_t=mv['t'][mval], val_target=data['y_val'][mval])

# knots and standard deviations, as train.py:144-152 computes them
knots = RT.angle_groups(data['train_angles'])
k_tr = RT.snap(mt['angle'], knots)
y64 = data['y_train'].astype(np.float64)
sd = np.array([max(np.std(y64[k_tr == j]), 0.1) for j in range(len(knots))])
facts['knots'] = knots
facts['sd'] = sd
facts['n_per_knot'] = [int(np.sum(k_tr == j)) for j in range(len(knots))]
facts['sd2_ratio_max_min'] = float((sd.max() / sd.min()) ** 2)
facts['sd_argmax'], facts['sd_argmin'] = int(np.argmax(sd)), int(np.argmin(sd))
for j in (0, int(np.argmax(sd))):
    mj = k_tr == j
    arrays[f'knot{j}_target'] = y64[mj]
    arrays[f'knot{j}_t'] = mt['t'][mj]
arrays.update(knots=knots, sd=sd)

save_npz('x05', **arrays)
save_facts('x05', {'x05': facts})
print('C01 == D01 arrays:', facts['c01_equals_d01'], facts['shapes'], facts['trial'])
print('knots', np.round(knots, 3), 'sd', np.round(sd, 2), 'n/knot', facts['n_per_knot'], 'ratio', round(facts['sd2_ratio_max_min'], 2))
print('example window', facts['example'])
