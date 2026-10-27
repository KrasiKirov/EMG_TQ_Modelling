"""Chapter 12 data: last week's pipeline (0.5 s window + retest passive map), EG seed 0.

Reads the saved run (EMG_TQ_Modelling_runs/window50_retest_map/EG_seed0): its config, package
(target standardisation), history and test predictions; rescores each test trial (same R2 / MAE as
presentation_metrics.py) and checks the numbers against the dashboard's presentation catalog.
Writes data/x09.npz, data/facts_x09.json.
"""
import json

import numpy as np

from common import DASHBOARD, LAST_WEEK_RUNS, save_facts, save_npz

run = LAST_WEEK_RUNS / 'EG_seed0'
cfg = json.loads((run / 'config.json').read_text())
pkg = json.loads((run / 'EG_lstm_seed0' / 'package.json').read_text())
hist = json.loads((run / 'EG_lstm_seed0' / 'history.json').read_text())
z = np.load(run / 'EG_lstm_seed0' / 'test_predictions.npz')
cat = json.loads((DASHBOARD / 'src' / 'data' / 'presentation-catalog.json').read_text(encoding='utf-8'))
catalog = {round(r['pos'], 3): r for r in cat['rows'] if r['exp'] == 'main/session-passive' and r['subj'] == 'EG'
           and r['seed'] == 0 and r['dir'] == 'S1toS2'}

trials, arrays, worst = {}, {}, 0.0
for tr in np.unique(z['trial']):
    m = z['trial'] == tr
    y, p = z['y_true'][m].astype(np.float64), z['y_pred'][m].astype(np.float64)
    e = p - y
    r2 = float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))
    mae = float(np.mean(np.abs(e)))
    trials[int(tr)] = dict(position_id=str(z['position_id'][m][0]), r2=r2, mae=mae, n=int(m.sum()),
                           t0=float(z['time_s'][m][0]), t1=float(z['time_s'][m][-1]))
    arrays[f't{int(tr)}_t'], arrays[f't{int(tr)}_y'], arrays[f't{int(tr)}_p'] = z['time_s'][m], y, p
# match trials to catalog positions by score (each catalog row is one trial of this run)
for tr, d in trials.items():
    a = min(catalog, key=lambda k: abs(catalog[k]['r2'] - d['r2']) + abs(catalog[k]['mae'] - d['mae']))
    d['position'] = a
    worst = max(worst, abs(catalog[a]['r2'] - d['r2']), abs(catalog[a]['mae'] - d['mae']))
eg = next(d for d in trials.values() if d['position'] == -0.038)
save_npz('x09', **arrays)
save_facts('x09', {'x09': dict(
    config={k: cfg[k] for k in ('window', 'train_stride', 'gap', 'edge_trim', 'batch', 'epochs', 'patience', 'lr',
                                'dropout', 'position_weights', 'session_passive_calibration', 'lp_cutoff')},
    target_offset=pkg['target_offset'], target_scale=pkg['target_scale'], epochs_executed=hist['epochs_executed'],
    best_epoch=hist['best_epoch_one_based'], optimizer_updates=hist['optimizer_updates'], trials=trials,
    eg_m0038=eg, check_vs_catalog_max_abs_diff=worst)})
print('EG -0.038:', eg, 'catalog check', worst, 'best epoch', hist['best_epoch_one_based'], 'of', hist['epochs_executed'])
