"""Arrays for the new-method figures, computed with the new method's own code (ML.research).

The new method lives in the experimentation-dorsiflexion worktree, whose ``ML`` package
differs from this branch's, so this step runs on its own and saves an .npz that
``pipeline_tree_new.py`` and ``autogain_zoom.py`` read.

Usage:
    python new_method_data.py <dorsiflexion worktree> <results folder | -> <out.npz> [subject] [angle]
EG at -0.04 rad (the tree) uses the C01 "auto-gain + equal weight" results, identical to the new
method D01 for EG (same passive table). JM at -0.36 rad (the auto-gain zoom) needs no predictions: pass -.
"""
import json
import sys
from pathlib import Path

import numpy as np

wt, results, out = sys.argv[1:4]
sys.path.insert(0, wt)
from ML.preprocessing import emg_agc                      # noqa: E402
from ML.research.data import load_subject                 # noqa: E402
from ML.research.emg_norm import normalise                # noqa: E402
from ML.research.passive import map_lookup                # noqa: E402

SUBJECT = sys.argv[4] if len(sys.argv) > 4 else 'EG'
ANGLE = {s: float(sys.argv[5]) if len(sys.argv) > 5 else -0.04 for s in ('S1', 'S2')}
b = load_subject(SUBJECT)
recs = b['active']
train = [r for r in recs if r['session'] == 'S1']
base, _ = normalise('trainmax', b, train, recs)
agc, info = normalise('agc_auto', b, train, recs)
pl, ph = np.asarray(info['prior_lo']), np.asarray(info['prior_hi'])
rmin = info['rmin_frac'] * np.maximum(ph - pl, 1e-3)
print(SUBJECT, info['auto_decision'], 'prior lo', np.round(pl, 3), 'hi', np.round(ph, 3))


def levels(env):
    """Running floor and effective top exactly as emg_norm._agc computes them."""
    lo, hi = emg_agc._running_lo_hi(env, pl, ph)
    clean, n_bad = env.copy(), 0
    for k in range(env.shape[1]):
        clean[:, k], n = emg_agc._despike(env[:, k], lo[:, k], np.maximum(hi[:, k] - lo[:, k], rmin[k]))
        n_bad += n
    if n_bad:
        lo, hi = emg_agc._running_lo_hi(clean, pl, ph)
    return lo, lo + np.maximum(hi - lo, rmin[None, :]), n_bad


def size(env, tq):
    """EMG push size per Nm of push: (95th - 10th percentile) / torque swing."""
    swing = np.percentile(tq, 95) - np.percentile(tq, 5)
    return (np.percentile(env, 95, axis=0) - np.percentile(env, 10, axis=0)) / max(swing, 1e-6)


print('S2/S1 calf EMG size per Nm, before -> after auto-gain (GM GL SOL)')
for r1 in sorted(train, key=lambda r: r['angle']):
    r2 = min((r for r in recs if r['session'] == 'S2'), key=lambda r: abs(r['angle'] - r1['angle']))
    before = size(base[r2['idx']], r2['tq']) / size(base[r1['idx']], r1['tq'])
    after = size(agc[r2['idx']], r2['tq']) / size(agc[r1['idx']], r1['tq'])
    print(f"  {r1['angle']:+.3f}  before {np.round(before[:3], 2)}  after {np.round(after[:3], 2)}")

save = dict(prior_lo=pl, prior_hi=ph, rmin=rmin)
pred = None
if results != '-':
    pred = np.load(Path(results) / SUBJECT / 'S1toS2' / 'seed0.npz')
    ti = json.loads((Path(results) / SUBJECT / 'S1toS2' / 'seed0.json').read_text())['train_info']
    save.update(knots=np.asarray(ti['knots']), sd=np.asarray(ti['sd_per_knot']), epochs=ti['epochs'],
                best_epoch=ti['best_epoch'])
    print('knots', np.round(ti['knots'], 3), 'sd', np.round(ti['sd_per_knot'], 2), 'epochs', ti['epochs'])
for s in ('S1', 'S2'):
    m = b['maps']['static'][s]
    save[f'map_{s}'] = np.asarray(m, float)
    r = min((r for r in recs if r['session'] == s), key=lambda r: abs(r['angle'] - ANGLE[s]))
    lo, top, n_bad = levels(base[r['idx']])
    passive = map_lookup(m, r['pos'])
    save.update({f'{s}_{k}': v for k, v in dict(
        idx=r['idx'], angle=r['angle'], t=r['t'], base=base[r['idx']], agc=agc[r['idx']], lo=lo, top=top,
        pos=r['pos'], tq=r['tq'], passive=passive, target=r['tq'] - passive).items()})
    hold = [h for h in b['holds'] if h['session'] == s and abs(h['angle'] - r['angle']) < 0.05]
    save[f'{s}_holds'] = np.asarray([[h['idx'], h['angle'], h['mean'], h['dur']] for h in hold], float)
    print(f"{s}: trial idx {r['idx']} angle {r['angle']:+.3f}, despiked {n_bad}, map at angle "
          f"{np.median(passive):+.2f} Nm, holds {[(h['idx'], round(h['angle'], 3), round(h['mean'], 2), round(h['dur'], 1)) for h in hold]}")
    if s == 'S2' and pred is not None:
        k = (pred['trial'] == r['idx']) & (pred['session'] == 1)
        save.update(S2_pred_t=pred['t'][k], S2_pred=pred['y_pred'][k], S2_pred_target=pred['y_own_target'][k])
        y, p = pred['y_own_target'][k].astype(float), pred['y_pred'][k].astype(float)
        r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
        save.update(S2_r2=r2, S2_mae=np.mean(abs(y - p)))
        print(f"  C01 seed0 on this trial: R2 {r2:.3f}, MAE {np.mean(abs(y - p)):.2f} Nm, "
              f"target check {np.max(abs(np.interp(pred['t'][k], r['t'], r['tq'] - passive) - y)):.3f}")
np.savez_compressed(out, **save)
print('wrote', out)
