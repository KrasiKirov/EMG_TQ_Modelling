"""Chapters 2 and 5 data: EMG scaling (÷ session-1 max) and auto-gain, with D01's own code.

normalise('trainmax' / 'agc_auto') from ML/research/emg_norm.py, the running floor and top from
ML/preprocessing/emg_agc.py, one rescale recomputed by hand and checked against the array value,
despike counts, EMG size per Nm of push before / after (definition of new_method_data.py), and the
check that the ÷ max cancels inside auto-gain. Writes data/x03.npz, data/facts_x03.json.
"""
import numpy as np

from common import SUBJECTS, save_facts, save_npz, use_d01_code

use_d01_code()
from ML.preprocessing import emg_agc                          # noqa: E402
from ML.research.data import load_subject                    # noqa: E402
from ML.research.emg_norm import _agc, _train_part, normalise, push_directions  # noqa: E402

facts, arrays = {}, {}


def size(env, tq):
    """EMG push size per Nm of push: (95th - 10th percentile of the envelope) / (95th - 5th of torque)."""
    swing = np.percentile(tq, 95) - np.percentile(tq, 5)
    return (np.percentile(env, 95, axis=0) - np.percentile(env, 10, axis=0)) / max(swing, 1e-6)


def levels(env, pl, ph, rmin):
    """Running floor and top exactly as emg_norm._agc computes them (emg_norm.py:60-70)."""
    lo, hi = emg_agc._running_lo_hi(env, pl, ph)
    clean, n_bad = env.copy(), 0
    for k in range(env.shape[1]):
        clean[:, k], n = emg_agc._despike(env[:, k], lo[:, k], np.maximum(hi[:, k] - lo[:, k], rmin[k]))
        n_bad += n
    if n_bad:
        lo, hi = emg_agc._running_lo_hi(clean, pl, ph)
    return clean, lo, hi, n_bad


def prepared(subject):
    b = load_subject(subject)
    recs = b['active']
    train = [r for r in recs if r['session'] == 'S1']
    base, info_b = normalise('trainmax', b, train, recs)          # S1toS2 settings, as run.py:62-65
    agc, info = normalise('agc_auto', b, train, recs)
    return b, recs, train, base, agc, info_b, info


per = {}
for s in SUBJECTS:
    b, recs, train, base, agc, info_b, info = prepared(s)
    d = dict(train_max=info_b['train_max'], decision=info['auto_decision'],
             push_directions=push_directions(b, train, base))
    if 'prior_lo' in info:
        pl, ph = np.asarray(info['prior_lo']), np.asarray(info['prior_hi'])
        rmin = info['rmin_frac'] * np.maximum(ph - pl, 1e-3)
        d.update(prior_lo=pl, prior_hi=ph, rmin=rmin,
                 despiked=sum(levels(base[r['idx']], pl, ph, rmin)[3] for r in recs))
    per[s] = d
facts['per_subject'] = per

# ── EG: step one, and the cancellation check ─────────────────────────────────
b, recs, train, base, agc, info_b, info = prepared('EG')
r21 = next(r for r in recs if r['idx'] == 21)
arrays.update(eg_t=r21['t'], eg_base=base[21], eg_env_raw=r21['env_raw'], eg_agc=agc[21], eg_tq=r21['tq'],
              eg2_t=next(r for r in recs if r['idx'] == 37)['t'], eg2_agc=agc[37], eg2_base=base[37])
raw = {r['idx']: r['env_raw'] for r in recs}
pl0, ph0 = emg_agc.agc_priors([raw[r['idx']][:_train_part(r)] for r in train])
undivided = {i: _agc(e, pl0, ph0, 0.35) for i, e in raw.items()}
facts['eg_cancel_max_abs_diff'] = float(max(np.max(np.abs(undivided[i] - agc[i])) for i in agc))
# where do they differ? only where the despiker's dropout rule (x < 0.25 floor, floor > 0.1) fires
pl1, ph1 = np.asarray(info['prior_lo']), np.asarray(info['prior_hi'])
n_diff, n_all, fired = 0, 0, {}
for r in recs:
    i = r['idx']
    d = np.abs(undivided[i] - agc[i]) > 1e-9
    n_diff, n_all = n_diff + int(d.sum()), n_all + d.size
    lo_d, _ = emg_agc._running_lo_hi(base[i], pl1, ph1)
    lo_u, _ = emg_agc._running_lo_hi(raw[i], pl0, ph0)
    f_d = int(((base[i] < 0.25 * lo_d) & (lo_d > 0.1)).sum())
    f_u = int(((raw[i] < 0.25 * lo_u) & (lo_u > 0.1)).sum())
    if d.any() or f_d or f_u:
        fired[i] = dict(differing=int(d.sum()), rule_fires_divided=f_d, rule_fires_undivided=f_u)
facts['eg_cancel'] = dict(fraction_differing=n_diff / n_all, n_differing=n_diff, n_samples=n_all, trials=fired,
                          only_where_rule_fires=all(v['rule_fires_divided'] > 0 for v in fired.values()))

sizes = {}
for subj in ('EG', 'JM'):
    b, recs, train, base, agc, info_b, info = prepared(subj)
    rows = []
    for r1 in sorted(train, key=lambda r: r['angle']):
        r2 = min((r for r in recs if r['session'] == 'S2'), key=lambda r: abs(r['angle'] - r1['angle']))
        before = size(base[r2['idx']], r2['tq']) / size(base[r1['idx']], r1['tq'])
        after = size(agc[r2['idx']], r2['tq']) / size(agc[r1['idx']], r1['tq'])
        rows.append(dict(angle=round(r1['angle'], 3), s1=r1['idx'], s2=r2['idx'], before=before, after=after))
    sizes[subj] = rows
facts['size_ratio'] = sizes
# typical mismatch over all angles and the three calf muscles: exp(median |ln ratio|)
facts['size_mismatch'] = {s: {k: float(np.exp(np.median(np.abs(np.log([r[k][:3] for r in rows])))))
                              for k in ('before', 'after')} for s, rows in sizes.items()}

# ── JM at -0.36 rad, session 2, soleus: floor, top, one worked rescale ───────
b, recs, train, base, agc, info_b, info = prepared('JM')
pl, ph = np.asarray(info['prior_lo']), np.asarray(info['prior_hi'])
rmin = info['rmin_frac'] * np.maximum(ph - pl, 1e-3)
r1 = min((r for r in recs if r['session'] == 'S1'), key=lambda r: abs(r['angle'] + 0.36))
r2 = min((r for r in recs if r['session'] == 'S2'), key=lambda r: abs(r['angle'] + 0.36))
clean, lo, hi, n_bad = levels(base[r2['idx']], pl, ph, rmin)
top = lo + np.maximum(hi - lo, rmin[None, :])
K = 2                                                          # soleus
j = 1000 + int(np.argmax(clean[1000:3000, K]))                  # strongest soleus sample 10-30 s into the trial
u = (j // 50) * 50                                              # last 0.5 s refresh before it
n_lo, n_hi = min(u, 2000), min(u, 4500)
w_lo, w_hi = n_lo / (n_lo + 500), n_hi / (n_hi + 500)
p10 = float(np.percentile(clean[u - n_lo:u, K], 10))
p95 = float(np.percentile(clean[u - n_hi:u, K], 95))
floor = w_lo * p10 + (1 - w_lo) * pl[K]
top_v = w_hi * p95 + (1 - w_hi) * ph[K]
rng = max(top_v - floor, rmin[K])
a = float(np.clip((clean[j, K] - floor) / rng, 0, 2))
facts['jm_example'] = dict(
    s1_idx=r1['idx'], s2_idx=r2['idx'], angle=round(r2['angle'], 3), j=j, u=u, t=float(r2['t'][j]),
    t_u=float(r2['t'][u]), t_in_trial=float(r2['t'][j] - r2['t'][0]), n_lo=n_lo, n_hi=n_hi, w_lo=w_lo, w_hi=w_hi,
    p10=p10, p95=p95, prior_lo=float(pl[K]), prior_hi=float(ph[K]), floor=floor, top=top_v,
    rmin=float(rmin[K]), prior_range=float(ph[K] - pl[K]), range=rng, x=float(clean[j, K]), a=a,
    a_array=float(agc[r2['idx']][j, K]), check=abs(a - float(agc[r2['idx']][j, K])), n_bad=n_bad,
    floor_array=float(lo[j, K]), top_array=float(top[j, K]))
arrays.update(jm_t=r2['t'], jm_base=base[r2['idx']], jm_lo=lo, jm_top=top, jm_agc=agc[r2['idx']], jm_tq=r2['tq'],
              jm_s1_t=r1['t'], jm_s1_base=base[r1['idx']], jm_s1_agc=agc[r1['idx']], jm_s1_tq=r1['tq'],
              jm_s1_lo=levels(base[r1['idx']], pl, ph, rmin)[1])
n = np.arange(0, 2001)
arrays.update(w_n=n, w=n / (n + 500.0))

save_npz('x03', **arrays)
save_facts('x03', {'x03': facts})
print('EG', per['EG']['decision'], 'CH', per['CH']['decision'], 'cancel diff', facts['eg_cancel_max_abs_diff'])
print('JM sizes', [(r['angle'], np.round(r['before'][:3], 2), np.round(r['after'][:3], 2)) for r in sizes['JM']])
print('worked example', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in facts['jm_example'].items()})
print('despiked', {s: per[s].get('despiked') for s in SUBJECTS})
