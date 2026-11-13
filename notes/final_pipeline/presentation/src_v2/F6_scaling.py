"""F6_scaling: explain the per-angle target scaling ("divide each angle's target
by its typical swing") used by the new pipeline.

Reproduces ML/research/train.py fit_predict (balance='scale'):
  knots   = training angles grouped (KNOT_TOL 0.02 rad)
  y_train = target at the last sample of every 20-sample window, first 80 %
            of the windows of each training trial
  sd_j    = max(std(y_train of angle j), 0.1)
  model learns y / sd_j; prediction multiplied back by sd of nearest knot.
Training targets of the new pipeline for S1toS2 = measured - B(t) on the
session-1 trials (truth.session_rest with max_frac=None, = shared truth).

Outputs:
  figures_v2/swing_per_angle.png
  figures_v2/swing_divide_example.png
  prints sigma per angle for all 6 subjects + ratio largest/smallest sigma^2
"""
import os
import sys
import json

sys.path.insert(0, os.environ['WT'])
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ML.research.data import load_subject
from ML.research import truth as T
from ML.research import config as C
from ML.research.train import angle_groups, snap

WT = os.environ['WT']
OUT = os.path.join(WT, 'notes', 'final_pipeline', 'presentation', 'figures_v2')
os.makedirs(OUT, exist_ok=True)
RES = os.path.join(WT, 'ML', 'research', 'results')

OLD, NEW, TRUTH, GREY, RELAX, BAD, GRID, SEC = ('#2a78d6', '#eb6834', '#1b1b1b', '#8a8984',
                                                '#1baf7a', '#c0392b', '#e6e5e1', '#52514e')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13, 'xtick.labelsize': 11.5,
                     'ytick.labelsize': 11.5, 'axes.titlesize': 14, 'axes.titleweight': 'bold',
                     'axes.labelsize': 13, 'legend.fontsize': 12, 'legend.frameon': False,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'axes.axisbelow': True,
                     'axes.unicode_minus': True})
MINUS = '−'


def m(s):
    """Unicode minus in a printed number/label."""
    return s.replace('-', MINUS)


def fmt_angle(a, deg=True):
    s = f'{a:+.2f} rad' if a != 0 else '0.00 rad'
    s = s.replace('+', '')
    if deg:
        s += f' ({np.degrees(a):+.0f}°)'.replace('+', '')
    return m(s)


def training_targets(b, tr, session='S1'):
    """Per training trial: (rec, y_train (windowed, first 80 %), t of those samples).
    Uses the shared truth (= new pipeline's training labels for S1toS2)."""
    W = C.WINDOW
    out = []
    for r in b['active']:
        if r['session'] != session:
            continue
        y = tr[r['idx']]['truth']
        n = len(y) - W + 1
        n_tr = int(n * C.TRAIN_FRAC)
        last = np.arange(n_tr) + W - 1
        out.append((r, y[last].astype(np.float64), r['t'][last]))
    return out


def sigma_per_angle(b, tr, session='S1'):
    items = training_targets(b, tr, session)
    angles = np.array([r['angle'] for r, _, _ in items])
    knots = angle_groups(angles)
    k = snap(angles, knots)
    sd, n = [], []
    for j in range(len(knots)):
        y = np.concatenate([items[i][1] for i in np.where(k == j)[0]])
        sd.append(max(np.std(y.astype(np.float32).astype(np.float64)), 0.1))
        n.append(len(y))
    return knots, np.array(sd), np.array(n), items


SUBJECTS = ['YES', 'JM', 'EG', 'IES', 'CH', 'HM']
summary = {}
cache = {}
for s in SUBJECTS:
    b = load_subject(s)
    tr = T.subject_truth(s)
    knots, sd, n, items = sigma_per_angle(b, tr)
    cache[s] = (b, tr, knots, sd, n, items)
    # check against the stored training info of the new pipeline (S1toS2)
    js = os.path.join(RES, 'final_check', s, 'S1toS2', 'FINAL', 'fast_seed0.json')
    ti = json.load(open(js))['train_info']
    sd_js, kn_js = np.array(ti['sd_per_knot']), np.array(ti['knots'])
    ok = (len(sd_js) == len(sd)) and np.allclose(sd_js, sd, rtol=1e-3) and np.allclose(kn_js, knots, atol=1e-3)
    var = sd ** 2
    share_plain = var / var.sum()
    share_plain_n = n * var / (n * var).sum()
    share_scaled_n = n / n.sum()
    summary[s] = dict(knots=knots.tolist(), sigma=sd.tolist(), n=n.tolist(),
                      share_plain=share_plain.tolist(), share_plain_with_n=share_plain_n.tolist(),
                      share_scaled_with_n=share_scaled_n.tolist(),
                      ratio_var=float(var.max() / var.min()),
                      ratio_sd=float(sd.max() / sd.min()),
                      match_json=bool(ok), max_rel_diff=float(np.max(np.abs(sd_js - sd) / sd_js))
                      if len(sd_js) == len(sd) else None)
    print(f'\n{s}: match stored train_info = {ok}  (max rel diff {summary[s]["max_rel_diff"]})')
    for j in range(len(knots)):
        print(f'  {knots[j]:+.3f} rad ({np.degrees(knots[j]):+5.1f} deg)  sigma {sd[j]:6.2f} Nm  '
              f'n {n[j]:5d}  plain share {100*share_plain[j]:6.2f}%  (with n {100*share_plain_n[j]:6.2f}%)  '
              f'scaled share 12.5% (with n {100*share_scaled_n[j]:5.2f}%)')
    print(f'  ratio largest/smallest sigma^2 = {var.max()/var.min():.0f}  (sigma ratio {sd.max()/sd.min():.1f})')

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'F6_scaling_numbers.json'), 'w') as f:
    json.dump(summary, f, indent=1)

# ───────────────────────── Figure 1: swing per angle (YES) ─────────────────────
b, tr, knots, sd, n, items = cache['YES']
var = sd ** 2
share_plain = 100 * var / var.sum()
share_scaled = np.full(len(sd), 100.0 / len(sd))
# same rounding as the other deck figures: angle keys are stored to 3 decimals,
# then printed with 2 decimals (e.g. 0.245 -> "0.24", -0.505 -> "-0.51")
knots_lbl = np.round(knots, 3)
labels = [m(f'{a:.2f}') + '\n' + m(f'{np.degrees(a):.0f}°') for a in knots_lbl]
ratio_yes = summary['YES']['ratio_var']
ratio_oth = [summary[s]['ratio_var'] for s in SUBJECTS if s != 'YES']
XLAB = 'ankle angle, rad (degrees)'


def angle_axis(ax):
    ax.set_xticks(np.arange(len(knots)), labels, fontsize=11)
    ax.set_xlabel(XLAB)
    for xf, ha, txt in ((0.0, 'left', '← toes down'), (1.0, 'right', 'toes up →')):
        ax.annotate(txt, xy=(xf, -0.1), xycoords=('axes fraction', ax.xaxis.label), ha=ha, va='top',
                    fontsize=11.5, color=SEC)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw=dict(width_ratios=[1, 1.1], wspace=0.24))
x = np.arange(len(knots))
a1.bar(x, sd, color=GREY, width=0.68, zorder=2)
for i, v in enumerate(sd):
    a1.text(i, v + 0.3, f'{v:.1f}', ha='center', va='bottom', fontsize=11.5)
angle_axis(a1)
a1.set_ylabel('typical swing σ (Nm)')
a1.set_ylim(0, sd.max() * 1.42)
a1.set_yticks(np.arange(0, sd.max() * 1.1, 2))   # no grid lines through the note at the top
a1.set_title('How much the voluntary torque\nswings at each angle', loc='left', fontsize=14)
a1.grid(axis='x', visible=False)
a1.text(0.02, 0.98, 'subject YES, session 1 (training part)', transform=a1.transAxes,
        ha='left', va='top', fontsize=11.5, color=TRUTH)
a1.text(0.02, 0.905, m(f'largest σ² ÷ smallest σ²: {ratio_yes:.0f}× here\n'
                       f'(other 5 subjects: {min(ratio_oth):.0f}–{max(ratio_oth):.0f}×)'),
        transform=a1.transAxes, ha='left', va='top', fontsize=11, color=SEC, linespacing=1.3)

w = 0.38
a2.bar(x - w / 2, share_plain, width=w, color=OLD, label='old: plain least squares (share ∝ σ²)', zorder=2)
a2.bar(x + w / 2, share_scaled, width=w, color=NEW, label='new: divided by σ (equal share)', zorder=2)
for i, v in enumerate(share_plain):
    if v < 1:
        txt = f'{v:.2f}%'
    elif v < 12.5:
        txt = f'{v:.1f}%'
    else:
        txt = f'{v:.0f}%'
    # labels of bars lower than the orange 12.5 % bar are turned upright so they
    # stay within the width of their own (blue) bar
    a2.text(i - w / 2, v + 0.5, txt, ha='center', va='bottom', fontsize=10.5, color=OLD,
            rotation=90 if v < 12.5 else 0)
a2.text(1.0, 12.5 + 0.6, '12.5% each', ha='center', va='bottom', fontsize=11.5, color=NEW,
        fontweight='bold')
angle_axis(a2)
a2.set_ylabel('share of the total squared error (%)')
a2.set_ylim(0, 45)
a2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35])
a2.set_title("Each angle's say in the fit\n(if the model is equally good at every angle)", loc='left', fontsize=14)
a2.grid(axis='x', visible=False)
a2.legend(loc='upper left', borderaxespad=0.2)
fig.savefig(os.path.join(OUT, 'swing_per_angle.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)

# ──────────────────── Figure 2: real 20 s traces, Nm vs divided ────────────────
j_strong = int(np.argmax(sd))
j_down = int(np.argmin(knots))
k_items = snap(np.array([r['angle'] for r, _, _ in items]), knots)
DUR = 20.0


def pick_window(rec, y, t, sd_j):
    """20 s window inside the training part whose spread is closest to the
    angle's sigma, with a median near 0, that STARTS while the person is not
    pushing (voluntary torque within 0.25 sigma of zero for the first 0.8 s),
    with penalties for upward bumps (> 0.3 sigma) and extreme dips (< -2.5 sigma).
    (The EMG relaxed-moment mask is not used here: in the toes-down trial it
    flags only 14 % of the training part, almost all in the first 30 s.)"""
    best, best_cost = None, np.inf
    for t0 in np.arange(t[0] + 2, t[-1] - DUR, 0.25):
        msk = (t >= t0) & (t < t0 + DUR)
        if not np.all(np.abs(y[(t >= t0) & (t < t0 + 0.8)]) < 0.25 * sd_j):
            continue
        seg = y[msk]
        cost = (abs(np.std(seg) - sd_j) / sd_j + 0.5 * abs(np.median(seg)) / sd_j
                + 2.0 * max(0.0, np.max(seg) / sd_j - 0.3)      # no big upward bumps
                + 2.0 * max(0.0, -2.5 - np.min(seg) / sd_j))    # no extreme dips
        if cost < best_cost:
            best, best_cost = t0, cost
    return best


traces, trace_info = [], []
for j in (j_strong, j_down):
    i = int(np.where(k_items == j)[0][0])
    rec, y, t = items[i]
    t0 = pick_window(rec, y, t, sd[j])
    msk = (t >= t0) & (t < t0 + DUR)
    traces.append((j, t[msk] - t0, y[msk], rec))
    info = dict(angle=float(knots[j]), trial_idx=int(rec['idx']), win_start_s=float(t0 - rec['t'][0]),
                win_end_s=float(t0 - rec['t'][0] + DUR), train_end_s=float(t[-1] - rec['t'][0]),
                std_win=float(np.std(y[msk])), sigma=float(sd[j]),
                min_div=float(np.min(y[msk]) / sd[j]), max_div=float(np.max(y[msk]) / sd[j]),
                start_val=float(y[msk][0]))
    trace_info.append(info)
    print('trace', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()})
summary['YES_traces'] = trace_info
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'F6_scaling_numbers.json'), 'w') as f:
    json.dump(summary, f, indent=1)

fig = plt.figure(figsize=(12.5, 5.6))
sf = fig.subfigures(2, 1, hspace=0.06)
row_titles = ['In Nm: the toes-down angle is almost invisible to the fit',
              'Divided by its typical swing: both count the same']
cols = [NEW, NEW]
ax_top = sf[0].subplots(1, 2, sharey=True, gridspec_kw=dict(wspace=0.08))
ax_bot = sf[1].subplots(1, 2, sharey=True, gridspec_kw=dict(wspace=0.08))
sf[0].suptitle(row_titles[0], fontsize=14, fontweight='bold', x=0.02, ha='left')
sf[1].suptitle(row_titles[1], fontsize=14, fontweight='bold', x=0.02, ha='left')
ax_top[1].text(1.0, 1.06, 'subject YES, session 1 (training part)', transform=ax_top[1].transAxes,
               ha='right', va='bottom', fontsize=11.5, color=SEC)
for c, (j, tt, y, rec) in enumerate(traces):
    name = 'strongest swing' if j == j_strong else 'most toes-down'
    ax_top[c].plot(tt, y, color=TRUTH, lw=1.3)
    ax_top[c].text(0.01, 0.04, f'{fmt_angle(knots_lbl[j])}, {name}:  σ = {sd[j]:.1f} Nm',
                   transform=ax_top[c].transAxes, fontsize=11.5, color=SEC, va='bottom')
    ax_bot[c].plot(tt, y / sd[j], color=NEW, lw=1.3)
    ax_bot[c].text(0.01, 0.04, m(f'same trace ÷ {sd[j]:.1f} Nm'),
                   transform=ax_bot[c].transAxes, fontsize=11.5, color=SEC, va='bottom')
    ax_bot[c].set_xlabel('time (s)')
    for a in (ax_top[c], ax_bot[c]):
        a.set_xlim(0, DUR)
        a.set_xticks([0, 5, 10, 15, 20])
        a.axhline(0, color=GREY, lw=0.8, zorder=1)
ax_top[0].set_ylabel('voluntary\ntorque (Nm)')
ax_bot[0].set_ylabel('torque ÷ σ\n(no unit)')
for a in ax_top:
    a.tick_params(labelbottom=False)
# give the labels room below the traces
ylo, yhi = ax_top[0].get_ylim()
ax_top[0].set_ylim(ylo - 0.25 * (yhi - ylo), yhi)
ylo, yhi = ax_bot[0].get_ylim()
ax_bot[0].set_ylim(ylo - 0.25 * (yhi - ylo), yhi)
fig.savefig(os.path.join(OUT, 'swing_divide_example.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('saved figures to', OUT)
