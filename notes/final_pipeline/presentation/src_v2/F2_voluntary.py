"""F2_voluntary: figures explaining what the torque sensor measures and how
voluntary torque is obtained (sketch + real data, subject YES, session 2).

Run (Git Bash):
  cd "$WT"; source ML/experiments/env.sh; export WT="$(pwd)"; $PY F2_voluntary.py
Writes 4 PNGs into notes/final_pipeline/presentation/figures_v2 and prints the
numbers shown in the figures.
"""
import os
import sys
import json

sys.path.insert(0, os.environ['WT'])

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from ML.research.data import load_subject
from ML.research import truth as T
from ML.research.passive import PassiveModel
from ML.preprocessing import active_torque as at

OUT = os.path.join(os.environ['WT'], 'notes', 'final_pipeline', 'presentation', 'figures_v2')
os.makedirs(OUT, exist_ok=True)

OLD, NEW, TRUTH, GREY, REST, BAD = '#2a78d6', '#eb6834', '#1b1b1b', '#8a8984', '#1baf7a', '#c0392b'
GRID, SEC = '#e6e5e1', '#52514e'
CALF, SHIN = '#7a4fa3', '#2f9fb5'   # muscle lines (neutral, not old/new)
ACT = '#3b3a37'                      # average-of-4-muscles activity (neutral dark grey)
EMG_YLAB = 'activity (each\nmuscle ÷ its own\nmax, session 2)'
M = '−'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 13,
    'axes.titlesize': 14, 'axes.titleweight': 'bold', 'axes.labelsize': 13,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.9,
    'axes.axisbelow': True, 'legend.frameon': False,
    'axes.unicode_minus': True, 'axes.edgecolor': '#9a9994',
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
})


def f(x, nd=1, plus=False):
    s = f'{x:+.{nd}f}' if plus else f'{x:.{nd}f}'
    return s.replace('-', M)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', p)
    return p


def darrow(ax, x, y0, y1, color, lw=2.0):
    ax.annotate('', xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle='<->', color=color, lw=lw, shrinkA=0, shrinkB=0))


NUMS = {}

# ════════════════════════════════════════════════════════════════════════
# (1) concept sketch
# ════════════════════════════════════════════════════════════════════════
def fig_concept():
    t = np.linspace(0, 26, 2601)
    P = -8.0                                    # passive level (Nm), sketch
    pushes = [(2.5, 6.5, 20.0), (9.0, 13.0, 22.0), (21.5, 25.0, 19.0)]
    vol = np.zeros_like(t)
    for a, e, h in pushes:
        k = 0.25
        vol -= h * 0.5 * (np.tanh((t - a) / k) - np.tanh((t - e) / k))
    rng = np.random.default_rng(3)
    wob = np.convolve(rng.normal(0, 1, len(t)), np.ones(40) / 40, 'same') * 1.2
    act = (vol < -0.5).astype(float)
    vol_n = vol + wob * act
    meas = P + vol_n

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12.5, 5.8), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2.1, 1.25], hspace=0.38))
    a1.plot(t, meas, color=GREY, lw=2.4, label='torque sensor reading')
    a1.axhline(P, color=NEW, ls='-', lw=2.0)   # solid orange = same as the in-trial passive line
    a1.axhline(0, color=SEC, lw=0.9)
    a1.text(26.2, 0, '0 Nm', va='center', ha='left', fontsize=11.5, color=SEC)
    a1.text(0.25, P / 2,
            'passive torque: what the sensor reads when the muscles are off\n'
            '(stretched tissues + weight of the foot)',
            color=NEW, fontsize=12, va='center', ha='left', fontweight='bold')
    # voluntary arrow on push 2
    a, e, h = pushes[1]
    xa = 11.0
    darrow(a1, xa, P, P - h, TRUTH, lw=2.2)
    yv = P - 0.62 * h
    a1.text(13.9, yv, 'voluntary torque:\nthe extra twist made\nby the muscles',
            color=TRUTH, fontsize=12.5, va='center', ha='left', fontweight='bold')
    a1.annotate('', xy=(xa + 0.15, yv), xytext=(13.8, yv),
                arrowprops=dict(arrowstyle='-', color=TRUTH, lw=1.0))
    a1.text(4.5, P - 9.5, 'push\n(toes down)', ha='center', va='center', fontsize=12, color=SEC)
    a1.text(17.3, P - 2.6, 'relaxed', ha='center', va='center', fontsize=12, color=SEC)
    a1.set_ylim(-33.5, 3.5)
    a1.set_ylabel('torque (Nm)')
    a1.set_title('Sketch (made-up numbers): what the torque sensor reads while the person pushes and relaxes',
                 loc='left', fontsize=14)

    a2.plot(t, vol_n, color=TRUTH, lw=2.4)
    a2.axhline(0, color=SEC, lw=0.9, ls=':')
    a2.set_ylim(-27, 5)
    a2.set_ylabel('torque (Nm)')
    a2.set_xlabel('time (s)')
    a2.set_title('After subtracting the passive level: voluntary torque', loc='left', fontsize=14)
    a2.text(17.25, -11.5, 'voluntary torque:\n0 when relaxed, negative\nwhen pushing toes down',
            ha='center', va='center', fontsize=12, color=TRUTH, fontweight='bold')
    a2.set_xlim(0, 26)
    return save(fig, 'vt_concept.png')


# ════════════════════════════════════════════════════════════════════════
# data
# ════════════════════════════════════════════════════════════════════════
B_ = load_subject('YES')
TR = T.subject_truth('YES')
S2 = [r for r in B_['active'] if r['session'] == 'S2']
EMG_MAX = np.max(np.stack([r['env_max1k'] for r in S2]), axis=0)
EMG_MAX = np.where(EMG_MAX > 0, EMG_MAX, 1.0)
SEGS = T.session_rest(B_, 'S2')
STATS = at.emg_rest_stats([r['env_raw'] / EMG_MAX for r in S2])
PM = PassiveModel('legacy', B_)


def rec_of(idx):
    return [r for r in S2 if r['idx'] == idx][0]


def activity_score(r):
    """Same score detect_rest uses (fraction of each muscle's session range)."""
    e = r['env_raw'] / EMG_MAX
    floor = np.percentile(e, 5.0, axis=0)
    return ((e - floor) / STATS['scale']).mean(axis=1)


def push_stretches(idx, min_len=100):
    """(start, end) samples between relaxed stretches (the pushes)."""
    sg = SEGS[idx]
    T_ = len(rec_of(idx)['tq'])
    bounds = [0] + [x for se in sg for x in se] + [T_]
    return [(a, e) for a, e in zip(bounds[0::2], bounds[1::2]) if e - a > min_len]


def shade_rest(ax, t, rest, y0=None, y1=None, alpha=0.18):
    ax.fill_between(t, 0, 1, where=rest, transform=ax.get_xaxis_transform(),
                    color=REST, alpha=alpha, lw=0, step='mid')


def deg(rad):
    return np.degrees(rad)


# ════════════════════════════════════════════════════════════════════════
# (2) real data, strong angle
# ════════════════════════════════════════════════════════════════════════
def fig_strong(idx=28, w0=4.0, w1=44.0, rest_pick=(21.5, 25.8), push_pick=(28.4, 33.0), xp=30.0):
    r = rec_of(idx); d = TR[idx]
    t = r['t'] - r['t'][0]
    m = (t >= w0) & (t <= w1)
    e = r['env_raw'] / EMG_MAX
    calf, shin = e[:, :3].mean(1), e[:, 3]
    tq, Bt, vol, rest = r['tq'], d['B'], d['truth'], d['rest']

    # numbers
    rs = (t >= rest_pick[0]) & (t <= rest_pick[1]) & rest
    X = float(np.median(tq[rs]))
    ps = (t >= push_pick[0]) & (t <= push_pick[1])
    Y = float(np.median(tq[ps]))          # plateau level of the marked push
    Yv = float(np.median(vol[ps]))
    kp = int(np.argmin(np.abs(t - xp)))   # arrow tips sit on the traces at this sample
    allrest = float(np.median(tq[rest]))
    pushes = push_stretches(idx)
    psz = [float(-np.percentile(vol[a:e], 5)) for a, e in pushes]
    ang = r['angle']
    NUMS['strong'] = dict(trial=idx, angle_rad=round(ang, 3), angle_deg=round(deg(ang), 1),
                          window_s=[w0, w1], relaxed_level_example_Nm=round(X, 2),
                          relaxed_level_trial_median_Nm=round(allrest, 2),
                          push_window_s=list(push_pick), arrow_at_s=xp,
                          push_reading_example_Nm=round(Y, 1), push_voluntary_example_Nm=round(Yv, 1),
                          trace_at_arrow_Nm=[round(float(tq[kp]), 2), round(float(vol[kp]), 2)],
                          push_size_median_Nm=round(float(np.median(psz)), 1),
                          B_range_Nm=[round(float(Bt.min()), 2), round(float(Bt.max()), 2)],
                          B_range_window_Nm=[round(float(Bt[m].min()), 2), round(float(Bt[m].max()), 2)],
                          n_relaxed_stretches=int(d['n_rest_segs']))

    fig, ax = plt.subplots(3, 1, figsize=(12.5, 7.6), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.0, 1.5, 1.2], hspace=0.42))
    a, b, c = ax
    tt = t[m]
    # (a)
    a.plot(tt, calf[m], color=CALF, lw=1.6, label='calf muscles\n(push toes down)')
    a.plot(tt, shin[m], color=SHIN, lw=1.6, label='shin muscle\n(pulls toes up)')
    shade_rest(a, tt, rest[m])
    a.set_ylabel(EMG_YLAB, fontsize=12)
    a.set_ylim(0, max(0.75, float(np.max(calf[m])) * 1.1))
    a.set_title('a. Muscle activity (EMG): calf on = pushing, everything low = relaxed', loc='left')
    a.legend(loc='upper left', bbox_to_anchor=(1.01, 1.05))
    # (b)
    b.plot(tt, tq[m], color=GREY, lw=1.6, label='measured torque')
    b.plot(tt, Bt[m], color=NEW, lw=2.4, label='passive torque,\nmeasured inside\nthe trial')
    shade_rest(b, tt, rest[m])
    b.axhline(0, color=SEC, lw=0.8, ls=':')
    b.set_ylim(-52, 15)
    b.set_ylabel('torque (Nm)')
    b.set_title('b. What the torque sensor measures', loc='left')
    xr = 0.5 * (rest_pick[0] + rest_pick[1])
    b.annotate(f'muscles quiet → sensor still reads ≈ {f(X)} Nm = passive',
               xy=(xr, X + 0.8), xytext=(xr - 1.5, 9.5), ha='left', va='center', fontsize=12,
               color=TRUTH, arrowprops=dict(arrowstyle='->', color=TRUTH, lw=1.2))
    b.annotate(f'push: reading goes to ≈ {f(Y, 0)} Nm', xy=(t[kp], float(tq[kp])), xytext=(xp, -46.5),
               ha='center', va='center', fontsize=12, color=TRUTH,
               arrowprops=dict(arrowstyle='->', color=TRUTH, lw=1.2, shrinkB=0))
    hb = [Line2D([], [], color=GREY, lw=1.8, label='measured torque'),
          Patch(color=REST, alpha=0.35, label='relaxed moments\n(found from the\nmuscle signals)'),
          Line2D([], [], color=NEW, lw=2.4, label='passive torque,\nmeasured inside\nthe trial')]
    b.legend(handles=hb, loc='upper left', bbox_to_anchor=(1.01, 1.05), labelspacing=0.9)
    # (c)
    c.plot(tt, vol[m], color=TRUTH, lw=1.6, label='voluntary torque')
    shade_rest(c, tt, rest[m])
    c.axhline(0, color=SEC, lw=0.8, ls=':')
    c.set_ylim(-45, 8)
    c.set_ylabel('torque (Nm)')
    c.set_xlabel('time in the trial (s)')
    c.set_title(f'c. Voluntary torque = measured {M} passive  (0 when relaxed)', loc='left')
    c.annotate(f'same push: ≈ {f(Yv, 0)} Nm made by the muscles', xy=(t[kp], float(vol[kp])), xytext=(xp, -40.5),
               ha='center', va='center', fontsize=12, color=TRUTH,
               arrowprops=dict(arrowstyle='->', color=TRUTH, lw=1.2, shrinkB=0))
    c.text(xr, 1.5, 'relaxed: 0 Nm', ha='center', va='bottom', fontsize=12, color=TRUTH)
    c.legend(handles=[Line2D([], [], color=TRUTH, lw=1.8, label='voluntary torque\n(what the model\nhas to predict)')],
             loc='upper left', bbox_to_anchor=(1.01, 1.05))
    c.set_xlim(w0, w1)
    fig.suptitle(f'Real data: subject YES, session 2, foot locked at {f(ang, 3, True)} rad '
                 f'({f(deg(ang), 0, True)}°, toes up), {int(w1 - w0)} s of the trial',
                 x=0.06, ha='left', y=0.985, fontsize=15, fontweight='bold')
    return save(fig, 'vt_real_strong.png')


# ════════════════════════════════════════════════════════════════════════
# (3) real data, most toes-down angle, old vs new
# ════════════════════════════════════════════════════════════════════════
def fig_toesdown(idx=36):
    r = rec_of(idx); d = TR[idx]
    t = r['t'] - r['t'][0]
    e = r['env_raw'] / EMG_MAX
    calf, shin = e[:, :3].mean(1), e[:, 3]
    tq, Bt, vol, rest = r['tq'], d['B'], d['truth'], d['rest']
    old_arr = PM.predict(r)
    OLDV = float(np.median(old_arr))
    rest_med = float(np.median(tq[rest]))
    gap = OLDV - rest_med
    pushes = push_stretches(idx)
    psz = np.array([float(-np.percentile(vol[a:e_], 5)) for a, e_ in pushes])
    typ = float(np.median(psz))
    # pick the push closest to the typical size, not at the very start
    cand = [i for i, (a, e_) in enumerate(pushes) if a > 500 and (e_ - a) > 200]
    k = min(cand, key=lambda i: abs(psz[i] - typ))
    pa, pe = pushes[k]
    seg_v = vol[pa:pe]
    kmin = pa + int(np.argmin(np.abs(seg_v - np.percentile(seg_v, 5))))
    ang = r['angle']
    NUMS['toesdown'] = dict(trial=idx, angle_rad=round(ang, 3), angle_deg=round(deg(ang), 1),
                            old_value_Nm=round(OLDV, 2), relaxed_level_median_Nm=round(rest_med, 2),
                            gap_old_minus_relaxed_Nm=round(gap, 2),
                            push_size_median_Nm=round(typ, 2),
                            push_sizes_Nm=[round(x, 2) for x in psz],
                            push_annotated_s=[round(pa / 100, 1), round(pe / 100, 1)],
                            push_annotated_size_Nm=round(float(-vol[kmin]), 2),
                            B_range_Nm=[round(float(Bt.min()), 2), round(float(Bt.max()), 2)],
                            n_relaxed_stretches=int(d['n_rest_segs']),
                            old_voluntary_relaxed_median_Nm=round(float(np.median(tq[rest] - OLDV)), 2),
                            new_voluntary_relaxed_median_Nm=round(float(np.median(vol[rest])), 3),
                            old_value_samples_off_by_0p1=int(np.sum(np.abs(old_arr - OLDV) > 0.1)))

    fig, ax = plt.subplots(3, 1, figsize=(12.5, 7.8), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.0, 1.55, 1.3], hspace=0.42))
    a, b, c = ax
    a.plot(t, calf, color=CALF, lw=1.3, label='calf muscles\n(push toes down)')
    a.plot(t, shin, color=SHIN, lw=1.3, label='shin muscle\n(pulls toes up)')
    shade_rest(a, t, rest)
    a.set_ylabel(EMG_YLAB, fontsize=12)
    a.set_ylim(0, 0.52)
    a.set_yticks([0, 0.2, 0.4])
    a.text(88.8, 0.515, 'each muscle is scaled to its own max, so line heights cannot be '
           'compared between calf and shin', ha='right', va='top', fontsize=11, color=SEC)
    a.set_title('a. Muscle activity (EMG): the pushes are much weaker at this angle', loc='left')
    a.legend(loc='upper left', bbox_to_anchor=(1.01, 1.05))

    b.plot(t, tq, color=GREY, lw=1.3)
    shade_rest(b, t, rest)
    b.axhline(OLDV, color=OLD, ls='--', lw=2.2)
    b.plot(t, Bt, color=NEW, lw=2.6)
    b.set_ylim(6.6, 11.6)
    b.set_ylabel('torque (Nm)')
    b.set_title('b. Measured torque and the two passive estimates', loc='left')
    b.text(45, OLDV + 0.12, f'old: value from the separate relaxed recording ({f(OLDV)} Nm)',
           color=OLD, fontsize=12, ha='center', va='bottom', fontweight='bold')
    b.text(88.8, float(Bt[-1]) + 0.3, 'new: measured inside this trial',
           color=NEW, fontsize=12, ha='right', va='bottom', fontweight='bold')
    # gap arrow (old vs real relaxed level) in a relaxed stretch
    # placed in the 44.0-46.4 s relaxed stretch, where the new line (8.53-8.54 Nm) sits at the
    # median relaxed level (8.56 Nm), so the arrow length equals the number written next to it
    xg = 45.2
    darrow(b, xg, rest_med, OLDV, OLD, lw=1.8)
    b.text(xg + 0.8, 0.5 * (OLDV + rest_med) + 0.05,
           f'gap ≈ {f(gap)} Nm', color=OLD, fontsize=12, ha='left', va='center', fontweight='bold')
    NUMS['toesdown']['gap_arrow'] = dict(x_s=xg, from_Nm=round(OLDV, 2), to_Nm=round(rest_med, 2),
                                         B_at_x_Nm=round(float(Bt[int(xg * 100)]), 3))
    # typical push arrow
    xpush = kmin / 100.0
    darrow(b, xpush, float(Bt[kmin]), float(tq[kmin]), TRUTH, lw=1.8)
    b.annotate(f'one typical push ≈ {f(-vol[kmin])} Nm', xy=(xpush, float(tq[kmin]) - 0.05),
               xytext=(xpush, 6.95), ha='center', va='center', fontsize=12, color=TRUTH,
               arrowprops=dict(arrowstyle='->', color=TRUTH, lw=1.0))
    hb = [Line2D([], [], color=GREY, lw=1.6, label='measured torque'),
          Patch(color=REST, alpha=0.35, label='relaxed moments'),
          Line2D([], [], color=OLD, lw=2.2, ls='--', label='old passive'),
          Line2D([], [], color=NEW, lw=2.6, label='new passive')]
    b.legend(handles=hb, loc='upper left', bbox_to_anchor=(1.01, 1.05), labelspacing=0.8)

    c.plot(t, tq - OLDV, color=OLD, lw=1.3)
    c.plot(t, vol, color=NEW, lw=1.3)
    shade_rest(c, t, rest)
    c.axhline(0, color=TRUTH, lw=1.0, ls=':')
    c.set_ylim(-4.7, 1.3)
    c.set_ylabel('torque (Nm)')
    c.set_xlabel('time in the trial (s)')
    c.set_title(f'c. Voluntary torque = measured {M} passive', loc='left')
    xs = 59.5
    c.annotate('relaxed should be 0', xy=(xs, 0.05), xytext=(xs + 4.5, 0.85), fontsize=12,
               color=TRUTH, ha='left', va='center', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color=TRUTH, lw=1.1))
    old_rest = float(np.median(tq[rest] - OLDV))
    c.text(0.5, -4.2, f'old: relaxed moments sit at ≈ {f(old_rest)} Nm, i.e. a "push" that is not there',
           color=OLD, fontsize=12, ha='left', va='center', fontweight='bold')
    hc = [Line2D([], [], color=OLD, lw=1.8, label='old voluntary\n(measured − old)'.replace('-', M)),
          Line2D([], [], color=NEW, lw=1.8, label='new voluntary\n(measured − new)'.replace('-', M))]
    c.legend(handles=hc, loc='upper left', bbox_to_anchor=(1.01, 1.05), labelspacing=0.9)
    c.set_xlim(0, t[-1])
    fig.suptitle(f'Real data: subject YES, session 2, foot locked at {f(ang, 3)} rad '
                 f'({f(deg(ang), 0)}°, most toes-down angle), whole trial',
                 x=0.06, ha='left', y=0.985, fontsize=15, fontweight='bold')
    return save(fig, 'vt_real_toesdown.png')


# ════════════════════════════════════════════════════════════════════════
# (4) the new method in 3 steps, same toes-down trial
# ════════════════════════════════════════════════════════════════════════
def fig_steps(idx=36):
    r = rec_of(idx); d = TR[idx]
    t = r['t'] - r['t'][0]
    tq, Bt, vol, rest = r['tq'], d['B'], d['truth'], d['rest']
    sc = activity_score(r) * 100.0
    sg = SEGS[idx]
    tc, med, n = at._seg_levels(tq, sg)
    keep = at._reject_outliers(tc, med, n, T.SIGMA_S * 100.0)
    NUMS['steps'] = dict(trial=idx, n_relaxed_stretches=len(sg), n_used=int(keep.sum()),
                         stretch_medians_Nm=[round(float(x), 2) for x in med],
                         stretch_lengths_s=[round(float(x) / 100, 2) for x in n],
                         B_start_Nm=round(float(Bt[0]), 2), B_end_Nm=round(float(Bt[-1]), 2),
                         B_range_Nm=[round(float(Bt.min()), 2), round(float(Bt.max()), 2)],
                         relaxed_fraction=round(float(rest.mean()), 3))

    fig, ax = plt.subplots(3, 1, figsize=(12.5, 8.0), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.1, 1.4, 1.2], hspace=0.52))
    a, b, c = ax
    a.plot(t, sc, color=ACT, lw=1.1)
    shade_rest(a, t, rest)
    a.axhline(3, color=REST, lw=1.8, ls='--')
    a.axhline(6, color=BAD, lw=1.4, ls=':')
    a.set_ylim(0, 25)
    ha_ = [Line2D([], [], color=ACT, lw=1.6, label='average activity of\nthe 4 muscles (pushes\ngo off the top)'),
           Line2D([], [], color=BAD, lw=1.6, ls=':', label='above 6%: active'),
           Line2D([], [], color=REST, lw=1.8, ls='--', label='below 3%: relaxed'),
           Patch(color=REST, alpha=0.35, label='relaxed moments')]
    a.legend(handles=ha_, loc='upper left', bbox_to_anchor=(1.01, 1.08), labelspacing=0.6)
    a.set_ylabel('average activity\nof 4 muscles (%)')
    a.set_title('1. Use the muscle signals to find the moments when all 4 muscles are switched off\n'
                '    (green = relaxed moments)', loc='left')

    b.plot(t, tq, color=GREY, lw=1.2)
    shade_rest(b, t, rest)
    b.plot(t, Bt, color=NEW, lw=2.6)
    tcs = tc / 100.0
    b.scatter(tcs[keep], med[keep], s=70, color=NEW, edgecolor='white', lw=1.2, zorder=5)
    if (~keep).any():
        b.scatter(tcs[~keep], med[~keep], s=70, facecolor='white', edgecolor=BAD, lw=1.6, zorder=5)
    b.set_ylim(6.55, 9.6)
    b.set_ylabel('torque (Nm)')
    b.set_title('2. In those moments nobody is pushing, so the sensor reads only passive torque:\n'
                '    take the middle value of each relaxed stretch (dots) and join them with a smooth line',
                loc='left')
    hb = [Line2D([], [], color=GREY, lw=1.6, label='measured torque'),
          Line2D([], [], marker='o', ls='', color=NEW, markeredgecolor='white', markersize=9,
                 label='middle value of one\nrelaxed stretch'),
          Line2D([], [], color=NEW, lw=2.6, label='passive torque line')]
    b.legend(handles=hb, loc='upper left', bbox_to_anchor=(1.01, 1.05), labelspacing=0.9)
    b.text(1.0, 6.62, f'{len(sg)} relaxed stretches in this trial; line goes from {f(Bt[0], 2)} '
           f'to {f(Bt[-1], 2)} Nm (the relaxed level slowly drifts down)',
           fontsize=11.5, color=SEC, ha='left', va='bottom')

    c.plot(t, vol, color=TRUTH, lw=1.2)
    shade_rest(c, t, rest)
    c.axhline(0, color=SEC, lw=0.9, ls=':')
    c.set_ylim(-1.9, 0.7)
    c.set_ylabel('torque (Nm)')
    c.set_xlabel('time in the trial (s)')
    c.set_title(f'3. Voluntary torque = measured torque {M} that line  (0 when relaxed, about {M}1 Nm per push)',
                loc='left')
    c.legend(handles=[Line2D([], [], color=TRUTH, lw=1.6, label='voluntary torque')],
             loc='upper left', bbox_to_anchor=(1.01, 1.05))
    c.set_xlim(0, t[-1])
    fig.suptitle(f'The new way to get voluntary torque, step by step  (subject YES, session 2, '
                 f'{f(r["angle"], 3)} rad = {f(deg(r["angle"]), 0)}°)',
                 x=0.06, ha='left', y=0.995, fontsize=15, fontweight='bold')
    fig.text(0.06, 0.005, 'Relaxed = average activity below 3% of each muscle\'s range for at least 0.5 s '
             '(counts as active again above 6%); the line is smoothed over ~8 s.',
             fontsize=12, color=SEC, ha='left', va='bottom')
    return save(fig, 'vt_newmethod_steps.png')


if __name__ == '__main__':
    files = [fig_concept(), fig_strong(), fig_toesdown(), fig_steps()]
    print(json.dumps(NUMS, indent=1))
