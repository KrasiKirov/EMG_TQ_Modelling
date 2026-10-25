"""F4_oldtable: how the OLD passive table was built and how we saw it was wrong.

Figures (written to notes/final_pipeline/presentation/figures_v2):
  oldtable_passive_trial.png   JM session-1 relaxed recordings, kept / wrongly kept parts
  oldtable_map_JM.png          JM old table vs clean static holds vs in-trial relaxed level
  oldtable_vs_intrial_all.png  old value - in-trial relaxed level, all 6 subjects, both sessions

Run (Git Bash):
  cd <worktree>; source ML/experiments/env.sh; export WT="$(pwd)"; $PY F4_oldtable.py
"""
import sys, os, json, warnings
sys.path.insert(0, os.environ['WT'])
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Ellipse
from matplotlib.legend_handler import HandlerTuple

from ML.config import SUBJECTS, DATA_DIR
from ML.preprocessing.flb_reader import read_flb
from ML.preprocessing import dataset_builder as db
from ML.research.data import load_subject
from ML.research import truth as T
from ML.research.passive import PassiveModel, map_lookup

WT = os.environ['WT']
OUT = os.path.join(WT, 'notes', 'final_pipeline', 'presentation', 'figures_v2')
os.makedirs(OUT, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))

OLD, NEW, TRUTH, GREY, RELAX, BAD = '#2a78d6', '#eb6834', '#1b1b1b', '#8a8984', '#1baf7a', '#c0392b'
GRID, SEC = '#e6e5e1', '#52514e'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 13, 'axes.titlesize': 14, 'axes.titleweight': 'bold',
    'axes.labelsize': 13, 'xtick.labelsize': 11.5, 'ytick.labelsize': 11.5, 'legend.fontsize': 12,
    'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.9, 'axes.axisbelow': True,
    'axes.unicode_minus': True, 'figure.facecolor': 'white', 'axes.edgecolor': '#6b6a66',
})
M = '−'


def f(x, nd=1, plus=False):
    s = f'{x:+.{nd}f}' if plus else f'{x:.{nd}f}'
    return s.replace('-', M)


def deg(rad):
    return f(np.degrees(rad), 0) + '°'


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', p)
    return p


def add_deg_axis(ax):
    sec = ax.secondary_xaxis('top', functions=(np.degrees, np.radians))
    sec.set_xlabel('ankle angle (degrees)', fontsize=11.5, color=SEC)
    sec.tick_params(labelsize=11, colors=SEC)
    return sec


# ════════════════════════════════════════════════════════════════════════════
# numbers: old value subtracted vs relaxed level measured inside each trial
# ════════════════════════════════════════════════════════════════════════════
SUBJ = ['YES', 'JM', 'EG', 'IES', 'CH', 'HM']
rows = []
bundles = {}
for s in SUBJ:
    b = load_subject(s)
    bundles[s] = b
    tr = T.subject_truth(s)
    pm = PassiveModel('legacy', b)
    for r in b['active']:
        old = float(np.mean(pm.base(r)))
        B = tr[r['idx']]['B']
        rows.append(dict(subject=s, session=r['session'], idx=r['idx'], angle=float(r['angle']),
                         old=old, intrial=float(np.median(B)), diff=old - float(np.median(B)),
                         has_table=bool(b['maps']['legacy'][r['session']])))
# toes-down rank per subject/session (0 = most toes down)
for s in SUBJ:
    for ss in ('S1', 'S2'):
        rr = sorted([r for r in rows if r['subject'] == s and r['session'] == ss], key=lambda r: r['angle'])
        for k, r in enumerate(rr):
            r['toes_down_rank'] = k
with open(os.path.join(HERE, 'F4_oldtable_numbers.json'), 'w') as fh:
    json.dump(rows, fh, indent=1)

# ════════════════════════════════════════════════════════════════════════════
# Figure 1: JM session-1 relaxed recordings that produced the bogus entry
# ════════════════════════════════════════════════════════════════════════════
fn, sid = SUBJECTS['JM']
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    trials = read_flb(os.path.join(DATA_DIR, fn), subject_id=sid)
byi = {df.attrs['trial_index']: df for df in trials}


def seg_info(df):
    """Segments exactly as the old rule makes them, with keep/reject flags."""
    out = []
    for seg in db._split_passive_segments(df):
        q = seg[db.TORQUE_COL].values
        p = seg[db.POSITION_COL].values
        keep = q.std() <= db._PASSIVE_STD_THRESHOLD and abs(q.mean()) <= db._PASSIVE_TQ_BOUND
        out.append(dict(t0=seg.index[0] / 1000, t1=(seg.index[-1] + 1) / 1000, pos=float(p.mean()),
                        mean=float(q.mean()), std=float(q.std()), keep=bool(keep)))
    return out


recA, recB = byi[14], byi[20]          # 'passive at p4, test' and 'passive at p7, test'
segA, segB = seg_info(recA), seg_info(recB)
qB = recB[db.TORQUE_COL].values
relaxB = float(qB[:19000].mean())       # first ~19 s of p7: foot still and relaxed
entry_S1 = [e for e in bundles['JM']['maps']['legacy']['S1'] if -0.1 < e[0] < -0.03][0]
entry_S2 = [e for e in bundles['JM']['maps']['legacy']['S2'] if -0.1 < e[0] < -0.03][0]
print('segments p4:', segA)
print('segments p7:', segB, 'relaxed first 19 s', relaxB)
print('bogus entries', entry_S1, entry_S2)

fig = plt.figure(figsize=(12.5, 5.9))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 2.3], hspace=0.12, wspace=0.10)
cols = [(recA, segA, 'Relaxed recording at position 4 (session 1)'),
        (recB, segB, 'Relaxed recording at position 7 (session 1)')]
axT = []
for c, (df, segs, ttl) in enumerate(cols):
    t = np.arange(len(df)) / 1000.0
    pos = df[db.POSITION_COL].values
    tq = df[db.TORQUE_COL].values
    a1 = fig.add_subplot(gs[0, c])
    a2 = fig.add_subplot(gs[1, c], sharex=a1)
    a1.plot(t[::10], pos[::10], color=TRUTH, lw=1.6)
    a1.set_ylim(-0.27, -0.02)
    a1.set_yticks([-0.2, -0.1])
    a1.set_yticklabels([f'{M}0.2 rad ({deg(-0.2)})', f'{M}0.1 rad ({deg(-0.1)})'] if c == 0 else [])
    a1.set_title(ttl, fontsize=14, pad=8)
    a1.tick_params(labelbottom=False)
    a2.plot(t[::10], tq[::10], color=GREY, lw=0.9)
    a2.axhline(0, color=SEC, lw=0.8)
    a2.set_ylim(-32, 66)
    a2.set_xlim(0, 90.1)
    a2.set_xlabel('time (s)')
    if c == 0:
        a1.set_ylabel('angle', labelpad=4)
        a2.set_ylabel('measured torque (Nm)')
    else:
        a2.tick_params(labelleft=False)
    for sg in segs:
        col = BAD if (sg['keep'] and sg['std'] > 1.0) else (RELAX if sg['keep'] else GREY)
        for a in (a1, a2):
            a.axvspan(sg['t0'], sg['t1'], color=col, alpha=0.13, lw=0)
        a2.plot([sg['t0'], sg['t1']], [sg['mean']] * 2, color=col, lw=2.6, ls='--')
    axT.append((a1, a2))

# annotations, left column (position 4)
a1, a2 = axT[0]
g, r = segA[0], segA[1]
a1.annotate('after 20 s the angle signal jumps\n' +f'{f(g["pos"], 2)} → {f(r["pos"], 3)} rad', xy=(20.6, -0.14),
            xytext=(27, -0.215), fontsize=11.5, color=SEC, va='center',
            arrowprops=dict(arrowstyle='->', color=SEC, lw=1))
a2.text(10, 62, f'kept:\n{f(g["mean"], 2)} Nm\nat {f(g["pos"], 2)}\nrad ({deg(g["pos"])})', color=RELAX,
        fontsize=12, ha='center', va='top', fontweight='bold', linespacing=1.15)
a2.text(56, 62, f'wrongly kept: {f(r["mean"], 1, True)} Nm', color=BAD, fontsize=12.5,
        ha='center', va='top', fontweight='bold')
a2.text(56, 54, f'not a relaxed hold: torque jumps ~40 Nm,\nbut its spread ({f(r["std"], 1)} Nm) was still\nunder the old 20 Nm limit',
        color=BAD, fontsize=11.5, ha='center', va='top')

# annotations, right column (position 7)
a1, a2 = axT[1]
s7 = segB[0]
a1.text(45, -0.12, f'angle signal stays near {f(s7["pos"], 2)} rad ({deg(s7["pos"])}): no jump detected,\nso the whole 90 s counts as one "steady" part',
        fontsize=11.5, color=SEC, ha='center', va='center')
a2.text(56, 62, f'wrongly kept: {f(s7["mean"], 1, True)} Nm', color=BAD, fontsize=12.5,
        ha='center', va='top', fontweight='bold')
a2.text(56, 54, f'whole 90 s averaged (spread {f(s7["std"], 1)} Nm,\nstill under the 20 Nm limit)',
        color=BAD, fontsize=11.5, ha='center', va='top')
a2.plot([0.3, 19], [relaxB] * 2, color=TRUTH, lw=2.4)
a2.annotate(f'first 19 s really relaxed: {f(relaxB, 1)} Nm', xy=(12, relaxB - 0.8), xytext=(22, -16),
            fontsize=11.5, color=TRUTH, ha='left', va='center',
            arrowprops=dict(arrowstyle='->', color=TRUTH, lw=1))

fig.suptitle('Old method, step 1: separate relaxed recording before each pushing trial; "steady" parts averaged',
             fontsize=15, fontweight='bold', y=1.0)
fig.text(0.5, -0.05,
         f'The two red values were averaged into one table entry at {f(entry_S1[0], 2)} rad ({deg(entry_S1[0])}): '
         f'({f(r["mean"], 1)} + {f(s7["mean"], 1)}) / 2 = {f(entry_S1[1], 1, True)} Nm.   '
         f'The real relaxed value there is about {f(relaxB, 0)} Nm.',
         ha='center', va='top', fontsize=12.5, color=TRUTH)
leg = [Patch(color=RELAX, alpha=0.35, label='steady part, kept (correct)'),
       Patch(color=BAD, alpha=0.35, label='kept by mistake (not a relaxed hold)'),
       Line2D([], [], color=GREY, lw=1.2, label='measured torque'),
       (Line2D([], [], color=RELAX, lw=2.4, ls=(0, (2.0, 1.0))),
        Line2D([], [], color=BAD, lw=2.4, ls=(0, (2.0, 1.0))))]
labs = ['steady part, kept (correct)', 'kept by mistake (not a relaxed hold)', 'measured torque',
        'average of each kept part']
fig.legend(handles=leg, labels=labs, loc='upper center', bbox_to_anchor=(0.5, -0.095), ncol=4, fontsize=11.5,
           handlelength=3.0, columnspacing=1.4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)})
save(fig, 'oldtable_passive_trial.png')

# ════════════════════════════════════════════════════════════════════════════
# Figure 2: JM old table vs static-hold map vs in-trial relaxed level
# ════════════════════════════════════════════════════════════════════════════
b = bundles['JM']
fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
fig.subplots_adjust(wspace=0.06)
xx = np.linspace(-0.82, 0.24, 400)
for ax, ss, lab in zip(axs, ('S1', 'S2'), ('session 1', 'session 2')):
    leg_map = b['maps']['legacy'][ss]
    st_map = b['maps']['static'][ss]
    act = sorted([r for r in rows if r['subject'] == 'JM' and r['session'] == ss], key=lambda r: r['angle'])
    ax.plot(xx, map_lookup(leg_map, xx), color=OLD, lw=2.2, zorder=3)
    ax.plot([p for p, _ in leg_map], [q for _, q in leg_map], 'o', color=OLD, ms=8, zorder=4)
    ax.plot(xx, map_lookup(st_map, xx), color=GREY, lw=2.0, ls='--', zorder=2)
    ax.plot([r['angle'] for r in act], [r['intrial'] for r in act], 'o', color=NEW, ms=9,
            markeredgecolor='white', markeredgewidth=1.2, zorder=5)
    for r in act:
        ax.plot([r['angle']] * 2, [-9.6, -8.8], color=TRUTH, lw=1.8, zorder=1)
    bp = [e for e in leg_map if -0.1 < e[0] < -0.03][0]
    ax.add_patch(Ellipse(bp, 0.085, 2.6, fill=False, color=BAD, lw=2.4, zorder=6))
    ax.annotate(f'bad entry: {f(bp[1], 2, True)} Nm\n(from the non-relaxed parts)', xy=(bp[0] - 0.045, bp[1] + 0.3),
                xytext=(-0.80, 12.6), color=BAD, fontsize=12, ha='left', va='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=BAD, lw=1.2))
    # the trial at about -0.08 rad: how much too much was subtracted
    rb = [r for r in act if -0.1 < r['angle'] < -0.03][0]
    ax.annotate('', xy=(rb['angle'], rb['intrial']), xytext=(rb['angle'], rb['old']),
                arrowprops=dict(arrowstyle='<->', color=TRUTH, lw=1.5, shrinkA=0, shrinkB=5.5), zorder=4.5)
    ymid = (rb['old'] + rb['intrial']) / 2
    ax.annotate(f'{f(rb["diff"], 1)} Nm too\nmuch subtracted\nfor the trial at\n{f(rb["angle"], 3)} rad',
                xy=(rb['angle'] - 0.004, ymid), xytext=(-0.215, 7.6), ha='right', va='center', fontsize=11.5,
                color=TRUTH, arrowprops=dict(arrowstyle='-', color=TRUTH, lw=0.9, shrinkA=3, shrinkB=0))
    print('JM', ss, 'arrow at', rb['angle'], 'from old', rb['old'], 'to intrial', rb['intrial'],
          'map at angle', float(map_lookup(leg_map, np.array([rb['angle']]))[0]))
    sx = np.linspace(-0.8, 0.2, 201)
    dd = np.abs(map_lookup(leg_map, sx) - map_lookup(st_map, sx))
    print('  legacy vs static map: max diff away from bad entry',
          float(dd[np.abs(sx - bp[0]) > 0.15].max()), ' n entries', len(leg_map), len(st_map))
    ax.axhline(0, color=SEC, lw=0.8, zorder=1)
    ax.set_xlim(-0.86, 0.27)
    ax.set_ylim(-10, 15.5)
    ax.set_title(f'JM, {lab}', pad=6)
    ax.set_xlabel('ankle angle (rad)    ← toes down  |  toes up →')
    add_deg_axis(ax)
axs[0].set_ylabel('passive (relaxed) torque (Nm)')
axs[0].text(-0.84, -8.0, 'angles of the 8\npushing trials', fontsize=11, color=SEC, va='bottom')
fig.suptitle('Old method, step 2: one number per angle, read off this curve', fontsize=15, fontweight='bold', y=1.07)
leg = [Line2D([], [], color=OLD, lw=2.2, marker='o', ms=7, label='old table (from the relaxed recordings)'),
       Line2D([], [], color=NEW, lw=0, marker='o', ms=9, label='relaxed level measured inside each pushing trial'),
       Line2D([], [], color=GREY, lw=2.0, ls='--',
              label='steady relaxed holds only (clean): same as the old table except around the bad entry, '
                    'so elsewhere it is hidden under the blue line')]
fig.legend(handles=leg, loc='upper center', bbox_to_anchor=(0.5, -0.0), ncol=1, fontsize=11.5,
           handlelength=1.8, columnspacing=1.6)
save(fig, 'oldtable_map_JM.png')

# ════════════════════════════════════════════════════════════════════════════
# Figure 3: old value - in-trial relaxed level, all subjects
# ════════════════════════════════════════════════════════════════════════════
YL = 5.6
fig, axs = plt.subplots(2, 3, figsize=(13, 6.9), sharey=True)
fig.subplots_adjust(hspace=0.55, wspace=0.08)
titles = {'YES': 'YES (table typed in from lab sheet)', 'JM': 'JM', 'EG': 'EG',
          'IES': 'IES', 'CH': 'CH', 'HM': 'HM'}
for ax, s in zip(axs.ravel(), SUBJ):
    ax.axhline(0, color=TRUTH, lw=1.3, zorder=2)
    for ss, mk, fc, lab in (('S1', 'o', OLD, 'session 1'), ('S2', 'D', 'white', 'session 2')):
        rr = sorted([r for r in rows if r['subject'] == s and r['session'] == ss], key=lambda r: r['angle'])
        x = np.array([r['angle'] for r in rr])
        y = np.array([r['diff'] for r in rr])
        yc = y
        if s == 'IES':
            ax.plot(x, yc, color=GREY, lw=1.0, alpha=0.8, zorder=3)
            ax.plot(x, yc, mk, color=GREY, mfc=fc if fc == 'white' else GREY, ms=7, mew=1.6, zorder=4)
        else:
            ax.plot(x, yc, color=OLD, lw=1.0, alpha=0.6, zorder=3)
            ax.plot(x, yc, mk, color=OLD, mfc=fc, ms=7.5, mew=1.6, zorder=4)
    ax.set_ylim(-YL, YL)
    ang = [r['angle'] for r in rows if r['subject'] == s]
    ax.set_xlim(min(ang) - 0.08, max(ang) + 0.08)
    ax.set_title(titles[s], fontsize=13.5, pad=6)
    if s == 'IES':
        ax.text(0.97, 0.04, 'no old table was found:\nold method subtracted\nnothing (0)', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=11, color=SEC)
    if s == 'JM':
        jm = sorted([r for r in rows if r['subject'] == 'JM' and r['diff'] > YL], key=lambda r: r['session'])
        ax.text(0.03, 0.04, f'off the scale at {f(jm[0]["angle"], 2)} rad:\n'
                f'{f(jm[0]["diff"], 1, True)} Nm (session 1)\n{f(jm[1]["diff"], 1, True)} Nm (session 2)',
                transform=ax.transAxes, ha='left', va='bottom', fontsize=11.5, color=BAD, fontweight='bold')
    ax.set_xlabel('ankle angle (rad)', fontsize=12)
fig.supylabel('old value − relaxed level measured in the trial (Nm)', fontsize=12.5, x=0.075)
fig.suptitle('How far off was the old passive value? (0 = correct)', fontsize=15, fontweight='bold', y=1.0)
leg = [Line2D([], [], color=OLD, marker='o', lw=0, ms=8, mew=1.6, label='session 1'),
       Line2D([], [], color=OLD, marker='D', mfc='white', lw=0, ms=7.5, mew=1.6, label='session 2')]
fig.legend(handles=leg, loc='upper center', bbox_to_anchor=(0.5, 0.965), ncol=2, fontsize=12, columnspacing=2)
fig.text(0.5, -0.03, 'above 0: old method subtracted too much (relaxed parts of the target sit below zero)   |   '
         'below 0: too little   |   left = toes down, right = toes up',
         ha='center', va='top', fontsize=11.5, color=SEC)
save(fig, 'oldtable_vs_intrial_all.png')

# ════════════════════════════════════════════════════════════════════════════
# summary
# ════════════════════════════════════════════════════════════════════════════
def summ(sel, name):
    d = np.array([r['diff'] for r in sel])
    print(f'{name:28s} n={len(d):3d} median|d|={np.median(np.abs(d)):.2f} mean|d|={np.mean(np.abs(d)):.2f} '
          f'max|d|={np.max(np.abs(d)):.2f} share>0.5={np.mean(np.abs(d) > 0.5):.2f} share>1={np.mean(np.abs(d) > 1):.2f}')


tab = [r for r in rows if r['has_table']]
summ(tab, 'all with table (5 subj)')
summ([r for r in tab if not (r['subject'] == 'JM' and -0.1 < r['angle'] < -0.03)], 'with table, excl JM bug')
summ(rows, 'all incl IES (0 subtracted)')
summ([r for r in tab if r['toes_down_rank'] <= 1], '2 most toes-down, with table')
summ([r for r in tab if r['toes_down_rank'] >= 2], 'other angles, with table')
for s in SUBJ:
    summ([r for r in rows if r['subject'] == s], s)
print('worst (with table):')
for r in sorted(tab, key=lambda r: -abs(r['diff']))[:8]:
    print(f"  {r['subject']} {r['session']} {r['angle']:+.3f} old={r['old']:+.2f} intrial={r['intrial']:+.2f} diff={r['diff']:+.2f}")
print('per subject table:')
for s in SUBJ:
    for ss in ('S1', 'S2'):
        rr = sorted([r for r in rows if r['subject'] == s and r['session'] == ss], key=lambda r: r['angle'])
        print(s, ss, ' '.join(f"{r['angle']:+.3f}:{r['diff']:+.2f}" for r in rr))
