"""Figures for the presentation (notes/final_pipeline/presentation/figures)."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse, Circle, Arc, Rectangle, FancyBboxPatch, FancyArrowPatch

WT = os.environ['WT']
sys.path.insert(0, WT)
OUT = os.path.join(WT, 'notes', 'final_pipeline', 'presentation', 'figures')
os.makedirs(OUT, exist_ok=True)

INK, INK2, MUTED, GRID = '#1b1b1b', '#52514e', '#8a8984', '#e6e5e1'
OLD, NEW, AQUA, GREEN = '#2a78d6', '#eb6834', '#1baf7a', '#1baf7a'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID,
                     'axes.axisbelow': True, 'legend.frameon': False})
M = chr(0x2212)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', p)


# ── 1. setup: ankle angles ────────────────────────────────────────────────────
def rot(pts, th):
    c, s = np.cos(th), np.sin(th)
    return pts @ np.array([[c, -s], [s, c]]).T


def draw_leg(ax, cx, th, label, sub, highlight=False):
    ax.add_patch(Ellipse((cx - 0.30, 2.15), 0.55, 1.55, fc='#e4e3df', ec=MUTED, lw=1.2, zorder=1))
    ax.add_patch(Rectangle((cx - 0.17, 0.0), 0.34, 3.2, fc='#e4e3df', ec=MUTED, lw=1.2, zorder=2))
    ax.plot([cx - 0.2, cx + 1.95], [0, 0], ls='--', color=MUTED, lw=1.2, zorder=0)
    foot = np.array([[-0.40, -0.24], [1.30, -0.24], [1.42, -0.16], [1.36, -0.06], [0.30, 0.18], [-0.34, 0.14]])
    foot = rot(foot, th) + np.array([cx, 0.0])
    ax.add_patch(Polygon(foot, closed=True, fc=INK2 if highlight else '#bdbcb7', ec=INK, lw=1.4, zorder=3))
    ax.add_patch(Circle((cx, 0.0), 0.09, fc='white', ec=INK, lw=1.4, zorder=4))
    if abs(th) > 1e-6:
        a0, a1 = (0, np.degrees(th)) if th > 0 else (np.degrees(th), 0)
        ax.add_patch(Arc((cx, 0), 3.6, 3.6, theta1=a0, theta2=a1, color=INK, lw=1.8, zorder=5))
        mid = np.radians((a0 + a1) / 2)
        ax.text(cx + 1.95 * np.cos(mid) + 0.08, 1.95 * np.sin(mid), f'{th:+.2f} rad'.replace('-', M),
                ha='left', va='center', fontsize=12, weight='bold', color=INK)
    ax.text(cx + 0.45, -1.3, label, ha='center', va='top', fontsize=14, weight='bold', color=INK)
    ax.text(cx + 0.45, -1.85, sub, ha='center', va='top', fontsize=12, color=INK2, linespacing=1.3)


def fig_setup():
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(-0.9, 8.7); ax.set_ylim(-3.6, 3.7)
    draw_leg(ax, 0.0, -0.50, 'Plantarflexed', 'toes DOWN, angle < 0\ncalf short = weak', highlight=True)
    draw_leg(ax, 3.0, 0.0, 'Neutral', 'foot flat\nangle = 0')
    draw_leg(ax, 6.0, +0.25, 'Dorsiflexed', 'toes UP, angle > 0\ncalf long = strong')
    ax.annotate('calf muscles\nGM, GL, SOL', xy=(2.55, 2.4), xytext=(1.2, 3.2), fontsize=11.5, color=INK2,
                ha='center', va='center', arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.2))
    ax.annotate('shin muscle TA\n(front of the leg)', xy=(3.17, 1.3), xytext=(4.6, 3.2), fontsize=11.5, color=INK2,
                ha='center', va='center', arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.2))
    save(fig, 'setup_angles.png')


# ── 2. last semester's problem: old per-angle R2 (own target, as scored then) ──
def fig_problem():
    S = json.load(open(os.path.join(WT, 'ML/research/results/screen/_summary/summary.json')))
    own = S['own']
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, sub in zip(axs, ['YES', 'JM']):
        x = own[f'{sub}|S1toS2|E01_old']
        ang = [float(k.split(':')[1]) for k in x['angles']]
        r2 = np.array(x['r2'])
        cols = ['#c0392b' if v < 0.5 else '#b9b8b3' for v in r2]
        ax.bar(range(8), np.maximum(r2, -1.2), color=cols, edgecolor=INK2, lw=0.8, zorder=3)
        for i, v in enumerate(r2):
            ax.text(i, max(v, -1.2) + (0.04 if v >= 0 else -0.06), f'{v:.2f}'.replace('-', M), ha='center',
                    va='bottom' if v >= 0 else 'top', fontsize=11, color=INK, weight='bold' if v < 0.5 else 'normal')
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(range(8)); ax.set_xticklabels([f'{a:+.2f}'.replace('-', M) for a in ang], fontsize=10.5)
        ax.set_xlabel('ankle angle (rad)   ' + chr(0x2190) + ' toes down    toes up ' + chr(0x2192))
        ax.set_title(f'Subject {sub}', loc='left', fontsize=14, weight='bold', color=INK)
        ax.set_ylim(-1.45, 1.15); ax.grid(axis='x', visible=False)
    axs[0].set_ylabel('R² per angle (1 = perfect)')
    fig.text(0.5, -0.06, 'Old pipeline, trained on session 1, tested on session 2, scored the way it was scored last semester '
             '(against its own target). Mean of 3 training runs.', ha='center', fontsize=11, color=INK2)
    save(fig, 'problem_old_r2.png')


# ── 3. how passive and voluntary torque are computed (YES S2 -0.486) ──────────
def fig_torque_split():
    from ML.research.data import load_subject
    from ML.research import truth as T
    from ML.research.passive import PassiveModel
    b = load_subject('YES')
    rec = min([r for r in b['active'] if r['session'] == 'S2'], key=lambda r: abs(r['angle'] + 0.486))
    old_p = PassiveModel('legacy', b).predict(rec)
    tr = T.subject_truth('YES')[rec['idx']]
    t, tq, B, rest = rec['t'], rec['tq'], tr['B'], tr['rest']
    keep = (t > t[0] + 0.5) & (t < t[-1] - 0.5)          # hide the filter edge transients
    t, tq, B, rest, old_p = t[keep], tq[keep], B[keep], rest[keep], old_p[keep]
    hold = rec['hold']['mean'] if rec['hold'] else None
    fig, axs = plt.subplots(2, 1, figsize=(12, 8.2), sharex=True,
                            gridspec_kw={'height_ratios': [1.15, 1], 'hspace': 0.45})
    ax = axs[0]
    ax.plot(t, tq, color=MUTED, lw=1.0, label='measured torque (sensor)')
    ax.scatter(t[rest], tq[rest], s=3, color=GREEN, zorder=3, label='relaxed moments (found from EMG only)')
    ax.plot(t, old_p, color=OLD, lw=2.4, ls='--', label=f'OLD passive: from separate relaxed recording ({old_p.mean():.1f} Nm)')
    ax.plot(t, B, color=NEW, lw=2.6, label='NEW passive: smooth line through the relaxed moments')
    ax.set_ylabel('torque (Nm)'); ax.set_ylim(7.0, 11.0)
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, -0.06), fontsize=10.5, ncol=2)
    ax.set_title('Subject YES, session 2, ankle at ' + M + '0.486 rad: one 90 s trial', loc='left', fontsize=14,
                 weight='bold', color=INK)
    ax = axs[1]
    ax.axhline(0, color=INK2, lw=1)
    ax.plot(t, tq - old_p, color=OLD, lw=1.4, label='OLD voluntary = measured ' + M + ' old passive')
    ax.plot(t, tq - B, color=NEW, lw=1.4, label='NEW voluntary = measured ' + M + ' new passive')
    ax.set_ylabel('voluntary torque (Nm)'); ax.set_xlabel('time in the trial (s)')
    ax.set_ylim(-3.9, 1.6)
    ax.legend(loc='lower right', fontsize=10.5, ncol=2)
    ax.text(0.99, 0.97, 'relaxed should be 0 Nm:  NEW sits at 0,  OLD sits ~2 Nm too low', transform=ax.transAxes,
            ha='right', va='top', fontsize=11, color=INK, bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=MUTED))
    save(fig, 'torque_split.png')
    return float(old_p.mean()), float(np.median(B)), hold


# ── 4. old vs new pipeline flow ──────────────────────────────────────────────
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(15, 6.2))
    ax.set_xlim(0, 15); ax.set_ylim(0, 6.2); ax.axis('off')

    def box(x, y, w, h, title, sub, fc, ec, lw=1.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.1', fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h - 0.22, title, ha='center', va='top', fontsize=12, weight='bold', color=INK)
        ax.text(x + w / 2, y + h - 0.62, sub, ha='center', va='top', fontsize=10, color=INK2, linespacing=1.3)

    def arr(x1, y, x2):
        ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle='-|>', mutation_scale=14, lw=1.4, color=INK2))
    steps_old = [('EMG envelope', '30-300 Hz filter,\nabsolute value,\n2 Hz smoothing'),
                 ('EMG scaling', '/ biggest value of\nthe training session'),
                 ('Target', 'measured torque\n' + M + ' passive value from\nseparate relaxed\nrecordings'),
                 ('Model', 'LSTM 8 ' + chr(0x2192) + ' LSTM 8\n' + chr(0x2192) + ' 1 output\n1,001 numbers'),
                 ('Training loss', 'plain MSE (Nm²):\nstrong angles\ndominate'),
                 ('Checking', 'train S1 ' + chr(0x2192) + ' test S2,\n1 run, R² vs its\nown target')]
    steps_new = [('EMG envelope', 'same'),
                 ('EMG scaling', '/ training max, then\nAUTO-GAIN (adapts to\nthe recent past);\noff if pushing both ways'),
                 ('Target', 'measured torque\n' + M + ' relaxed level found\nINSIDE the same trial'),
                 ('Model', 'LSTM 16 ' + chr(0x2192) + ' LSTM 16\n' + chr(0x2192) + ' output + 1 expert\nper angle (3,665)'),
                 ('Training loss', 'MSE on targets\n/ each angle\'s spread:\nall angles count'),
                 ('Checking', 'both directions, 3 runs,\none SHARED truth,\n+ RMSE, rest, 3 extra\nsubjects')]
    ax.text(0.05, 6.0, 'OLD (last semester)', fontsize=14, weight='bold', color=OLD, va='top')
    ax.text(0.05, 2.95, 'NEW (FINAL)', fontsize=14, weight='bold', color=NEW, va='top')
    w, h, gap, x0 = 2.25, 1.85, 0.25, 0.1
    for row, steps, y in ((0, steps_old, 3.6), (1, steps_new, 0.5)):
        for i, (tt, ss) in enumerate(steps):
            x = x0 + i * (w + gap)
            changed = row == 1 and tt != 'EMG envelope'
            box(x, y, w, h, tt, ss, '#fbe0d4' if changed else ('#dbe8fb' if row == 0 else '#f0efec'),
                NEW if changed else (OLD if row == 0 else MUTED), lw=2.0 if changed else 1.4)
            if i < len(steps) - 1:
                arr(x + w, y + h / 2, x + w + gap)
    ax.text(15.0, 0.05, 'orange = changed in the new pipeline', ha='right', fontsize=10.5, color=NEW)
    save(fig, 'pipeline_old_vs_new.png')


# ── 5. before / after at the hardest angles ──────────────────────────────────
def fig_overlays():
    from ML.research import truth as T
    from ML.research.evaluate import metrics
    R = os.path.join(WT, 'ML/research/results')
    cases = [('YES', 'S1toS2', -0.486, 'screen'), ('JM', 'S2toS1', -0.787, 'screen'),
             ('EG', 'S1toS2', -0.739, 'screen'), ('HM', 'S1toS2', -0.789, 'confirm')]
    fig, axs = plt.subplots(len(cases), 1, figsize=(12, 10.5), sharex=True)
    for ax, (sub, prot, ang, st) in zip(axs, cases):
        tr = T.subject_truth(sub)
        zo = np.load(f'{R}/{st}/{sub}/{prot}/E01_old/fast_seed0.npz')
        zf = np.load(f'{R}/final_check/{sub}/{prot}/FINAL/fast_seed0.npz')
        ti = min(np.unique(zf['trial']), key=lambda i: abs(tr[int(i)]['angle'] - ang))
        mo, mf = zo['trial'] == ti, zf['trial'] == ti
        t = zf['t'][mf]
        d = tr[int(ti)]
        y = d['truth'][np.rint((t - d['t0']) * 100).astype(int)]
        po, pf = zo['y_pred'][mo], zf['y_pred'][mf]
        so, sf = metrics(y, po), metrics(y, pf)
        ax.axhline(0, color=GRID, lw=1)
        ax.plot(t, y, color=INK, lw=1.3, label='truth (voluntary torque)')
        ax.plot(zo['t'][mo], po, color=OLD, lw=1.2, label=f"OLD   R² {so['r2']:.2f}  error {so['rmse']:.2f} Nm".replace('-', M))
        ax.plot(t, pf, color=NEW, lw=1.2, label=f"FINAL R² {sf['r2']:.2f}  error {sf['rmse']:.2f} Nm".replace('-', M))
        test = prot[-2:]
        tag = 'held-out subject' if st == 'confirm' else 'main subject'
        ax.set_title(f"{sub} ({tag}), trained {prot[:2]} " + chr(0x2192) + f" tested {test}, angle {d['angle']:+.3f} rad".replace('-', M),
                     loc='left', fontsize=12.5, weight='bold', color=INK)
        ax.set_ylabel('Nm')
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=10.5)
    axs[-1].set_xlabel('time in the trial (s)')
    fig.text(0.0, -0.02, 'One training run each (seed 0). Negative = pushing toes down; 0 = relaxed.', fontsize=10.5, color=INK2)
    fig.tight_layout()
    save(fig, 'overlays_hard_angles.png')


if __name__ == '__main__':
    fig_setup()
    fig_problem()
    print('torque split (old passive mean, new passive median, hold):', fig_torque_split())
    fig_pipeline()
    fig_overlays()
