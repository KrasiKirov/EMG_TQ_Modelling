"""F5 - EMG size changes between session 1 and session 2, and what auto-gain does.

Figures (written to notes/final_pipeline/presentation/figures_v2):
  emg_size_change.png  per subject / per angle: calf EMG size in session 2 relative to session 1
                       for the same voluntary push (slope ratio), dots + median bar, log axis
  agc_timeseries.png   JM -0.64 rad, session 1 vs session 2: calf EMG with auto-gain floor/top,
                       auto-gain output, voluntary torque
  agc_scatter.png      JM -0.64 rad: calf EMG vs voluntary push, before vs after auto-gain

Method (size-change ratio, one number per subject x angle)
  * EMG: envelopes divided by the SESSION-1 per-channel maximum (normalise('trainmax') with the
    session-1 records = the same fixed scale for both sessions).  calf = mean of GM, GL, SOL.
  * push = -(reference truth) = voluntary torque with toes-down positive (Nm).
    CH at the angles where it pulled toes UP in BOTH sessions: push = +truth and the TA (shin)
    channel replaces the calf.  CH's -0.31/-0.33 rad pair is excluded (toes up in session 1,
    toes down in session 2: not the same push).
  * relaxed level = median EMG over the EMG-detected relaxed samples of that trial (truth 'rest'
    mask); if a trial has < 50 relaxed samples (0.5 s), its 5th percentile is used instead.
  * pushing samples = push inside the range BOTH sessions reached at that angle:
    from max(0.5 Nm, 10 % of top) to top, top = the smaller of the two sessions' 98th
    percentile of push.  The first and last 1 s of each trial are dropped.
  * per session: least-squares line through the origin of (EMG - relaxed level) vs push on the
    pushing samples; ratio = slope(session 2) / slope(session 1).
  * angles where the push std is < 0.5 Nm in either session are skipped (too small to measure).
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

WT = os.environ['WT']
sys.path.insert(0, WT)
from ML.research.data import load_subject
from ML.research import truth as T
from ML.research.emg_norm import normalise, _train_part
from ML.preprocessing import emg_agc

OUT = os.path.join(WT, 'notes', 'final_pipeline', 'presentation', 'figures_v2')
os.makedirs(OUT, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))

OLD, NEW, TRUTH, GREY, GREEN, RED, GRID, INK2 = ('#2a78d6', '#eb6834', '#1b1b1b', '#8a8984',
                                                 '#1baf7a', '#c0392b', '#e6e5e1', '#52514e')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': GREY, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'xtick.labelsize': 11.5,
                     'ytick.labelsize': 11.5, 'axes.grid': True, 'grid.color': GRID,
                     'axes.axisbelow': True, 'legend.frameon': False, 'legend.fontsize': 11.5,
                     'axes.titlesize': 14, 'axes.titleweight': 'bold'})
M = '−'      # minus sign
X = '×'      # times sign
DEG = '°'
SUBJECTS = ['YES', 'JM', 'EG', 'IES', 'CH', 'HM']
MAIN = {'YES', 'JM', 'EG'}
EDGE = 100        # 1 s trimmed at both ends of every trial
MIN_STD = 0.5     # Nm
MIN_REST = 50     # samples


def fm(x, fmt='{:.2f}'):
    return fmt.format(x).replace('-', M)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', p)
    return p


# ----------------------------------------------------------------------------------------------
# (1) size-change ratio per subject x angle
# ----------------------------------------------------------------------------------------------
def pair_trials(b):
    s1 = sorted([r for r in b['active'] if r['session'] == 'S1'], key=lambda r: r['angle'])
    s2 = [r for r in b['active'] if r['session'] == 'S2']
    pairs = []
    for r1 in s1:
        r2 = min(s2, key=lambda r: abs(r['angle'] - r1['angle']))
        pairs.append((r1, r2))
    return pairs


def relaxed_level(y, rest):
    m = rest.copy()
    m[:EDGE] = False
    m[-EDGE:] = False
    if m.sum() >= MIN_REST:
        return float(np.median(y[m])), int(m.sum())
    return float(np.percentile(y[EDGE:-EDGE], 5)), int(m.sum())


def slope_through_origin(x, y):
    return float(np.sum(x * y) / np.sum(x * x))


def size_ratios(subject):
    b = load_subject(subject)
    tr = T.subject_truth(subject)
    cols = [c.lower() for c in b['emg_cols']]
    calf = [cols.index('gm'), cols.index('gl'), cols.index('sol')]
    ta = cols.index('ta')
    s1recs = [r for r in b['active'] if r['session'] == 'S1']
    base, _ = normalise('trainmax', b, s1recs, b['active'])
    agc, _ = normalise('agc', b, s1recs, b['active'])
    rows = []
    for r1, r2 in pair_trials(b):
        t1, t2 = tr[r1['idx']]['truth'], tr[r2['idx']]['truth']
        # direction of the push in each session (median of the 5 % most extreme samples)
        d1 = np.sign(np.median(t1[np.abs(t1) >= np.percentile(np.abs(t1), 95)]))
        d2 = np.sign(np.median(t2[np.abs(t2) >= np.percentile(np.abs(t2), 95)]))
        row = dict(subject=subject, angle1=float(r1['angle']), angle2=float(r2['angle']),
                   idx1=int(r1['idx']), idx2=int(r2['idx']))
        if d1 != d2:
            row.update(status='excluded: push direction differs between sessions')
            rows.append(row)
            continue
        toes_up = d1 > 0
        sign = 1.0 if toes_up else -1.0
        chans = [ta] if toes_up else calf
        p1, p2 = sign * t1, sign * t2
        s1, s2 = p1[EDGE:-EDGE].std(), p2[EDGE:-EDGE].std()
        row.update(toes_up=bool(toes_up), muscle='TA (shin)' if toes_up else 'calf (GM, GL, SOL)',
                   push_std1=float(s1), push_std2=float(s2))
        if min(s1, s2) < MIN_STD:
            row.update(status=f'skipped: push std < {MIN_STD} Nm')
            rows.append(row)
            continue
        top = min(np.percentile(p1[EDGE:-EDGE], 98), np.percentile(p2[EDGE:-EDGE], 98))
        lo = max(0.5, 0.1 * top)
        out = {}
        for tag, E in (('raw', base), ('agc', agc)):
            sl, lev = [], []
            for rec, p in ((r1, p1), (r2, p2)):
                y = E[rec['idx']][:, chans].mean(axis=1)
                rest = tr[rec['idx']]['rest']
                lvl, nrest = relaxed_level(y, rest)
                m = np.zeros(len(p), bool)
                m[EDGE:-EDGE] = True
                m &= (p >= lo) & (p <= top)
                sl.append(slope_through_origin(p[m], y[m] - lvl))
                lev.append((lvl, nrest, int(m.sum())))
            out[tag] = (sl, lev)
        # per-channel raw ratios (calf only), for the notes
        per_ch = {}
        if not toes_up:
            for k, nm in zip(calf, ['GM', 'GL', 'SOL']):
                sl = []
                for rec, p in ((r1, p1), (r2, p2)):
                    y = base[rec['idx']][:, k]
                    lvl, _ = relaxed_level(y, tr[rec['idx']]['rest'])
                    m = np.zeros(len(p), bool)
                    m[EDGE:-EDGE] = True
                    m &= (p >= lo) & (p <= top)
                    sl.append(slope_through_origin(p[m], y[m] - lvl))
                per_ch[nm] = sl[1] / sl[0] if sl[0] > 0 else float('nan')
        row.update(status='ok', push_lo=float(lo), push_hi=float(top),
                   slope1=out['raw'][0][0], slope2=out['raw'][0][1],
                   ratio=out['raw'][0][1] / out['raw'][0][0],
                   ratio_agc=out['agc'][0][1] / out['agc'][0][0],
                   rest_level1=out['raw'][1][0][0], rest_level2=out['raw'][1][1][0],
                   n_rest1=out['raw'][1][0][1], n_rest2=out['raw'][1][1][1],
                   n_push1=out['raw'][1][0][2], n_push2=out['raw'][1][1][2], per_channel=per_ch)
        rows.append(row)
    # the 2 most toes-down angles of this subject
    angs = sorted(r['angle1'] for r in rows)
    for r in rows:
        r['most_toes_down'] = r['angle1'] in angs[:2]
    return rows


def summarise(all_rows):
    summary = {}
    for s in SUBJECTS:
        rows = [r for r in all_rows if r['subject'] == s]
        ok = [r for r in rows if r['status'] == 'ok']
        rat = np.array([r['ratio'] for r in ok])
        ra = np.array([r['ratio_agc'] for r in ok])
        td = [r['ratio'] for r in ok if r['most_toes_down']]
        summary[s] = dict(median=float(np.median(rat)), min=float(rat.min()), max=float(rat.max()),
                          n=len(ok), n_total=len(rows), median_agc=float(np.median(ra)),
                          min_agc=float(ra.min()), max_agc=float(ra.max()),
                          toes_down_ratios=[float(x) for x in td],
                          spread_log2=float(np.std(np.log2(rat))), spread_log2_agc=float(np.std(np.log2(ra))))
    return summary


def _log_axis(ax, lo, hi):
    ax.set_yscale('log')
    ticks = [t for t in [0.25, 0.33, 0.5, 0.67, 1, 1.5, 2, 3, 4] if lo <= t <= hi]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: ('same size  1' + X if v == 1 else f'{v:g}' + X)))
    ax.set_ylim(lo, hi)


def fig_size_change(all_rows, with_agc=False):
    from matplotlib.lines import Line2D
    summary = summarise(all_rows)
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    groups = [('ratio', 0.0, 0.62)] if not with_agc else [('ratio', -0.21, 0.38), ('ratio_agc', 0.21, 0.38)]
    xl = []
    box = dict(boxstyle='round,pad=0.15', fc='white', ec='none')
    for i, s in enumerate(SUBJECTS):
        rows = sorted([r for r in all_rows if r['subject'] == s], key=lambda r: r['angle1'])
        ok = [r for r in rows if r['status'] == 'ok']
        n = len(rows)
        for key, dx, width in groups:
            x0 = i + dx
            offs = np.linspace(-width / 2 + 0.05, width / 2 - 0.05, n)
            med = float(np.median([r[key] for r in ok]))
            # auto-gain is switched off for CH in the new pipeline: fade its 'after' group
            fade = 0.35 if (with_agc and s == 'CH' and key == 'ratio_agc') else 1.0
            if with_agc:
                col = OLD if key == 'ratio' else NEW
                ax.bar(x0, med - 1.0, bottom=1.0, width=width, color=col, alpha=0.18 * fade, lw=0, zorder=1)
                ax.plot([x0 - width / 2, x0 + width / 2], [med, med], color=col, lw=2.6, alpha=fade, zorder=3)
            else:
                ax.bar(x0, med - 1.0, bottom=1.0, width=width, color=GREY, alpha=0.30, lw=0, zorder=1)
                ax.plot([x0 - width / 2, x0 + width / 2], [med, med], color=TRUTH, lw=2.6, zorder=3)
            for o, r in zip(offs, rows):
                if r['status'] != 'ok':
                    continue
                if with_agc:
                    c = OLD if key == 'ratio' else NEW
                else:
                    c = RED if r['most_toes_down'] else INK2
                if r.get('toes_up'):
                    ax.scatter(x0 + o, r[key], s=50, facecolor='white', edgecolor=c, lw=1.8, alpha=fade, zorder=4)
                else:
                    ax.scatter(x0 + o, r[key], s=46, color=c, lw=0, alpha=fade, zorder=4)
                # EG at its most toes-down angle: the calf average hides opposite changes per muscle
                if s == 'EG' and key == 'ratio' and abs(r['angle1'] + 0.739) < 0.01:
                    pc = r['per_channel']
                    ax.annotate(f'EG {M}0.74 rad: only GM got bigger\n'
                                f"GM {pc['GM']:.1f}{X}, GL {pc['GL']:.2f}{X}, SOL {pc['SOL']:.2f}{X}",
                                xy=(x0 + o, r[key]), xytext=(x0 + o - 0.16, r[key] * 1.03),
                                ha='right', va='center', multialignment='left', fontsize=11, color=INK2,
                                arrowprops=dict(arrowstyle='-', color=INK2, lw=1.0, shrinkB=4), zorder=6)
            if fade < 1.0:
                ax.text(x0, 0.86, 'auto-gain\nnot used\nfor CH', ha='center', va='top', fontsize=10.5,
                        color=NEW, zorder=6)
            if not with_agc:
                ax.text(x0 + width / 2 + 0.03, med, fm(med, '{:.2f}') + X, va='center', ha='left',
                        fontsize=12, weight='bold', color=TRUTH, bbox=box, zorder=5)
        lab = s if s in MAIN else s + ' (extra)'
        if n - len(ok):
            lab += f'\n{len(ok)} of {n} angles'
        xl.append(lab)
    ax.axhline(1.0, color=TRUTH, lw=1.2, ls='--', zorder=2)
    _log_axis(ax, 0.36, 3.0)
    ax.set_xticks(range(len(SUBJECTS)))
    ax.set_xticklabels(xl, fontsize=12)
    ax.set_xlim(-0.6, len(SUBJECTS) - 0.35 + (0.0 if with_agc else 0.12))
    ax.grid(axis='x', visible=False)
    ax.set_ylabel('calf EMG size in session 2\n(same push, session 1 = 1' + X + ')')
    if with_agc:
        ax.set_title('Same push, different EMG size in session 2 ' + chr(0x2014) + ' before and after auto-gain',
                     loc='left', fontsize=15)
        h = [Line2D([], [], ls='', marker='o', ms=7.5, color=OLD, label='before (EMG ' + chr(0x00F7) + ' session-1 max)'),
             Line2D([], [], ls='', marker='o', ms=7.5, color=NEW, label='after auto-gain'),
             Line2D([], [], ls='', marker='o', ms=7.5, mfc='white', mec=INK2, mew=1.8,
                    label='CH pulling toes up (shin muscle)'),
             Line2D([], [], color=INK2, lw=2.6, label='median of the subject')]
        note = ('Each dot = one ankle angle, left to right = most toes-down to most toes-up. '
                'Auto-gain is switched off for CH in the new pipeline (it pushes both ways).\n'
                'The calf average (GM, GL, SOL) can hide muscles changing in opposite directions, e.g. EG ' + M +
                '0.74 rad. Auto-gain rescales each muscle on its own.')
        name = 'emg_size_change_before_after_agc.png'
    else:
        ax.set_title('Same push, different EMG size in session 2', loc='left', fontsize=15)
        h = [Line2D([], [], ls='', marker='o', ms=7.5, color=INK2, label='one ankle angle'),
             Line2D([], [], ls='', marker='o', ms=7.5, color=RED, label='2 most toes-down angles'),
             Line2D([], [], ls='', marker='o', ms=7.5, mfc='white', mec=INK2, mew=1.8,
                    label='CH pulling toes up (shin muscle)'),
             Line2D([], [], color=TRUTH, lw=2.6, label='median of the subject')]
        note = ('Dots left to right = most toes-down to most toes-up angle. Not shown: YES ' + M +
                '0.49 rad (push too small to measure)\nand CH ' + M + '0.31 rad (toes up in session 1, '
                'toes down in session 2). EMG divided by the session-1 maximum in both sessions.\n'
                'The calf average (GM, GL, SOL) can hide muscles changing in opposite directions, '
                'e.g. EG ' + M + '0.74 rad.')
        name = 'emg_size_change.png'
    leg = ax.legend(handles=h, loc='upper left', bbox_to_anchor=(0.0, -0.17), ncol=4, fontsize=11.5,
                    handletextpad=0.4, columnspacing=1.6)
    fig.canvas.draw()
    bb = leg.get_window_extent().transformed(fig.transFigure.inverted())
    fig.text(bb.x0 + 0.005, bb.y0 - 0.015, note, ha='left', va='top', fontsize=11, color=INK2)
    save(fig, name)
    return summary


# ----------------------------------------------------------------------------------------------
# (2) + (3) JM -0.64 rad
# ----------------------------------------------------------------------------------------------
def agc_with_lines(env, prior_lo, prior_hi, rmin_frac=0.35):
    """Exactly ML.research.emg_norm._agc, but also returns the floor and the level that maps to 1."""
    rmin = rmin_frac * np.maximum(prior_hi - prior_lo, 1e-3)
    lo, hi = emg_agc._running_lo_hi(env, prior_lo, prior_hi)
    clean, n_bad = env.copy(), 0
    for k in range(env.shape[1]):
        clean[:, k], n = emg_agc._despike(env[:, k], lo[:, k], np.maximum(hi[:, k] - lo[:, k], rmin[k]))
        n_bad += n
    if n_bad:
        lo, hi = emg_agc._running_lo_hi(clean, prior_lo, prior_hi)
    rng = np.maximum(hi - lo, rmin[None, :])
    out = np.clip((clean - lo) / rng, 0.0, emg_agc.CLIP_HI)
    return out, lo, lo + rng, (hi - lo < rmin[None, :]), n_bad


def jm_data():
    b = load_subject('JM')
    tr = T.subject_truth('JM')
    cols = [c.lower() for c in b['emg_cols']]
    calf = [cols.index('gm'), cols.index('gl'), cols.index('sol')]
    s1recs = [r for r in b['active'] if r['session'] == 'S1']
    base, info_b = normalise('trainmax', b, s1recs, b['active'])
    agc, info_a = normalise('agc', b, s1recs, b['active'])
    pl, ph = np.array(info_a['prior_lo']), np.array(info_a['prior_hi'])
    d = {}
    for sess, ang in (('S1', -0.645), ('S2', -0.644)):
        rec = min([r for r in b['active'] if r['session'] == sess], key=lambda r: abs(r['angle'] - ang))
        out, lo, top, rmin_on, nbad = agc_with_lines(base[rec['idx']], pl, ph)
        assert np.allclose(out, agc[rec['idx']]), 'replica differs from normalise("agc")'
        t = rec['t'] - rec['t'][0]
        d[sess] = dict(rec=rec, t=t, calf=base[rec['idx']][:, calf].mean(axis=1),
                       floor=lo[:, calf].mean(axis=1), top=top[:, calf].mean(axis=1),
                       agc=agc[rec['idx']][:, calf].mean(axis=1), push=-tr[rec['idx']]['truth'],
                       rest=tr[rec['idx']]['rest'], rmin_frac_on=float(rmin_on[:, calf].mean()),
                       n_despiked=int(nbad), angle=float(rec['angle']))
    return d


def fig_agc_timeseries(d, t0=10.0, dur=50.0):
    fig, axs = plt.subplots(3, 2, figsize=(12.5, 7.0), sharex='col',
                            gridspec_kw={'hspace': 0.38, 'wspace': 0.08, 'height_ratios': [1.1, 1, 1]})
    stats = {}
    for j, sess in enumerate(('S1', 'S2')):
        q = d[sess]
        t = q['t']
        w = (t >= t0) & (t <= t0 + dur)
        tt = t[w] - t0
        ax = axs[0, j]
        ax.plot(tt, q['calf'][w], color=OLD, lw=1.1, label='calf EMG (mean of GM, GL, SOL)')
        ax.plot(tt, q['floor'][w], color=GREEN, lw=2.2, ls='--', label='floor = recent relaxed level')
        ax.plot(tt, q['top'][w], color=TRUTH, lw=2.2, ls='--', label='top = recent strong-push level')
        ax = axs[1, j]
        ax.plot(tt, q['agc'][w], color=NEW, lw=1.2)
        ax.axhline(0, color=GREEN, lw=1.6, ls='--')
        ax.axhline(1, color=TRUTH, lw=1.6, ls='--')
        ax = axs[2, j]
        ax.plot(tt, q['push'][w], color=TRUTH, lw=1.2)
        ax.axhline(0, color=GREY, lw=1)
        pm = w & (q['push'] >= 1.5) & (q['push'] <= 2.3)
        rm = w & q['rest']
        stats[sess] = dict(calf_p95=float(np.percentile(q['calf'][w], 95)),
                           agc_p95=float(np.percentile(q['agc'][w], 95)),
                           push_p95=float(np.percentile(q['push'][w], 95)),
                           calf_push_median=float(np.median(q['calf'][pm])),
                           calf_rest_median=float(np.median(q['calf'][rm])),
                           calf_rise=float(np.median(q['calf'][pm]) - np.median(q['calf'][rm])),
                           agc_push_median=float(np.median(q['agc'][pm])),
                           floor_mean=float(q['floor'][w].mean()), top_mean=float(q['top'][w].mean()),
                           angle=q['angle'], rmin_frac_on=q['rmin_frac_on'], n_despiked=q['n_despiked'])
    ylims = [(0, 0.85), (-0.05, 1.85), (-0.25, 3.0)]
    for i in range(3):
        for j in range(2):
            axs[i, j].set_ylim(*ylims[i])
            if j == 1:
                axs[i, j].tick_params(labelleft=False)
    for j in range(2):
        axs[2, j].set_xlabel(f'time (s), seconds {int(t0)}{chr(0x2013)}{int(t0 + dur)} of the trial')
    axs[0, 0].set_title('Session 1 (learned from)', loc='left', fontsize=14)
    axs[0, 1].set_title('Session 2 (never seen)', loc='left', fontsize=14)
    axs[0, 0].set_ylabel('calf EMG\n(' + chr(0x00F7) + ' session-1 max)')
    axs[1, 0].set_ylabel('after\nauto-gain')
    axs[2, 0].set_ylabel('voluntary push,\ntoes down (Nm)')
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower left', bbox_to_anchor=(0.12, 0.935), ncol=3, fontsize=11.5,
               handlelength=1.8, columnspacing=1.5)
    axs[0, 1].text(0.98, 0.97, 'same pushes, signal about half the size', transform=axs[0, 1].transAxes,
                   ha='right', va='top', fontsize=12.5, weight='bold', color=RED)
    axs[1, 1].text(0.98, 0.97, 'after auto-gain: much closer (about 0.8' + X + ')', transform=axs[1, 1].transAxes,
                   ha='right', va='top', fontsize=12.5, weight='bold', color=NEW)
    axs[2, 1].text(0.98, 0.97, 'same pushes in both sessions', transform=axs[2, 1].transAxes,
                   ha='right', va='top', fontsize=12.5, weight='bold', color=TRUTH)
    tf = axs[1, 1].get_yaxis_transform()
    axs[1, 1].text(1.015, 1.0, '1 = recent\nstrong push', transform=tf, ha='left', va='center',
                   fontsize=10.8, color=TRUTH)
    axs[1, 1].text(1.015, 0.0, '0 = recent\nrelaxed level', transform=tf, ha='left', va='center',
                   fontsize=10.8, color=GREEN)
    fig.suptitle(f'Subject JM, ankle at {M}0.64 rad ({M}37{DEG}): what auto-gain does',
                 x=0.125, ha='left', y=1.03, fontsize=15, weight='bold', color=TRUTH)
    save(fig, 'agc_timeseries.png')
    stats['ratio_p95_raw'] = stats['S2']['calf_p95'] / stats['S1']['calf_p95']
    stats['ratio_p95_agc'] = stats['S2']['agc_p95'] / stats['S1']['agc_p95']
    stats['ratio_rise_raw'] = stats['S2']['calf_rise'] / stats['S1']['calf_rise']
    stats['ratio_push_agc'] = stats['S2']['agc_push_median'] / stats['S1']['agc_push_median']
    return stats


def fig_agc_scatter(d):
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.0), sharex=True)
    out, lev, band = {}, {}, {}
    edges = np.linspace(0, 2.2, 9)
    for sess, col in (('S1', OLD), ('S2', NEW)):
        q = d[sess]
        push = q['push']
        sel = np.arange(EDGE, len(push) - EDGE)
        for ax, key, name in ((axs[0], 'calf', 'before'), (axs[1], 'agc', 'after')):
            y = q[key]
            ax.scatter(push[sel][::6], y[sel][::6], s=5, color=col, alpha=0.22, lw=0)
            mids, meds = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (push[sel] >= lo) & (push[sel] < hi)
                if m.sum() > 30:
                    mids.append((lo + hi) / 2)
                    meds.append(np.median(y[sel][m]))
            lab = 'session 1 (learned from)' if sess == 'S1' else 'session 2 (never seen)'
            ax.plot(mids, meds, color=col, lw=3, marker='o', ms=6.5, label=lab)
            mid_m = (push[sel] >= 1.0) & (push[sel] <= 1.8)
            rest_m = push[sel] < 0.2
            lev[(sess, name)] = float(np.median(y[sel][rest_m]))
            band[(sess, name)] = float(np.median(y[sel][mid_m]))
            out[(sess, name)] = band[(sess, name)] - lev[(sess, name)]
    r_before = out[('S2', 'before')] / out[('S1', 'before')]
    r_after = out[('S2', 'after')] / out[('S1', 'after')]
    # relaxed level (median EMG while the push is < 0.2 Nm) and the rise above it for 1.0-1.8 Nm pushes
    for ax, name in ((axs[0], 'before'), (axs[1], 'after')):
        for sess, col, xa in (('S1', OLD, 1.36), ('S2', NEW, 1.50)):
            ax.axhline(lev[(sess, name)], color=GREEN, lw=1.4, ls='--', zorder=2)
            ax.annotate('', xy=(xa, band[(sess, name)]), xytext=(xa, lev[(sess, name)]),
                        arrowprops=dict(arrowstyle='<|-|>', color=col, lw=2.2, mutation_scale=13,
                                        shrinkA=0, shrinkB=0), zorder=5)
        ax.text(2.72, max(lev[('S1', name)], lev[('S2', name)]), 'relaxed level', ha='right', va='bottom',
                fontsize=11, color=GREEN, zorder=6,
                bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.8))
    axs[0].set_title('Before: EMG ' + chr(0x00F7) + ' session-1 maximum', loc='left')
    axs[1].set_title('After auto-gain (uses only past EMG)', loc='left')
    for ax in axs:
        ax.set_xlabel('voluntary push, toes down (Nm)')
        ax.legend(loc='upper left', fontsize=11.5, markerscale=0.9)
        ax.axvspan(1.0, 1.8, color=GREY, alpha=0.10, lw=0, zorder=0)
        ax.set_xlim(-0.25, 2.75)
    axs[0].set_ylim(0, 0.82)
    axs[0].set_ylabel('calf EMG (mean of GM, GL, SOL)')
    axs[1].set_ylabel('calf EMG after auto-gain')
    bx = dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.88)
    s1b, s2b = out[('S1', 'before')], out[('S2', 'before')]
    s1a, s2a = out[('S1', 'after')], out[('S2', 'after')]
    axs[0].text(0.03, 0.75, f'rise above relaxed level (arrows):\nsession 2 {s2b:.3f} vs session 1 {s1b:.3f}'
                f'\n= {r_before:.2f}{X}', transform=axs[0].transAxes,
                ha='left', va='top', fontsize=12, weight='bold', color=RED, zorder=7)
    axs[1].text(0.03, 0.75, f'rise above relaxed level (arrows):\nsession 2 {s2a:.2f} vs session 1 {s1a:.2f}'
                f'\n= {r_after:.2f}{X}', transform=axs[1].transAxes,
                ha='left', va='top', fontsize=12, weight='bold', color=TRUTH, zorder=7)
    fig.text(0.07, -0.05, f'Subject JM, ankle at {M}0.64 rad ({M}37{DEG}). Shaded: pushes of 1.0{chr(0x2013)}1.8 Nm. '
             'Arrows = median EMG for those pushes minus the relaxed level. Lines = median per push size.',
             fontsize=11, color=INK2)
    fig.subplots_adjust(wspace=0.22)
    save(fig, 'agc_scatter.png')
    return dict(before=r_before, after=r_after, rises={f'{k[0]}_{k[1]}': v for k, v in out.items()},
                relaxed_levels={f'{k[0]}_{k[1]}': v for k, v in lev.items()},
                band_medians={f'{k[0]}_{k[1]}': v for k, v in band.items()})


if __name__ == '__main__':
    which = sys.argv[1:] or ['size', 'jm']
    res = {}
    if 'size' in which:
        rows = []
        for s in SUBJECTS:
            rows += size_ratios(s)
        for r in rows:
            if r['status'] == 'ok':
                print(f"{r['subject']:4s} {r['angle1']:+.3f}/{r['angle2']:+.3f} {r['muscle'][:4]} "
                      f"push {r['push_lo']:.2f}-{r['push_hi']:.2f} std {r['push_std1']:.2f}/{r['push_std2']:.2f} "
                      f"ratio {r['ratio']:.2f} agc {r['ratio_agc']:.2f} nrest {r['n_rest1']}/{r['n_rest2']} "
                      f"npush {r['n_push1']}/{r['n_push2']} {'TD' if r['most_toes_down'] else ''} "
                      + ' '.join(f'{k}={v:.2f}' for k, v in r['per_channel'].items()))
            else:
                print(f"{r['subject']:4s} {r['angle1']:+.3f}/{r['angle2']:+.3f} {r['status']}")
        res['summary'] = fig_size_change(rows)
        fig_size_change(rows, with_agc=True)
        for s, v in res['summary'].items():
            print(s, {k: round(x, 3) if isinstance(x, float) else x for k, x in v.items()})
        res['rows'] = rows
    if 'jm' in which:
        d = jm_data()
        res['jm_timeseries'] = fig_agc_timeseries(d)
        res['jm_scatter'] = fig_agc_scatter(d)
        print(json.dumps(res['jm_timeseries'], indent=1))
        print(json.dumps(res['jm_scatter'], indent=1))
    with open(os.path.join(HERE, 'F5_results_' + '_'.join(which) + '.json'), 'w') as f:
        json.dump(res, f, indent=1, default=float)
