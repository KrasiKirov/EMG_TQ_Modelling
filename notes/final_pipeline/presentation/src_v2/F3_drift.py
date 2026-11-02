"""F3 drift: passive (relaxed) torque level changing during one ~90 s trial.

For every active trial of all 6 subjects:
  * B(t)  = in-trial passive line (shared truth, Gaussian sigma 8 s through
            the medians of the EMG-detected relaxed stretches)
  * change      = mean B over the last 15 s - mean B over the first 15 s (Nm)
  * change_pct  = change / push size, push size = std of the trial's
                  voluntary torque (shared truth)
  * change_raw  = mean of the raw relaxed-stretch medians in the last third
                  of the trial - mean in the first third (no smoothing, no
                  outlier rejection) -> check that the drift is not made by
                  the smoothing.
Figures: figures_v2/drift_examples.png and figures_v2/drift_all.png
Run:  source ML/experiments/env.sh; export WT=...; $PY F3_drift.py
"""
import os
import sys
import json
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.environ['WT'])
from ML.research.data import load_subject
from ML.research import truth as T
from ML.research.passive import PassiveModel
from ML.preprocessing import active_torque as at

FS = 100.0
WIN_S = 15.0
SUBJ = ['YES', 'JM', 'EG', 'IES', 'CH', 'HM']
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'F3_drift_data.pkl')
FIGDIR = os.path.join(os.environ['WT'], 'notes', 'final_pipeline', 'presentation', 'figures_v2')
os.makedirs(FIGDIR, exist_ok=True)

OLD, NEW, TRUTH, GREY = '#2a78d6', '#eb6834', '#1b1b1b', '#8a8984'
RELAX, BAD, GRID, SEC = '#1baf7a', '#c0392b', '#e6e5e1', '#52514e'
MINUS = '−'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 13, 'axes.titlesize': 14,
    'axes.titleweight': 'bold', 'axes.labelsize': 13, 'xtick.labelsize': 11.5,
    'ytick.labelsize': 11.5, 'legend.fontsize': 12, 'legend.frameon': False,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.9, 'axes.axisbelow': True,
    'axes.unicode_minus': True, 'axes.edgecolor': '#6b6a66',
})


def f(x, nd=2, sign=True):
    s = f'{x:+.{nd}f}' if sign else f'{x:.{nd}f}'
    return s.replace('-', MINUS)


# ── data ────────────────────────────────────────────────────────────────────

def compute():
    rows, examples = [], {}
    w = int(WIN_S * FS)
    for s in SUBJ:
        b = load_subject(s)
        tr = T.subject_truth(s)
        has_old = any(len(b['maps']['legacy'][ss]) for ss in ('S1', 'S2'))
        pm = PassiveModel('legacy', b)
        for sess in ('S1', 'S2'):
            segs_all = T.session_rest(b, sess)
            recs = [r for r in b['active'] if r['session'] == sess]
            angs = sorted(float(r['angle']) for r in recs)
            for r in recs:
                i = r['idx']
                d = tr[i]
                B = d['B'].astype(float)
                n = len(B)
                segs = segs_all[i]
                if segs:
                    tc, med, nn = at._seg_levels(np.asarray(r['tq'], float), segs)
                else:
                    tc, med, nn = np.array([]), np.array([]), np.array([])
                first, last = med[tc < n / 3.0], med[tc >= 2 * n / 3.0]
                raw = float(last.mean() - first.mean()) if len(first) and len(last) else np.nan
                push = float(np.std(d['truth']))
                peak = float(np.percentile(np.abs(d['truth']), 95))
                old = float(np.mean(pm.predict(r))) if has_old else np.nan
                old_used = float(np.mean(pm.predict(r)))
                dB = float(B[-w:].mean() - B[:w].mean())
                ang = float(r['angle'])
                keep = (at._reject_outliers(tc, med, nn, T.SIGMA_S * FS)
                        if len(tc) else np.array([], bool))
                tck = tc[keep] if len(tc) else tc
                row = dict(subj=s, idx=int(i), session=sess, angle=ang, dur=n / FS,
                           B_first=float(B[:w].mean()), B_last=float(B[-w:].mean()),
                           change=dB, push=push, peak=peak, pct=100 * dB / push,
                           change_raw=raw, n_rest=len(segs), n_first=len(first), n_last=len(last),
                           # relaxed-moment centres inside the two compared 15 s windows
                           n_wf=int(np.sum(tc < w)), n_wl=int(np.sum(tc >= n - w)),
                           n_wf_kept=int(np.sum(tck < w)), n_wl_kept=int(np.sum(tck >= n - w)),
                           t_first_rest=float(tck.min() / FS) if len(tck) else np.nan,
                           t_last_rest=float(tck.max() / FS) if len(tck) else np.nan,
                           old=old, old_used=old_used,
                           toes_down2=ang <= angs[1], most_td=ang == angs[0])
                rows.append(row)
                if sess == 'S2' and ang == angs[0]:
                    examples[s] = dict(t=np.arange(n) / FS, B=B, tc=tc / FS, med=med, keep=keep,
                                       row=row)
        print(s, 'done', flush=True)
    with open(CACHE, 'wb') as fh:
        pickle.dump({'rows': rows, 'examples': examples}, fh)
    return rows, examples


def load():
    if os.path.exists(CACHE) and '--recompute' not in sys.argv:
        with open(CACHE, 'rb') as fh:
            d = pickle.load(fh)
        if 'row' in next(iter(d['examples'].values())) and 'n_wf_kept' in d['rows'][0]:
            return d['rows'], d['examples']
    return compute()


# ── summary numbers ─────────────────────────────────────────────────────────

def flat_end(r):
    """True when one of the two compared 15 s windows contains no (kept)
    relaxed-moment centre, i.e. the line there is just extended flat."""
    return not (r['n_wf_kept'] >= 1 and r['n_wl_kept'] >= 1)


def summarise(rows):
    ch = np.array([r['change'] for r in rows])
    pct = np.array([r['pct'] for r in rows])
    raw = np.array([r['change_raw'] for r in rows])
    B0 = np.array([r['B_first'] for r in rows])
    ok = np.isfinite(raw)
    rel = np.array([r['n_first'] >= 1 and r['n_last'] >= 1 for r in rows])
    # "shrinks" = relaxed torque moves toward zero (stress relaxation)
    clear = np.abs(B0) >= 0.5
    shrink = (ch * np.sign(B0) < 0)
    out = {
        'n_trials': int(len(rows)),
        'median_change_Nm': float(np.median(ch)),
        'median_abs_change_Nm': float(np.median(np.abs(ch))),
        'p90_abs_change_Nm': float(np.percentile(np.abs(ch), 90)),
        'pct_trials_decreasing': float(100 * np.mean(ch < 0)),
        'pct_trials_size_shrinks_(|B| toward zero, |B|>=0.5 Nm)': float(100 * np.mean(shrink[clear])),
        'n_trials_|B|>=0.5': int(clear.sum()),
        'median_abs_pct_of_push': float(np.median(np.abs(pct))),
        'n_trials_abs_change_>=0.3Nm': int(np.sum(np.abs(ch) >= 0.3)),
        'n_trials_abs_pct_>=10': int(np.sum(np.abs(pct) >= 10)),
        'raw_vs_smooth_pearson_r': float(np.corrcoef(ch[ok], raw[ok])[0, 1]),
        'raw_vs_smooth_median_abs_diff_Nm': float(np.median(np.abs(ch[ok] - raw[ok]))),
        'raw_vs_smooth_same_sign_pct': float(100 * np.mean(np.sign(ch[ok]) == np.sign(raw[ok]))),
        'n_trials_raw_available': int(ok.sum()),
        'n_trials_no_rest_in_first_or_last_third': int((~rel).sum()),
    }
    out['n_trials_no_rest_in_a_15s_window_(all centres)'] = int(sum(
        not (r['n_wf'] >= 1 and r['n_wl'] >= 1) for r in rows))
    out['n_trials_no_rest_in_a_15s_window_(kept centres)'] = int(sum(flat_end(r) for r in rows))
    pos = ch > 0
    out['n_positive_changes'] = int(pos.sum())
    out['n_positive_changes_with_negative_passive'] = int(np.sum(pos & (B0 < 0)))
    out['n_negative_changes_with_positive_passive'] = int(np.sum(~pos & (B0 > 0)))
    out['n_negative_changes'] = int((~pos).sum())
    big = np.abs(ch) >= 0.3
    out['n_|change|>=0.3_moving_toward_zero'] = int(np.sum(big & shrink))
    out['n_|change|>=0.3'] = int(big.sum())
    out['pct_toward_zero_all_trials'] = float(100 * np.mean(shrink))
    td = [r for r in rows if r['toes_down2']]
    oth = [r for r in rows if not r['toes_down2']]
    out['toes_down2_median_abs_pct'] = float(np.median([abs(r['pct']) for r in td]))
    out['others_median_abs_pct'] = float(np.median([abs(r['pct']) for r in oth]))
    out['toes_down2_median_abs_Nm'] = float(np.median([abs(r['change']) for r in td]))
    out['others_median_abs_Nm'] = float(np.median([abs(r['change']) for r in oth]))
    out['toes_down2_pct_decreasing'] = float(100 * np.mean([r['change'] < 0 for r in td]))
    return out


# ── figure 1: examples ──────────────────────────────────────────────────────

def fig_examples(examples):
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0), sharex=True)
    # one common y-span (in Nm) for every panel, so the drops can be compared
    BOT, TOP = 0.30, 0.06          # fraction of the span kept free for notes
    rng_needed = []
    for s in SUBJ:
        e, r = examples[s], examples[s]['row']
        vals = np.concatenate([e['B'], e['med']] + ([[r['old']]] if np.isfinite(r['old']) else []))
        rng_needed.append(np.nanmax(vals) - np.nanmin(vals))
    span = max(rng_needed) / (1 - BOT - TOP)
    span = np.ceil(span * 4) / 4.0
    for ax, s in zip(axes.ravel(), SUBJ):
        e = examples[s]
        r = e['row']
        t, B = e['t'], e['B']
        tend = t[-1]
        for a0, a1 in ((0, WIN_S), (tend - WIN_S, tend)):
            ax.axvspan(a0, a1, color=GREY, alpha=0.13, lw=0, zorder=0)
        ax.plot(t, B, color=NEW, lw=2.6, zorder=3)
        k = e['keep'].astype(bool)
        ax.scatter(e['tc'][k], e['med'][k], s=34, color=RELAX, edgecolor='white', lw=0.6, zorder=4)
        if (~k).any():
            ax.scatter(e['tc'][~k], e['med'][~k], s=34, facecolor='white', edgecolor=RELAX,
                       lw=1.4, zorder=4)
        notes = []
        if np.isfinite(r['old']):
            ax.axhline(r['old'], color=OLD, ls='--', lw=2.0, zorder=2)
        else:
            notes.append((f"no old table (old used {f(r['old_used'], 0, False)} Nm)", OLD))
        if r['n_wf_kept'] == 0:
            notes.append((f"no relaxed moment before {int(r['t_first_rest'])} s:\n"
                          "line held flat at the start", SEC))
        if r['n_wl_kept'] == 0:
            notes.append((f"no relaxed moment after {int(r['t_last_rest'])} s:\n"
                          "line held flat at the end", SEC))
        yy = 0.97
        for txt, col in notes:
            ax.text(0.98, yy, txt, transform=ax.transAxes, ha='right', va='top',
                    fontsize=11, color=col)
            yy -= 0.08 * (txt.count('\n') + 1) + 0.03
        vals = np.concatenate([B, e['med']] + ([[r['old']]] if np.isfinite(r['old']) else []))
        lo = np.nanmin(vals)
        y0 = lo - BOT * span
        ax.set_ylim(y0, y0 + span)
        ax.set_xlim(0, tend)
        deg = np.degrees(r['angle'])
        verb = 'drops' if r['change'] < 0 else 'rises'
        ax.set_title(f"{s}, {f(r['angle'], 2, False)} rad ({f(deg, 0, False)}°)\n"
                     f"{verb} {f(abs(r['change']), 2, False)} Nm in 90 s",
                     fontsize=13.5, loc='left')
        ax.text(0.02, 0.035,
                f"= {f(abs(r['pct']), 0, False)}% of the push size ({f(r['push'], 2, False)} Nm)\n"
                f"= {f(abs(r['change']) / r['peak'] * 100, 0, False)}% of the peak push "
                f"({f(r['peak'], 2, False)} Nm)",
                transform=ax.transAxes, ha='left', va='bottom', fontsize=11.5, color=SEC,
                linespacing=1.25)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, p: f(v, 1, False)))
    for ax in axes[:, 0]:
        ax.set_ylabel('passive torque (Nm)')
    for ax in axes[1]:
        ax.set_xlabel('time in the trial (s)')
    handles = [
        Line2D([], [], marker='o', ls='', color=RELAX, markeredgecolor='white', ms=8,
               label='relaxed moment (median torque)'),
        Line2D([], [], marker='o', ls='', markerfacecolor='white', markeredgecolor=RELAX,
               mew=1.4, ms=8, label='relaxed moment ignored (outlier)'),
        Line2D([], [], color=NEW, lw=2.6, label='new: smooth passive line'),
        Line2D([], [], color=OLD, lw=2.0, ls='--', label='old: table value (separate passive trials)'),
        Patch(facecolor=GREY, alpha=0.25, label='first / last 15 s (compared)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02),
               handletextpad=0.5, columnspacing=1.6)
    fig.text(0.5, -0.045,
             f"Every panel shows the same {f(span, 2, False)} Nm range on the y-axis, so the drops "
             "can be compared by eye.\nPush size = standard deviation of that trial's voluntary "
             "torque;  peak push = 95th percentile of the size of the voluntary torque.",
             ha='center', va='top', fontsize=11.5, color=SEC, linespacing=1.3)
    fig.suptitle('Session 2, most toes-down angle of each person: the relaxed (passive) torque '
                 'is not constant within one trial', fontsize=14.5, fontweight='bold', x=0.5, y=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97), h_pad=1.6, w_pad=1.2)
    p = os.path.join(FIGDIR, 'drift_examples.png')
    fig.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return p


# ── figure 2: all trials ────────────────────────────────────────────────────

CLIP_PCT = -32.0      # right panel: points below this are drawn at the bottom edge


def fig_all(rows):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.4))
    rng = np.random.default_rng(3)
    xpos = {s: k for k, s in enumerate(SUBJ)}
    jit = {(r['subj'], r['idx']): rng.uniform(-0.22, 0.22) for r in rows}
    off_scale = []
    for ax, key in ((a1, 'change'), (a2, 'pct')):
        ax.axhline(0, color=TRUTH, lw=1.1, zorder=1)
        for r in rows:
            x = xpos[r['subj']] + jit[(r['subj'], r['idx'])]
            y = r[key]
            if key == 'pct' and y < CLIP_PCT:
                off_scale.append(r)
                continue
            c = BAD if r['toes_down2'] else GREY
            hollow = flat_end(r)
            mk = 'o' if r['B_first'] >= 0 else '^'
            ax.scatter(x, y, s=(52 if r['toes_down2'] else 40) * (1.25 if mk == '^' else 1),
                       marker=mk, facecolor='white' if hollow else c, edgecolor=c,
                       lw=1.4 if hollow else 0.6, alpha=0.95, zorder=3 if r['toes_down2'] else 2)
        ax.set_xticks(range(len(SUBJ)))
        ax.set_xticklabels(SUBJ)
        ax.set_xlim(-0.6, len(SUBJ) - 0.4)
        ax.grid(axis='x', visible=False)
    a1.set_ylabel('change in the relaxed level from\nthe first to the last 15 s (Nm)')
    a1.set_title('Change in Nm (every trial, both sessions)', loc='left')
    a2.set_ylabel('change as % of the push size\n(SD of that trial\'s voluntary torque)')
    a2.set_title('Same change, as % of the push size', loc='left')
    a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f(v, 1, False)))
    tz = sum(r['change'] * np.sign(r['B_first']) < 0 for r in rows)
    a1.text(0.45, 1.47, f"{tz} of {len(rows)} trials ({100 * tz / len(rows):.0f}%):\n"
            "the relaxed torque\nmoves toward zero", ha='left', va='top', fontsize=11.5,
            color=SEC, linespacing=1.2)
    # right panel: clip the axis, show the YES -0.49 rad points below a break line
    a2.set_ylim(-54, 25)
    a2.set_yticks([-30, -20, -10, 0, 10, 20])
    a2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f(v, 0, False)))
    a2.axhline(-33, color='#b9b8b3', lw=1.0, ls=(0, (5, 4)), zorder=1)
    off_scale = sorted({(r['subj'], r['idx']): r for r in off_scale}.values(),
                       key=lambda r: r['session'])
    x0 = xpos['YES']
    a2.text(x0 - 0.3, -37.0, f"below the scale: YES {f(off_scale[0]['angle'], 2, False)} rad",
            ha='left', va='center', fontsize=11, color=BAD)
    for yrow, r in zip((-42.5, -48.5), off_scale):
        hollow = flat_end(r)
        a2.scatter(x0, yrow, s=52, marker='o' if r['B_first'] >= 0 else '^',
                   facecolor='white' if hollow else BAD, edgecolor=BAD, lw=0.6, zorder=4)
        a2.text(x0 + 0.2, yrow, f"session {r['session'][-1]}: {f(r['pct'], 0)}%  "
                f"(push size only {f(r['push'], 2, False)} Nm)",
                ha='left', va='center', fontsize=11, color=BAD)
    handles = [
        Line2D([], [], marker='o', ls='', color=BAD, ms=8, label='2 most toes-down angles'),
        Line2D([], [], marker='o', ls='', color=GREY, ms=7, label='other angles'),
        Line2D([], [], marker='o', ls='', markerfacecolor='white', markeredgecolor=GREY, mew=1.4,
               ms=7, label='no relaxed moment in the first or last 15 s (line extended flat)'),
        Line2D([], [], marker='o', ls='', markerfacecolor='none', markeredgecolor=TRUTH, mew=1.2,
               ms=7, label=f'relaxed torque above 0: moving toward zero = below the 0 line'),
        Line2D([], [], marker='^', ls='', markerfacecolor='none', markeredgecolor=TRUTH, mew=1.2,
               ms=8, label=f'relaxed torque below 0: moving toward zero = above the 0 line'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.11),
               handletextpad=0.4, columnspacing=2.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=2.5)
    p = os.path.join(FIGDIR, 'drift_all.png')
    fig.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return p


if __name__ == '__main__':
    rows, ex = load()
    summ = summarise(rows)
    p1 = fig_examples(ex)
    p2 = fig_all(rows)
    print(json.dumps(summ, indent=1))
    print('\nLargest |change|:')
    for r in sorted(rows, key=lambda r: -abs(r['change']))[:8]:
        print(f"  {r['subj']:4s} {r['session']} {r['angle']:+.3f} change={r['change']:+.2f} Nm "
              f"raw={r['change_raw']:+.2f} pct={r['pct']:+.0f}% push={r['push']:.2f} "
              f"rest(first,last)=({r['n_first']},{r['n_last']})")
    print('\nLargest |% of push|:')
    for r in sorted(rows, key=lambda r: -abs(r['pct']))[:8]:
        print(f"  {r['subj']:4s} {r['session']} {r['angle']:+.3f} change={r['change']:+.2f} Nm "
              f"pct={r['pct']:+.0f}% push={r['push']:.2f}")
    print('\n2 most toes-down angles (per session):')
    for r in rows:
        if r['toes_down2']:
            print(f"  {r['subj']:4s} {r['session']} {r['angle']:+.3f} ({np.degrees(r['angle']):+.0f} deg) "
                  f"change={r['change']:+.2f} Nm raw={r['change_raw']:+.2f} pct={r['pct']:+.0f}% "
                  f"push={r['push']:.2f} B_first={r['B_first']:+.2f} old={r['old']:+.2f}")
    print('\nExamples (session 2, most toes-down):')
    for s in SUBJ:
        r = ex[s]['row']
        print(f"  {s:4s} {r['angle']:+.3f} change={r['change']:+.2f} pct={r['pct']:+.0f}% "
              f"push={r['push']:.2f} old={r['old']:+.2f} B_first={r['B_first']:+.2f} B_last={r['B_last']:+.2f} "
              f"rest(first,last)=({r['n_first']},{r['n_last']})")
    print(p1, p2)
