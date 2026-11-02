"""F9_inspection: full-trial inspection figures (old vs new) + hardest-angle figures.

Old  = R/<screen|confirm>/<SUBJ>/<prot>/E01_old/fast_seed0.npz
New  = R/final_check/<SUBJ>/<prot>/FINAL/fast_seed0.npz
Truth = shared reference truth (measured torque - in-trial passive baseline B(t)).
Every panel's R² and RMSE ("error") are computed here against the shared truth.
"""
import sys, os, json
sys.path.insert(0, os.environ['WT'])
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ML.research import truth as T

WT = os.environ['WT']
R = os.path.join(WT, 'ML', 'research', 'results')
OUT = os.path.join(WT, 'notes', 'final_pipeline', 'presentation', 'figures_v2')
os.makedirs(OUT, exist_ok=True)
SP = os.path.dirname(os.path.abspath(__file__))

OLD_C, NEW_C, TRUTH_C = '#2a78d6', '#eb6834', '#1b1b1b'
GRID, SEC = '#e6e5e1', '#52514e'
MINUS = '−'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 13,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.edgecolor': '#8a8984', 'axes.labelcolor': '#1b1b1b',
    'xtick.color': SEC, 'ytick.color': SEC,
    'legend.frameon': False, 'axes.unicode_minus': True,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
})

SUBJECTS = ['YES', 'JM', 'EG', 'IES', 'CH', 'HM']
MAIN = {'YES', 'JM', 'EG'}
PROTS = ['S1toS2', 'S2toS1']
PROT_TXT = {'S1toS2': 'learned from session 1 → tested on session 2',
            'S2toS1': 'learned from session 2 → tested on session 1'}


def m(x, fmt):
    """Format number with a Unicode minus sign."""
    s = format(x, fmt)
    return s.replace('-', MINUS)


def fr2(x):
    """R² label: 2 decimals, whole number when |R²| >= 10 (keeps titles short)."""
    return m(x, '.0f' if abs(x) >= 10 else '.2f')


def ferr(x):
    return m(x, '.1f' if abs(x) >= 10 else '.2f')


def load_pred(path, tr):
    """Return {trial idx: (t, pred, truth)} aligned to the shared truth."""
    d = np.load(path)
    out = {}
    for i in np.unique(d['trial']):
        i = int(i)
        sel = d['trial'] == i
        t = d['t'][sel]
        yp = d['y_pred'][sel].astype(float)
        k = np.rint((t - tr[i]['t0']) * 100).astype(int)
        ok = (k >= 0) & (k < len(tr[i]['truth']))
        out[i] = (t[ok], yp[ok], np.asarray(tr[i]['truth'], float)[k[ok]])
    return out


def scores(yt, yp):
    ok = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[ok], yp[ok]
    sse = np.sum((yt - yp) ** 2)
    sst = np.sum((yt - yt.mean()) ** 2)
    return 1.0 - sse / sst, float(np.sqrt(np.mean((yt - yp) ** 2)))


def old_path(s, prot):
    study = 'screen' if s in MAIN else 'confirm'
    return os.path.join(R, study, s, prot, 'E01_old', 'fast_seed0.npz')


def new_path(s, prot):
    return os.path.join(R, 'final_check', s, prot, 'FINAL', 'fast_seed0.npz')


results = {}
hardest = {}

for s in SUBJECTS:
    tr = T.subject_truth(s)
    for prot in PROTS:
        test_sess = 'S2' if prot == 'S1toS2' else 'S1'
        old = load_pred(old_path(s, prot), tr)
        new = load_pred(new_path(s, prot), tr)
        trials = sorted([i for i in tr if tr[i]['session'] == test_sess],
                        key=lambda i: float(tr[i]['angle']))
        assert set(trials) == set(old) == set(new), (s, prot)
        # common y-limits per angle (shared between old and new image).
        # Truth is never clipped. A prediction is clipped only when a SHORT
        # spike (outside its own 1-99 percentiles, i.e. < 1 % of the trial) would otherwise stretch
        # the axis by > 25 %; the clipped spike is then marked with an arrow.
        # A sustained offset (e.g. JM -0.08 rad, old) stays inside the
        # percentiles and is therefore never clipped.
        ylims, clipped, toes_up = {}, {}, {}
        for i in trials:
            yt_all = np.concatenate([old[i][2], new[i][2]])
            full_lo = min(np.nanmin(yt_all), np.nanmin(old[i][1]), np.nanmin(new[i][1]))
            full_hi = max(np.nanmax(yt_all), np.nanmax(old[i][1]), np.nanmax(new[i][1]))
            r_lo = min(np.nanmin(yt_all), np.nanpercentile(old[i][1], 1),
                       np.nanpercentile(new[i][1], 1))
            r_hi = max(np.nanmax(yt_all), np.nanpercentile(old[i][1], 99),
                       np.nanpercentile(new[i][1], 99))
            if (r_hi - r_lo) < 0.75 * (full_hi - full_lo):
                lo, hi = r_lo, r_hi
                pad = 0.07 * (hi - lo)
                lims = (lo - pad, hi + pad)
                clipped[i] = {}
                for tag, pr in (('old', old[i]), ('new', new[i])):
                    tt, yp = pr[0], pr[1]
                    below, above = yp < lims[0], yp > lims[1]
                    if below.any() or above.any():
                        j = int(np.argmin(yp)) if below.any() and (
                            lims[0] - yp.min() >= yp.max() - lims[1]) else int(np.argmax(yp))
                        clipped[i][tag] = (float(tt[j]), float(yp[j]))
                # empty band on the spike side, where the label goes
                band = 0.30 * (lims[1] - lims[0])
                if any(v[1] < lims[0] for v in clipped[i].values()):
                    lims = (lims[0] - band, lims[1])
                else:
                    lims = (lims[0], lims[1] + band)
                print('CLIP', s, prot, f'{float(tr[i]["angle"]):+.4f}', clipped[i],
                      f'full {full_lo:.1f}..{full_hi:.1f} -> {lims[0]:.1f}..{lims[1]:.1f}')
            else:
                lo, hi = full_lo, full_hi
                pad = 0.07 * (hi - lo if hi > lo else 1.0)
                lims = (lo - pad, hi + pad)
            # "pulled toes up": voluntary torque is mostly positive while active
            act = ~np.asarray(tr[i]['rest'], bool)
            tv = np.asarray(tr[i]['truth'], float)[act]
            toes_up[i] = bool(np.nanmedian(tv) > 0)
            if toes_up[i]:
                # head-room above the curves for the "pulled toes up" tag
                lims = (lims[0], lims[1] + 0.30 * (lims[1] - lims[0]))
                print('TOESUP', s, prot, f'{float(tr[i]["angle"]):+.4f}',
                      f'median active truth {np.nanmedian(tv):+.2f} Nm')
            ylims[i] = lims

        for tag, pred, col, name in [('old', old, OLD_C, 'old model'),
                                     ('new', new, NEW_C, 'new model')]:
            fig, axes = plt.subplots(4, 2, figsize=(6.3, 5.7), sharex=True)
            fig.subplots_adjust(left=0.105, right=0.985, top=0.835, bottom=0.085,
                                hspace=0.50, wspace=0.24)
            rows = []
            for n, ax in enumerate(axes.flat):
                if n >= len(trials):
                    ax.axis('off'); continue
                i = trials[n]
                t, yp, yt = pred[i]
                ax.plot(t, yt, color=TRUTH_C, lw=0.8, zorder=2)
                ax.plot(t, yp, color=col, lw=0.8, alpha=0.95, zorder=3)
                r2, rmse = scores(yt, yp)
                ang = float(tr[i]['angle'])
                rows.append({'trial': i, 'angle_rad': round(ang, 4),
                             'angle_label': m(ang, '.2f'),
                             'angle_deg': round(float(np.degrees(ang)), 1),
                             'r2': float(r2), 'rmse': rmse,
                             'toes_up': toes_up[i]})
                ax.set_title(f'{m(ang, ".2f")} rad · R² {fr2(r2)} · '
                             f'error {ferr(rmse)} Nm', fontsize=10, pad=3,
                             color='#1b1b1b')
                ax.set_ylim(*ylims[i])
                if toes_up[i]:
                    ax.text(0.015, 0.97, 'pulled toes up', transform=ax.transAxes,
                            fontsize=9.5, color=SEC, ha='left', va='top',
                            fontstyle='italic', zorder=6)
                if tag in clipped.get(i, {}):
                    ts, ys = clipped[i][tag]
                    lo_, hi_ = ylims[i]
                    rng = hi_ - lo_
                    down = ys < lo_
                    # label sits in the empty band (bottom or top 23 % of the panel)
                    y_txt = lo_ + 0.115 * rng if down else hi_ - 0.115 * rng
                    y_edge = lo_ + 0.01 * rng if down else hi_ - 0.01 * rng
                    left = ts > 45
                    ax.annotate(f'{tag} spike to {m(ys, ".0f")} Nm',
                                xy=(ts, y_edge), xytext=(ts - 2.5 if left else ts + 2.5, y_txt),
                                fontsize=9.5, color=col,
                                ha='right' if left else 'left', va='center',
                                arrowprops=dict(arrowstyle='->', color=col, lw=1.0,
                                                shrinkA=1, shrinkB=0), zorder=7)
                ax.set_xlim(0, 91)
                ax.set_xticks([0, 30, 60, 90])
                ax.tick_params(labelsize=9.5, length=2.5, pad=2)
                ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
                if n % 2 == 0:
                    ax.set_ylabel('Nm', fontsize=10.5, labelpad=2)
                if n >= 6:
                    ax.set_xlabel('time (s)', fontsize=10.5, labelpad=2)
            fig.text(0.015, 0.975, f'{s} · {name}', fontsize=13.5,
                     fontweight='bold', ha='left', va='top')
            fig.text(0.015, 0.925, PROT_TXT[prot], fontsize=11, color=SEC,
                     ha='left', va='top')
            handles = [Line2D([], [], color=TRUTH_C, lw=1.8, label='truth'),
                       Line2D([], [], color=col, lw=1.8, label='prediction')]
            fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.99, 0.99),
                       ncol=1, fontsize=10.5, handlelength=1.6, borderaxespad=0.1,
                       labelspacing=0.3)
            fn = os.path.join(OUT, f'inspect_{s}_{prot}_{tag}.png')
            fig.savefig(fn, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            results[f'{s}_{prot}_{tag}'] = rows

        if prot == 'S1toS2':
            i = trials[0]  # most toes-down angle of the test session
            to, po, yt = old[i]
            tn, pn, ytn = new[i]
            r2o, eo = scores(yt, po)
            r2n, en = scores(ytn, pn)
            ang_raw = float(tr[i]['angle'])
            hardest[s] = {'trial': i, 'angle_raw': ang_raw,
                          'angle_rad': round(ang_raw, 4),
                          'angle_label': f'{m(ang_raw, ".2f")} rad = '
                                         f'{m(np.degrees(ang_raw), ".0f")}°',
                          'angle_deg': round(float(np.degrees(ang_raw)), 1),
                          'toes_up': toes_up[i],
                          'old_r2': float(r2o), 'old_rmse': eo,
                          'new_r2': float(r2n), 'new_rmse': en,
                          'data': (to, yt, po, tn, pn)}


def hardest_fig(subs, fn):
    # Fixed canvas (no tight bbox) so hardest_1 and hardest_2 have identical
    # pixel size and identical plot-area positions.
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 6.4), sharex=True)
    fig.subplots_adjust(left=0.065, right=0.755, top=0.955, bottom=0.095, hspace=0.42)
    for ax, s in zip(axes, subs):
        h = hardest[s]
        to, yt, po, tn, pn = h['data']
        ax.plot(to, yt, color=TRUTH_C, lw=0.8, zorder=2,
                label='truth')
        ax.plot(to, po, color=OLD_C, lw=0.8, alpha=0.9, zorder=3,
                label=f'old  R² {fr2(h["old_r2"])} · error {ferr(h["old_rmse"])} Nm')
        ax.plot(tn, pn, color=NEW_C, lw=0.8, alpha=0.9, zorder=4,
                label=f'new  R² {fr2(h["new_r2"])} · error {ferr(h["new_rmse"])} Nm')
        ax.set_title(f'{s} · most toes-down angle ({h["angle_label"]}) · '
                     f'learned from session 1, tested on session 2',
                     fontsize=13, fontweight='bold', loc='left', pad=5)
        ax.set_ylabel('Nm', fontsize=12)
        ax.set_xlim(0, 91)
        ax.tick_params(labelsize=11)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        leg = ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=11.5,
                        handlelength=1.8, labelspacing=0.45, borderaxespad=0.0)
        for ln in leg.get_lines():
            ln.set_linewidth(2.2)
        if h['toes_up']:
            ax.text(1.022, 0.0, f'{s} pulled toes UP at this angle,\n'
                    'so its voluntary torque is positive', transform=ax.transAxes,
                    fontsize=11, color=SEC, fontstyle='italic', ha='left', va='bottom')
    axes[-1].set_xlabel('time in the trial (s)', fontsize=12.5)
    # check nothing falls outside the fixed canvas
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    W, H = fig.get_size_inches()
    print('BBOX', os.path.basename(fn), [round(v, 3) for v in bb.extents], 'canvas', W, H)
    assert bb.x0 >= 0 and bb.y0 >= 0 and bb.x1 <= W and bb.y1 <= H, 'content cut off'
    fig.savefig(fn, dpi=200, facecolor='white')
    plt.close(fig)


hardest_fig(['YES', 'JM', 'EG'], os.path.join(OUT, 'hardest_1.png'))
hardest_fig(['IES', 'CH', 'HM'], os.path.join(OUT, 'hardest_2.png'))

def rnd(d):
    return {k: (v if k in ('angle_rad', 'angle_raw') else round(v, 3))
            if isinstance(v, float) else v for k, v in d.items()}


summ = {'note': "angle_label = exactly what the figure prints (formatted from the raw "
                "angle); angle_rad keeps 4 decimals. R2/rmse vs shared truth, seed 0, "
                "whole trial, not floored.",
        'per_image': {k: [rnd(r) for r in rows] for k, rows in results.items()},
        'hardest': {k: rnd({kk: vv for kk, vv in v.items() if kk != 'data'})
                    for k, v in hardest.items()}}
with open(os.path.join(SP, 'F9_numbers.json'), 'w', encoding='utf-8') as f:
    json.dump(summ, f, indent=1, ensure_ascii=False)

# printed labels come from the same formatter as the figure titles
for k, rows in results.items():
    print(k, ' '.join(f"{r['angle_label']}:{fr2(r['r2'])}/{ferr(r['rmse'])}" for r in rows))
for k, v in hardest.items():
    print('HARD', k, v['angle_label'],
          f"old {fr2(v['old_r2'])}/{ferr(v['old_rmse'])}", f"new {fr2(v['new_r2'])}/{ferr(v['new_rmse'])}")
