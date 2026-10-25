"""F7 'how the model is checked': three_runs.png, phantom.png, answer_key.png.

Reads saved predictions only (no training). Old = screen/confirm E01_old,
new = screen E22_restanchor_final (main subjects, 3 runs), confirm
E22_restanchor_final (IES, HM) and confirm E32_restanchor_noagc_moe (CH).
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

WT = os.environ['WT']
sys.path.insert(0, WT)
from ML.research import truth as T

R = os.path.join(WT, 'ML', 'research', 'results')
OUT = os.path.join(WT, 'notes', 'final_pipeline', 'presentation', 'figures_v2')
os.makedirs(OUT, exist_ok=True)
SCR = os.path.dirname(os.path.abspath(__file__))

OLD, NEW, TRUTH, GREY, GREEN, BAD = '#2a78d6', '#eb6834', '#1b1b1b', '#8a8984', '#1baf7a', '#c0392b'
GRID, INK2 = '#e6e5e1', '#52514e'
OLD3 = ['#2a78d6', '#164a8c', '#8dbbee']   # three shades of blue
NEW3 = ['#eb6834', '#a8411a', '#f6ab86']   # three shades of orange
M = '−'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 13, 'axes.titlesize': 14, 'axes.titleweight': 'bold',
    'axes.labelsize': 13, 'xtick.labelsize': 11.5, 'ytick.labelsize': 11.5, 'legend.fontsize': 11.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': GREY,
    'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True,
    'grid.color': GRID, 'axes.axisbelow': True, 'legend.frameon': False, 'axes.unicode_minus': True})

NUM = {}


def f(x, d=2):
    """number -> string with the Unicode minus."""
    return f'{x:.{d}f}'.replace('-', M)


def deg(rad):
    return f'{np.degrees(rad):.0f}'.replace('-', M)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', p)


def npz(study, s, p, c, k):
    z = np.load(f'{R}/{study}/{s}/{p}/{c}/fast_seed{k}.npz')
    return {kk: z[kk] for kk in z.files}


def tjson(study, s, p, c, k):
    with open(f'{R}/{study}/{s}/{p}/{c}/fast_seed{k}_truth.json') as fh:
        return json.load(fh)


def trial_series(z, tr, i):
    """time in trial (s), shared truth, rest mask, prediction, own target for trial i."""
    m = z['trial'] == i
    t = z['t'][m].astype(np.float64)
    k = np.rint((t - tr[i]['t0']) * 100).astype(int)
    return {'t': t - tr[i]['t0'], 'truth': tr[i]['truth'][k].astype(float), 'rest': tr[i]['rest'][k],
            'B': tr[i]['B'][k].astype(float), 'pred': z['y_pred'][m].astype(float),
            'own': z['y_true'][m].astype(float)}


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def sess_txt(p):
    return 'learned from session 1 → tested on session 2' if p == 'S1toS2' else \
        'learned from session 2 → tested on session 1'


MAIN, EXTRA = ['YES', 'JM', 'EG'], ['IES', 'CH', 'HM']
NEWCFG = {'IES': 'E22_restanchor_final', 'HM': 'E22_restanchor_final', 'CH': 'E32_restanchor_noagc_moe'}


# ----------------------------------------------------------------------------- 1
def fig_three_runs():
    s, p, ang_target = 'JM', 'S2toS1', -0.360
    tr = T.subject_truth(s)
    zo = [npz('screen', s, p, 'E01_old', k) for k in range(3)]
    zn = [npz('screen', s, p, 'E22_restanchor_final', k) for k in range(3)]
    i = min(np.unique(zo[0]['trial']), key=lambda j: abs(tr[j]['angle'] - ang_target))
    ang = tr[i]['angle']
    so = [trial_series(z, tr, i) for z in zo]
    sn = [trial_series(z, tr, i) for z in zn]
    for a in so + sn:
        assert np.allclose(a['t'], so[0]['t'])
    r2o = [r2(a['truth'], a['pred']) for a in so]
    r2n = [r2(a['truth'], a['pred']) for a in sn]
    # 25 s window with the largest run-to-run spread of the old model
    spread = np.stack([a['pred'] for a in so]).std(0)
    w = 2500
    cs = np.convolve(spread, np.ones(w), 'valid') / w
    j0 = int(np.argmax(cs))
    t0w = np.floor(so[0]['t'][j0])
    win = (so[0]['t'] >= t0w) & (so[0]['t'] <= t0w + 25)
    NUM['three_runs_trial'] = {'subject': s, 'direction': p, 'angle_rad': float(ang), 'window_s': [float(t0w), float(t0w + 25)],
                               'old_R2_whole_trial_runs': r2o, 'new_R2_whole_trial_runs': r2n,
                               'old_run_spread_in_window_Nm(mean sd)': float(spread[win].mean()),
                               'new_run_spread_in_window_Nm(mean sd)':
                                   float(np.stack([a['pred'] for a in sn]).std(0)[win].mean())}

    # dot plot data: R^2 per angle per run (floored at -1), mean over 2 most toes-down angles
    rows = []
    for subj in MAIN:
        for prot in ('S1toS2', 'S2toS1'):
            d = {}
            for cfg, key in (('E01_old', 'old'), ('E22_restanchor_final', 'new')):
                vals, alls = [], []
                for k in range(3):
                    pa = tjson('screen', subj, prot, cfg, k)['per_angle']
                    ks = sorted(pa, key=lambda x: float(x[3:]))
                    v = np.maximum([pa[x]['r2'] for x in ks], -1.0)
                    vals.append(float(np.mean(v[:2])))
                    alls.append(float(np.mean(v)))
                d[key] = vals
                d[key + '_all'] = alls
            rows.append((subj, prot, d))
    NUM['three_runs_dots'] = {f'{a}_{b}': d for a, b, d in rows}

    fig = plt.figure(figsize=(12.5, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], height_ratios=[1, 1], hspace=0.5, wspace=0.62,
                          top=0.77, bottom=0.1, left=0.07, right=0.98)
    axo = fig.add_subplot(gs[0, 0])
    axn = fig.add_subplot(gs[1, 0], sharex=axo, sharey=axo)
    axd = fig.add_subplot(gs[:, 1])
    for ax, ser, cols, rr, name in ((axo, so, OLD3, r2o, 'Old model'), (axn, sn, NEW3, r2n, 'New model')):
        ax.plot(ser[0]['t'][win], ser[0]['truth'][win], color=TRUTH, lw=2.2, label='shared answer key', zorder=5)
        for k, (a, c) in enumerate(zip(ser, cols)):
            ax.plot(a['t'][win], a['pred'][win], color=c, lw=1.6, alpha=0.95, zorder=4 + k * 0.1,
                    label=f'run {k + 1}: R\u00b2 {f(rr[k], 2 if abs(rr[k]) < 10 else 1)}')
        ax.set_title(f'{name}: 3 runs', loc='left')
        ax.set_ylabel('voluntary\ntorque (Nm)')
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.04), fontsize=11, handlelength=1.3, borderaxespad=0.2)
    axn.set_xlabel('time in trial (s)')
    axo.set_xlim(t0w, t0w + 25)
    floor = float(so[2]['pred'][win].min())
    NUM['three_runs_trial']['old_run3_floor_in_window_Nm'] = floor
    NUM['three_runs_trial']['old_runs_min_whole_test_session_Nm'] = [float(z['y_pred'].min()) for z in zo]
    axo.set_ylim(floor - 4.6, 1.6)
    axo.set_yticks([0, -5, -10, -15])
    axo.text(t0w + 0.3, floor - 0.9, f'run 3 keeps dropping to its lowest possible output (≈ {f(floor, 1)} Nm)',
             fontsize=11, color=OLD3[1], ha='left', va='top')
    axo.tick_params(labelbottom=True)
    fig.text(0.07, 0.85, f'Left: the test trial where the old runs disagree most: JM, {f(ang)} rad ({deg(ang)}\u00b0), '
             f'{sess_txt(p)}.\n'
             'R\u00b2 next to each run = whole trial (~90 s). The other 47 test trials (3 subjects) disagree less.\n'
             'Right: R\u00b2 averaged over the 2 most toes-down angles of each subject '
             '(an angle with R\u00b2 below \u22121 counts as \u22121).',
             fontsize=11.5, color=INK2, ha='left', va='bottom', linespacing=1.5)

    # dot plot (vertical): x = subject and direction, y = R^2
    xt, xl = [], []
    for g, (subj, prot, d) in enumerate(rows):
        xt.append(g)
        arrow = '1 \u2192 2' if prot == 'S1toS2' else '2 \u2192 1'
        xl.append(subj + '\n' + arrow)
        for key, cols, dx, ys in (('old', OLD3, -0.17, 1.27), ('new', NEW3, 0.17, 1.15)):
            v = d[key]
            axd.plot([g + dx] * 2, [min(v), max(v)], color=cols[0], lw=3.0, alpha=0.25, solid_capstyle='round',
                     zorder=2)
            for vv, c in zip(v, cols):
                axd.scatter(g + dx, vv, s=60, color=c, edgecolor='white', lw=0.8, zorder=3)
            axd.text(g, ys, f(max(v) - min(v)), color=OLD if key == 'old' else NEW,
                     fontsize=11, va='center', ha='center')
    axd.text(-0.55, 1.39, 'spread (best \u2212 worst run):', fontsize=11, color=INK2, ha='left', va='center')
    for g in range(1, len(rows)):
        if g % 2 == 0:
            axd.plot([g - 0.5] * 2, [-1.1, 1.07], color=GREY, lw=0.8, ls=':', zorder=1)
    axd.set_xticks(xt)
    axd.set_xticklabels(xl, fontsize=11.5)
    axd.set_xlim(-0.6, len(rows) - 0.4)
    axd.set_ylim(-1.1, 1.47)
    axd.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
    axd.axhline(0, color=GREY, lw=0.8, zorder=1)
    axd.set_ylabel('R\u00b2 at the 2 most toes-down angles')
    axd.set_xlabel('learned from session \u2192 tested on session', fontsize=12)
    axd.set_title('Each dot = one training run', loc='left')
    axd.grid(axis='x', visible=False)
    axd.legend(handles=[Line2D([], [], ls='', marker='o', ms=8, color=OLD, label='old'),
                        Line2D([], [], ls='', marker='o', ms=8, color=NEW, label='new')],
               loc='lower right', fontsize=11.5, handletextpad=0.2, borderaxespad=0.3, ncol=1)
    fig.suptitle('Same data, 3 training runs (different random start)', x=0.07, ha='left', y=1.015,
                 fontsize=16, fontweight='bold')
    save(fig, 'three_runs.png')


# ----------------------------------------------------------------------------- 2
def fig_phantom():
    s, p, ang_target = 'YES', 'S1toS2', -0.381
    tr = T.subject_truth(s)
    zo, zn = npz('screen', s, p, 'E01_old', 0), npz('screen', s, p, 'E22_restanchor_final', 0)
    i = min(np.unique(zo['trial']), key=lambda j: abs(tr[j]['angle'] - ang_target))
    ang = tr[i]['angle']
    a, b = trial_series(zo, tr, i), trial_series(zn, tr, i)
    w = 2000
    score = np.convolve(a['rest'] * np.abs(a['pred']), np.ones(w), 'valid') / w
    j0 = int(np.argmax(score))
    t0w = np.floor(a['t'][j0])
    win = (a['t'] >= t0w) & (a['t'] <= t0w + 20)
    NUM['phantom_trial'] = {'subject': s, 'direction': p, 'angle_rad': float(ang), 'window_s': [float(t0w), float(t0w + 20)],
                            'old_phantom_whole_trial_Nm': float(np.abs(a['pred'][a['rest']]).mean()),
                            'new_phantom_whole_trial_Nm': float(np.abs(b['pred'][b['rest']]).mean()),
                            'old_phantom_in_window_Nm': float(np.abs(a['pred'][win & a['rest']]).mean()),
                            'new_phantom_in_window_Nm': float(np.abs(b['pred'][win & b['rest']]).mean())}

    # per subject phantom torque
    ph = {}
    for subj in MAIN + EXTRA:
        out = {}
        for key in ('old', 'new'):
            vals = []
            for prot in ('S1toS2', 'S2toS1'):
                if subj in MAIN:
                    cfg = 'E01_old' if key == 'old' else 'E22_restanchor_final'
                    for k in range(3):
                        pa = tjson('screen', subj, prot, cfg, k)['per_angle']
                        vals.append(np.nanmean([v['false_rest'] for v in pa.values()]))
                else:
                    cfg = 'E01_old' if key == 'old' else NEWCFG[subj]
                    pa = tjson('confirm', subj, prot, cfg, 0)['per_angle']
                    vals.append(np.nanmean([v['false_rest'] for v in pa.values()]))
            out[key] = float(np.mean(vals))
        ph[subj] = out
    NUM['phantom_per_subject_Nm'] = ph

    fig = plt.figure(figsize=(12.5, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.25)
    ax = fig.add_subplot(gs[0])
    tt = a['t'][win]
    rest = a['rest'][win]
    # shade relaxed runs
    edges = np.flatnonzero(np.diff(np.r_[0, rest.astype(int), 0]))
    for s0, e0 in zip(edges[::2], edges[1::2]):
        ax.axvspan(tt[s0], tt[e0 - 1], color=GREEN, alpha=0.18, lw=0, zorder=0)
    ax.axhline(0, color=GREY, lw=0.8, zorder=1)
    ax.plot(tt, a['truth'][win], color=TRUTH, lw=2.2, label='shared answer key', zorder=5)
    ax.plot(tt, a['pred'][win], color=OLD, lw=1.8, label='old prediction', zorder=4)
    ax.plot(tt, b['pred'][win], color=NEW, lw=1.8, label='new prediction', zorder=4)
    ax.set_xlim(t0w, t0w + 20)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.set_xlabel('time in trial (s)')
    ax.set_ylabel('voluntary torque (Nm)')
    ax.set_title(f'One test trial: YES, {f(ang)} rad ({deg(ang)}°)\n{sess_txt(p)}', loc='left', fontsize=13)
    h, l = ax.get_legend_handles_labels()
    h.append(Patch(color=GREEN, alpha=0.3, lw=0)); l.append('relaxed moments')
    ax.legend(h, l, loc='lower left', ncol=2, fontsize=11.5, columnspacing=1.2)
    # annotation: arrow to the old prediction in the longest relaxed run of the window
    runs = [(s0, e0) for s0, e0 in zip(edges[::2], edges[1::2])]
    s0, e0 = max(runs, key=lambda r: (r[1] - r[0]) * np.abs(a['pred'][win][r[0]:r[1]]).mean())
    mid = (s0 + e0) // 2
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 0.25 * (ymax - ymin), ymax + 0.42 * (ymax - ymin))
    ax.annotate('phantom torque = torque the model predicts\nwhile the person is relaxed (should be 0)',
                xy=(tt[mid], a['pred'][win][mid]), xytext=(0.03, 0.97), textcoords='axes fraction',
                fontsize=11.5, color=OLD, va='top', ha='left',
                arrowprops=dict(arrowstyle='->', color=OLD, lw=1.4, shrinkB=3))

    axb = fig.add_subplot(gs[1])
    subs = MAIN + EXTRA
    x = np.arange(len(subs))
    bw = 0.38
    vo = [ph[sj]['old'] for sj in subs]
    vn = [ph[sj]['new'] for sj in subs]
    axb.bar(x - bw / 2, vo, bw, color=OLD, label='old')
    axb.bar(x + bw / 2, vn, bw, color=NEW, label='new')
    top = max(vo)
    for xi, v1, v2 in zip(x, vo, vn):
        axb.text(xi - bw / 2, v1 + 0.03 * top, f(v1), ha='center', va='bottom', fontsize=10.5, color=OLD, rotation=90)
        axb.text(xi + bw / 2, v2 + 0.03 * top, f(v2), ha='center', va='bottom', fontsize=10.5, color=NEW, rotation=90)
    axb.set_xticks(x)
    axb.set_xticklabels(subs)
    axb.set_ylim(0, top * 1.42)
    axb.axvline(2.5, color=GREY, lw=0.8, ls=':')
    axb.text(1.0, top * 1.41, 'main subjects\n3 runs ×\nboth directions', ha='center', va='top', fontsize=10.5,
             color=INK2)
    axb.text(4.0, top * 1.41, 'extra subjects\n1 run ×\nboth directions', ha='center', va='top', fontsize=10.5,
             color=INK2)
    axb.set_ylabel('phantom torque (Nm)')
    axb.set_title('Phantom torque per subject', loc='left')
    axb.grid(axis='x', visible=False)
    axb.legend(loc='upper right', bbox_to_anchor=(1.0, 0.8), fontsize=11.5)
    save(fig, 'phantom.png')


# ----------------------------------------------------------------------------- 3
def fig_answer_key():
    s, p, ang_target = 'YES', 'S1toS2', -0.486
    tr = T.subject_truth(s)
    zo = npz('screen', s, p, 'E01_old', 0)
    i = min(np.unique(zo['trial']), key=lambda j: abs(tr[j]['angle'] - ang_target))
    ang = tr[i]['angle']
    a = trial_series(zo, tr, i)
    r2_own, r2_shared = r2(a['own'], a['pred']), r2(a['truth'], a['pred'])
    rmse_own = float(np.sqrt(np.mean((a['own'] - a['pred']) ** 2)))
    rmse_shared = float(np.sqrt(np.mean((a['truth'] - a['pred']) ** 2)))
    shift = float(np.median(a['own'] - a['truth']))
    with open(f'{R}/screen/{s}/{p}/E01_old/fast_seed0.json') as fh:
        saved_own = json.load(fh)['per_angle']
    k_own = min(saved_own, key=lambda x: abs(float(x[3:]) - ang))
    NUM['answer_key'] = {'subject': s, 'direction': p, 'angle_rad': float(ang),
                         'R2_old_vs_own_target': r2_own, 'R2_old_vs_shared_truth': r2_shared,
                         'error_old_vs_own_target_Nm': rmse_own, 'error_old_vs_shared_truth_Nm': rmse_shared,
                         'median_shift_own_target_minus_truth_Nm': shift,
                         'median_shift_old_pred_minus_truth_Nm': float(np.median(a['pred'] - a['truth'])),
                         'saved_json_R2_own': saved_own[k_own]['r2']}

    fig = plt.figure(figsize=(12.5, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.6, 1], wspace=0.3)
    ax = fig.add_subplot(gs[0])
    t = a['t']
    ax.axhline(0, color=GREY, lw=0.8, zorder=1)
    ax.plot(t, a['own'], color=OLD3[2], lw=1.6, label="old model's own target", zorder=3)
    ax.plot(t, a['pred'], color=OLD, lw=1.8, label='old prediction', zorder=4)
    ax.plot(t, a['truth'], color=TRUTH, lw=2.0, label='shared answer key', zorder=5)
    ax.set_xlim(t[0], t[-1] - 0.4)
    ax.set_xlabel('time in trial (s)')
    ax.set_ylabel('voluntary torque (Nm)')
    ax.set_title(f'One test trial: YES, {f(ang)} rad ({deg(ang)}°), {sess_txt(p)}', loc='left', fontsize=13)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 0.05 * (ymax - ymin), ymax + 0.45 * (ymax - ymin))
    ax.legend(loc='upper left', ncol=3, fontsize=11.5, columnspacing=1.2, handlelength=1.6)
    # shift bracket just outside the right edge of the plot (last 10 s of the trial)
    te = t[-1] - 0.4
    m10 = (t > te - 10) & (t < te)
    yt = float(np.median(a['truth'][m10]))
    yo = float(np.median(a['own'][m10]))
    xs = te + 0.015 * (te - t[0])
    ax.annotate('', xy=(xs, yo), xytext=(xs, yt), arrowprops=dict(arrowstyle='<->', color=BAD, lw=1.6),
                annotation_clip=False)
    ax.text(xs + 0.012 * (te - t[0]), (yt + yo) / 2, f'≈{f(abs(shift), 1)} Nm\nshift', color=BAD,
            fontsize=11.5, ha='left', va='center', clip_on=False)

    axs = fig.add_subplot(gs[1])
    axs.axis('off')
    axs.text(0.12, 0.99, 'Same old prediction,\nscored two ways', fontsize=13, fontweight='bold', va='top')
    axs.text(0.12, 0.76, 'against its own target', fontsize=12, color=INK2, va='top')
    axs.text(0.12, 0.69, '(measured − old passive table)', fontsize=11, color=INK2, va='top')
    axs.text(0.12, 0.60, f'R² = {f(r2_own, 1)}', fontsize=17, fontweight='bold', color=OLD, va='top')
    axs.text(0.12, 0.48, f'error {f(rmse_own)} Nm', fontsize=12, color=OLD, va='top')
    axs.text(0.12, 0.35, 'against the shared answer key', fontsize=12, color=INK2, va='top')
    axs.text(0.12, 0.28, '(measured − in-trial passive)', fontsize=11, color=INK2, va='top')
    axs.text(0.12, 0.19, f'R² = {f(r2_shared, 1)}', fontsize=17, fontweight='bold', color=BAD, va='top')
    axs.text(0.12, 0.07, f'error {f(rmse_shared)} Nm', fontsize=12, color=BAD, va='top')
    save(fig, 'answer_key.png')


if __name__ == '__main__':
    fig_three_runs()
    fig_phantom()
    fig_answer_key()
    with open(os.path.join(SCR, 'F7_numbers.json'), 'w') as fh:
        json.dump(NUM, fh, indent=1)
    print(json.dumps(NUM, indent=1))
