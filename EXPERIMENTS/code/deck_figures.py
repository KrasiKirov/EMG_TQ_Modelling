"""
Figures for the presentation / report deck (EXPERIMENTS/presentation/fig/*.png).

    python EXPERIMENTS/code/presentation_metrics.py     # first: scores every run
    python EXPERIMENTS/code/deck_figures.py

Conventions (supervisor feedback, 20 Oct 2026)
  * every R2 shown is floored at 0 for display (R2 < 0 -> 0); medians over training runs;
    whiskers = 5th-95th percentile over runs;
  * MAE (Nm) instead of RMSE; the peak voluntary torque of each position is printed;
  * answer key = passive trials of the tested session (score.py 'passive_trial');
  * 'toes-down' = the 2 most plantarflexed positions of each subject.
Writes also presentation/numbers.json (all numbers quoted in the deck text).
"""
import os
import sys
import glob
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(EXP))
sys.path.insert(0, HERE)
import presentation_metrics as PM                       # noqa: E402

RES = os.path.join(EXP, 'results')
OUT = os.path.join(EXP, 'presentation', 'fig')
os.makedirs(OUT, exist_ok=True)
SUBJ = PM.SUBJ
OLD, NEW, KEY, MUTED, GRID = '#2a78d6', '#eb6834', '#1b1b1b', '#9a988f', '#e6e5e1'
OURS = PM.MODELS['ours']
plt.rcParams.update({'font.family': ['Segoe UI', 'DejaVu Sans'], 'font.size': 13, 'axes.titlesize': 14,
                     'axes.labelsize': 13, 'xtick.labelsize': 11.5, 'ytick.labelsize': 11.5, 'legend.fontsize': 12,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                     'axes.axisbelow': True, 'legend.frameon': False, 'figure.dpi': 100})
N = {}                                                  # numbers quoted in the deck


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=125, bbox_inches='tight')
    plt.close(fig)
    try:                                   # 256-colour palette PNG: about 3x smaller, no visible change for line plots
        from PIL import Image
        im = Image.open(path).convert('RGB')
        im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(path, optimize=True)
    except Exception as e:
        print('png quantise skipped:', e)


def fl(v):
    return np.maximum(v, 0.0)


def near(x, p):
    return (x - p).abs() < 0.005


# ── data ────────────────────────────────────────────────────────────────────
D = pd.read_csv(os.path.join(RES, 'presentation_runs.csv'))
D['r2f'] = fl(D.r2)


def sel(model, d='S1toS2', s=None):
    x = D[(D.model == model) & (D.direction == d)]
    return x if s is None else x[x.subject == s]


def load_run(model, s, d, seed):
    """{position: (t, key, pred)} for one saved run (same alignment as presentation_metrics)"""
    exp, f = PM.run_file(model, s, d, seed)
    return {round(pos, 3): (t, y, p) for pos, t, y, p in PM.trials(exp, s, f)}


def med_run(model, s, d, pos):
    """seed of the typical run (median R2) at a position"""
    x = sel(model, d, s)
    x = x[near(x.position_rad, pos)].sort_values('r2')
    return int(x.iloc[(len(x) - 1) // 2].seed)


# ── 1. box grids: R2 and MAE per position, all subjects ─────────────────────
def box_grid(models, labels, colors, metric, d, name, title, peak=False):
    fig, axs = plt.subplots(2, 3, figsize=(17, 8.6))
    for ax, s in zip(axs.flat, SUBJ):
        x = D[(D.direction == d) & (D.subject == s)]
        pos = sorted(x[x.model == models[0]].position_rad.unique())
        w = 0.8 / len(models)
        tops = []                                            # (x, top of whisker, median) of every box, to clip extreme outliers
        for j, m in enumerate(models):
            xm = x[x.model == m]
            data = [xm[near(xm.position_rad, p)][metric].values for p in pos]
            if metric == 'r2':
                data = [fl(v) for v in data]
            xs = np.arange(len(pos)) + (j - (len(models) - 1) / 2) * w
            ok = [k for k, v in enumerate(data) if len(v)]
            if not ok:
                continue
            bp = ax.boxplot([data[k] for k in ok], positions=xs[ok], widths=w * 0.82, whis=(5, 95), patch_artist=True,
                            showfliers=False, medianprops=dict(color='k', lw=1.6))
            for b in bp['boxes']:
                b.set(facecolor=colors[j], alpha=0.6, edgecolor=colors[j])
            for k in ok:
                ax.scatter(np.full(len(data[k]), xs[k]), data[k], s=8, color=colors[j], edgecolor='k', lw=0.3, zorder=3)
                tops.append((xs[k], float(np.max(data[k])), float(np.median(data[k]))))
        if metric != 'r2' and len(tops) > 2:
            # one extreme box (JM -0.08 rad, the wrong passive label) would squash the axis: clip it and print its median
            srt = sorted(t[1] for t in tops)
            if srt[-1] > 3 * srt[-2]:
                lim = 1.25 * srt[-2]
                ax.set_ylim(0, lim)
                for xx, top, md in tops:
                    if top > lim:
                        ax.annotate(f'↑ {md:.1f} Nm', (xx, lim), xytext=(0, -14), textcoords='offset points', ha='center', fontsize=9.5, color=KEY)
        if peak:
            ymax = ax.get_ylim()[1]
            for i, p in enumerate(pos):
                pk = x[near(x.position_rad, p)].peak_torque.iloc[0]
                ax.text(i, ymax, f'{pk:.0f}', ha='center', va='bottom', fontsize=9.5, color=MUTED)
            ax.text(-0.6, ymax * 1.07, 'peak torque (Nm):', fontsize=9.5, color=MUTED, ha='left')
        ax.set_xticks(range(len(pos)))
        ax.set_xticklabels([f'{p:+.2f}' for p in pos], rotation=40)
        for i in range(2):
            ax.axvspan(i - 0.5, i + 0.5, color='#fbe3d6', alpha=0.45, zorder=0)
        if metric == 'r2':
            ax.set_ylim(-0.03, 1.05)
        ax.set_title(s + (' (no passive trials)' if s == 'IES' else ''), loc='left', weight='bold', pad=22 if peak else 6)
    for ax in axs[:, 0]:
        ax.set_ylabel('R² (below 0 shown as 0)' if metric == 'r2' else 'MAE (Nm, lower is better)')
    for ax in axs[1]:
        ax.set_xlabel('ankle position (rad); shaded = 2 most toes-down')
    handles = [Patch(facecolor=c, alpha=0.6, label=l) for c, l in zip(colors, labels)]
    fig.legend(handles=handles, loc='upper center', ncol=len(models), bbox_to_anchor=(0.5, 1.0), fontsize=13)
    fig.suptitle(title, y=1.045, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, name)


# ── helpers for per-subject summaries ───────────────────────────────────────
def summary(model, d='S1toS2', s=None, toes=None, metric='r2f', seeds=None, subjects=None):
    """Group value = median over the positions of the group of each position's median over runs.
    (First the median over training runs at every position, then the median across positions; for one
    person's 2 toes-down positions that is the average of the two.  Pooling runs x positions into one
    median would land between two positions that behave differently, so it is not used.)"""
    x = sel(model, d, s)
    if subjects is not None:
        x = x[x.subject.isin(subjects)]
    if seeds is not None:
        x = x[x.seed.isin(seeds)]
    if toes is not None:
        x = x[x.toes_down == toes]
    if x.empty:
        return np.nan
    per = x.groupby(['subject', 'position_rad'])[metric].median()
    return float(np.median(per.values))


def per_position(model, s, d='S1toS2', metric='r2f', toes=None, seeds=None):
    """{position: median over runs}"""
    x = sel(model, d, s)
    if seeds is not None:
        x = x[x.seed.isin(seeds)]
    if toes is not None:
        x = x[x.toes_down == toes]
    return {round(p, 3): float(v) for p, v in x.groupby('position_rad')[metric].median().items()}


def bars(ax, groups, series, labels, colors, fmt='{:.2f}', ylim=None, hatch=None):
    w = 0.8 / len(series)
    for j, (vals, lab, col) in enumerate(zip(series, labels, colors)):
        xs = np.arange(len(groups)) + (j - (len(series) - 1) / 2) * w
        b = ax.bar(xs, vals, width=w * 0.92, color=col, label=lab, hatch=(hatch[j] if hatch else None), edgecolor='white')
        for xx, v in zip(xs, vals):
            if np.isfinite(v):
                ax.text(xx, v, fmt.format(v), ha='center', va='bottom', fontsize=9.5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    if ylim:
        ax.set_ylim(*ylim)


def main():
    seeds = {m: sel(m).groupby('subject').seed.nunique().min() for m in ('original', 'ours', 'original_fw')}
    N['seeds'] = {k: int(v) for k, v in seeds.items()}

    # 1. last semester, all subjects
    box_grid(['original'], [f'last semester\'s model (original code, {seeds["original"]} runs)'], [OLD], 'r2', 'S1toS2',
             'baseline_box_r2.png', 'Last semester\'s model · learn session 1 → test session 2 · R² per position')
    # final vs last semester, R2 and MAE
    lab2 = [f'last semester ({seeds["original"]} runs)', f'ours: clean passive + auto-gain + equal weight ({seeds["ours"]} runs)']
    box_grid(['original', 'ours'], lab2, [OLD, NEW], 'r2', 'S1toS2', 'final_box_r2.png',
             'Learn session 1 → test session 2 · box = middle 50 % of runs, line = median, whiskers = 5th–95th percentile, dots = runs')
    box_grid(['original', 'ours'], lab2, [OLD, NEW], 'mae', 'S1toS2', 'final_box_mae.png',
             'Learn session 1 → test session 2 · MAE per position (lower is better)', peak=True)

    # 2. single changes vs last semester's model (same framework, same seeds 0-2)
    singles = [('X05_autogain', 'auto-gain only'), ('X04_equal_angle_weight', 'equal weight only')]
    for exp, lab in singles:
        if exp in set(D.model):
            x = D[D.model.isin(['original_fw', exp]) & (D.direction == 'S1toS2') & (D.seed <= 2)]
            subs = sorted(set(x[x.model == exp].subject), key=SUBJ.index)
            fig, axs = plt.subplots(1, 2, figsize=(16, 5.2))
            S3 = [0, 1, 2]
            for ax, metric, ttl in ((axs[0], 'r2f', 'R² at the 2 toes-down positions (average of the two)'), (axs[1], 'mae', 'MAE (Nm) at the 2 toes-down positions (average of the two)')):
                ref = [summary('original_fw', s=s, toes=1, metric=metric, seeds=S3) for s in subs]
                new = [summary(exp, s=s, toes=1, metric=metric, seeds=S3) for s in subs]
                bars(ax, subs, [ref, new], ['last semester\'s model, re-run in our framework', lab], [OLD, NEW], ylim=(0, 1.1) if metric == 'r2f' else None)
                ax.set_title(ttl, loc='left', weight='bold')
            axs[0].legend(loc='upper left', ncol=2, bbox_to_anchor=(0, -0.08))
            save(fig, f'single_{exp}.png')
            N[exp] = {s: dict(ref_td=summary('original_fw', s=s, toes=1, seeds=S3), new_td=summary(exp, s=s, toes=1, seeds=S3),
                              ref_other=summary('original_fw', s=s, toes=0, seeds=S3), new_other=summary(exp, s=s, toes=0, seeds=S3),
                              ref_mae_td=summary('original_fw', s=s, toes=1, metric='mae', seeds=S3), new_mae_td=summary(exp, s=s, toes=1, metric='mae', seeds=S3),
                              ref_mae_other=summary('original_fw', s=s, toes=0, metric='mae', seeds=S3), new_mae_other=summary(exp, s=s, toes=0, metric='mae', seeds=S3),
                              ref_pos_r2=per_position('original_fw', s, toes=1, seeds=S3), new_pos_r2=per_position(exp, s, toes=1, seeds=S3),
                              ref_pos_mae=per_position('original_fw', s, metric='mae', toes=1, seeds=S3), new_pos_mae=per_position(exp, s, metric='mae', toes=1, seeds=S3),
                              ref_pos_r2raw=per_position('original_fw', s, metric='r2', toes=1, seeds=S3), new_pos_r2raw=per_position(exp, s, metric='r2', toes=1, seeds=S3))
                      for s in subs}

    # 3. clean passive table (JM): per position
    if 'X01_clean_passive' in set(D.model):
        x = D[(D.subject == 'JM') & (D.direction == 'S1toS2') & D.model.isin(['original_fw', 'X01_clean_passive'])]
        pos = sorted(x.position_rad.unique())
        fig, axs = plt.subplots(1, 2, figsize=(16, 5))
        for ax, metric, ttl in ((axs[0], 'r2f', 'JM: median R² per position'), (axs[1], 'mae', 'JM: median MAE (Nm) per position')):
            ser = [[np.median(x[(x.model == m) & near(x.position_rad, p)][metric]) for p in pos] for m in ('original_fw', 'X01_clean_passive')]
            bars(ax, [f'{p:+.2f}' for p in pos], ser, ['original passive table (with the wrong entry)', 'clean passive table'], [OLD, NEW],
                 fmt='{:.2f}', ylim=(0, 1.15) if metric == 'r2f' else None)
            ax.set_title(ttl, loc='left', weight='bold')
            ax.set_xlabel('ankle position (rad)')
        axs[0].legend(loc='upper left')
        if axs[1].get_ylim()[1] > 6:
            axs[1].set_yscale('log')
        save(fig, 'single_X01_clean_passive.png')
        j = lambda m, p: float(np.median(x[(x.model == m) & near(x.position_rad, p)]['r2']))
        jm = lambda m, p: float(np.median(x[(x.model == m) & near(x.position_rad, p)]['mae']))
        bad = min(pos, key=lambda p: abs(p + 0.078))
        N['X01'] = dict(pos=bad, r2_before=j('original_fw', bad), r2_after=j('X01_clean_passive', bad),
                        mae_before=jm('original_fw', bad), mae_after=jm('X01_clean_passive', bad))

    # 4. ladder: last semester -> + clean table -> + auto-gain -> + equal weight
    # like for like: every step in our framework, the same 3 seeds (0-2) for every bar
    steps = [('last semester (re-run, our framework)', {s: 'original_fw' for s in SUBJ}),
             ('+ clean passive table', {s: ('X01_clean_passive' if s == 'JM' else 'original_fw') for s in SUBJ}),
             ('+ EMG auto-gain', {s: ('S2_cleanpassive+autogain' if s == 'JM' else 'X05_autogain') for s in SUBJ}),
             ('+ equal weight = ours', {s: 'ours' for s in SUBJ})]
    cols = [OLD, '#b9c9de', '#8d9bb0', NEW]
    lad = {}
    for metric, toes, fname, ttl, yl in (('r2f', 1, 'ladder_r2_toesdown.png', 'R² at the 2 toes-down positions (average of the two)', (0, 1.12)),
                                         ('mae', 1, 'ladder_mae_toesdown.png', 'MAE (Nm) at the 2 toes-down positions (average of the two)', None),
                                         ('r2f', 0, 'ladder_r2_other.png', 'R² at the other 6 positions (median of the positions)', (0, 1.12)),
                                         ('mae', 0, 'ladder_mae_other.png', 'MAE (Nm) at the other 6 positions (median of the positions)', None)):
        ser = []
        for lab, mp in steps:
            ser.append([summary(mp[s], s=s, toes=toes, metric=metric, seeds=[0, 1, 2]) if mp[s] in set(D.model) and not sel(mp[s], s=s).empty else np.nan for s in SUBJ])
        lad[fname] = ser
        fig, ax = plt.subplots(figsize=(16, 5))
        bars(ax, SUBJ, ser, [l for l, _ in steps], cols, ylim=yl)
        ax.set_title(ttl + ' · learn session 1 → test session 2 · same 3 seeds', loc='left', weight='bold')
        ax.legend(loc='upper left', ncol=4, bbox_to_anchor=(0, -0.08))
        save(fig, fname)
    N['ladder'] = {k: [[None if not np.isfinite(v) else round(v, 3) for v in row] for row in ser] for k, ser in lad.items()}

    # 5. other single changes (screening on YES + EG; every run that exists, reference = same seeds)
    other = [('X02a_dropout_0.0', 'dropout 0 (no dropout)'), ('X02b_dropout_0.1', 'dropout 0.1'), ('X02c_dropout_0.5', 'dropout 0.5'),
             ('X03_lr_0.001', 'learning rate 0.001'), ('X06_window_0.4s', 'window 0.4 s'), ('T01_passive_value_input', 'passive value as 6th input'),
             ('X05_autogain', 'auto-gain'), ('X04_equal_angle_weight', 'equal weight')]
    rows = []
    for exp, lab in other:
        if exp not in set(D.model):
            continue
        r = dict(exp=exp, label=lab)
        for dname in ('S1toS2', 'S2toS1'):
            sfx = '' if dname == 'S1toS2' else '_rev'
            sd = sorted(set(sel(exp, dname, 'YES').seed) & set(sel(exp, dname, 'EG').seed))
            for toes in (1, 0):
                if not sd:
                    r[f'd_r2_{toes}{sfx}'] = r[f'd_mae_{toes}{sfx}'] = np.nan
                    continue
                # deck convention: median over runs at each position, then over the 4 (or 12) positions of YES + EG; same seeds
                g = lambda m, metric: summary(m, dname, None, toes, metric, seeds=sd, subjects=['YES', 'EG'])
                r[f'd_r2_{toes}{sfx}'] = g(exp, 'r2f') - g('original_fw', 'r2f')
                r[f'd_mae_{toes}{sfx}'] = 100 * (g(exp, 'mae') / g('original_fw', 'mae') - 1)
            r[f'runs{sfx}'] = len(sd)
        rows.append(r)
    O = pd.DataFrame(rows)
    if not O.empty:
        fig, axs = plt.subplots(1, 2, figsize=(16, 5.4), sharey=True)
        y = np.arange(len(O))
        for ax, col, ttl in ((axs[0], 'd_mae_1', 'change in MAE at the 2 toes-down positions (%)'), (axs[1], 'd_mae_0', 'change in MAE at the other 6 positions (%)')):
            v = O[col].values
            ax.barh(y, v, color=[NEW if vv < 0 else OLD for vv in v], height=0.6)
            for yy, vv in zip(y, v):
                ax.text(vv, yy, f' {vv:+.0f} % ', va='center', ha='left' if vv >= 0 else 'right', fontsize=11)
            ax.axvline(0, color='k', lw=0.8)
            ax.set_title(ttl, loc='left', weight='bold')
            ax.set_xlim(min(-60, v.min() * 1.4), max(60, v.max() * 1.4))
        axs[0].set_yticks(y)
        axs[0].set_yticklabels(O.label)
        axs[0].invert_yaxis()
        fig.text(0.01, -0.03, 'Each change alone vs last semester\'s model re-run in our framework (same seeds), YES and EG, learn session 1 → test session 2; '
                 'MAE = median over runs at each position, then over the positions. 2 runs (dropout, 0.4 s window, passive input) or 3 runs (the others). '
                 'Orange = error went down (better), blue = error went up (worse).', fontsize=11, wrap=True)
        save(fig, 'screening_changes.png')
        N['screening'] = O.round(3).to_dict('records')

    # 6. window 0.2 s (ours) vs 0.5 s (ours + window)
    if 'D02_D01+window_0.5s' in set(D.model):
        W = 'D02_D01+window_0.5s'
        subs = [s for s in SUBJ if not sel(W, s=s).empty]

        def msum(m, s, toes, metric='r2f'):
            # matched seeds: the 0.2 s model is restricted to the seeds the 0.5 s model has for that person
            return summary(m, s=s, toes=toes, metric=metric, seeds=sorted(set(sel(W, s=s).seed)))
        fig, axs = plt.subplots(1, 2, figsize=(16, 5))
        for ax, (metric, toes, ttl) in zip(axs, (('mae', 1, 'MAE (Nm), 2 toes-down positions'), ('mae', 0, 'MAE (Nm), other 6 positions'))):
            ser = [[msum(m, s, toes, metric) for s in subs] for m in ('ours', W)]
            bars(ax, subs, ser, ['ours, 0.2 s window (20 samples)', 'ours, 0.5 s window (50 samples)'], [NEW, '#7a5195'])
            ax.set_title(ttl + ' · same seeds', loc='left', weight='bold')
        axs[0].legend(loc='upper left', bbox_to_anchor=(0, -0.08), ncol=2)
        save(fig, 'window_0.5s.png')
        N['window'] = {s: dict(td_02=msum('ours', s, 1), td_05=msum(W, s, 1), ot_02=msum('ours', s, 0), ot_05=msum(W, s, 0),
                               mae_td_02=msum('ours', s, 1, 'mae'), mae_td_05=msum(W, s, 1, 'mae'),
                               mae_ot_02=msum('ours', s, 0, 'mae'), mae_ot_05=msum(W, s, 0, 'mae'))
                       for s in subs}
        nr = [sel(W, s=s).seed.nunique() for s in subs]
        N['window_runs'] = f'{min(nr)}' if min(nr) == max(nr) else f'{min(nr)}–{max(nr)}'
        # consistency: how many 0.5 s runs lie below the best MATCHED 0.2 s run (other-position MAE, run-level medians over positions)
        cons = {}
        for s in subs:
            ws = sorted(set(sel(W, s=s).seed))

            def run_level(m):
                x = sel(m, s=s)
                x = x[(x.toes_down == 0) & x.seed.isin(ws)]
                return x.groupby('seed').mae.median()
            r02, r05 = run_level('ours'), run_level(W)
            cons[s] = dict(below=int((r05 < r02.min()).sum()), n=int(len(r05)))
        N['window_consistency'] = cons
        # all-people summary under the deck convention (same seeds: the 0.5 s runs' seeds)
        wseeds = sorted(set(sel(W).seed))
        N['window_all'] = {k: {m: summary(m, toes=t, metric=met, seeds=wseeds) for m in ('ours', W)}
                           for k, t, met in (('td_r2', 1, 'r2f'), ('td_mae', 1, 'mae'), ('ot_r2', 0, 'r2f'), ('ot_mae', 0, 'mae'))}
        po2, po5 = (sel(m)[sel(m).seed.isin(wseeds)].groupby(['subject', 'position_rad', 'toes_down']).mae.median() for m in ('ours', W))
        rr = (po5 / po2).dropna()
        N['window_pos_better'] = {t: [int((rr.xs(t, level='toes_down') < 1).sum()), int(len(rr.xs(t, level='toes_down')))] for t in (0, 1)}
        ratios = []
        for s in subs:
            exp02 = PM.MODELS['ours'][s]
            for sd in wseeds:
                f2, f5 = (os.path.join(RES, e, s, 'S1toS2', f'seed{sd}.json') for e in (exp02, W))
                if os.path.exists(f2) and os.path.exists(f5):
                    a, b = json.load(open(f2)), json.load(open(f5))
                    ratios.append((b['train_s'] / a['train_s'], (b['train_s'] / b['train_info']['epochs']) / (a['train_s'] / a['train_info']['epochs'])))
        N['window_slowdown'] = dict(per_run=float(np.median([r[0] for r in ratios])), per_epoch=float(np.median([r[1] for r in ratios]))) if ratios else {}
        N['window_yes_pos'] = {'02': per_position('ours', 'YES', toes=1, seeds=sorted(set(sel(W, s='YES').seed))),
                               '05': per_position(W, 'YES', toes=1)}

    # 7. directions: last semester re-run in our framework vs ours, the same 3 seeds (0-2) in both directions
    S3 = [0, 1, 2]
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    for ax, toes, ttl in ((axs[0], 1, 'R², 2 toes-down positions (average of the two)'), (axs[1], 0, 'R², other 6 positions (median)')):
        ser = [[summary(m, d, s, toes, seeds=S3) for s in SUBJ] for m, d in (('original_fw', 'S1toS2'), ('ours', 'S1toS2'), ('original_fw', 'S2toS1'), ('ours', 'S2toS1'))]
        bars(ax, SUBJ, ser, ['last semester (re-run), learn 1 → test 2', 'ours, learn 1 → test 2', 'last semester (re-run), learn 2 → test 1', 'ours, learn 2 → test 1'],
             [OLD, NEW, '#9cc0ea', '#f5b48f'], ylim=(0, 1.15), fmt='{:.2f}')
        ax.set_title(ttl + ' · same 3 seeds', loc='left', weight='bold')
    axs[0].legend(loc='upper left', ncol=2, bbox_to_anchor=(0, -0.08))
    save(fig, 'directions.png')
    N['directions'] = {d: {m: dict(td=summary(m, d, None, 1, seeds=S3), ot=summary(m, d, None, 0, seeds=S3), mae_td=summary(m, d, None, 1, 'mae', seeds=S3),
                                   mae_ot=summary(m, d, None, 0, 'mae', seeds=S3)) for m in ('original_fw', 'ours')} for d in ('S1toS2', 'S2toS1')}
    N['directions_per_subject'] = {d: {s: {m: dict(td=summary(m, d, s, 1, seeds=S3), mae_td=summary(m, d, s, 1, 'mae', seeds=S3))
                                           for m in ('original_fw', 'ours')} for s in SUBJ} for d in ('S1toS2', 'S2toS1')}
    N['directions_untouched'] = {d: {m: dict(td=summary(m, d, None, 1, seeds=S3, subjects=['IES', 'HM']), mae_td=summary(m, d, None, 1, 'mae', seeds=S3, subjects=['IES', 'HM']))
                                     for m in ('original_fw', 'ours')} for d in ('S1toS2', 'S2toS1')}
    # YES at its most toes-down position: raw median R2, MAE, bias (learn 1->2: 10 vs 10 runs; learn 2->1: re-run vs ours, 3 vs 3)
    def yes49(m, d, col):
        pp = per_position(m, 'YES', d, col, 1, None if d == 'S1toS2' else S3)
        return pp[min(pp)]
    N['yes49'] = {d: {m: {c: yes49(m, d, c) for c in ('r2', 'mae', 'bias')} for m in (('original', 'ours') if d == 'S1toS2' else ('original_fw', 'ours'))}
                  for d in ('S1toS2', 'S2toS1')}
    # reverse direction, per toes-down position (re-run vs ours, same 3 seeds), for the per-person wording
    N['rev_pos'] = {s: {m: {c: {f'{p:+.3f}': v for p, v in per_position(m, s, 'S2toS1', c, 1, S3).items()} for c in ('r2f', 'r2', 'mae')}
                        for m in ('original_fw', 'ours')} for s in SUBJ}

    # 8. waveforms per subject: all positions (typical run), then best & worst position
    N['subjects'] = {}
    for s in SUBJ:
        pos = sorted(sel('ours', s=s).position_rad.unique())
        runs_new = {p: load_run('ours', s, 'S1toS2', med_run('ours', s, 'S1toS2', p)) for p in pos}
        runs_old = {p: load_run('original', s, 'S1toS2', med_run('original', s, 'S1toS2', p)) for p in pos}
        fig, axs = plt.subplots(2, 4, figsize=(18, 7.4), sharex=True)
        stats = {}
        for ax, p in zip(axs.flat, pos):
            t, y, pn = runs_new[p][p]
            to, yo, po = runs_old[p].get(p, (None, None, None))
            ax.plot(t, y, color=KEY, lw=0.9, label='measured')
            if po is not None:
                ax.plot(to, po, color=OLD, lw=0.75, alpha=0.85, label='last semester')
            ax.plot(t, pn, color=NEW, lw=0.8, label='ours')
            xn, xo = sel('ours', s=s), sel('original', s=s)
            rn, ro = np.median(xn[near(xn.position_rad, p)].r2), np.median(xo[near(xo.position_rad, p)].r2)
            mn, mo = np.median(xn[near(xn.position_rad, p)].mae), np.median(xo[near(xo.position_rad, p)].mae)
            stats[p] = dict(r2_new=rn, r2_old=ro, mae_new=mn, mae_old=mo, peak=float(np.max(np.abs(y))))
            ax.set_title(f'{p:+.2f} rad{"  (toes-down)" if p in pos[:2] else ""}\nR² {max(ro, 0):.2f} → {max(rn, 0):.2f} · MAE {mo:.2f} → {mn:.2f} Nm',
                         loc='left', fontsize=11.5)
        for ax in axs[:, 0]:
            ax.set_ylabel('voluntary torque (Nm)')
        for ax in axs[1]:
            ax.set_xlabel('time in trial (s)')
        axs[0, 0].legend(loc='lower left', ncol=3, bbox_to_anchor=(0, 1.22), fontsize=12)
        fig.tight_layout()
        save(fig, f'wave_all_{s}.png')
        best, worst = max(pos, key=lambda p: stats[p]['r2_new']), min(pos, key=lambda p: stats[p]['r2_new'])
        fig, axs = plt.subplots(2, 1, figsize=(17, 7.6))
        for ax, p, tag in ((axs[0], best, 'BEST position'), (axs[1], worst, 'WORST position')):
            t, y, pn = runs_new[p][p]
            to, yo, po = runs_old[p].get(p, (None, None, None))
            ax.plot(t, y, color=KEY, lw=1.0, label='measured voluntary torque')
            if po is not None:
                ax.plot(to, po, color=OLD, lw=0.85, alpha=0.85, label='last semester (typical run)')
            ax.plot(t, pn, color=NEW, lw=0.9, label='ours (typical run)')
            st = stats[p]
            ax.set_title(f'{tag}: {p:+.2f} rad · peak {st["peak"]:.1f} Nm · R² {max(st["r2_old"], 0):.2f} → {max(st["r2_new"], 0):.2f} · '
                         f'MAE {st["mae_old"]:.2f} → {st["mae_new"]:.2f} Nm', loc='left', weight='bold')
            ax.set_ylabel('Nm')
        axs[0].legend(loc='lower left', ncol=3, bbox_to_anchor=(0, 1.13))
        axs[1].set_xlabel('time in trial (s)')
        fig.tight_layout()
        save(fig, f'wave_bestworst_{s}.png')
        for p in pos:                                      # bias and key spread per position (for the worst-position wording)
            xn, xo = sel('ours', s=s), sel('original', s=s)
            stats[p].update(bias_new=float(np.median(xn[near(xn.position_rad, p)].bias)), bias_old=float(np.median(xo[near(xo.position_rad, p)].bias)),
                            key_std=float(xn[near(xn.position_rad, p)].key_std.iloc[0]))
        N['subjects'][s] = dict(best=best, worst=worst, positions={f'{p:+.3f}': v for p, v in stats.items()},
                                td_positions=[f'{p:+.3f}' for p in pos[:2]],
                                td_r2_old=summary('original', s=s, toes=1), td_r2_new=summary('ours', s=s, toes=1),
                                ot_r2_old=summary('original', s=s, toes=0), ot_r2_new=summary('ours', s=s, toes=0),
                                td_mae_old=summary('original', s=s, toes=1, metric='mae'), td_mae_new=summary('ours', s=s, toes=1, metric='mae'),
                                ot_mae_old=summary('original', s=s, toes=0, metric='mae'), ot_mae_new=summary('ours', s=s, toes=0, metric='mae'))

    # 9. run-to-run spread: best and worst run at the same position (most toes-down)
    for m, lab, col in (('original', 'last semester', OLD), ('ours', 'ours', NEW)):
        fig, axs = plt.subplots(3, 2, figsize=(18, 9.6))
        for ax, s in zip(axs.T.flat, SUBJ):
            x = sel(m, s=s)
            p = sorted(x.position_rad.unique())[0]
            x = x[near(x.position_rad, p)]
            b, w = x.loc[x.r2.idxmax()], x.loc[x.r2.idxmin()]
            rb, rw = load_run(m, s, 'S1toS2', int(b.seed)), load_run(m, s, 'S1toS2', int(w.seed))
            t, y, pb = rb[p]
            ax.plot(t, y, color=KEY, lw=0.9, label='measured')
            ax.plot(t, pb, color='#1f9d55', lw=0.8, label='best run')
            ax.plot(rw[p][0], rw[p][2], color='#c0392b', lw=0.8, label='worst run')
            ax.set_title(f'{s} · {p:+.2f} rad · {len(x)} runs · best R² {max(b.r2, 0):.2f} (MAE {b.mae:.2f}) · worst R² {max(w.r2, 0):.2f} (MAE {w.mae:.2f})',
                         loc='left', fontsize=11.5, weight='bold')
        axs[0, 0].legend(loc='lower left', ncol=3, bbox_to_anchor=(0, 1.15))
        fig.tight_layout()
        save(fig, f'seed_bestworst_{m}.png')

    # 10. bias / variance split over runs + ensemble; demeaned scores
    rows = []
    for m in ('original', 'ours'):
        for s in SUBJ:
            sds = sorted(sel(m, s=s).seed.unique())
            R = {sd: load_run(m, s, 'S1toS2', sd) for sd in sds}
            for p in sorted(R[sds[0]]):
                t, y, _ = R[sds[0]][p]
                P = np.vstack([R[sd][p][2] for sd in sds])
                pbar = P.mean(0)
                mse_runs = np.mean((P - y) ** 2)
                bias2 = np.mean((pbar - y) ** 2)
                var = np.mean(P.var(0))
                ss = np.sum((y - y.mean()) ** 2)
                rows.append(dict(model=m, subject=s, position_rad=p, runs=len(sds), mse_runs=mse_runs, systematic=bias2, run_to_run=var,
                                 r2_ensemble=1 - np.sum((pbar - y) ** 2) / ss, mae_ensemble=np.mean(np.abs(pbar - y)),
                                 offset2=(pbar - y).mean() ** 2))
    BV = pd.DataFrame(rows)
    BV['toes_down'] = 0
    for (m, s), x in BV.groupby(['model', 'subject']):
        BV.loc[x.index[x.position_rad.isin(sorted(x.position_rad)[:2])], 'toes_down'] = 1
    BV.to_csv(os.path.join(RES, 'bias_variance_runs.csv'), index=False)
    fig, axs = plt.subplots(1, 2, figsize=(17, 5.6))
    for ax, toes, ttl in ((axs[0], 1, '2 toes-down positions'), (axs[1], 0, 'other 6 positions')):
        g = BV[BV.toes_down == toes].groupby(['model', 'subject'])[['mse_runs', 'systematic', 'run_to_run', 'offset2']].mean()
        w = 0.38
        for j, (m, col) in enumerate((('original', OLD), ('ours', NEW))):
            xs = np.arange(len(SUBJ)) + (j - 0.5) * w
            tot = np.array([g.loc[(m, s), 'mse_runs'] for s in SUBJ])
            off = 100 * np.array([g.loc[(m, s), 'offset2'] for s in SUBJ]) / tot
            sy = 100 * np.array([g.loc[(m, s), 'systematic'] for s in SUBJ]) / tot
            va = 100 * np.array([g.loc[(m, s), 'run_to_run'] for s in SUBJ]) / tot
            ax.bar(xs, off, width=w * 0.92, color=col, edgecolor='white')
            ax.bar(xs, sy - off, bottom=off, width=w * 0.92, color=col, alpha=0.45, edgecolor='white')
            ax.bar(xs, va, bottom=sy, width=w * 0.92, color='white', edgecolor=col, hatch='///', lw=0.8)
            for xx, t in zip(xs, tot):
                ax.text(xx, 101.5, f'{t:.2f}', ha='center', va='bottom', fontsize=9.5, color=col)
        ax.set_xticks(range(len(SUBJ)))
        ax.set_xticklabels(SUBJ)
        ax.set_ylim(0, 112)
        ax.set_ylabel('share of the error (%)')
        ax.set_title(f'Where the error comes from · {ttl}', loc='left', weight='bold', pad=22)
        ax.text(0, 1.015, 'number on top = mean squared error of one run (Nm²)', transform=ax.transAxes, fontsize=10, color=MUTED)
    handles = [Patch(facecolor='#777', label='offset (whole curve shifted up/down)'),
               Patch(facecolor='#777', alpha=0.45, label='other systematic error (same in every run)'),
               Patch(facecolor='white', edgecolor='#777', hatch='///', label='run-to-run spread (random seed)'),
               Patch(facecolor=OLD, label='last semester'), Patch(facecolor=NEW, label='ours')]
    fig.legend(handles=handles, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.08), fontsize=11.5)
    fig.tight_layout()
    save(fig, 'bias_variance.png')
    agg = BV.groupby(['model', 'toes_down'])[['mse_runs', 'systematic', 'run_to_run', 'offset2']].sum()
    persub = BV.groupby(['model', 'toes_down', 'subject']).mse_runs.sum()
    N['biasvar'] = {f'{m}_{t}': dict(var_share=float(agg.loc[(m, t), 'run_to_run'] / agg.loc[(m, t), 'mse_runs']),
                                     offset_share=float(agg.loc[(m, t), 'offset2'] / agg.loc[(m, t), 'mse_runs']),
                                     systematic_share=float(agg.loc[(m, t), 'systematic'] / agg.loc[(m, t), 'mse_runs']),
                                     top=[[s, float(v / agg.loc[(m, t), 'mse_runs'])] for s, v in persub.loc[(m, t)].sort_values(ascending=False).head(2).items()])
                    for m in ('original', 'ours') for t in (0, 1)}
    N['ensemble'] = {m: dict(td_r2_runs=summary(m, toes=1), td_r2_ens=float(np.median(fl(BV[(BV.model == m) & (BV.toes_down == 1)].r2_ensemble))),
                             td_mae_runs=summary(m, toes=1, metric='mae'), td_mae_ens=float(np.median(BV[(BV.model == m) & (BV.toes_down == 1)].mae_ensemble)),
                             ot_r2_ens=float(np.median(fl(BV[(BV.model == m) & (BV.toes_down == 0)].r2_ensemble))))
                     for m in ('original', 'ours')}

    # demeaned
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    for ax, m, lab in ((axs[0], 'original', 'last semester'), (axs[1], 'ours', 'ours')):
        raw = [summary(m, s=s, toes=1) for s in SUBJ]
        dem = [summary(m, s=s, toes=1, metric='r2_demeaned') for s in SUBJ]
        bars(ax, SUBJ, [raw, list(fl(np.array(dem)))], ['R² as measured', 'R² after removing each trial\'s mean (demeaned)'],
             [OLD if m == 'original' else NEW, '#5b5b5b'], ylim=(0, 1.15))
        ax.set_title(f'{lab}: R² at the 2 toes-down positions (average of the two)', loc='left', weight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(0, -0.08))
    save(fig, 'demeaned.png')
    D['r2df'] = fl(D.r2_demeaned)
    N['demeaned'] = {m: dict(td_raw=summary(m, toes=1), td_dm=summary(m, toes=1, metric='r2df'),
                             ot_raw=summary(m, toes=0), ot_dm=summary(m, toes=0, metric='r2df'),
                             td_mae_raw=summary(m, toes=1, metric='mae'), td_mae_dm=summary(m, toes=1, metric='mae_demeaned'),
                             person_jump=float(np.mean([summary(m, s=s, toes=1, metric='r2df') - summary(m, s=s, toes=1) for s in SUBJ])),
                             jumps={s: summary(m, s=s, toes=1, metric='r2df') - summary(m, s=s, toes=1) for s in SUBJ})
                     for m in ('original', 'ours')}

    # 11. learning curves (if run): fit / validation / other session, dropout off
    lc = sorted(glob.glob(os.path.join(RES, 'learning_curves', '*', '*', 'S1toS2', 'seed0.json')))
    if lc:
        LC = [json.load(open(f)) for f in lc]
        fig, axs = plt.subplots(2, 3, figsize=(17, 8))
        colp = {'fit': '#1f9d55', 'val': '#2a78d6', 'test': '#c0392b'}
        labp = {'fit': 'fitting part (first 80 % of session 1)', 'val': 'validation part (last 20 % of session 1)', 'test': 'session 2 (test, only watched)'}
        rows = []
        for ax, s in zip(axs.flat, SUBJ):
            for r in LC:
                if r['subject'] != s:
                    continue
                h = pd.DataFrame(r['history'])
                ours = not r['experiment'].startswith('X00')
                for p in ('fit', 'val', 'test'):
                    ax.plot(h.epoch, h[f'mae_{p}'], color=colp[p], lw=1.6 if ours else 1.0, ls='-' if ours else ':',
                            label=(labp[p] if ours else None))
                if ours:
                    ax.axvline(r['best_epoch'], color='k', lw=0.8, ls='--')
                fin = {p: pd.DataFrame(r['final_per_trial'][p]) for p in ('fit', 'val', 'test')}
                rows.append(dict(model='ours' if ours else 'last semester', subject=s, epochs=r['epochs'], best_epoch=r['best_epoch'],
                                 **{f'mae_{p}': float(fin[p].mae.mean()) for p in fin}, **{f'r2_{p}': float(fin[p].r2.clip(lower=0).median()) for p in fin}))
            ax.set_title(s, loc='left', weight='bold')
            ax.set_ylim(bottom=0)
        for ax in axs[:, 0]:
            ax.set_ylabel('MAE (Nm)')
        for ax in axs[1]:
            ax.set_xlabel('epoch (dashed line = epoch kept by early stopping)')
        h1, l1 = axs.flat[0].get_legend_handles_labels()
        fig.legend(h1 + [plt.Line2D([], [], color='k', ls=':', lw=1)], l1 + ["dotted = last semester's model"], loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.04))
        fig.tight_layout()
        save(fig, 'learning_curves.png')
        N['learning_curves'] = rows

    # run times
    tt, eps, best = [], [], []
    for f in glob.glob(os.path.join(RES, '*', '*', 'S*to*', 'seed*.json')):
        if 'learning_curves' in f:
            continue
        r = json.load(open(f))
        tt.append(r['train_s'] / 60)
        ti = r.get('train_info', {})
        eps.append(ti.get('epochs', r.get('epochs_run')))
        best.append(ti.get('best_epoch', r.get('best_epoch')))
    N['runtime'] = dict(runs=len(tt), median_min=float(np.median(tt)), min_min=float(np.min(tt)), max_min=float(np.max(tt)),
                        cpu_hours=float(np.sum(tt) / 60), best_epoch_median=float(np.median(best)), best_epoch_min=int(np.min(best)),
                        best_epoch_max=int(np.max(best)), epochs_max=int(np.max(eps)))

    # overall headline numbers (median over positions of the per-position median over runs)
    def width(m, toes, metric, lo, hi):
        x = sel(m)
        x = x[x.toes_down == toes]
        g = x.groupby(['subject', 'position_rad'])[metric]
        return float(np.mean(g.quantile(hi) - g.quantile(lo)))
    N['overall'] = {m: dict(td_r2=summary(m, toes=1), ot_r2=summary(m, toes=0), td_mae=summary(m, toes=1, metric='mae'),
                            ot_mae=summary(m, toes=0, metric='mae'),
                            td_w_r2=width(m, 1, 'r2f', .05, .95), td_w_mae=width(m, 1, 'mae', .05, .95),
                            td_w_r2_99=width(m, 1, 'r2f', .01, .99), td_w_mae_99=width(m, 1, 'mae', .01, .99))
                    for m in ('original', 'original_fw', 'ours')}
    for grp, ss in (('untouched_IES_HM', ['IES', 'HM']), ('without_YES_EG', ['JM', 'IES', 'CH', 'HM'])):
        N[grp] = {m: dict(td_r2=summary(m, toes=1, subjects=ss), td_mae=summary(m, toes=1, metric='mae', subjects=ss))
                  for m in ('original', 'ours')}
    # positions that got worse (median MAE over 10 vs 10 runs), and the other positions without JM -0.08 rad
    po, pn = sel('original').groupby(['subject', 'position_rad']).mae.median(), sel('ours').groupby(['subject', 'position_rad']).mae.median()
    ch = (pn / po - 1).dropna()
    N['worse_positions'] = dict(n_lower=int((ch < 0).sum()), n_total=int(len(ch)),
                                worse10=[[s, float(p), float(po[(s, p)]), float(pn[(s, p)])] for (s, p), v in ch.sort_values(ascending=False).items() if v > 0.10],
                                worse_any=[[s, float(p), float(v)] for (s, p), v in ch.sort_values(ascending=False).items() if v > 0])
    def other_wo_jm08(m, metric):
        x = sel(m)
        x = x[(x.toes_down == 0) & ~((x.subject == 'JM') & ((x.position_rad + 0.078).abs() < 0.01))]
        return float(np.median(x.groupby(['subject', 'position_rad'])[metric].median()))
    N['other_without_JM08'] = {m: dict(ot_mae=other_wo_jm08(m, 'mae'), ot_r2=other_wo_jm08(m, 'r2f')) for m in ('original', 'ours')}
    if N.get('learning_curves'):
        L = pd.DataFrame(N['learning_curves'])
        rat = lambda m, a, b: float(np.median(L[L.model == m][a] / L[L.model == m][b]))
        N['lc_ratios'] = dict(ours_val_fit=rat('ours', 'mae_val', 'mae_fit'), ours_test_val=rat('ours', 'mae_test', 'mae_val'),
                              old_val_fit=rat('last semester', 'mae_val', 'mae_fit'), old_test_val=rat('last semester', 'mae_test', 'mae_val'))
    json.dump(N, open(os.path.join(EXP, 'presentation', 'numbers.json'), 'w'), indent=1, default=float)
    print('figures in', OUT)


if __name__ == '__main__':
    main()
