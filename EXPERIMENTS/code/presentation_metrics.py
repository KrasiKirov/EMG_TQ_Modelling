"""
Statistics in the form the supervisor asked for (feedback on the 20 Oct 2026 presentation).

    python EXPERIMENTS/code/presentation_metrics.py

What it reports, per model x subject x direction x position, over the training runs
(random seeds; the test trials are always the same):
  * median R2, and the 5th-95th and 1st-99th percentiles over seeds (no SD, no mean);
  * R2 below 0 is DISPLAYED as 0 (tables and plots); the CSVs keep the raw values;
  * MAE (Nm) instead of RMSE, and the peak voluntary torque at that position
    (max |answer key| in the tested trial) so the MAE can be read against it;
  * the same R2 / MAE after removing each trial's mean from prediction and answer key
    ("demeaned": scores the shape of the torque only, not its level).

Percentiles use numpy's linear interpolation, so with few seeds the outer percentiles are
close to the extreme runs: with 10 seeds the 1st percentile = min + 0.09 x (2nd lowest - min)
and the 5th = min + 0.45 x (2nd lowest - min); with 3 seeds both are almost the minimum.

Models
  original      X00_baseline: the frozen original code (learn S1, test S2 only)
  original_fw   X00f_baseline_framework: the same model rebuilt in our framework (both directions)
  ours          D01_cleanpassive+autogain+equalweight (identical to C01 for YES and EG,
                whose clean and legacy passive tables are the same, so C01 runs are used there)
Answer key: passive_trial (measured torque - passive torque from the passive trials of the
tested session; IES has no passive trials, nothing is subtracted) - the same key as score.py.

Every other experiment folder in results/ is scored as well (model = experiment name) and
written to presentation_runs.csv; the tables and box plots show only the three models above.

Writes results/presentation_runs.csv (one row per run x position),
results/presentation_summary.csv, reports/presentation_tables.md,
plots/pres_box_r2_<dir>.png, plots/pres_box_mae_<dir>.png, plots/pres_best_worst_<model>.png
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

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(EXP))
from ML.research.data import load_subject                 # noqa: E402
from ML.research.passive import map_lookup, PassiveModel  # noqa: E402

FS = 100.0
RES, REP, PLOTS = (os.path.join(EXP, d) for d in ('results', 'reports', 'plots'))
SUBJ = ['YES', 'EG', 'JM', 'IES', 'CH', 'HM']
MODELS = {'original': {s: 'X00_baseline' for s in SUBJ},
          'original_fw': {s: 'X00f_baseline_framework' for s in SUBJ},
          'ours': {s: ('C01_autogain+equalweight' if s in ('YES', 'EG') else 'D01_cleanpassive+autogain+equalweight') for s in SUBJ}}
MAIN = list(MODELS)                       # the models shown in the tables and box plots
# every other experiment is scored too (model name = experiment name), for the step-by-step figures
for _e in sorted(os.listdir(RES)):
    if os.path.isdir(os.path.join(RES, _e)) and _e[:1] in 'XCDST' and _e not in ('X00_baseline',) and not _e.startswith('learning'):
        MODELS.setdefault(_e, {s: _e for s in SUBJ})
LABEL = {'original': 'original LSTM-8 (original code)', 'original_fw': 'original LSTM-8 (our framework)',
         'ours': 'ours (clean passive + auto-gain + equal weight)'}
COL = {'original': '#2a78d6', 'original_fw': '#7aa9dd', 'ours': '#eb6834'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e5e1', 'axes.axisbelow': True, 'legend.frameon': False})
_cache = {}


def subj(s):
    """answer key (passive_trial) of every active trial, as in score.py"""
    if s not in _cache:
        b = load_subject(s)
        st = PassiveModel('static', b)
        recs = {r['idx']: dict(session=r['session'], angle=float(r['angle']), t0=float(r['t'][0]),
                               key=(r['tq'] - st.predict(r)).astype(np.float64)) for r in b['active']}
        _cache[s] = dict(bundle=b, recs=recs)
    return _cache[s]


def metrics(y, p):
    e = p - y
    ss = np.sum((y - y.mean()) ** 2)
    ed = (p - p.mean()) - (y - y.mean())                   # demeaned: level removed from both
    return dict(r2=float(1 - np.sum(e ** 2) / ss), mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e ** 2))),
                bias=float(np.mean(e)), r2_demeaned=float(1 - np.sum(ed ** 2) / ss), mae_demeaned=float(np.mean(np.abs(ed))),
                peak_torque=float(np.max(np.abs(y))), torque_range=float(np.max(y) - np.min(y)), key_std=float(np.std(y)),
                n=int(len(y)))


def trials(exp, s, f):
    """yield (position, t_rel, key, pred) for every test trial of one saved run file"""
    S = subj(s)
    z = np.load(f)
    if exp == 'X00_baseline':
        # original code: windows are identified by their position, in time order;
        # passive_trial key = original target + (original table - clean table)
        maps = S['bundle']['maps']
        pos, snapped = z['position'].astype(np.float64), z['position_snapped'].astype(np.float64)
        y = z['y_true_original'].astype(np.float64) + map_lookup(maps['legacy']['S2'], pos) - map_lookup(maps['static']['S2'], pos)
        p = z['y_pred'].astype(np.float64)
        s2 = [R['angle'] for R in S['recs'].values() if R['session'] == 'S2']
        for a in np.unique(snapped):
            m = snapped == a
            yield min(s2, key=lambda x: abs(x - a)), (np.arange(m.sum()) + 19) / FS, y[m], p[m]
    else:
        trial, t, p = z['trial'].astype(int), z['t'].astype(np.float64), z['y_pred'].astype(np.float64)
        for i in np.unique(trial):
            m = trial == i
            R = S['recs'][i]
            k = np.rint((t[m] - R['t0']) * FS).astype(int)
            yield R['angle'], k / FS, R['key'][k], p[m]


def run_file(model, s, d, seed):
    exp = MODELS[model][s] if model in MODELS else model
    return exp, os.path.join(RES, exp, s, d, f'seed{seed}.npz')


def runs():
    """yield (model, subject, direction, seed, position, t_rel, key, pred) for every saved run x test trial"""
    for model, per in MODELS.items():
        for s in SUBJ:
            exp = per[s]
            for f in sorted(glob.glob(os.path.join(RES, exp, s, 'S*to*', 'seed*.npz'))):
                d, seed = os.path.basename(os.path.dirname(f)), int(os.path.basename(f)[4:-4])
                for pos, t, y, p in trials(exp, s, f):
                    yield model, s, d, seed, pos, t, y, p


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else np.nan


def show_r2(v):
    """R2 for display: below 0 is shown as 0 (supervisor's request); raw values stay in the CSVs"""
    return f'{max(v, 0.0):.2f}'


def main():
    rows, curves = [], {}
    for model, s, d, seed, pos, t, y, p in runs():
        rows.append(dict(model=model, subject=s, direction=d, seed=seed, position_rad=round(pos, 3), **metrics(y, p)))
        if model in MAIN:                                  # keep waveforms only for the plotted models
            curves[(model, s, d, seed, round(pos, 3))] = (t, y, p)
    df = pd.DataFrame(rows)
    # problematic = the 2 most toes-down positions of each subject's tested session (as in score.py)
    df['toes_down'] = 0
    for (s, d), x in df.groupby(['subject', 'direction']):
        df.loc[x.index[x.position_rad.isin(sorted(x.position_rad.unique())[:2])], 'toes_down'] = 1
    df.to_csv(os.path.join(RES, 'presentation_runs.csv'), index=False, encoding='utf-8-sig')

    agg = []
    for (model, s, d, pos), x in df.groupby(['model', 'subject', 'direction', 'position_rad']):
        r = dict(model=model, subject=s, direction=d, position_rad=pos, toes_down=int(x.toes_down.iloc[0]), seeds=len(x),
                 peak_torque=x.peak_torque.iloc[0])
        for mname in ('r2', 'mae', 'r2_demeaned', 'mae_demeaned'):
            v = x[mname].values
            r.update({f'{mname}_median': float(np.median(v)), f'{mname}_p5': pct(v, 5), f'{mname}_p95': pct(v, 95),
                      f'{mname}_p1': pct(v, 1), f'{mname}_p99': pct(v, 99)})
        r['mae_pct_of_peak'] = 100 * r['mae_median'] / r['peak_torque']
        agg.append(r)
    A = pd.DataFrame(agg)
    A.to_csv(os.path.join(RES, 'presentation_summary.csv'), index=False, encoding='utf-8-sig')

    # ── tables ──────────────────────────────────────────────────────────────
    L = ['# Results in the format asked for by the supervisor', '',
         'Generated by `code/presentation_metrics.py`. Each cell summarises the training runs (different random seeds) '
         'of one model; the test trials are always the same. **R² below 0 is shown as 0**; raw values are in '
         '`results/presentation_runs.csv`. MAE in Nm. Peak = largest |voluntary torque| in the tested trial at that position. '
         'Window = 20 samples = 0.2 s (100 Hz) for every model here.', '']
    seeds = df[df.model.isin(MAIN)].groupby('model').seed.nunique().to_dict()
    L += ['Seeds per model: ' + ', '.join(f'{LABEL[m]} {n}' for m, n in seeds.items()), '']
    for d in ('S1toS2', 'S2toS1'):
        Ad = A[A.direction == d]
        if Ad.empty:
            continue
        models = [m for m in MAIN if m in set(Ad.model)]
        L += [f'## {"learn session 1 → test session 2 (primary)" if d == "S1toS2" else "learn session 2 → test session 1 (supplementary)"}', '']
        hdr = '| subject | position (rad) | peak (Nm) | ' + ' | '.join(f'{m}: R² median [5–95 %] | {m}: MAE median [5–95 %]' for m in models) + ' |'
        L += [hdr, '|' + '---|' * (3 + 2 * len(models))]
        for s in SUBJ:
            for pos in sorted(Ad[Ad.subject == s].position_rad.unique()):
                cells = []
                for m in models:
                    r = Ad[(Ad.model == m) & (Ad.subject == s) & ((Ad.position_rad - pos).abs() < 0.005)]
                    if r.empty:
                        cells += ['–', '–']
                        continue
                    r = r.iloc[0]
                    cells += [f'{show_r2(r.r2_median)} [{show_r2(r.r2_p5)}–{show_r2(r.r2_p95)}]',
                              f'{r.mae_median:.2f} [{r.mae_p5:.2f}–{r.mae_p95:.2f}]']
                pk = Ad[(Ad.subject == s) & ((Ad.position_rad - pos).abs() < 0.005)].peak_torque.iloc[0]
                td = ' ⬇' if Ad[(Ad.subject == s) & ((Ad.position_rad - pos).abs() < 0.005)].toes_down.iloc[0] else ''
                L.append(f'| {s} | {pos:+.3f}{td} | {pk:.1f} | ' + ' | '.join(cells) + ' |')
        L += ['', '⬇ = one of the 2 most toes-down positions.', '']
        # overall: median over every run x position of the group
        L += ['**Over all subjects** (median over every run × position; 5th–95th / 1st–99th percentile):', '',
              '| model | positions | R² median | R² 5–95 % | R² 1–99 % | MAE median (Nm) | MAE 5–95 % | demeaned R² median | demeaned MAE median |',
              '|---|---|---|---|---|---|---|---|---|']
        Dd = df[df.direction == d]
        for m in models:
            for grp, gname in ((1, 'toes-down'), (0, 'other')):
                x = Dd[(Dd.model == m) & (Dd.toes_down == grp)]
                L.append(f'| {m} | {gname} | {show_r2(np.median(x.r2))} | {show_r2(pct(x.r2, 5))}–{show_r2(pct(x.r2, 95))} | '
                         f'{show_r2(pct(x.r2, 1))}–{show_r2(pct(x.r2, 99))} | {np.median(x.mae):.2f} | {pct(x.mae, 5):.2f}–{pct(x.mae, 95):.2f} | '
                         f'{show_r2(np.median(x.r2_demeaned))} | {np.median(x.mae_demeaned):.2f} |')
        L.append('')
    open(os.path.join(REP, 'presentation_tables.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')

    # ── box plots: whiskers = 5th-95th percentile, dots = each run ─────────────
    for d in ('S1toS2', 'S2toS1'):
        Dd = df[df.direction == d]
        models = [m for m in ('original', 'ours') if m in set(Dd.model)] or [m for m in MAIN if m in set(Dd.model)]
        if d == 'S2toS1':
            models = [m for m in ('original_fw', 'ours') if m in set(Dd.model)]
        for metric, ylab, fn in (('r2', 'R² (below 0 shown as 0)', 'pres_box_r2'), ('mae', 'MAE (Nm)', 'pres_box_mae')):
            fig, axs = plt.subplots(2, 3, figsize=(16, 8))
            for ax, s in zip(axs.flat, SUBJ):
                x = Dd[Dd.subject == s]
                pos = sorted(x.position_rad.unique())
                w = 0.8 / len(models)
                for j, m in enumerate(models):
                    data = [np.clip(x[(x.model == m) & ((x.position_rad - p).abs() < 0.005)][metric].values, 0 if metric == 'r2' else None, None) for p in pos]
                    xs = np.arange(len(pos)) + (j - (len(models) - 1) / 2) * w
                    ok = [k for k, v in enumerate(data) if len(v)]
                    bp = ax.boxplot([data[k] for k in ok], positions=xs[ok], widths=w * 0.85, whis=(5, 95), patch_artist=True,
                                    showfliers=False, medianprops=dict(color='k', lw=1.4))
                    for b in bp['boxes']:
                        b.set(facecolor=COL[m], alpha=0.55, edgecolor=COL[m])
                    for k in ok:
                        ax.scatter(np.full(len(data[k]), xs[k]), data[k], s=7, color=COL[m], edgecolor='k', lw=0.3, zorder=3)
                    ax.plot([], [], color=COL[m], lw=6, alpha=0.55, label=LABEL[m])
                ax.set_xticks(range(len(pos)))
                ax.set_xticklabels([f'{p:+.2f}' for p in pos], rotation=40, fontsize=8.5)
                if metric == 'r2':
                    ax.set_ylim(-0.03, 1.03)
                ax.set_title(s + (' (no passive trials)' if s == 'IES' else ''), loc='left', fontsize=11, weight='bold')
            for ax in axs[:, 0]:
                ax.set_ylabel(ylab)
            for ax in axs[1]:
                ax.set_xlabel('ankle position (rad, negative = toes down)')
            axs[0, 0].legend(fontsize=8.5, loc='lower right')
            n = Dd[Dd.model.isin(models)].groupby('model').seed.nunique().to_dict()
            fig.suptitle(f'{"learn session 1 → test session 2" if d == "S1toS2" else "learn session 2 → test session 1"} · '
                         f'box = 25th–75th percentile, line = median, whiskers = 5th–95th percentile, dots = each training run '
                         f'(seeds: ' + ', '.join(f'{m} {k}' for m, k in n.items()) + ')', fontsize=10.5)
            fig.tight_layout()
            fig.savefig(os.path.join(PLOTS, f'{fn}_{d}.png'), dpi=130)
            plt.close(fig)

    # ── best and worst run of each model at the same position (most toes-down, learn S1 -> test S2) ──
    for m in ('original', 'ours'):
        Dm = df[(df.model == m) & (df.direction == 'S1toS2')]
        if Dm.empty:
            continue
        fig, axs = plt.subplots(len(SUBJ), 1, figsize=(14, 2.6 * len(SUBJ)))
        for ax, s in zip(axs, SUBJ):
            x = Dm[Dm.subject == s]
            pos = sorted(x.position_rad.unique())[0]
            x = x[(x.position_rad - pos).abs() < 0.005]
            best, worst = x.loc[x.r2.idxmax()], x.loc[x.r2.idxmin()]
            t, y, _ = curves[(m, s, 'S1toS2', int(best.seed), pos)]
            ax.plot(t, y, color='#1b1b1b', lw=0.9, label=f'measured voluntary torque (peak {np.max(np.abs(y)):.1f} Nm)')
            for r, col, nm in ((best, '#1f9d55', 'best run'), (worst, '#c0392b', 'worst run')):
                tt, _, p = curves[(m, s, 'S1toS2', int(r.seed), pos)]
                ax.plot(tt, p, color=col, lw=0.8, label=f'{nm} (seed {int(r.seed)}): R² {show_r2(r.r2)}, MAE {r.mae:.2f} Nm')
            ax.set_title(f'{s} · {pos:+.2f} rad (most toes-down) · {len(x)} runs', loc='left', fontsize=10.5, weight='bold')
            ax.set_ylabel('Nm')
            ax.legend(fontsize=8, loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=3)
        axs[-1].set_xlabel('time in trial (s)')
        fig.suptitle(f'{LABEL[m]} · learn session 1 → test session 2 · best and worst training run at the same position', fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS, f'pres_best_worst_{m}.png'), dpi=110)
        plt.close(fig)
    print('written: results/presentation_runs.csv, results/presentation_summary.csv, reports/presentation_tables.md, plots/pres_*.png')
    print(df[df.model.isin(MAIN)].groupby(['model', 'direction']).seed.nunique())


if __name__ == '__main__':
    main()
