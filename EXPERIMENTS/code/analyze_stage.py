"""
Compare experiments with the baseline using the rules in SUCCESS_CRITERIA.md.

    python EXPERIMENTS/code/analyze_stage.py --stage A --subjects YES JM \
        --reference X00f_baseline_framework --exps X01_clean_passive X02a_dropout_0.0 ...

For each experiment (primary key = passive_trial, primary direction = S1toS2; the
reverse direction is reported as supplementary) it writes, per subject x position:
seed-mean R2, MSE, RMSE, NRMSE for the reference and the experiment and their
differences, then checks the criteria:
  1. problematic-position seed-mean R2 (mean over subjects x positions) rises >= 0.10
     and their mean RMSE does not rise;
  2. R2 improves in >= 3/4 (subject x problematic position) cells;
  3. no other position loses > 0.05 R2; mean other-position R2 drops <= 0.01;
     no other position's RMSE rises > 15 %;
  4. mean problematic improvement > 2 x the reference's seed SD there.
Outputs reports/stage<A>_<exp>.md, results/stage<A>_comparison.csv, plots/stage<A>_<exp>.png
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
RES, REP, PLOTS = (os.path.join(EXP, d) for d in ('results', 'reports', 'plots'))


def per_pos(ev, exp, subjects, direction, key):
    x = ev[(ev.experiment == exp) & ev.subject.isin(subjects) & (ev.direction == direction) & (ev.key == key)]
    return x.groupby(['subject', 'position_rad', 'problematic']).agg(
        r2=('r2', 'mean'), r2_sd=('r2', 'std'), mse=('mse', 'mean'), rmse=('rmse', 'mean'), nrmse=('nrmse', 'mean'),
        seeds=('seed', 'nunique')).reset_index()


def match(a, b):
    """join by subject and nearest position (labels can differ in the 3rd decimal)."""
    out = []
    for r in a.itertuples():
        c = b[b.subject == r.subject]
        if c.empty:
            continue
        j = (c.position_rad - r.position_rad).abs().idxmin()
        out.append(dict(subject=r.subject, position_rad=r.position_rad, problematic=r.problematic,
                        ref_r2=r.r2, ref_r2_sd=r.r2_sd, ref_mse=r.mse, ref_rmse=r.rmse, ref_nrmse=r.nrmse,
                        r2=c.loc[j, 'r2'], r2_sd=c.loc[j, 'r2_sd'], mse=c.loc[j, 'mse'], rmse=c.loc[j, 'rmse'],
                        nrmse=c.loc[j, 'nrmse'], seeds=c.loc[j, 'seeds']))
    m = pd.DataFrame(out)
    if not m.empty:
        m['d_r2'] = m.r2 - m.ref_r2
        m['d_rmse_pct'] = 100 * (m.rmse / m.ref_rmse - 1)
    return m


def criteria(m):
    pf, ot = m[m.problematic == 1], m[m.problematic == 0]
    c1 = (pf.d_r2.mean() >= 0.10) and (pf.rmse.mean() <= pf.ref_rmse.mean())
    c2 = (pf.d_r2 > 0).sum() >= 3
    c3 = (ot.d_r2.min() >= -0.05) and (ot.r2.mean() - ot.ref_r2.mean() >= -0.01) and (ot.d_rmse_pct.max() <= 15)
    noise = pf.ref_r2_sd.mean()
    c4 = pf.d_r2.mean() > 2 * (noise if np.isfinite(noise) else 0)
    return dict(c1=bool(c1), c2=bool(c2), c3=bool(c3), c4=bool(c4), pf_d_r2=pf.d_r2.mean(), pf_ref_r2=pf.ref_r2.mean(),
                pf_r2=pf.r2.mean(), pf_d_rmse=pf.rmse.mean() - pf.ref_rmse.mean(), pf_cells_improved=int((pf.d_r2 > 0).sum()),
                other_min_d_r2=ot.d_r2.min(), other_mean_d_r2=ot.r2.mean() - ot.ref_r2.mean(), other_max_d_rmse_pct=ot.d_rmse_pct.max(),
                ref_noise_sd=noise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='A')
    ap.add_argument('--subjects', nargs='+', required=True)
    ap.add_argument('--reference', default='X00f_baseline_framework')
    ap.add_argument('--exps', nargs='+', required=True)
    ap.add_argument('--key', default='passive_trial')
    ap.add_argument('--seeds', nargs='*', type=int, default=None, help='restrict to these seeds (e.g. the seeds available when a decision was made)')
    a = ap.parse_args()
    ev = pd.read_csv(os.path.join(RES, 'all_evaluations.csv'))
    if a.seeds is not None:
        ev = ev[ev.seed.isin(a.seeds)]
    allc, summ = [], []
    for exp in a.exps:
        L = [f'# Stage {a.stage}: {exp} vs {a.reference}', '', f'Subjects {", ".join(a.subjects)}; answer key **{a.key}**; seed-mean values over MATCHED seeds (same seeds for reference and experiment); R² not floored.', '']
        verdicts = {}
        for d in ('S1toS2', 'S2toS1'):
            # MATCHED SEEDS: per subject, only seeds present for both the reference and the experiment
            evd = ev[(ev.direction == d) & (ev.key == a.key)]
            keep = []
            for sj in a.subjects:
                sr = set(evd[(evd.experiment == a.reference) & (evd.subject == sj)].seed)
                se = set(evd[(evd.experiment == exp) & (evd.subject == sj)].seed)
                common = sr & se
                keep.append(evd[(evd.subject == sj) & evd.experiment.isin([a.reference, exp]) & evd.seed.isin(common)])
            evm = pd.concat(keep) if keep else evd.iloc[:0]
            ref, ex = per_pos(evm, a.reference, a.subjects, d, a.key), per_pos(evm, exp, a.subjects, d, a.key)
            if ref.empty or ex.empty:
                continue
            m = match(ref, ex)
            m['experiment'], m['direction'] = exp, d
            allc.append(m)
            cr = criteria(m)
            verdicts[d] = cr
            tag = 'PRIMARY' if d == 'S1toS2' else 'supplementary'
            L += [f'## {d} ({tag})', '',
                  f"Criteria: 1 {'✔' if cr['c1'] else '✘'} · 2 {'✔' if cr['c2'] else '✘'} · 3 {'✔' if cr['c3'] else '✘'} · 4 {'✔' if cr['c4'] else '✘'}  "
                  f"(problematic R² {cr['pf_ref_r2']:.3f} → {cr['pf_r2']:.3f}, Δ {cr['pf_d_r2']:+.3f}; cells improved {cr['pf_cells_improved']}/4; "
                  f"RMSE Δ {cr['pf_d_rmse']:+.3f} Nm; other positions: worst ΔR² {cr['other_min_d_r2']:+.3f}, mean ΔR² {cr['other_mean_d_r2']:+.3f}, "
                  f"worst RMSE change {cr['other_max_d_rmse_pct']:+.0f} %; reference run-to-run SD {cr['ref_noise_sd']:.3f})", '',
                  '| subject | position | problematic | R² ref | R² exp | ΔR² | MSE ref | MSE exp | RMSE ref | RMSE exp | ΔRMSE % | NRMSE ref | NRMSE exp | seeds |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
            for r in m.sort_values(['subject', 'position_rad']).itertuples():
                L.append(f'| {r.subject} | {r.position_rad:+.3f} | {"yes" if r.problematic else ""} | {r.ref_r2:.3f} | {r.r2:.3f} | {r.d_r2:+.3f} | '
                         f'{r.ref_mse:.3f} | {r.mse:.3f} | {r.ref_rmse:.3f} | {r.rmse:.3f} | {r.d_rmse_pct:+.0f} | {r.ref_nrmse:.3f} | {r.nrmse:.3f} | {r.seeds} |')
            L.append('')
        p = verdicts.get('S1toS2')
        if p:
            s = verdicts.get('S2toS1')
            status = 'promising' if all(p[c] for c in ('c1', 'c2', 'c3', 'c4')) else ('trade-off' if p['c1'] and p['c2'] else 'not promising')
            summ.append(dict(experiment=exp, status=status, **{k: v for k, v in p.items()},
                             supp_pf_d_r2=(s['pf_d_r2'] if s else np.nan)))
        open(os.path.join(REP, f'stage{a.stage}_{exp}.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
        # plot: ΔR² per position, both directions
        sub = [x for x in allc if (x.experiment == exp).all()]
        if sub:
            fig, axs = plt.subplots(1, len(a.subjects), figsize=(5.5 * len(a.subjects), 3.6), squeeze=False)
            for i, sj in enumerate(a.subjects):
                ax = axs[0, i]
                for x, col in zip(sub, ('#eb6834', '#8a8984')):
                    y = x[x.subject == sj].sort_values('position_rad')
                    ax.bar(np.arange(len(y)) + (0 if col == '#eb6834' else 0.4), y.d_r2.clip(-1, 1), width=0.4, color=col,
                           label=x.direction.iloc[0])
                    ax.set_xticks(np.arange(len(y)) + 0.2)
                    ax.set_xticklabels([f'{v:+.2f}' for v in y.position_rad], rotation=40, fontsize=8)
                ax.axhline(0, color='k', lw=0.6)
                ax.set_title(f'{sj}: change in R² vs baseline', fontsize=10, loc='left')
                ax.legend(fontsize=8)
            fig.suptitle(exp, fontsize=11)
            fig.tight_layout()
            fig.savefig(os.path.join(PLOTS, f'stage{a.stage}_{exp}.png'), dpi=110)
            plt.close(fig)
    if allc:
        pd.concat(allc).to_csv(os.path.join(RES, f'stage{a.stage}_comparison.csv'), index=False)
    s = pd.DataFrame(summ)
    s.to_csv(os.path.join(RES, f'stage{a.stage}_summary.csv'), index=False)
    print(s[['experiment', 'status', 'pf_ref_r2', 'pf_r2', 'pf_d_r2', 'pf_cells_improved', 'pf_d_rmse', 'other_min_d_r2',
             'other_mean_d_r2', 'other_max_d_rmse_pct', 'supp_pf_d_r2']].round(3).to_string() if not s.empty else 'no results')


if __name__ == '__main__':
    main()
