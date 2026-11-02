"""
Final figures: baseline (X00f) vs the best candidate (D01; equals C01 for YES and EG) for
every subject, position, direction and seed.  Primary key = passive trials of the tested
session.  Writes EXPERIMENTS/plots/final_*.png and results/final_per_position.csv.

    python EXPERIMENTS/code/final_figures.py [--candidate D01_cleanpassive+autogain+equalweight]
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(EXP))
RES, PLOTS = os.path.join(EXP, 'results'), os.path.join(EXP, 'plots')
OLD, NEW, KEY, GRID = '#2a78d6', '#eb6834', '#1b1b1b', '#e6e5e1'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'axes.axisbelow': True, 'legend.frameon': False})
SUBJ = ['YES', 'EG', 'JM', 'IES', 'CH', 'HM']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', default='D01_cleanpassive+autogain+equalweight')
    ap.add_argument('--alias', default='C01_autogain+equalweight', help='identical config used for YES and EG')
    a = ap.parse_args()
    ev = pd.read_csv(os.path.join(RES, 'all_evaluations.csv'))
    ev = ev[ev.key == 'passive_trial']
    base = ev[ev.experiment == 'X00f_baseline_framework'].assign(model='baseline')
    cand = ev[(ev.experiment == a.candidate) | ((ev.experiment == a.alias) & ev.subject.isin(['YES', 'EG']))].assign(model='candidate')
    d = pd.concat([base, cand])
    d.to_csv(os.path.join(RES, 'final_per_position.csv'), index=False)

    # 1. R2 per position, both directions, seed dots
    for metric, lab, fname in (('r2', 'R² (1 = perfect)', 'final_r2_per_position'), ('rmse', 'RMSE (Nm, lower is better)', 'final_rmse_per_position')):
        for direction in ('S1toS2', 'S2toS1'):
            fig, axs = plt.subplots(2, 3, figsize=(15, 7.5))
            for ax, s in zip(axs.flat, SUBJ):
                x = d[(d.subject == s) & (d.direction == direction)]
                pos = sorted(x.position_rad.unique())
                for j, (mdl, col) in enumerate((('baseline', OLD), ('candidate', NEW))):
                    y = x[x.model == mdl]
                    for i, p in enumerate(pos):
                        v = y[(y.position_rad - p).abs() < 0.005][metric].values
                        if len(v) == 0:
                            continue
                        vv = np.clip(v, -1.5, None) if metric == 'r2' else v
                        ax.bar(i + (j - 0.5) * 0.38, np.mean(vv), width=0.36, color=col, alpha=0.85,
                               label=('baseline LSTM-8' if mdl == 'baseline' else 'candidate') if i == 0 else None)
                        ax.scatter(np.full(len(vv), i + (j - 0.5) * 0.38), vv, s=9, color='k', zorder=3)
                ax.set_xticks(range(len(pos)))
                ax.set_xticklabels([f'{p:+.2f}' for p in pos], rotation=40, fontsize=9)
                if metric == 'r2':
                    ax.set_ylim(-1.6, 1.08)
                    ax.axhline(0, color='k', lw=0.6)
                ax.set_title(s + (' (no passive trials)' if s == 'IES' else ''), loc='left', fontsize=12, weight='bold')
            axs[0, 0].set_ylabel(lab)
            axs[1, 0].set_ylabel(lab)
            axs[0, 0].legend(fontsize=9, loc='lower right')
            fig.suptitle(f'{"learned session 1 → tested session 2" if direction == "S1toS2" else "learned session 2 → tested session 1"} · '
                         'bars = mean of 3 training runs, dots = each run · leftmost 2 positions = toes-down (problematic)', fontsize=11)
            fig.tight_layout()
            fig.savefig(os.path.join(PLOTS, f'{fname}_{direction}.png'), dpi=130)
            plt.close(fig)

    # 2. problematic vs other positions, per subject and direction
    g = d.groupby(['model', 'subject', 'direction', 'problematic']).agg(r2=('r2', lambda v: np.mean(np.clip(v, -1, None))), rmse=('rmse', 'mean')).reset_index()
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, pr, ttl in ((axs[0], 1, 'toes-down (problematic) positions'), (axs[1], 0, 'other positions')):
        for j, (mdl, col) in enumerate((('baseline', OLD), ('candidate', NEW))):
            y = g[(g.model == mdl) & (g.problematic == pr)]
            labels = [f'{s}\n{"1→2" if dd == "S1toS2" else "2→1"}' for s in SUBJ for dd in ('S1toS2', 'S2toS1')]
            vals = [y[(y.subject == s) & (y.direction == dd)].r2.mean() for s in SUBJ for dd in ('S1toS2', 'S2toS1')]
            ax.bar(np.arange(len(vals)) + (j - 0.5) * 0.38, vals, width=0.36, color=col, label=mdl)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_title(f'R² at {ttl} (each position floored at −1)', loc='left', fontsize=11, weight='bold')
    axs[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, 'final_problematic_vs_other.png'), dpi=130)
    plt.close(fig)

    # 3. prediction curves at the 2 toes-down positions, seed 0, learned S1 -> tested S2
    from ML.research.data import load_subject
    from ML.research.passive import PassiveModel
    fig, axs = plt.subplots(6, 2, figsize=(15, 17))
    for i, s in enumerate(SUBJ):
        b = load_subject(s)
        pm = PassiveModel('static', b)
        recs = {r['idx']: r for r in b['active']}
        pf = sorted([r for r in b['active'] if r['session'] == 'S2'], key=lambda r: r['angle'])[:2]
        for j, r in enumerate(pf):
            ax = axs[i, j]
            key = r['tq'] - pm.predict(r)
            ax.plot(r['t'] - r['t'][0], key, color=KEY, lw=0.8, label='answer key')
            for exp, col, lab in (('X00f_baseline_framework', OLD, 'baseline'), (a.alias if s in ('YES', 'EG') else a.candidate, NEW, 'candidate')):
                f = os.path.join(RES, exp, s, 'S1toS2', 'seed0.npz')
                if not os.path.exists(f):
                    continue
                z = np.load(f)
                m = z['trial'] == r['idx']
                yy = key[np.rint((z['t'][m] - r['t'][0]) * 100).astype(int)]
                r2 = 1 - np.sum((z['y_pred'][m] - yy) ** 2) / np.sum((yy - yy.mean()) ** 2)
                ax.plot(z['t'][m] - r['t'][0], z['y_pred'][m], color=col, lw=0.8, label=f'{lab}  R² {r2:.2f}')
            ax.set_title(f'{s} · {r["angle"]:+.2f} rad · learned S1 → tested S2 · seed 0', loc='left', fontsize=10.5, weight='bold')
            ax.legend(fontsize=8, loc='lower right', ncol=3)
    axs[-1, 0].set_xlabel('time in trial (s)')
    axs[-1, 1].set_xlabel('time in trial (s)')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, 'final_toesdown_curves_S1toS2.png'), dpi=120)
    plt.close(fig)
    print('figures written')


if __name__ == '__main__':
    main()
