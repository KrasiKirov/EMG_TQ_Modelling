"""
Figures for the study report (all generated from saved results).

    python -m ML.research.figures --study screen --out notes/report2/figures [--only truth]

Colour rules: OLD (original pipeline) blue, NEW/candidate orange, truth near-
black, extra series aqua/yellow/purple, and every series is labelled (never
colour alone).  This is a validated categorical palette, the same one as the
first report.
"""

import os
import sys
import json
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ML.research import config as C
from ML.research import truth as T

OLD, NEW, AQUA, YELLOW, PURPLE = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#8a5cc2'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e6e5e1'
SERIES = [OLD, NEW, AQUA, YELLOW, PURPLE]


def setup():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
        'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
        'grid.color': GRID, 'grid.linewidth': 0.7, 'xtick.color': INK2, 'ytick.color': INK2,
        'legend.frameon': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'axes.axisbelow': True})


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('saved', p)
    return p


# ── 1. reference truth inspection (every trial) ──────────────────────────────

def truth_inspection(subject, out_dir):
    """Measured torque, EMG-detected rest, in-trial passive baseline B(t),
    and the passive levels the configurations subtract (before-hold mean,
    static map), for every active trial of both sessions."""
    from ML.research.data import load_subject
    from ML.research.passive import map_lookup
    b = load_subject(subject)
    tr = T.subject_truth(subject)
    fig, axs = plt.subplots(8, 2, figsize=(15, 20), sharex=True)
    for col, sess in enumerate(('S1', 'S2')):
        recs = sorted([r for r in b['active'] if r['session'] == sess], key=lambda r: r['angle'])
        for row, r in enumerate(recs[:8]):
            ax = axs[row, col]
            d = tr[r['idx']]
            ax.plot(r['t'], r['tq'], color=MUTED, lw=0.6, label='measured torque')
            ax.plot(r['t'][d['rest']], r['tq'][d['rest']], '.', ms=1.5, color=AQUA, label='EMG-detected rest')
            ax.plot(r['t'], d['B'], color=INK, lw=1.6, label='in-trial passive baseline B(t)')
            static = map_lookup(b['maps']['static'][sess], np.array([r['angle']]))[0]
            ax.axhline(static, color=OLD, ls='--', lw=1.0, label='static passive map')
            if r['hold'] is not None:
                ax.axhline(r['hold']['mean'], color=YELLOW, ls=':', lw=1.4, label='before-hold mean')
            ax.set_title(f"{sess}  {r['angle']:+.3f} rad   rest {d['rest'].mean():.0%}", fontsize=9, loc='left')
            ax.set_ylabel('Nm')
    axs[0, 0].legend(fontsize=7, loc='lower right', markerscale=4)
    axs[-1, 0].set_xlabel('time (s)'); axs[-1, 1].set_xlabel('time (s)')
    fig.suptitle(f'{subject}: reference truth = measured − B(t)  (angles from most plantarflexed, top)', y=1.0)
    fig.tight_layout()
    return save(fig, out_dir, f'truth_inspection_{subject}.png')


# ── 2. prediction overlays for chosen trials ─────────────────────────────────

def overlay(study, subject, protocol, configs, angles, out_dir, name, seed=0, labels=None):
    """Truth vs predictions of several configurations for selected angles."""
    tr = T.subject_truth(subject)
    fig, axs = plt.subplots(len(angles), 1, figsize=(14, 3.0 * len(angles)), squeeze=False)
    for ai, ang in enumerate(angles):
        ax = axs[ai, 0]
        drawn = False
        for ci, cfg in enumerate(configs):
            f = os.path.join(C.RESULTS_DIR, study, subject, protocol, cfg, f'fast_seed{seed}.npz')
            if not os.path.exists(f):
                continue
            z = np.load(f)
            trials = np.unique(z['trial'])
            ti = min(trials, key=lambda i: abs(tr[int(i)]['angle'] - ang))
            m = z['trial'] == ti
            t = z['t'][m]
            if not drawn:
                d = tr[int(ti)]
                k = np.rint((t - d['t0']) * C.FS).astype(int)
                ax.plot(t, d['truth'][k], color=INK, lw=1.0, label='reference truth')
                drawn = True
            res = T.rescore(subject, f)['per_angle']
            key = f"{'S2' if z['session'][m][0] == 1 else 'S1'}:{float(z['angle'][m][0]):+.3f}"
            lab = (labels or configs)[ci]
            ax.plot(t, z['y_pred'][m], color=SERIES[ci % len(SERIES)], lw=0.9,
                    label=f"{lab}  (R² {res[key]['r2']:+.2f}, RMSE {res[key]['rmse']:.2f} Nm)")
        ax.set_title(f'{subject} {protocol}  angle {ang:+.3f} rad', loc='left')
        ax.set_ylabel('active torque (Nm)')
        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    axs[-1, 0].set_xlabel('time (s)')
    fig.tight_layout()
    return save(fig, out_dir, name)


def target_vs_truth(study, cases, out_dir, name, config='E01_old', seed=0):
    """Why a common truth is needed: for each case (subject, protocol, angle)
    plot the configuration's OWN test target, the reference truth and the
    configuration's prediction."""
    fig, axs = plt.subplots(len(cases), 1, figsize=(14, 3.2 * len(cases)), squeeze=False)
    for ax, (sub, prot, ang) in zip(axs[:, 0], cases):
        f = os.path.join(C.RESULTS_DIR, study, sub, prot, config, f'fast_seed{seed}.npz')
        z = np.load(f)
        tr = T.subject_truth(sub)
        ti = min(np.unique(z['trial']), key=lambda i: abs(tr[int(i)]['angle'] - ang))
        m = z['trial'] == ti
        t = z['t'][m]
        d = tr[int(ti)]
        k = np.rint((t - d['t0']) * C.FS).astype(int)
        own = json.load(open(f[:-4] + '.json'))['per_angle']
        key = f"{'S2' if z['session'][m][0] == 1 else 'S1'}:{float(z['angle'][m][0]):+.3f}"
        res = T.rescore(sub, f)['per_angle'][key]
        ax.plot(t, z['y_true'][m], color=OLD, lw=0.9, label=f"{config} own target (R² {own[key]['r2']:+.2f})")
        ax.plot(t, d['truth'][k], color=INK, lw=0.9, label=f"reference truth (R² {res['r2']:+.2f})")
        ax.plot(t, z['y_pred'][m], color=NEW, lw=0.9, label=f'{config} prediction')
        ax.set_title(f'{sub}, trained on the other session, tested on {key[:2]}, angle {ang:+.3f} rad', loc='left')
        ax.set_ylabel('active torque (Nm)')
        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    axs[-1, 0].set_xlabel('time (s)')
    fig.tight_layout()
    return save(fig, out_dir, name)


def full_trials_vs_truth(npz_path):
    """Full-trial inspection plot of one run against the REFERENCE truth
    (the plots written by run.py show the configuration's own target).
    Saved next to the run as full_trial_inspection_truth_<S1|S2>_<stem>.png."""
    from ML.research.evaluate import plot_trials
    parts = os.path.normpath(npz_path).split(os.sep)
    sub, prot, cfg, stem = parts[-4], parts[-3], parts[-2], os.path.basename(npz_path)[:-4]
    z = np.load(npz_path)
    tr = T.subject_truth(sub)
    y = np.empty(len(z['y_pred']))
    for i in np.unique(z['trial']):
        m = z['trial'] == i
        y[m] = tr[int(i)]['truth'][np.rint((z['t'][m] - tr[int(i)]['t0']) * C.FS).astype(int)]
    out = []
    for code, sess in ((0, 'S1'), (1, 'S2')):
        m = z['session'] == code
        if not m.any():
            continue
        meta = {'angle': z['angle'][m].astype(np.float64)}
        p = os.path.join(os.path.dirname(npz_path), f'full_trial_inspection_truth_{sess}_{stem}.png')
        plot_trials(meta, y[m], z['y_pred'][m].astype(np.float64), p,
                    f'{sub} | {cfg} | {prot} -> test {sess} | {stem} | vs REFERENCE truth')
        out.append(p)
    return out


# ── 3. summary charts from analyze.py's summary.json ─────────────────────────

def load_summary(study):
    with open(os.path.join(C.RESULTS_DIR, study, '_summary', 'summary.json')) as f:
        return json.load(f)


def config_dotplot(study, configs, out_dir, name, which='truth', labels=None, title=None):
    """Primary metric per configuration: one dot per (subject, fold) plus
    the mean; configurations sorted as given."""
    s = load_summary(study)
    tab = s[which]
    subs = s['subjects']
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(configs) + 1.5))
    marks = {'S1toS2': 'o', 'S2toS1': 's'}
    for yi, c in enumerate(configs[::-1]):
        vals = []
        for si, sub in enumerate(subs):
            for p in ('S1toS2', 'S2toS1'):
                x = tab.get(f'{sub}|{p}|{c}')
                if x:
                    ax.plot(x['primary'], yi + (si - 1) * 0.12, marks[p], color=SERIES[si], ms=5, alpha=0.85,
                            label=f'{sub} {p}' if yi == 0 else None)
                    vals.append(x['primary'])
        if vals:
            ax.plot([np.mean(vals)] * 2, [yi - 0.32, yi + 0.32], color=INK, lw=2.2,
                    label='mean' if yi == 0 else None)
            ax.text(1.02, yi, f'{np.mean(vals):+.2f}', va='center', fontsize=8, color=INK2,
                    transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels((labels or configs)[::-1], fontsize=9)
    ax.set_xlabel('mean per-angle R² vs reference truth (each angle floored at −1)')
    lo = min([tab[k]['primary'] for k in tab if k.split('|')[2] in configs and k.split('|')[1] != 'pooled'] + [0])
    ax.set_xlim(max(-1.05, lo - 0.05), 1.0)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.legend(fontsize=8, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.06 if len(configs) > 12 else -0.2))
    if title:
        ax.set_title(title, loc='left')
    fig.tight_layout()
    return save(fig, out_dir, name)


def angle_heatmap(study, configs, out_dir, name, which='truth', labels=None):
    """Per-angle seed-mean R² (rows = configurations, columns = subject/fold/angle)."""
    s = load_summary(study)
    tab = s[which]
    cols, M = [], []
    for sub in s['subjects']:
        for p in ('S1toS2', 'S2toS1'):
            x = tab.get(f'{sub}|{p}|{configs[0]}')
            if x:
                cols += [f"{sub} {p[-2:]} {float(k.split(':')[1]):+.2f}" for k in x['angles']]
    for c in configs:
        row = []
        for sub in s['subjects']:
            for p in ('S1toS2', 'S2toS1'):
                x = tab.get(f'{sub}|{p}|{c}')
                row += (x['r2'] if x else [np.nan] * 8)
        M.append(row[:len(cols)])
    M = np.maximum(np.array(M, float), -1)
    fig, ax = plt.subplots(figsize=(0.32 * len(cols) + 3, 0.42 * len(configs) + 2))
    im = ax.imshow(M, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
    ax.set_yticks(range(len(configs))); ax.set_yticklabels(labels or configs, fontsize=8)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90, fontsize=6.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f'{M[i, j]:.1f}', ha='center', va='center', fontsize=5.5,
                        color='white' if abs(M[i, j]) > 0.6 else INK)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.02); cb.set_label('R² (floored at −1)')
    ax.set_title('Per-angle R² vs reference truth (mean of 3 seeds); columns: subject, test session, angle', loc='left')
    fig.tight_layout()
    return save(fig, out_dir, name)


def calibration_plot(study, configs, out_dir, name, labels=None):
    s = load_summary(study)['calibration_truth']
    fig, ax = plt.subplots(figsize=(7, 4))
    for ci, c in enumerate(configs):
        if c not in s:
            continue
        ks = sorted(int(k) for k in s[c] if k not in ('raw', 'n_units'))
        ax.plot([0] + ks, [s[c]['raw']] + [s[c][str(k)] for k in ks], 'o-', color=SERIES[ci % 5],
                label=(labels or configs)[ci])
    ax.set_xlabel('labelled calibration data per test trial (s)  [0 = none]')
    ax.set_ylabel('mean per-trial R² (t ≥ 30 s)')
    ax.legend(fontsize=8)
    ax.set_title('Effect of a short labelled calibration (linear a·pred + b)', loc='left')
    fig.tight_layout()
    return save(fig, out_dir, name)


def _cells(tab, subjects, config, key, protocols=('S1toS2', 'S2toS1')):
    return [tab[f'{s}|{p}|{config}'][key] for s in subjects for p in protocols if f'{s}|{p}|{config}' in tab]


def rest_ablation_plot(study, configs, out_dir, name, labels=None):
    """Relaxed-window ablation: R² on all windows, R² on active windows only,
    and the phantom torque predicted at rest.  Two panels (R² and Nm), one
    bar per configuration, mean over subjects x folds, dots = subject x fold."""
    s = load_summary(study)
    tab, subs = s['truth'], s['subjects']
    labels = labels or configs
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, title, unit in ((axs[0], 'primary', 'R², all windows', 'R²'),
                                 (axs[1], 'r2_active', 'R², active windows only', 'R²'),
                                 (axs[2], 'false_rest', 'Phantom torque at rest', 'mean |prediction| (Nm)')):
        for i, c in enumerate(configs):
            v = _cells(tab, subs, c, key)
            if not v:
                continue
            ax.bar(i, np.mean(v), color=SERIES[i % 5], width=0.65)
            ax.plot(np.full(len(v), i) + np.linspace(-0.18, 0.18, len(v)), v, 'o', color=INK, ms=3)
            ax.text(i, max(v) + (0.008 if unit == 'R²' else 0.015), f'mean {np.mean(v):.2f}', ha='center',
                    va='bottom', fontsize=8, color=INK, fontweight='bold')
        ax.set_xticks(range(len(configs))); ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
        ax.set_title(title, loc='left'); ax.set_ylabel(unit)
        if unit == 'R²':
            ax.set_ylim(0.5, 1.03)
    fig.tight_layout()
    return save(fig, out_dir, name)


def pooled_vs_cross_plot(study, configs, out_dir, name, labels=None):
    """Same test windows (last 20 % of every trial), models trained on both
    sessions (pooled) vs on the other session only (cross-session)."""
    s = load_summary(study)['pooled_vs_cross_last20']
    configs = [c for c in configs if c in s]
    labels = labels or configs
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(configs))
    ax.bar(x - 0.18, [s[c]['cross'] for c in configs], 0.34, color=OLD, label='trained on the OTHER session only')
    ax.bar(x + 0.18, [s[c]['pooled'] for c in configs], 0.34, color=NEW, label='trained on BOTH sessions (pooled)')
    for i, c in enumerate(configs):
        ax.text(i - 0.18, s[c]['cross'], f"{s[c]['cross']:.2f}", ha='center', va='bottom', fontsize=8)
        ax.text(i + 0.18, s[c]['pooled'], f"{s[c]['pooled']:.2f}", ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('mean per-angle R² (floored at −1)')
    ax.set_title('Same test windows (last 20 % of each trial), reference truth', loc='left')
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    return save(fig, out_dir, name)


def subject_bars(study, configs, subjects, out_dir, name, labels=None, key='primary', ylabel=None, title=None):
    """Per-subject bars (mean over the two folds), one group per subject."""
    s = load_summary(study)
    tab = s['truth']
    labels = labels or configs
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.8 / len(configs)
    for ci, c in enumerate(configs):
        for si, sub in enumerate(subjects):
            v = _cells(tab, [sub], c, key)
            if v:
                ax.bar(si + (ci - (len(configs) - 1) / 2) * w, np.mean(v), w * 0.92, color=SERIES[ci % 5],
                       label=labels[ci] if si == 0 else None)
                ax.text(si + (ci - (len(configs) - 1) / 2) * w, np.mean(v), f'{np.mean(v):.2f}', ha='center',
                        va='bottom' if np.mean(v) >= 0 else 'top', fontsize=7)
    ax.set_xticks(range(len(subjects))); ax.set_xticklabels(subjects)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel(ylabel or key)
    if title:
        ax.set_title(title, loc='left')
    ax.legend(fontsize=8, ncol=len(configs), loc='upper center', bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    return save(fig, out_dir, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--study', default='screen')
    ap.add_argument('--out', default=os.path.join('notes', 'report2', 'figures'))
    ap.add_argument('--only', default=None)
    a = ap.parse_args()
    setup()
    if a.only in (None, 'truth'):
        for s in C.STUDY_SUBJECTS:
            truth_inspection(s, a.out)
    if a.only in (None, 'trials'):        # reference-truth trial plots for every seed-0 run
        for f in sorted(glob.glob(os.path.join(C.RESULTS_DIR, a.study, '*', '*', '*', 'fast_seed0.npz'))):
            full_trials_vs_truth(f)


if __name__ == '__main__':
    main()
