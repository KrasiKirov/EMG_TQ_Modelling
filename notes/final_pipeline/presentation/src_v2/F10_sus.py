"""F10_sus: are the new results too good to be true?

Uses existing results only (no training).  Per test trial (subject x direction x
angle) of the NEW model: R2, error (RMSE), torque swing (std of the shared
truth), R2 on pushing parts only, phantom torque.  Compares the strong trials
(R2 >= 0.98) with the OLD pipeline (own target and shared truth) and with the
ridge baseline (screen E27_ridge, main subjects).  Figure: sus_r2_vs_swing.png
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.environ['WT'])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WT = os.environ['WT']
R = os.path.join(WT, 'ML', 'research', 'results')
OUT = os.path.join(WT, 'notes', 'final_pipeline', 'presentation', 'figures_v2')
os.makedirs(OUT, exist_ok=True)

MAIN, EXTRA = ['YES', 'JM', 'EG'], ['IES', 'CH', 'HM']
PROTS = ['S1toS2', 'S2toS1']
MINUS = '−'


def fm(x, nd=2):
    return f'{x:.{nd}f}'.replace('-', MINUS)


def src(subj, kind):
    """(study, config, seeds) for 'new', 'old', 'ridge'."""
    if kind == 'new':
        if subj in MAIN:
            return 'screen', 'E22_restanchor_final', [0, 1, 2]
        return 'confirm', ('E32_restanchor_noagc_moe' if subj == 'CH' else 'E22_restanchor_final'), [0]
    if kind == 'old':
        return ('screen', 'E01_old', [0, 1, 2]) if subj in MAIN else ('confirm', 'E01_old', [0])
    if kind == 'ridge':
        return ('screen', 'E27_ridge', [0, 1, 2]) if subj in MAIN else None


def load_per_angle(subj, prot, kind, truth=True):
    s = src(subj, kind)
    if s is None:
        return None
    study, cfg, seeds = s
    out = []
    for k in seeds:
        f = os.path.join(R, study, subj, prot, cfg, f'fast_seed{k}' + ('_truth.json' if truth else '.json'))
        if not os.path.exists(f):
            continue
        out.append(json.load(open(f))['per_angle'])
    return out


FIELDS = ['r2', 'rmse', 'std_true', 'r2_active', 'false_rest', 'rest_frac']


def seed_mean(per_list, key, fields):
    d = {}
    for f in fields:
        v = [p[key].get(f, np.nan) for p in per_list if key in p]
        d[f] = float(np.nanmean(v)) if len(v) else np.nan
    d['n_seeds'] = sum(key in p for p in per_list)
    return d


rows = []
for subj in MAIN + EXTRA:
    for prot in PROTS:
        new = load_per_angle(subj, prot, 'new')
        old_t = load_per_angle(subj, prot, 'old')
        old_own = load_per_angle(subj, prot, 'old', truth=False)
        rid = load_per_angle(subj, prot, 'ridge')
        keys = sorted(new[0].keys(), key=lambda s: float(s[3:]))
        for i, key in enumerate(keys):
            r = dict(subj=subj, prot=prot, key=key, angle=float(key[3:]), rank=i)
            r.update({f'new_{f}': v for f, v in seed_mean(new, key, FIELDS).items()})
            r.update({f'oldT_{f}': v for f, v in seed_mean(old_t, key, ['r2', 'rmse', 'r2_active', 'false_rest']).items()})
            r.update({f'oldOwn_{f}': v for f, v in seed_mean(old_own, key, ['r2', 'rmse', 'std_true']).items()})
            if rid:
                r.update({f'rid_{f}': v for f, v in seed_mean(rid, key, ['r2', 'rmse', 'r2_active']).items()})
                r['rid_r2_seeds'] = [p[key]['r2'] for p in rid]
            # spread of new R2 over seeds (main subjects)
            r['new_r2_seeds'] = [p[key]['r2'] for p in new]
            rows.append(r)

N = len(rows)
print(f'trials: {N}')

# ── spot check: recompute one truth-json entry from the npz + shared truth ──
from ML.research import truth as T
tr = T.subject_truth('YES')
z = np.load(os.path.join(R, 'screen', 'YES', 'S1toS2', 'E22_restanchor_final', 'fast_seed0.npz'))
tri = int(z['trial'][0]); m = z['trial'] == tri
k = np.rint((z['t'][m].astype(float) - tr[tri]['t0']) * 100).astype(int)
y = tr[tri]['truth'][k].astype(float); p = z['y_pred'][m].astype(float)
r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
key0 = f"S2:{float(z['angle'][m][0]):+.3f}"
js = json.load(open(os.path.join(R, 'screen', 'YES', 'S1toS2', 'E22_restanchor_final', 'fast_seed0_truth.json')))['per_angle'][key0]
print(f'spot check YES S1toS2 seed0 {key0}: recomputed R2 {r2:.4f} vs json {js["r2"]:.4f}; '
      f'rmse {np.sqrt(np.mean((y-p)**2)):.4f} vs {js["rmse"]:.4f}')
# also: does the new model's OWN test target equal the shared truth? (restanchor test target = truth)
yt = z['y_true'][m].astype(float)
print(f'  max |own target - shared truth| on that trial: {np.max(np.abs(yt - y)):.2e} Nm')

# ── (a) table of every trial ────────────────────────────────────────────────
print('\n(a) per trial, NEW model (main: mean of 3 runs; extra: 1 run)')
print(f'{"subj":4s} {"dir":6s} {"angle":>14s} {"R2":>6s} {"err":>5s} {"swing":>6s} {"R2push":>7s} {"phantom":>7s} {"rest%":>5s}')
for r in rows:
    print(f"{r['subj']:4s} {r['prot']:6s} {r['key']:>8s} ({np.degrees(r['angle']):+4.0f}°) {r['new_r2']:6.3f} {r['new_rmse']:5.2f} "
          f"{r['new_std_true']:6.2f} {r['new_r2_active']:7.3f} {r['new_false_rest']:7.3f} {100*r['new_rest_frac']:5.0f}")

# ── (b) strong trials ───────────────────────────────────────────────────────
strong = [r for r in rows if r['new_r2'] >= 0.98]
ns = len(strong)
print(f'\n(b) trials with new R2 >= 0.98: {ns} of {N}')
by_subj = {s: sum(r['subj'] == s for r in strong) for s in MAIN + EXTRA}
print('   per subject:', by_subj)
by_rank = {}
for r in strong:
    by_rank[r['rank']] = by_rank.get(r['rank'], 0) + 1
print('   by angle rank (0 = most toes-down):', dict(sorted(by_rank.items())))
e = np.array([r['new_rmse'] for r in strong]); sw = np.array([r['new_std_true'] for r in strong])
print(f'   error range {e.min():.2f}-{e.max():.2f} Nm (median {np.median(e):.2f}); swing range {sw.min():.2f}-{sw.max():.2f} Nm (median {np.median(sw):.2f})')
print(f'   error as % of swing: {100*np.min(e/sw):.0f}-{100*np.max(e/sw):.0f}% (median {100*np.median(e/sw):.0f}%)')
ra = np.array([r['new_r2_active'] for r in strong]); ph = np.array([r['new_false_rest'] for r in strong])
print(f'   pushing-only R2 range {ra.min():.3f}-{ra.max():.3f} (median {np.median(ra):.3f}); phantom torque {ph.min():.2f}-{ph.max():.2f} Nm (median {np.median(ph):.2f})')
weak = [r for r in rows if r['new_r2'] < 0.98]
sww = np.array([r['new_std_true'] for r in weak])
print(f'   other {len(weak)} trials: swing median {np.median(sww):.2f} Nm (range {sww.min():.2f}-{sww.max():.2f})')
# R2 vs swing bins
allsw = np.array([r['new_std_true'] for r in rows]); allr2 = np.array([r['new_r2'] for r in rows])
alle = np.array([r['new_rmse'] for r in rows])
for lo, hi in [(0, 1), (1, 3), (3, 6), (6, 100)]:
    mm = (allsw >= lo) & (allsw < hi)
    if mm.any():
        print(f'   swing {lo}-{hi} Nm: n={mm.sum():2d}  median R2 {np.median(allr2[mm]):+.3f}  median error {np.median(alle[mm]):.2f} Nm')
rho = np.corrcoef(np.argsort(np.argsort(allsw)), np.argsort(np.argsort(allr2)))[0, 1]
print(f'   rank correlation swing vs R2 (all {N}): {rho:.2f}')

# seed spread of new R2 on strong main-subject trials
sp = [np.ptp(r['new_r2_seeds']) for r in strong if len(r['new_r2_seeds']) == 3]
print(f'   spread of R2 over the 3 runs on strong main-subject trials: max {max(sp):.4f}, median {np.median(sp):.4f}')

# ── (c) old pipeline at the same trials ────────────────────────────────────
oo = np.array([r['oldOwn_r2'] for r in strong]); ot = np.array([r['oldT_r2'] for r in strong])
oe = np.array([r['oldT_rmse'] for r in strong]); oe_own = np.array([r['oldOwn_rmse'] for r in strong])
print(f'\n(c) OLD pipeline at those {ns} trials')
print(f'   own target (last semester scoring): R2 {oo.min():.3f}-{oo.max():.3f}, median {np.median(oo):.3f}; '
      f'>=0.97 in {np.sum(oo >= 0.97)}/{ns}, >=0.98 in {np.sum(oo >= 0.98)}/{ns}; error {oe_own.min():.2f}-{oe_own.max():.2f} Nm')
print(f'   shared truth: R2 {ot.min():.3f}-{ot.max():.3f}, median {np.median(ot):.3f}; >=0.97 in {np.sum(ot >= 0.97)}/{ns}; '
      f'error {oe.min():.2f}-{oe.max():.2f} Nm (median {np.median(oe):.2f})')
print(f'   new on same trials: R2 median {np.median([r["new_r2"] for r in strong]):.3f}, error median {np.median(e):.2f} Nm')
# last semester overall: how many trials had own-target R2 >= 0.97 / 0.98
ao = np.array([r['oldOwn_r2'] for r in rows])
print(f'   old own-target R2 over all {N} trials: >=0.97 in {np.sum(ao >= 0.97)}, >=0.98 in {np.sum(ao >= 0.98)}')

# ── (d) ridge (simple formula) at the strong main-subject trials ────────────
sm = [r for r in strong if 'rid_r2' in r]
rr = np.array([r['rid_r2'] for r in sm]); re_ = np.array([r['rid_rmse'] for r in sm])
same = all(np.ptp(r['rid_r2_seeds']) < 1e-9 for r in rows if 'rid_r2_seeds' in r)
print(f'\n(d) ridge at the {len(sm)} strong main-subject trials (seeds 0,1,2 present; identical across seeds: {same})')
print(f'   R2 {rr.min():.3f}-{rr.max():.3f}, median {np.median(rr):.3f}; >=0.95 in {np.sum(rr >= 0.95)}, >=0.97 in {np.sum(rr >= 0.97)}; '
      f'error {re_.min():.2f}-{re_.max():.2f} Nm (median {np.median(re_):.2f})')
print(f'   new at same {len(sm)}: R2 median {np.median([r["new_r2"] for r in sm]):.3f}, error median {np.median([r["new_rmse"] for r in sm]):.2f}')
for r in sm:
    print(f"   {r['subj']} {r['prot']} {r['key']}: new {r['new_r2']:.3f}/{r['new_rmse']:.2f}  ridge {r['rid_r2']:.3f}/{r['rid_rmse']:.2f}  "
          f"old(own) {r['oldOwn_r2']:.3f}  old(truth) {r['oldT_r2']:.3f}  swing {r['new_std_true']:.2f}")

# ── (e) pushing-only vs all-sample average R2 ──────────────────────────────
def avg(rs, f):
    return float(np.mean([max(r[f], -1.0) for r in rs]))


print('\n(e) average R2 (floored at -1, averaged over angles)')
for name, rs in [('main subjects, both directions', [r for r in rows if r['subj'] in MAIN]),
                 ('main subjects, session 1 -> 2', [r for r in rows if r['subj'] in MAIN and r['prot'] == 'S1toS2']),
                 ('all 6 subjects, both directions', rows),
                 ('all 6 subjects, session 1 -> 2', [r for r in rows if r['prot'] == 'S1toS2'])]:
    print(f'   {name:34s}: all samples {avg(rs, "new_r2"):.3f}  pushing only {avg(rs, "new_r2_active"):.3f}  '
          f'(old vs truth: {avg(rs, "oldT_r2"):.3f} / {avg(rs, "oldT_r2_active"):.3f})')
print(f'   strong trials: all samples median {np.median([r["new_r2"] for r in strong]):.3f}, pushing only median {np.median(ra):.3f}')

# ── figure ──────────────────────────────────────────────────────────────────
def fs(x, nd=2):
    """signed number with a true minus sign (+0.14 / −0.49)"""
    return f'{x:+.{nd}f}'.replace('-', MINUS)


plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13, 'xtick.labelsize': 11.5,
                     'ytick.labelsize': 11.5, 'axes.titlesize': 14, 'axes.titleweight': 'bold',
                     'axes.unicode_minus': True})
# subject colours: none of them is the deck's OLD blue / NEW orange; CH and HM clearly apart
SCOL = {'YES': '#6a3d9a', 'JM': '#17a2a2', 'EG': '#c4a000', 'IES': '#d6457a', 'CH': '#7a5230', 'HM': '#66a61e'}
GRID, SEC, BAD, INK = '#e6e5e1', '#52514e', '#c0392b', '#1b1b1b'
MK = {'S1toS2': 'o', 'S2toS1': '^'}
DIRTXT = {'S1toS2': 'learned session 1 → tested session 2', 'S2toS1': 'learned session 2 → tested session 1'}

fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.4))
YMIN = 0.6            # y-axis of the left panel starts here; anything lower is drawn as an open ▽ under the axis
Y_OFF = YMIN - 0.018  # where the off-scale marker sits
XMAX = max(allsw) * 2.0
for ax in axs:
    ax.set_xscale('log')
    ax.grid(True, color=GRID, lw=0.8, which='major')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_xlabel('How much the voluntary torque moves in the trial (std, Nm)')
    ax.set_xticks([0.3, 1, 3, 10, 30]); ax.set_xticklabels(['0.3', '1', '3', '10', '30'])
    ax.set_xlim(0.2, XMAX)

clipped = []
for r in rows:
    c = SCOL[r['subj']]; mk = MK[r['prot']]
    y = r['new_r2']
    if y < YMIN:
        clipped.append(r)
        axs[0].scatter(r['new_std_true'], Y_OFF, marker='v', s=50, facecolor='white', edgecolor=c, lw=1.5, zorder=3, clip_on=False)
    else:
        axs[0].scatter(r['new_std_true'], y, marker=mk, s=42, color=c, alpha=0.85, lw=0, zorder=3)
    axs[1].scatter(r['new_std_true'], r['new_rmse'], marker=mk, s=42, color=c, alpha=0.85, lw=0, zorder=3)

# left panel: R2
a0 = axs[0]
a0.axhspan(0.98, 1.0, color='#f1f0ec', zorder=0, lw=0)
a0.axhline(0.98, color=SEC, lw=1, ls='--', zorder=1)
a0.set_ylim(YMIN - 0.035, 1.022)
a0.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0]); a0.set_yticklabels(['0.6', '0.7', '0.8', '0.9', '1.0'])
a0.spines['left'].set_bounds(YMIN, 1.0)
a0.set_ylabel('R² of the new model')
a0.set_title('R² is high where the torque moves a lot...', loc='left')
a0.text(XMAX * 0.95, 0.627, f'Shaded band (R² ≥ 0.98):\n{ns} of {N} trials', color=SEC, fontsize=11, va='bottom', ha='right')
# label the off-scale trial(s) directly
for r in clipped:
    a0.text(r['new_std_true'] * 1.17, Y_OFF,
            f"{r['subj']} {fs(r['angle'])} rad ({fs(np.degrees(r['angle']), 0)}°), toes down: R² {r['new_r2']:.2f} (off scale)\n"
            f"error only {r['new_rmse']:.1f} Nm, but the torque swing is only {r['new_std_true']:.1f} Nm",
            color=SEC, fontsize=10.5, va='center', ha='left')

# right panel: error in Nm
a1 = axs[1]
a1.set_ylabel('Error (Nm)')
a1.set_ylim(0, None)
a1.text(0.21, 3.0, f'Trials with R² ≥ 0.98: error {e.min():.1f}–{e.max():.1f} Nm\n({100*np.min(e/sw):.0f}–{100*np.max(e/sw):.0f}% of the torque swing)',
        color=SEC, fontsize=11, va='top', ha='left')
a1.set_title('...but the error in Nm is not zero there', loc='left')

# example: the strong main-subject trial with the largest swing; black ring in both panels
ex = max([r for r in strong if r['subj'] in MAIN], key=lambda r: r['new_std_true'])
tdir = 'toes up' if ex['angle'] > 0 else 'toes down'
lab = (f"{ex['subj']}, {fs(ex['angle'])} rad ({fs(np.degrees(ex['angle']), 0)}°), {tdir}\n"
       f"{DIRTXT[ex['prot']]}\n"
       f"R² {ex['new_r2']:.3f}, error {ex['new_rmse']:.1f} Nm")
for ax, yv in [(a0, ex['new_r2']), (a1, ex['new_rmse'])]:
    ax.scatter(ex['new_std_true'], yv, s=170, facecolor='none', edgecolor=INK, lw=1.6, zorder=4)
a0.annotate(lab, xy=(ex['new_std_true'], ex['new_r2']), xytext=(1.45, 0.735), va='bottom', ha='left',
            fontsize=11, color=INK,
            arrowprops=dict(arrowstyle='->', color=INK, lw=1.1, shrinkA=2, shrinkB=7, relpos=(0.93, 1.0)))
a1.annotate(f"same trial:\nswing {ex['new_std_true']:.0f} Nm,\nerror {ex['new_rmse']:.1f} Nm",
            xy=(ex['new_std_true'], ex['new_rmse']), xytext=(XMAX * 0.93, 0.55), va='bottom', ha='right',
            fontsize=11, color=INK,
            arrowprops=dict(arrowstyle='->', color=INK, lw=1.1, shrinkA=2, shrinkB=7, relpos=(0.35, 1.0)))

from matplotlib.lines import Line2D
h = [Line2D([], [], marker='o', ls='', color=SCOL[s], ms=8, label=s) for s in MAIN + EXTRA]
h += [Line2D([], [], marker='o', ls='', color=SEC, ms=7, label=DIRTXT['S1toS2']),
      Line2D([], [], marker='^', ls='', color=SEC, ms=7, label=DIRTXT['S2toS1'])]
fig.legend(handles=h, loc='lower center', ncol=8, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.06),
           handletextpad=0.3, columnspacing=1.0)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fp = os.path.join(OUT, 'sus_r2_vs_swing.png')
fig.savefig(fp, dpi=200, bbox_inches='tight', facecolor='white')
print('\nsaved', fp, ' example:', ex['subj'], ex['prot'], ex['key'], f"{ex['new_r2']:.4f} {ex['new_rmse']:.3f} swing {ex['new_std_true']:.2f}")
print('off-scale:', [(r['subj'], r['prot'], r['key'], round(r['new_r2'], 3)) for r in clipped])
print('lowest R2 values:', sorted(round(r['new_r2'], 3) for r in rows)[:4])
