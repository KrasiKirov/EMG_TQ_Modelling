"""Figures for the weekly-update deck, from the real recordings.

Usage (repository root on PYTHONPATH):
    python make_figures.py <data dir> <overrides.json> <emg_agc.py> <output dir>
"""
import importlib.util
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ML.preprocessing.calibration import fit_passive, fit_passive_session
from ML.preprocessing.pipeline import PipelineConfig, process_block
from ML.preprocessing.trial_manifest import load_inventory

data_dir, overrides, agc_path, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
spec = importlib.util.spec_from_file_location('emg_agc', agc_path)
agc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agc)

INK, MUTED, GRID, BG = '#1B2430', '#5B6573', '#D9DDE2', '#F7F5F0'
S1, S2, ACTIVE = '#2F6DB5', '#C2410C', '#1F7A72'
MUSCLE = {'gm': '#1F4E79', 'gl': '#2F6DB5', 'sol': '#6FA8DC', 'ta': '#C2410C'}
NAMES = {'gm': 'GM · calf', 'gl': 'GL · calf', 'sol': 'SOL · calf', 'ta': 'TA · shin'}
plt.rcParams.update({'font.family': 'Arial', 'font.size': 19, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.facecolor': BG, 'axes.facecolor': BG,
                     'savefig.facecolor': BG, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8})
CFG = PipelineConfig(window=50)


def subject(name):
    trials, manifest = load_inventory(name, data_dir, overrides)
    return trials, manifest


def active(manifest, session):
    return [(i, r) for i, r in enumerate(manifest['trials'])
            if r['kind'] == 'active' and r['status'] == 'included' and r['session'] == session]


def signals(trials, i, record):
    start, stop = record['valid_intervals'][0]
    s, index = process_block(trials[i], start, stop, CFG)
    return s, (index - start) / 1000.0


def nearest(records, angle):
    return min(records, key=lambda ir: abs(ir[1]['mean_position_rad'] - angle))


def save(fig, name):
    fig.savefig(os.path.join(out, name), dpi=200, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    print('wrote', name)


# 1. What the model sees: one session-1 push trial, four muscles + torque, one 0.5 s window.
trials, manifest = subject('HM')
i, rec = nearest(active(manifest, 'test'), -0.36)
s, t = signals(trials, i, rec)
keep = (t >= 20) & (t <= 44)
fig, axes = plt.subplots(5, 1, figsize=(10, 6.4), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1, 1.8]})
for k, (ax, m) in enumerate(zip(axes, ['gm', 'gl', 'sol', 'ta'])):
    env = s[:, k] / np.percentile(s[:, k], 99.5)
    ax.plot(t[keep], env[keep], color=MUSCLE[m], lw=2)
    ax.set_yticks([]); ax.set_ylim(0, 1.15); ax.grid(False)
    ax.text(-0.01, 0.5, NAMES[m], transform=ax.transAxes, ha='right', va='center', color=MUSCLE[m], fontsize=18)
    ax.spines['left'].set_visible(False)
axes[-1].plot(t[keep], s[keep, 5], color=INK, lw=2.2)
axes[-1].set_ylabel('Torque (Nm)')
axes[-1].set_xlabel('Time in the trial (s)')
for ax in axes:
    ax.axvspan(33.5, 34.0, color=ACTIVE, alpha=0.18, lw=0)
axes[-1].plot([34.0], [s[np.argmin(abs(t - 34.0)), 5]], 'o', color=ACTIVE, ms=11, zorder=5)
save(fig, 'fig_data.png')
print(f'  HM trial {rec["trial"]} at {rec["mean_position_rad"]:+.3f} rad')

# 2. Passive maps of one person, session 1 vs session 2 (EG: both maps from recorded still holds).
trials, manifest = subject('EG')
p1, p2 = fit_passive(trials, manifest, data_dir), fit_passive_session(trials, manifest, 'retest')
fig, ax = plt.subplots(figsize=(8.6, 6))
for cal, color, label in [(p1, S1, 'Session 1 map'), (p2, S2, 'Session 2 map (retest)')]:
    ax.plot(cal.positions, cal.torques, '-o', color=color, lw=2.6, ms=9, label=label)
ax.axhline(0, color=MUTED, lw=1)
ax.set_xlabel('Ankle angle (rad)   ← toes down · toes up →')
ax.set_ylabel('Torque with muscles relaxed (Nm)')
ax.legend(frameon=False, loc='lower left')
example = -0.040
ax.axvline(example, color=MUTED, lw=1.4, ls=':')
save(fig, 'fig_passive_map.png')
print('  EG maps s1', np.round(p1.positions, 3), np.round(p1.torques, 2), ' s2', np.round(p2.positions, 3), np.round(p2.torques, 2))

# 3. How the target is computed for a session-2 trial (EG, -0.04 rad); top panel zoomed on the relaxed level.
i, rec = nearest(active(manifest, 'retest'), example)
s, t = signals(trials, i, rec)
keep = (t >= 20) & (t <= 50)
angle = float(np.median(s[:, 4]))
pas2, pas1 = float(p2.predict([angle])[0]), float(p1.predict([angle])[0])
measured, tt = s[keep, 5], t[keep]
zoom = (tt >= 40.0) & (tt <= 44.0)
rest = float(np.median(measured[zoom & (measured > max(pas1, pas2) - 1.5)]))
fig = plt.figure(figsize=(11.5, 7.6))
gs = fig.add_gridspec(2, 2, width_ratios=[2.1, 1], hspace=0.62, wspace=0.32)
a1, a2, a3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
a1.plot(tt, measured, color=INK, lw=1.8)
a1.axvspan(40.0, 44.0, color=MUTED, alpha=0.15, lw=0)
a1.text(42.0, np.min(measured) * 0.98, 'zoom', ha='center', va='bottom', fontsize=16, color=MUTED)
a1.set_ylabel('Torque (Nm)'); a1.set_xlabel('Time (s)')
a1.set_title('1 · Measured torque (pushes go down)', loc='left', fontsize=19, color=INK)
a2.plot(tt[zoom], measured[zoom], color=INK, lw=2.2)
a2.axhline(pas2, color=S2, lw=2.8, ls='--'); a2.axhline(pas1, color=S1, lw=2.6, ls=':')
a2.set_xlim(40.0, 44.0); a2.set_ylim(pas1 - 0.7, rest + 0.6)
a2.set_yticks([round(pas1, 1), round(pas2, 1), round(rest, 1)])
a2.set_xlabel('Time (s)')
for y, color, label in [(rest, INK, 'relaxed'), (pas2, S2, 'session 2 map'), (pas1, S1, 'session 1 map')]:
    a2.text(44.15, y, f'{label}', va='center', fontsize=16, color=color, clip_on=False)
a2.set_title('2 · Zoom: muscles relaxed', loc='left', fontsize=19, color=INK)
target = measured - pas2
a3.fill_between(tt, 0, target, color=ACTIVE, alpha=0.25, lw=0)
a3.plot(tt, target, color=ACTIVE, lw=2)
a3.axhline(0, color=MUTED, lw=1)
a3.set_title('3 · What the model is scored on = measured − session 2 resting torque', loc='left', fontsize=19, color=INK)
a3.set_ylabel('Active (Nm)'); a3.set_xlabel('Time in the session-2 trial (s)')
save(fig, 'fig_target.png')
print(f'  EG trial {rec["trial"]} at {angle:+.3f} rad: passive s2 {pas2:+.2f}, s1 {pas1:+.2f}, '
      f'active SD {np.nanstd(target):.2f} Nm, measured {np.percentile(s[keep, 5], 5):.1f}..{np.percentile(s[keep, 5], 95):.1f}')

# 4. How auto-gain works on one session-2 trial (JM, -0.36 rad, soleus).
trials, manifest = subject('JM')
train_envs, scale = [], np.zeros(4)
for i, rec in active(manifest, 'test'):
    s, _ = signals(trials, i, rec)
    part = s[: int(len(s) * 0.8), :4]
    train_envs.append(part); scale = np.maximum(scale, part.max(axis=0))
prior_lo, prior_hi = agc.agc_priors(train_envs)
i, rec = nearest(active(manifest, 'retest'), -0.36)
s, t = signals(trials, i, rec)
env = s[:, :4].astype(np.float64)
lo, hi = agc._running_lo_hi(env, prior_lo, prior_hi)
out_env = agc.apply_agc(env, prior_lo, prior_hi)[0]
k = 2  # soleus
fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
top.plot(t, env[:, k] / scale[k], color=MUSCLE['sol'], lw=1.8, label='Soleus activity')
top.plot(t, hi[:, k] / scale[k], color=INK, lw=2.4, label='Strong level (95th pct, past 45 s)')
top.plot(t, lo[:, k] / scale[k], color=MUTED, lw=2.4, ls='--', label='Quiet level (10th pct, past 20 s)')
top.set_ylabel('Scaled by\nsession-1 max')
top.set_ylim(0, max(0.45, 1.15 * np.percentile(env[:, k] / scale[k], 99.5)))
top.legend(frameon=False, fontsize=16, loc='lower left', bbox_to_anchor=(0, 1.02), ncol=3)
bottom.plot(t, out_env[:, k], color=ACTIVE, lw=1.8)
bottom.axhline(1, color=MUTED, lw=1.2, ls=':')
bottom.set_ylabel('After\nauto-gain'); bottom.set_xlabel('Time in the session-2 trial (s)')
save(fig, 'fig_agc_how.png')
print(f'  JM trial {rec["trial"]} at {rec["mean_position_rad"]:+.3f} rad: soleus 95th pct before {np.percentile(env[:, k] / scale[k], 95):.2f}, '
      f'after {np.percentile(out_env[:, k], 95):.2f}')

# 5. Share of the training loss per angle, plain vs balanced (YES, session 1).
trials, manifest = subject('YES')
p1 = fit_passive(trials, manifest, data_dir)
rows = []
for i, rec in active(manifest, 'test'):
    s, _ = signals(trials, i, rec)
    part = s[: int(len(s) * 0.8)]
    y = part[:, 5] - p1.predict(part[:, 4])
    rows.append((rec['mean_position_rad'], np.var(y[np.isfinite(y)]), np.isfinite(y).sum()))
rows.sort()
angles = np.array([r[0] for r in rows]); var = np.array([r[1] for r in rows]); n = np.array([r[2] for r in rows])
plain = var * n / np.sum(var * n) * 100
balanced = n / n.sum() * 100
fig, ax = plt.subplots(figsize=(11, 5.6))
x = np.arange(len(angles))
ax.bar(x - 0.2, plain, 0.4, color=MUTED, label='Plain loss (last week)')
ax.bar(x + 0.2, balanced, 0.4, color=ACTIVE, label='Every angle counts the same (new)')
ax.set_axisbelow(True); ax.grid(axis='x', visible=False)
ax.set_xticks(x); ax.set_xticklabels([f'{a:+.2f}' for a in angles])
ax.set_xlabel('Ankle angle (rad)   ← toes down · toes up →')
ax.set_ylabel('Share of the training loss (%)')
ax.legend(frameon=False, loc='upper left')
for xi, v in zip(x[:2], plain[:2]):
    ax.text(xi - 0.2, v + 1.2, f'{v:.1f}%', ha='center', color=INK, fontsize=17)
save(fig, 'fig_balance.png')
print('  YES angles', np.round(angles, 3), 'torque SD', np.round(np.sqrt(var), 2), 'plain share %', np.round(plain, 1))
