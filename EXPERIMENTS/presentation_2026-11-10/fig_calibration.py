"""Calibration-line figure for the 10 Nov deck: one real session-2 trial (A0, YES, -0.381 rad, seed 0)."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXP, 'normalization'))
from calibration_test import correct, GAIN_LIMITS  # noqa: E402

BG, INK, SOFT, TEAL, ORANGE, GRID = '#F7F5F0', '#1B2430', '#4A5568', '#1F7A72', '#C2622D', '#DDD8CC'
SUBJ, ANGLE, SEED, CAL, SCORE = 'YES', -0.381, 0, 10.0, 30.0

z = np.load(os.path.join(EXP, 'results', 'A0_active_trainmax', SUBJ, 'S1toS2', f'seed{SEED}.npz'))
y, p, ang, t = (z[k].astype(float) for k in ('y_own_target', 'y_pred', 'angle', 't'))
m = np.isclose(ang, min(np.unique(ang), key=lambda a: abs(a - ANGLE)))
y, p, t = y[m], p[m], t[m] - t[m][0]
cal = t < CAL
a_raw, _ = np.polyfit(p[cal], y[cal], 1)
a = float(np.clip(a_raw, *GAIN_LIMITS)); b = float(np.mean(y[cal] - a * p[cal]))
pc = correct(p[cal], y[cal], p, 'gain+offset')
sc = t >= SCORE
mae_raw, mae_cal = np.mean(np.abs(p[sc] - y[sc])), np.mean(np.abs(pc[sc] - y[sc]))
print(f'a={a:.3f} (raw {a_raw:.3f}) b={b:.3f}  MAE from 30 s: {mae_raw:.3f} -> {mae_cal:.3f}')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 19, 'axes.edgecolor': SOFT, 'axes.labelcolor': INK,
                     'xtick.color': SOFT, 'ytick.color': SOFT, 'axes.spines.top': False, 'axes.spines.right': False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6.6), gridspec_kw=dict(width_ratios=[1, 2.1]), facecolor=BG)
for ax in (ax1, ax2):
    ax.set_facecolor(BG); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)

# (1) the calibration line, fitted on the first 10 s
ax1.scatter(p[cal][::3], y[cal][::3], s=22, color=TEAL, alpha=0.5, lw=0, label='first 10 s')
lo, hi = min(p[cal].min(), y[cal].min()) - 0.5, max(p[cal].max(), y[cal].max()) + 0.5
xx = np.array([lo, hi])
ax1.plot(xx, xx, ls='--', color=SOFT, lw=1.5, label='perfect')
ax1.plot(xx, a * xx + b, color=ORANGE, lw=3, label=f'line: {a:.2f} × ŷ {b:+.2f}')
ax1.set_xlim(lo, hi); ax1.set_ylim(lo, hi)
ax1.set_xlabel('LSTM prediction (Nm)'); ax1.set_ylabel('measured torque (Nm)')
ax1.set_title('1 · Fit a line on the first 10 s', loc='left', fontsize=21, color=INK, fontweight='bold')
ax1.legend(frameon=False, fontsize=16, loc='lower right')

# (2) the same trial over time
for x0, x1, c, lab in ((0, CAL, TEAL, 'calibration\n(10 s)'), (CAL, SCORE, SOFT, 'not scored'), (SCORE, t[-1], None, 'scored')):
    if c:
        ax2.axvspan(x0, x1, color=c, alpha=0.10 if c == TEAL else 0.06, lw=0)
    ax2.text((x0 + x1) / 2, 0.97, lab, transform=ax2.get_xaxis_transform(), ha='center', va='top', fontsize=16, color=SOFT)
ax2.plot(t, y, color=INK, lw=1.8, label='measured')
ax2.plot(t, p, color=ORANGE, lw=2.0, alpha=0.9, label=f'before (MAE {mae_raw:.2f} Nm)')
ax2.plot(t, pc, color=TEAL, lw=2.2, label=f'after calibration (MAE {mae_cal:.2f} Nm)')
ax2.set_xlim(0, t[-1]); ax2.set_ylim(min(y.min(), p.min()) - 0.3, max(y.max(), p.max()) + 1.6); ax2.set_xlabel('time in the session-2 trial (s)'); ax2.set_ylabel('active torque (Nm)')
ax2.set_title('2 · Apply the same line to the rest of the trial', loc='left', fontsize=21, color=INK, fontweight='bold')
ax2.legend(frameon=False, fontsize=16, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3)
fig.tight_layout(w_pad=3)
fig.savefig(os.path.join(HERE, 'fig_calibration.png'), dpi=110, facecolor=BG)
