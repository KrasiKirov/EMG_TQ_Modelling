"""Backup-slide figures: rest detection step by step, and what auto-gain does."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WT = os.environ['WT']
sys.path.insert(0, WT)
OUT = os.path.join(WT, 'notes', 'final_pipeline', 'presentation', 'figures')

INK, INK2, MUTED, GRID = '#1b1b1b', '#52514e', '#8a8984', '#e6e5e1'
OLD, NEW, GREEN = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID,
                     'axes.axisbelow': True, 'legend.frameon': False})
M = chr(0x2212)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', p)


def fig_rest_detect():
    """YES, session 2, -0.486 rad: the exact steps of truth.session_rest + baseline."""
    from ML.research.data import load_subject
    from ML.research import truth as T
    from ML.preprocessing import active_torque as at
    b = load_subject('YES')
    recs = [r for r in b['active'] if r['session'] == 'S2']
    emg_max = np.max(np.stack([r['env_max1k'] for r in recs]), axis=0)
    emg_max = np.where(emg_max > 0, emg_max, 1.0)
    envs = {r['idx']: r['env_raw'] / emg_max for r in recs}
    stats = at.emg_rest_stats(list(envs.values()))
    rec = min(recs, key=lambda r: abs(r['angle'] + 0.486))
    env = envs[rec['idx']]
    floor = np.percentile(env, 5.0, axis=0)
    score = ((env - floor) / stats['scale']).mean(axis=1)
    segs = at.detect_rest(env, stats)
    # same segments as the cached reference truth (sanity check)
    tr = T.subject_truth('YES')[rec['idx']]
    assert tr['n_rest_segs'] == len(segs), (tr['n_rest_segs'], len(segs))
    B = T.baseline(rec, segs)
    assert np.allclose(B, tr['B'], atol=1e-3)
    tc, med, n = at._seg_levels(rec['tq'], segs)
    t = rec['t'] - rec['t'][0]
    tq = rec['tq']
    keep = (t > 0.5) & (t < t[-1] - 0.5)

    fig, axs = plt.subplots(3, 1, figsize=(12, 9.6), sharex=True,
                            gridspec_kw={'height_ratios': [1, 1.1, 1], 'hspace': 0.42})
    for ax in axs:
        for s, e in segs:
            ax.axvspan(t[s], t[min(e, len(t) - 1)], color=GREEN, alpha=0.16, lw=0, zorder=0)
    ax = axs[0]
    ax.plot(t[keep], score[keep], color=INK2, lw=1.0)
    ax.axhline(0.03, color=GREEN, lw=1.8, ls='--')
    ax.axhline(0.06, color=NEW, lw=1.8, ls='--')
    tf = ax.get_yaxis_transform()
    ax.text(1.01, 0.012, 'enter "relaxed"\nbelow 0.03', transform=tf, va='center', fontsize=10.5, color=GREEN)
    ax.text(1.01, 0.09, 'leave "relaxed"\nabove 0.06', transform=tf, va='center', fontsize=10.5, color=NEW)
    ax.set_ylim(0, 0.25)
    ax.set_ylabel('muscle activity')
    ax.set_title('Step 1 ' + chr(0x2014) + ' EMG only: average activity of the 4 muscles (0 = quietest, 1 = the session'
                 ' range; cut at 0.25)', loc='left', fontsize=12.5, weight='bold', color=INK)
    ax = axs[1]
    ax.plot(t[keep], tq[keep], color=MUTED, lw=1.0, label='measured torque')
    ax.scatter(tc / 100.0, med, s=34, color=INK, zorder=4, label='median torque of each relaxed stretch')
    ax.plot(t[keep], B[keep], color=NEW, lw=2.6, zorder=5, label='smooth line through the dots (' + chr(0x03C3) + ' = 8 s) = passive B(t)')
    ax.set_ylim(7.0, 11.0)
    ax.set_ylabel('torque (Nm)')
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, -0.04), fontsize=10.5, ncol=3, handlelength=1.6)
    ax.set_title('Steps 2' + chr(0x2013) + '3 ' + chr(0x2014) + ' inside each green stretch nobody pushes, so the torque there is passive',
                 loc='left', fontsize=12.5, weight='bold', color=INK)
    ax = axs[2]
    ax.axhline(0, color=INK2, lw=1)
    ax.plot(t[keep], (tq - B)[keep], color=NEW, lw=1.3)
    ax.set_ylabel('voluntary (Nm)')
    ax.set_xlabel('time in the trial (s)')
    ax.set_ylim(-3.2, 1.2)
    ax.set_title('Step 4 ' + chr(0x2014) + ' voluntary torque = measured ' + M + ' B(t)   (0 when relaxed, negative = pushing toes down)',
                 loc='left', fontsize=12.5, weight='bold', color=INK)
    fig.text(0.125, 0.035, f'Subject YES, session 2, ankle at {M}0.486 rad. Green = {len(segs)} relaxed stretches '
             f'(at least 0.5 s; the first 0.3 s and the last 0.1 s of each are skipped).',
             fontsize=10.5, color=INK2)
    save(fig, 'rest_detection_steps.png')
    return len(segs), float(np.mean([e - s for s, e in segs]) / 100.0), float(B[keep].min()), float(B[keep].max())


def fig_agc():
    """JM, trained on S1, angle -0.645 / -0.644 rad: calf EMG vs voluntary torque, before / after auto-gain."""
    from ML.research.data import load_subject
    from ML.research import truth as T
    from ML.research.emg_norm import normalise
    b = load_subject('JM')
    train = [r for r in b['active'] if r['session'] == 'S1']
    base, _ = normalise('trainmax', b, train, b['active'])
    agc, _ = normalise('agc', b, train, b['active'])
    tr = T.subject_truth('JM')
    cols = list(b['emg_cols'])
    calf = [i for i, c in enumerate(cols) if 'TA' not in str(c).upper()]
    out = {}
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.9), sharex=True)
    for sess, col, ang in (('S1', OLD, -0.645), ('S2', NEW, -0.644)):
        rec = min([r for r in b['active'] if r['session'] == sess], key=lambda r: abs(r['angle'] - ang))
        push = -tr[rec['idx']]['truth']                      # push size (Nm), positive = toes down
        sel = np.arange(100, len(push) - 100)
        for ax, E, name in ((axs[0], base, 'before'), (axs[1], agc, 'after')):
            y = E[rec['idx']][:, calf].mean(axis=1)
            ax.scatter(push[sel][::6], y[sel][::6], s=4, color=col, alpha=0.25, lw=0)
            edges = np.linspace(0, 2.2, 9)
            mids, means = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (push[sel] >= lo) & (push[sel] < hi)
                if m.sum() > 30:
                    mids.append((lo + hi) / 2); means.append(np.median(y[sel][m]))
            lab = ('session 1 (training)' if sess == 'S1' else 'session 2 (never seen)')
            ax.plot(mids, means, color=col, lw=3, marker='o', ms=6, label=lab)
            mid_m = (push[sel] >= 1.0) & (push[sel] <= 1.8)       # push range both sessions cover
            rest_m = push[sel] < 0.2
            out[(sess, name)] = float(np.median(y[sel][mid_m]) - np.median(y[sel][rest_m]))
    axs[0].set_title('Before: EMG ' + chr(0x00F7) + ' training-session maximum', loc='left', fontsize=13, weight='bold', color=INK)
    axs[1].set_title('After auto-gain (uses only the past EMG)', loc='left', fontsize=13, weight='bold', color=INK)
    for ax in axs:
        ax.set_xlabel('voluntary push (Nm, toes down)')
        ax.legend(loc='upper left', fontsize=10.5)
    axs[0].set_ylabel('calf EMG (mean of GM, GL, SOL)')
    r_before = out[('S2', 'before')] / out[('S1', 'before')]
    r_after = out[('S2', 'after')] / out[('S1', 'after')]
    fig.text(0.07, -0.04, f'Subject JM, ankle at {M}0.64 rad. For the same push (1.0{chr(0x2013)}1.8 Nm), the calf EMG rises above its relaxed level '
             f'only {r_before:.2f}{chr(0x00D7)} as much in session 2 as in session 1; after auto-gain {r_after:.2f}{chr(0x00D7)}. '
             'Lines = median per push size.', fontsize=10.5, color=INK2)
    fig.subplots_adjust(wspace=0.22)
    save(fig, 'agc_before_after.png')
    return r_before, r_after, cols


if __name__ == '__main__':
    print('rest', fig_rest_detect())
    print('agc', fig_agc())
