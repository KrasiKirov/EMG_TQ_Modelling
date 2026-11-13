"""
Shared plotting utilities for sEMG-to-torque model evaluation.

Used by both train.py (within-subject) and cross_subject.py (cross-subject).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ML.config import POSITION_FEATURE_INDEX, TIME_STEP_S
from ML.evaluation.metrics import snap_to_operating_points, compute_metrics


def plot_full_trials(X_test, y_test, y_pred, out_path,
                     known_positions=None, title=None):
    """Full 90 s inspection per position: true vs predicted + per-subplot metrics."""
    positions = X_test[:, -1, POSITION_FEATURE_INDEX]
    if known_positions is not None:
        pos_snapped, valid_pos = snap_to_operating_points(positions, known_positions)
    else:
        pos_snapped = np.round(positions / 0.10) * 0.10
        valid_pos = np.sort(np.unique(pos_snapped))
    n_pos = len(valid_pos)

    fig, axes = plt.subplots(n_pos, 1, figsize=(14, 3 * n_pos), squeeze=False)

    for i, pos in enumerate(valid_pos):
        ax = axes[i][0]
        mask = pos_snapped == pos
        yt = y_test[mask]
        yp = y_pred[mask]
        time_s = np.arange(len(yt)) * TIME_STEP_S

        m = compute_metrics(yt, yp)

        ax.plot(time_s, yt, linewidth=0.6, color='steelblue', label='True')
        ax.plot(time_s, yp, linewidth=0.6, color='red', alpha=0.7, label='Predicted')
        ax.set_xlim(0, 90)
        ax.set_ylabel('Torque (Nm)', fontsize=8)
        ax.set_title(f'Position {pos:+.3f}  |  R\u00b2={m["r2"]:.4f}  |  '
                     f'RMSE={m["rmse"]:.3f} Nm  |  N={mask.sum()}', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, linewidth=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc='upper right')

    axes[-1][0].set_xlabel('Time (s)', fontsize=9)
    fig.suptitle(title or 'Full Trial Inspection \u2014 All Positions (Retest Session)',
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Full trial inspection saved \u2192 {out_path}')


def plot_per_position_metrics(X_test, y_test, y_pred, out_path,
                              known_positions=None, pooled_r2=None, title=None):
    """Bar charts of R\u00b2, RMSE, MAE per position. Returns {pos: r2} dict."""
    positions = X_test[:, -1, POSITION_FEATURE_INDEX]

    if known_positions is not None:
        pos_snapped, unique_pos = snap_to_operating_points(positions, known_positions)
    else:
        pos_snapped = np.round(positions / 0.10) * 0.10
        unique_pos = np.sort(np.unique(pos_snapped))

    pos_labels, r2s, rmses, maes = [], [], [], []

    for pos in unique_pos:
        mask = pos_snapped == pos
        if mask.sum() == 0:
            continue
        m = compute_metrics(y_test[mask], y_pred[mask])
        pos_labels.append(pos)
        r2s.append(m['r2'])
        rmses.append(m['rmse'])
        maes.append(m['mae'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(range(len(pos_labels)), r2s, color='steelblue')
    axes[0].set_xticks(range(len(pos_labels)))
    axes[0].set_xticklabels([f'{p:+.2f}' for p in pos_labels], rotation=45, fontsize=8)
    axes[0].set_ylabel('R\u00b2')
    axes[0].set_title('R\u00b2 per Position')
    axes[0].set_xlabel('Ankle Position')
    axes[0].grid(True, axis='y', linewidth=0.4)
    axes[0].axhline(y=0, color='k', linewidth=0.5)
    if pooled_r2 is not None:
        axes[0].axhline(y=pooled_r2, color='coral', linestyle='--', linewidth=1.2,
                        label=f'Pooled R\u00b2 = {pooled_r2:.4f}')
        axes[0].legend(fontsize=7)

    axes[1].bar(range(len(pos_labels)), rmses, color='coral')
    axes[1].set_xticks(range(len(pos_labels)))
    axes[1].set_xticklabels([f'{p:+.2f}' for p in pos_labels], rotation=45, fontsize=8)
    axes[1].set_ylabel('RMSE (Nm)')
    axes[1].set_title('RMSE per Position')
    axes[1].set_xlabel('Ankle Position')
    axes[1].grid(True, axis='y', linewidth=0.4)

    axes[2].bar(range(len(pos_labels)), maes, color='mediumpurple')
    axes[2].set_xticks(range(len(pos_labels)))
    axes[2].set_xticklabels([f'{p:+.2f}' for p in pos_labels], rotation=45, fontsize=8)
    axes[2].set_ylabel('MAE (Nm)')
    axes[2].set_title('MAE per Position')
    axes[2].set_xlabel('Ankle Position')
    axes[2].grid(True, axis='y', linewidth=0.4)

    fig.suptitle(title or 'Per-Position Test Metrics (Retest Trials)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Per-position metrics saved \u2192 {out_path}')

    return dict(zip(pos_labels, r2s))


def plot_r2_summary(pos_r2_dict, pooled_r2, out_path, title=None):
    """Scatter+line plot of R\u00b2 vs ankle position with pooled R\u00b2 reference."""
    positions = np.array(sorted(pos_r2_dict.keys()))
    r2_vals   = np.array([pos_r2_dict[p] for p in positions])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(positions, r2_vals, 'o-', color='steelblue', markersize=7, linewidth=1.5,
            label='Per-position R\u00b2')
    ax.axhline(y=pooled_r2, color='coral', linestyle='--', linewidth=1.2,
               label=f'Pooled R\u00b2 = {pooled_r2:.4f}')

    for p, r in zip(positions, r2_vals):
        ax.annotate(f'{r:.3f}', (p, r), textcoords='offset points',
                    xytext=(0, 10), fontsize=8, ha='center')

    ax.set_xlabel('Ankle Position', fontsize=10)
    ax.set_ylabel('R\u00b2', fontsize=10)
    ax.set_title(title or 'R\u00b2 vs Ankle Position (Retest Session)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.4)
    ax.set_ylim(min(0, min(r2_vals) - 0.05), 1.05)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'R\u00b2 summary saved \u2192 {out_path}')


def plot_r2_overlay(results, out_path):
    """Cross-subject comparison: R\u00b2 vs position index for all scenarios overlaid."""
    subject_colors = {'HM': 'coral', 'EG': 'seagreen'}
    markers = ['o', 's', '^', 'v', 'D', 'P', 'X', '*', 'h']

    fig, ax = plt.subplots(figsize=(10, 5))

    for res in results:
        label = res['label']
        pp = res['per_position']
        positions = np.array(sorted(pp.keys()))
        r2_vals = np.array([pp[p]['r2'] for p in positions])

        train_subj, test_subj = label.split('\u2192')
        color = subject_colors.get(train_subj, 'gray')
        ls = '-' if train_subj == test_subj else '--'
        x = np.arange(1, len(positions) + 1)
        ax.plot(x, r2_vals, color=color, linestyle=ls,
                marker=markers[hash(label) % len(markers)],
                markersize=6, linewidth=1.5, label=label)

    ax.set_xlabel('Position Index (sorted dorsiflexion \u2192 plantarflexion)', fontsize=10)
    ax.set_ylabel('R\u00b2', fontsize=10)
    ax.set_title('Cross-Subject Transferability: R\u00b2 per Position', fontsize=11)
    ax.set_xticks(range(1, 9))
    ax.set_xticklabels([f'p{i}' for i in range(1, 9)])
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.4)
    ax.set_ylim(-0.1, 1.05)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'R\u00b2 overlay saved \u2192 {out_path}')
