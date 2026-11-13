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


def plot_pred_vs_true(splits, out_path, known_positions=None, title=None):
    """Scatter plots of predicted vs true torque for train / val / test splits.

    Parameters
    ----------
    splits : list of dict
        Each dict has keys: 'X', 'y_true', 'y_pred', 'label' (e.g. 'Train').
    out_path : str
    known_positions : array-like or None
    title : str or None
    """
    n_splits = len(splits)
    fig, axes = plt.subplots(1, n_splits, figsize=(6 * n_splits, 5), squeeze=False)

    for col, split in enumerate(splits):
        ax = axes[0][col]
        y_true = split['y_true']
        y_pred = split['y_pred']
        label = split['label']

        # colour by position if available
        if 'X' in split and split['X'] is not None and known_positions is not None:
            positions = split['X'][:, -1, POSITION_FEATURE_INDEX]
            pos_snapped, valid_pos = snap_to_operating_points(positions, known_positions)
            cmap = plt.cm.coolwarm
            norm = plt.Normalize(vmin=valid_pos.min(), vmax=valid_pos.max())
            sc = ax.scatter(y_true, y_pred, c=pos_snapped, cmap=cmap, norm=norm,
                            s=4, alpha=0.3, edgecolors='none')
            if col == n_splits - 1:
                cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
                cbar.set_label('Position (rad)', fontsize=8)
        else:
            ax.scatter(y_true, y_pred, s=4, alpha=0.3, color='steelblue',
                       edgecolors='none')

        # identity line
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        margin = 0.05 * (hi - lo)
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                'k--', linewidth=0.8, label='Identity')

        m = compute_metrics(y_true, y_pred)
        ax.set_title(f'{label}  |  R\u00b2={m["r2"]:.4f}  RMSE={m["rmse"]:.2f} Nm',
                     fontsize=10)
        ax.set_xlabel('True Torque (Nm)', fontsize=9)
        ax.set_ylabel('Predicted Torque (Nm)', fontsize=9)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linewidth=0.3)
        ax.legend(fontsize=7, loc='upper left')

    fig.suptitle(title or 'Predicted vs True Torque', fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Pred vs true scatter saved \u2192 {out_path}')


def plot_training_curves(history, out_path, title=None):
    """Plot training & validation loss and MAE over epochs.

    Parameters
    ----------
    history : keras History object (or dict with same keys)
    out_path : str
    title : str or None
    """
    h = history.history if hasattr(history, 'history') else history

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(h['loss'], label='Train Loss', linewidth=1.2)
    axes[0].plot(h['val_loss'], label='Val Loss', linewidth=1.2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Loss Convergence')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linewidth=0.3)
    axes[0].set_yscale('log')

    # MAE
    axes[1].plot(h['mae'], label='Train MAE', linewidth=1.2)
    axes[1].plot(h['val_mae'], label='Val MAE', linewidth=1.2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (Nm)')
    axes[1].set_title('MAE Convergence')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linewidth=0.3)

    # annotate best val epoch
    best_epoch = int(np.argmin(h['val_loss']))
    best_val = h['val_loss'][best_epoch]
    axes[0].axvline(best_epoch, color='gray', linestyle=':', linewidth=0.8)
    axes[0].annotate(f'Best epoch {best_epoch}\nval_loss={best_val:.5f}',
                     xy=(best_epoch, best_val),
                     xytext=(best_epoch + len(h['loss']) * 0.05, best_val),
                     fontsize=8, arrowprops=dict(arrowstyle='->', color='gray'))

    fig.suptitle(title or 'Training Convergence', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Training curves saved \u2192 {out_path}')


def plot_capacity_comparison(sweep_results, known_positions, out_path, title=None):
    """Per-position R² across LSTM unit counts for train / validation / test.

    Each subplot (train, val, test) shows one line per unit count so you can
    see whether adding capacity helps at the extreme positions and whether any
    gain carries through to the held-out retest session.

    Parameters
    ----------
    sweep_results : list of dict
        Each dict has keys:
          'n_units'       : int
          'per_position'  : {'train': {pos: metrics}, 'val': ..., 'test': ...}
          'overall'       : {'train': metrics, 'val': metrics, 'test': metrics}
    known_positions : array-like
        Sorted operating positions (used as x-axis labels).
    out_path : str
    title : str or None
    """
    positions  = np.array(sorted(known_positions))
    pos_labels = [f'{p:+.2f}' for p in positions]
    x          = np.arange(len(positions))

    splits       = ['train', 'val', 'test']
    split_titles = {'train': 'Train', 'val': 'Validation', 'test': 'Test (retest)'}
    colors       = ['steelblue', 'darkorange', 'seagreen', 'crimson', 'mediumpurple']
    markers      = ['o', 's', '^', 'D', 'v']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for col, split in enumerate(splits):
        ax = axes[col]
        for i, res in enumerate(sweep_results):
            n_units = res['n_units']
            pp      = res['per_position'][split]
            r2_vals = [pp[p]['r2'] for p in positions]
            overall_r2 = res['overall'][split]['r2']
            ax.plot(x, r2_vals,
                    marker=markers[i % len(markers)],
                    color=colors[i % len(colors)],
                    linewidth=1.8, markersize=7,
                    label=f'{n_units} units  (pooled R²={overall_r2:.3f})')

        ax.set_xticks(x)
        ax.set_xticklabels(pos_labels, rotation=45, fontsize=8)
        ax.set_xlabel('Ankle Position (rad)', fontsize=9)
        if col == 0:
            ax.set_ylabel('R²', fontsize=9)
        ax.set_title(split_titles[split], fontsize=11)
        ax.grid(True, linewidth=0.3)
        ax.axhline(0, color='k', linewidth=0.4)

        all_r2 = [res['per_position'][split][p]['r2']
                  for res in sweep_results for p in positions]
        ax.set_ylim(min(0, min(all_r2) - 0.05), 1.05)
        ax.legend(fontsize=8)

    fig.suptitle(title or 'Capacity Sweep: Per-Position R²', fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Capacity comparison saved → {out_path}')


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

    ax.set_xlabel('Position Index (sorted negative angle \u2192 positive angle)', fontsize=10)
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


# ── Passive torque diagnostic plots ───────────────────────────────────────────


def plot_passive_coverage(raw_segments, passive_entries, operating_positions,
                          out_path):
    """Scatter + bar chart showing how many passive segments exist per cluster.

    Top panel : scatter of all raw segment means (accepted = blue circle,
                rejected = red ×).  Final cluster positions shown as vertical
                dashed lines.
    Bottom panel : bar chart of accepted segment count per cluster.
                   Green = ≥2 segments (reliable), orange = 1 (flagged).
                   Rejected count annotated above each bar in red.
    """
    passive_entries = sorted(passive_entries, key=lambda x: x[0])
    cluster_pos  = [p for p, _ in passive_entries]
    pos_tolerance = 0.06   # slightly wider than merge tolerance for display snapping

    # Snap each raw segment to its nearest cluster
    def _snap(pos):
        return min(cluster_pos, key=lambda cp: abs(cp - pos))

    accepted = [s for s in raw_segments if s['passed']]
    rejected = [s for s in raw_segments if not s['passed']]

    # Count accepted / rejected per cluster
    from collections import Counter
    acc_counts = Counter(_snap(s['pos']) for s in accepted)
    rej_counts = Counter(_snap(s['pos']) for s in rejected)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    # ── Top: scatter of raw segment means ────────────────────────────────────
    ax = axes[0]
    if accepted:
        ax.scatter([s['pos'] for s in accepted],
                   [s['tq']  for s in accepted],
                   c='steelblue', s=60, zorder=3, label='Accepted segment')
    if rejected:
        ax.scatter([s['pos'] for s in rejected],
                   [s['tq']  for s in rejected],
                   c='red', marker='x', s=60, linewidths=1.5, zorder=3,
                   label='Rejected (high std / implausible)')
    for cp, ct in passive_entries:
        ax.axvline(cp, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    for op in operating_positions:
        ax.axvline(op, color='coral', linestyle=':', linewidth=1.0, alpha=0.6)
    ax.set_ylabel('Passive Torque (Nm)', fontsize=9)
    ax.set_title('Raw passive segments — accepted vs rejected', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.3)
    ax.axhline(0, color='k', linewidth=0.4)

    # ── Bottom: bar chart of counts per cluster ───────────────────────────────
    ax2 = axes[1]
    x  = np.arange(len(cluster_pos))
    counts = [acc_counts.get(cp, 0) for cp in cluster_pos]
    colors = ['seagreen' if c >= 2 else 'darkorange' for c in counts]
    bars = ax2.bar(x, counts, color=colors)
    for i, cp in enumerate(cluster_pos):
        rj = rej_counts.get(cp, 0)
        if rj:
            ax2.text(i, counts[i] + 0.05, f'+{rj} rej', ha='center',
                     fontsize=7, color='red')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{p:+.3f}' for p in cluster_pos], rotation=45, fontsize=8)
    ax2.set_ylabel('Accepted segment count', fontsize=9)
    ax2.set_xlabel('Cluster position (rad)', fontsize=9)
    ax2.set_title('Accepted segments per passive cluster', fontsize=10)
    ax2.axhline(2, color='gray', linestyle='--', linewidth=0.7,
                label='Minimum reliable threshold (2)')
    ax2.legend(fontsize=8)
    ax2.grid(True, axis='y', linewidth=0.3)
    ax2.set_ylim(0, max(counts) + 1.5)

    # Legend patches
    from matplotlib.patches import Patch
    axes[1].legend(handles=[
        Patch(color='seagreen',    label='≥2 segments (reliable)'),
        Patch(color='darkorange',  label='1 segment (flagged)'),
        plt.Line2D([0], [0], color='gray', linestyle='--', label='Threshold = 2'),
    ], fontsize=8)

    fig.suptitle('Passive Torque Coverage', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Passive coverage saved → {out_path}')


def plot_passive_interpolation_range(passive_entries, operating_positions,
                                     out_path):
    """Show where operating positions fall relative to the passive measurement range.

    Blue curve: passive torque evaluated on a dense position grid.
    Black circles: measured cluster positions.
    Vertical lines: operating positions, green = interpolated, red = extrapolated.
    Gray shading: extrapolation zones (outside measured range).
    """
    from ML.preprocessing.dataset_builder import lookup_passive_torque

    passive_entries = sorted(passive_entries, key=lambda x: x[0])
    cluster_pos    = np.array([p for p, _ in passive_entries])
    cluster_tq     = np.array([t for _, t in passive_entries])
    pos_min, pos_max = cluster_pos.min(), cluster_pos.max()

    grid_lo = min(pos_min, min(operating_positions)) - 0.05
    grid_hi = max(pos_max, max(operating_positions)) + 0.05
    grid    = np.linspace(grid_lo, grid_hi, 500)
    curve   = np.array([lookup_passive_torque(passive_entries, p) for p in grid])

    fig, ax = plt.subplots(figsize=(11, 5))

    # Extrapolation zones
    ax.axvspan(grid_lo, pos_min, color='gray', alpha=0.12, label='Extrapolation zone')
    ax.axvspan(pos_max, grid_hi, color='gray', alpha=0.12)

    ax.plot(grid, curve, color='steelblue', linewidth=1.8,
            label='Passive torque (interpolated / extrapolated)')
    ax.scatter(cluster_pos, cluster_tq, color='black', s=70, zorder=5,
               label='Measured clusters')

    # Operating position lines
    for op in sorted(operating_positions):
        inside = pos_min <= op <= pos_max
        color  = 'seagreen' if inside else 'red'
        label  = '(interp)' if inside else '(EXTRAP)'
        ax.axvline(op, color=color, linestyle='--', linewidth=1.2, alpha=0.8)
        tq_at_op = lookup_passive_torque(passive_entries, op)
        ax.annotate(f'{op:+.3f}\n{label}', xy=(op, tq_at_op),
                    xytext=(0, 14), textcoords='offset points',
                    fontsize=7, ha='center', color=color)

    ax.set_xlabel('Ankle Position (rad)', fontsize=10)
    ax.set_ylabel('Passive Torque (Nm)', fontsize=10)
    ax.set_title('Passive Torque — Interpolation vs Extrapolation', fontsize=11)
    ax.axhline(0, color='k', linewidth=0.4)
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Interpolation range plot saved → {out_path}')


def plot_passive_variability(raw_segments, passive_entries, out_path,
                             flag_threshold=2.0):
    """Strip plot of accepted passive segment values per cluster.

    One subplot per cluster.  Individual accepted segment means shown as
    jittered points; the merged cluster mean as an orange dashed line.
    Title of each subplot includes the within-cluster std.
    """
    passive_entries = sorted(passive_entries, key=lambda x: x[0])
    cluster_pos = [p for p, _ in passive_entries]
    accepted    = [s for s in raw_segments if s['passed']]

    def _snap(pos):
        return min(cluster_pos, key=lambda cp: abs(cp - pos))

    # Group accepted segments by cluster
    from collections import defaultdict
    groups = defaultdict(list)
    for s in accepted:
        groups[_snap(s['pos'])].append(s['tq'])

    n_clusters = len(cluster_pos)
    fig, axes = plt.subplots(1, n_clusters,
                             figsize=(max(8, 2.5 * n_clusters), 4),
                             sharey=True)
    if n_clusters == 1:
        axes = [axes]

    for i, (cp, ct) in enumerate(passive_entries):
        ax    = axes[i]
        vals  = groups.get(cp, [])
        n     = len(vals)
        std   = float(np.std(vals)) if n > 1 else float('nan')

        # Jittered strip
        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.08, 0.08, size=n) if n > 1 else [0]
        ax.scatter(jitter, vals, color='steelblue', s=50, zorder=3, alpha=0.8)
        ax.axhline(ct, color='darkorange', linestyle='--', linewidth=1.5,
                   label=f'Cluster mean\n{ct:+.3f} Nm')

        flag = '' if np.isnan(std) else (
            '  ⚠' if std > flag_threshold else '')
        std_str = 'N/A' if np.isnan(std) else f'{std:.3f} Nm'
        ax.set_title(f'{cp:+.3f} rad\nn={n}  σ={std_str}{flag}',
                     fontsize=8,
                     color='red' if (not np.isnan(std) and std > flag_threshold)
                           else 'black')
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.grid(True, axis='y', linewidth=0.3)
        ax.axhline(0, color='k', linewidth=0.4)
        if i == 0:
            ax.set_ylabel('Passive Torque (Nm)', fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle('Within-Cluster Passive Torque Variability', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Passive variability plot saved → {out_path}')


def plot_passive_torque_curve(passive_entries, operating_positions, out_path):
    """Passive torque curve with physiological sanity checks.

    Shows the measured cluster points and the full interpolated/extrapolated
    curve.  Prints monotonicity and slope diagnostics.
    """
    from ML.preprocessing.dataset_builder import lookup_passive_torque

    passive_entries = sorted(passive_entries, key=lambda x: x[0])
    cluster_pos = np.array([p for p, _ in passive_entries])
    cluster_tq  = np.array([t for _, t in passive_entries])

    grid_lo = min(cluster_pos.min(), min(operating_positions)) - 0.05
    grid_hi = max(cluster_pos.max(), max(operating_positions)) + 0.05
    grid    = np.linspace(grid_lo, grid_hi, 500)
    curve   = np.array([lookup_passive_torque(passive_entries, p) for p in grid])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grid, curve, color='steelblue', linewidth=1.8,
            label='Passive torque curve')
    ax.scatter(cluster_pos, cluster_tq, color='black', s=80, zorder=5,
               label='Measured clusters')

    for op in operating_positions:
        tq = lookup_passive_torque(passive_entries, op)
        ax.scatter([op], [tq], color='coral', s=40, zorder=4, marker='D')

    ax.axhline(0, color='k', linestyle='--', linewidth=0.6)
    ax.set_xlabel('Ankle Position (rad)', fontsize=10)
    ax.set_ylabel('Passive Torque (Nm)', fontsize=10)
    ax.set_title('Passive Torque Curve — Physiological Check', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.3)

    # Monotonicity annotation
    diffs = np.diff(cluster_tq)
    if np.all(diffs > 0):
        mono_str = 'monotonically increasing ✓'
        mono_color = 'seagreen'
    elif np.all(diffs < 0):
        mono_str = 'monotonically decreasing ✓'
        mono_color = 'seagreen'
    else:
        mono_str = 'NON-MONOTONIC ✗'
        mono_color = 'red'
    ax.text(0.02, 0.97, mono_str, transform=ax.transAxes,
            fontsize=9, color=mono_color, va='top')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Passive torque curve saved → {out_path}')


def plot_passive_residual_by_position(X_array, y_array, operating_positions,
                                      out_path, split_label='Train+Val'):
    """Box + IQR plots showing active torque distribution per position.

    Top panel  : box plot of active torque values per position (reveals range
                 differences — a narrow range makes high R² intrinsically harder).
    Bottom panel: bar chart of within-position IQR.
    """
    positions   = X_array[:, -1, POSITION_FEATURE_INDEX]
    pos_snapped, valid_pos = snap_to_operating_points(positions, operating_positions)

    groups     = {p: y_array[pos_snapped == p] for p in valid_pos}
    pos_labels = [f'{p:+.3f}' for p in valid_pos]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    # Top: box plot
    ax = axes[0]
    box_data = [groups[p] for p in valid_pos]
    bp = ax.boxplot(box_data, labels=pos_labels, patch_artist=True,
                    medianprops=dict(color='black', linewidth=1.5))
    cmap = plt.cm.coolwarm
    for patch, pos in zip(bp['boxes'], valid_pos):
        norm_val = (pos - valid_pos.min()) / (valid_pos.max() - valid_pos.min() + 1e-12)
        patch.set_facecolor(cmap(norm_val))
        patch.set_alpha(0.7)
    ax.set_ylabel('Active Torque (Nm)', fontsize=9)
    ax.set_title(f'Active torque distribution per position — {split_label}', fontsize=10)
    ax.tick_params(axis='x', labelsize=8)
    ax.axhline(0, color='k', linewidth=0.4)
    ax.grid(True, axis='y', linewidth=0.3)

    # Bottom: IQR bar chart
    ax2 = axes[1]
    iqrs = [float(np.percentile(groups[p], 75) - np.percentile(groups[p], 25))
            for p in valid_pos]
    bar_colors = [cmap((p - valid_pos.min()) / (valid_pos.max() - valid_pos.min() + 1e-12))
                  for p in valid_pos]
    ax2.bar(np.arange(len(valid_pos)), iqrs, color=bar_colors, alpha=0.8)
    ax2.set_xticks(np.arange(len(valid_pos)))
    ax2.set_xticklabels(pos_labels, rotation=45, fontsize=8)
    ax2.set_ylabel('IQR of active torque (Nm)', fontsize=9)
    ax2.set_title('Within-position IQR (narrow IQR → R² is intrinsically harder)',
                  fontsize=10)
    ax2.axhline(2.0, color='gray', linestyle='--', linewidth=0.8,
                label='Low-variance flag threshold (2 Nm)')
    ax2.legend(fontsize=8)
    ax2.grid(True, axis='y', linewidth=0.3)

    fig.suptitle(f'Active Torque Residual by Position — {split_label}', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Residual-by-position plot saved → {out_path}')


def plot_extreme_vs_middle_summary(passive_entries, raw_segments,
                                   operating_positions, X_array, y_array,
                                   out_path):
    """2×3 grid comparing all passive quality metrics: extreme vs middle positions.

    Extreme positions (leftmost and rightmost) are highlighted in red;
    middle positions in steelblue.
    """
    from ML.preprocessing.dataset_builder import lookup_passive_torque
    from collections import Counter, defaultdict

    passive_entries = sorted(passive_entries, key=lambda x: x[0])
    cluster_pos = [p for p, _ in passive_entries]
    cluster_tq  = [t for _, t in passive_entries]
    pos_min, pos_max = cluster_pos[0], cluster_pos[-1]

    accepted = [s for s in raw_segments if s['passed']]

    def _snap(pos):
        return min(cluster_pos, key=lambda cp: abs(cp - pos))

    acc_counts = Counter(_snap(s['pos']) for s in accepted)
    groups     = defaultdict(list)
    for s in accepted:
        groups[_snap(s['pos'])].append(s['tq'])

    # Per-position IQR of active torque
    positions   = X_array[:, -1, POSITION_FEATURE_INDEX]
    pos_snapped, valid_pos = snap_to_operating_points(positions, operating_positions)
    iqrs = {p: float(np.percentile(y_array[pos_snapped == p], 75) -
                     np.percentile(y_array[pos_snapped == p], 25))
            for p in valid_pos}

    # Extrapolation distances
    extrap_dist = {}
    for op in operating_positions:
        if op < pos_min:
            extrap_dist[op] = pos_min - op
        elif op > pos_max:
            extrap_dist[op] = op - pos_max
        else:
            extrap_dist[op] = 0.0

    ops  = np.array(sorted(operating_positions))
    xlabels = [f'{p:+.3f}' for p in ops]
    x    = np.arange(len(ops))

    def _bar_colors(positions_arr):
        extreme = {positions_arr[0], positions_arr[-1]}
        return ['red' if p in extreme else 'steelblue' for p in positions_arr]

    metrics = [
        ('Segment count per cluster',       [acc_counts.get(_snap(p), 0) for p in ops], 'count'),
        ('Within-cluster std (Nm)',          [float(np.std(groups.get(_snap(p), [0])))
                                              if len(groups.get(_snap(p), [])) > 1
                                              else float('nan') for p in ops], 'Nm'),
        ('Extrapolation distance (rad)',     [extrap_dist.get(p, 0.0) for p in ops], 'rad'),
        ('Passive torque at position (Nm)',  [lookup_passive_torque(passive_entries, p) for p in ops], 'Nm'),
        ('Active torque IQR (Nm)',           [iqrs.get(p, float('nan')) for p in ops], 'Nm'),
        ('Interp=0 / Extrap=1',             [1 if extrap_dist.get(p, 0) > 0 else 0 for p in ops], 'flag'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    for i, (title, values, unit) in enumerate(metrics):
        ax = axes[i]
        colors = _bar_colors(ops)
        finite = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
        if unit == 'flag':
            ax.scatter(x, values, c=colors, s=80, zorder=3)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Interp', 'Extrap'], fontsize=8)
        else:
            ax.bar(x, [0 if (isinstance(v, float) and np.isnan(v)) else v
                       for v in values], color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=45, fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(unit, fontsize=8)
        ax.grid(True, axis='y', linewidth=0.3)

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color='red', label='Extreme positions'),
                        Patch(color='steelblue', label='Middle positions')],
               loc='lower center', ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Passive Torque Quality — Extreme vs Middle Positions', fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Extreme vs middle summary saved → {out_path}')


# ── EMG-torque relationship diagnostic plots ──────────────────────────────────


def plot_nrmse_and_residual_boxplots(X, y_true, y_pred, operating_positions,
                                     out_path, title=None):
    """3-panel error overview: NRMSE bars, signed residual boxes, bias ± std bars.

    Panel 1 — NRMSE (%) per position: range-normalised error that removes the
              small-signal R² artifact at the most dorsiflexed position.
    Panel 2 — Signed residual box plot: reveals whether errors are symmetric
              and proportional to the torque range.
    Panel 3 — Bias (mean error) ± std per position: separates systematic from
              random error components.
    """
    positions = X[:, -1, POSITION_FEATURE_INDEX]
    pos_snapped, valid_pos = snap_to_operating_points(positions, operating_positions)
    cmap = plt.cm.coolwarm
    pos_min, pos_max = valid_pos.min(), valid_pos.max()

    def _norm(p):
        return (p - pos_min) / (pos_max - pos_min + 1e-12)

    labels   = [f'{p:+.3f}' for p in valid_pos]
    nrmses, biases, stds, iqrs = [], [], [], []
    residual_groups = []

    for p in valid_pos:
        mask = pos_snapped == p
        yt, yp = y_true[mask], y_pred[mask]
        res = yp - yt
        m = compute_metrics(yt, yp)
        nrmses.append(m['nrmse'] * 100)
        biases.append(float(res.mean()))
        stds.append(float(res.std()))
        iqrs.append(float(np.percentile(yt, 75) - np.percentile(yt, 25)))
        residual_groups.append(res)

    x = np.arange(len(valid_pos))
    colors = [cmap(_norm(p)) for p in valid_pos]
    mean_nrmse = float(np.mean(nrmses))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── Panel 1: NRMSE ────────────────────────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(x, nrmses, color=colors, alpha=0.85)
    for xi, v in zip(x, nrmses):
        ax.text(xi, v + 0.2, f'{v:.1f}%', ha='center', fontsize=7)
    ax.axhline(mean_nrmse, color='gray', linestyle='--', linewidth=1.0,
               label=f'Mean = {mean_nrmse:.1f}%')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_ylabel('NRMSE (%)', fontsize=9)
    ax.set_title('NRMSE per Position\n(range-normalised — removes small-signal artifact)',
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', linewidth=0.3)

    # ── Panel 2: Residual box plot ────────────────────────────────────────────
    ax2 = axes[1]
    tick_labels = [f'{p:+.3f}\nIQR={iq:.1f}Nm'
                   for p, iq in zip(valid_pos, iqrs)]
    bp = ax2.boxplot(residual_groups, labels=tick_labels, patch_artist=True,
                     medianprops=dict(color='black', linewidth=1.5))
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax2.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax2.set_ylabel('Residual = y_pred − y_true (Nm)', fontsize=9)
    ax2.set_title('Error Distribution per Position\n(signed residuals)', fontsize=9)
    ax2.tick_params(axis='x', labelsize=7)
    ax2.grid(True, axis='y', linewidth=0.3)

    # ── Panel 3: Bias ± std ───────────────────────────────────────────────────
    ax3 = axes[2]
    bar_colors = ['coral' if b >= 0 else 'steelblue' for b in biases]
    ax3.bar(x, biases, color=bar_colors, alpha=0.85, zorder=3)
    ax3.errorbar(x, biases, yerr=stds, fmt='none', color='black',
                 capsize=4, linewidth=1.2, zorder=4)
    for xi, b in zip(x, biases):
        ax3.text(xi, b + (0.05 if b >= 0 else -0.15), f'{b:+.2f}',
                 ha='center', fontsize=7)
    ax3.axhline(0, color='gray', linestyle='--', linewidth=1.0)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=45, fontsize=8)
    ax3.set_ylabel('Mean Residual (Nm)', fontsize=9)
    ax3.set_title('Bias (mean error) ± Std per Position\n'
                  '(non-zero bias = systematic under/over-prediction)', fontsize=9)
    ax3.grid(True, axis='y', linewidth=0.3)

    fig.suptitle(title or 'Error Overview', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'NRMSE + residual overview saved → {out_path}')


def plot_emg_torque_scatter_by_position(X, y_true, operating_positions, out_path,
                                        emg_channel_names=None, title=None):
    """EMG-torque scatter per position, coloured by EMG sum.

    For each position one subplot shows the dominant EMG channel (highest
    |correlation| with torque) vs active torque.  The linear fit R² and slope
    are annotated.  Points coloured by total EMG activation reveal saturation.
    """
    import math
    ch_names = emg_channel_names or ['MG', 'LG', 'SOL', 'TA']
    positions = X[:, -1, POSITION_FEATURE_INDEX]
    pos_snapped, valid_pos = snap_to_operating_points(positions, operating_positions)

    n_pos  = len(valid_pos)
    n_cols = min(4, n_pos)
    n_rows = math.ceil(n_pos / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4.5 * n_rows),
                             squeeze=False)

    for idx, p in enumerate(valid_pos):
        row, col = divmod(idx, n_cols)
        ax   = axes[row][col]
        mask = pos_snapped == p
        if mask.sum() < 10:
            ax.set_title(f'{p:+.3f} rad\nn={mask.sum()} (too few)', fontsize=8)
            ax.set_visible(False)
            continue

        emg  = X[mask, -1, 0:4]          # (N, 4)
        tq   = y_true[mask]
        esum = emg.sum(axis=1)

        # Dominant channel by |correlation|
        corrs   = [float(np.corrcoef(emg[:, k], tq)[0, 1]) for k in range(4)]
        dom_ch  = int(np.argmax(np.abs(corrs)))
        emg_dom = emg[:, dom_ch]

        sc = ax.scatter(emg_dom, tq, c=esum, cmap='viridis',
                        s=3, alpha=0.4, edgecolors='none',
                        vmin=0, vmax=4)

        # Linear fit
        coeffs = np.polyfit(emg_dom, tq, 1)
        xfit   = np.linspace(emg_dom.min(), emg_dom.max(), 100)
        yfit   = np.polyval(coeffs, xfit)
        ss_res = np.sum((tq - np.polyval(coeffs, emg_dom)) ** 2)
        ss_tot = np.sum((tq - tq.mean()) ** 2)
        lin_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        ax.plot(xfit, yfit, 'k--', linewidth=1.0,
                label=f'R²={lin_r2:.2f}  slope={coeffs[0]:.1f}')

        # Saturation zone
        ax.axvline(0.9, color='red', linestyle=':', linewidth=0.8, alpha=0.7)

        ax.set_title(f'{p:+.3f} rad  n={mask.sum()}\n'
                     f'{ch_names[dom_ch]} (r={corrs[dom_ch]:.2f})', fontsize=8)
        ax.set_xlabel('EMG envelope (norm.)', fontsize=7)
        ax.set_ylabel('Active Torque (Nm)', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc='upper left')
        ax.grid(True, linewidth=0.25)

        plt.colorbar(sc, ax=ax, shrink=0.7).set_label('EMG sum', fontsize=6)

    # Hide unused axes
    for idx in range(n_pos, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle(title or 'EMG-Torque Scatter per Position', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'EMG-torque scatter saved → {out_path}')


def plot_emg_torque_linearity_by_position(X, y_true, operating_positions, out_path,
                                          title=None):
    """Linear EMG-torque R² and fit residuals per position.

    Panel 1 — bar chart of R²(EMG_sum → torque) per position.
              Low R² at a position means the relationship is intrinsically
              nonlinear there, independent of the LSTM.
    Panel 2 — box plot of residuals from the linear fit per position.
              Large residuals confirm the linear model is insufficient.
    """
    positions = X[:, -1, POSITION_FEATURE_INDEX]
    pos_snapped, valid_pos = snap_to_operating_points(positions, operating_positions)
    cmap = plt.cm.coolwarm
    pos_min, pos_max = valid_pos.min(), valid_pos.max()

    labels, lin_r2s, res_groups = [], [], []

    for p in valid_pos:
        mask = pos_snapped == p
        if mask.sum() < 10:
            continue
        tq   = y_true[mask]
        esum = X[mask, -1, 0:4].sum(axis=1)
        coeffs = np.polyfit(esum, tq, 1)
        fitted = np.polyval(coeffs, esum)
        ss_res = np.sum((tq - fitted) ** 2)
        ss_tot = np.sum((tq - tq.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        lin_r2s.append(r2)
        res_groups.append(tq - fitted)
        labels.append(f'{p:+.3f}')

    x      = np.arange(len(labels))
    colors = [cmap((i / max(len(labels) - 1, 1))) for i in range(len(labels))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1 — Linear R²
    ax = axes[0]
    ax.bar(x, lin_r2s, color=colors, alpha=0.85)
    for xi, v in zip(x, lin_r2s):
        ax.text(xi, v + 0.01, f'{v:.3f}', ha='center', fontsize=7)
    ax.axhline(0.9, color='gray', linestyle='--', linewidth=0.8,
               label='High-linearity threshold (0.90)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_ylabel('Linear fit R²', fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_title('Linear EMG-Torque R² per Position\n'
                 '(low R² = intrinsic nonlinearity, not just LSTM failure)', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', linewidth=0.3)

    # Panel 2 — Residuals from linear fit
    ax2 = axes[1]
    bp = ax2.boxplot(res_groups, labels=labels, patch_artist=True,
                     medianprops=dict(color='black', linewidth=1.5))
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax2.axhline(0, color='red', linestyle='--', linewidth=1.2)
    ax2.set_ylabel('Residual from linear EMG-torque fit (Nm)', fontsize=9)
    ax2.set_title('Residuals from Linear Fit per Position\n'
                  '(large residuals = nonlinear EMG-torque relationship)', fontsize=9)
    ax2.tick_params(axis='x', labelsize=8)
    ax2.grid(True, axis='y', linewidth=0.3)

    fig.suptitle(title or 'EMG-Torque Linearity per Position', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'EMG-torque linearity saved → {out_path}')


def plot_plantarflexion_residual_detail(X, y_true, y_pred, operating_positions,
                                        target_pos, out_path, title=None):
    """Deep-dive on one position: true vs pred scatter + residual vs true torque.

    Panel 1 — True vs predicted scatter with identity line and linear fit.
              Slope < 1 means the model compresses the dynamic range.
    Panel 2 — Residual (y_pred - y_true) vs true torque, coloured by EMG sum.
              A curved trend confirms nonlinearity; a funnel suggests
              heteroscedastic noise; a constant offset means systematic bias.
    """
    positions = X[:, -1, POSITION_FEATURE_INDEX]
    pos_snapped, valid_pos = snap_to_operating_points(positions, operating_positions)

    # Snap target_pos to nearest operating position
    target_arr = np.array([target_pos])
    snapped, _ = snap_to_operating_points(target_arr, operating_positions)
    pos = float(snapped[0])
    mask = pos_snapped == pos

    if mask.sum() < 10:
        print(f'WARNING: only {mask.sum()} windows at {pos:+.3f} rad — plot may be sparse.')

    yt   = y_true[mask]
    yp   = y_pred[mask]
    res  = yp - yt
    esum = X[mask, -1, 0:4].sum(axis=1)
    m    = compute_metrics(yt, yp)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel 1: True vs predicted ────────────────────────────────────────────
    ax = axes[0]
    lo = min(yt.min(), yp.min())
    hi = max(yt.max(), yp.max())
    mg = 0.05 * (hi - lo)
    ax.scatter(yt, yp, s=4, alpha=0.3, color='steelblue', edgecolors='none')
    ax.plot([lo - mg, hi + mg], [lo - mg, hi + mg], 'k--', linewidth=0.9,
            label='Identity')
    # Linear regression of y_pred on y_true
    coeffs = np.polyfit(yt, yp, 1)
    xfit   = np.linspace(lo, hi, 200)
    ax.plot(xfit, np.polyval(coeffs, xfit), 'r--', linewidth=1.2,
            label=f'Fit: slope={coeffs[0]:.3f}, intercept={coeffs[1]:+.3f} Nm')
    ax.set_xlabel('True Torque (Nm)', fontsize=9)
    ax.set_ylabel('Predicted Torque (Nm)', fontsize=9)
    ax.set_title(f'{pos:+.3f} rad  |  R²={m["r2"]:.4f}  RMSE={m["rmse"]:.3f} Nm',
                 fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.3)

    # ── Panel 2: Residual vs true torque ─────────────────────────────────────
    ax2 = axes[1]
    sc = ax2.scatter(yt, res, c=esum, cmap='viridis', s=4, alpha=0.4,
                     edgecolors='none', vmin=0, vmax=4)
    plt.colorbar(sc, ax=ax2).set_label('EMG sum', fontsize=8)
    ax2.axhline(0, color='red', linestyle='--', linewidth=1.2)

    # Quadratic trend line
    try:
        qcoeffs = np.polyfit(yt, res, 2)
        xq      = np.linspace(yt.min(), yt.max(), 200)
        ax2.plot(xq, np.polyval(qcoeffs, xq), 'k-', linewidth=2.0,
                 label='Quadratic trend')
    except np.linalg.LinAlgError:
        ax2.axhline(float(res.mean()), color='black', linewidth=1.5,
                    label=f'Mean residual = {res.mean():+.3f} Nm')

    bias = float(res.mean())
    ax2.set_xlabel('True Torque (Nm)', fontsize=9)
    ax2.set_ylabel('Residual = y_pred − y_true (Nm)', fontsize=9)
    ax2.set_title(f'Residual vs True Torque  |  bias={bias:+.3f} Nm', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, linewidth=0.3)

    fig.suptitle(title or f'Deep Dive — {pos:+.3f} rad', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Plantarflexion detail saved → {out_path}')
