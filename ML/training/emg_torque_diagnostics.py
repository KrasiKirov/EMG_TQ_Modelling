"""
EMG-torque relationship diagnostics.

Investigates why the LSTM performs worse at the extreme ankle positions by
examining the raw EMG-torque relationship independently of the model.

Five plots are generated:

  01_nrmse_residual_overview.png   — NRMSE bars + signed residual boxes + bias±std
  02_emg_torque_scatter.png        — EMG vs torque scatter per position (dominant channel)
  03_emg_torque_linearity.png      — linear EMG-torque R² and fit residuals per position
  04_detail_plantarflexion.png     — deep dive: true vs pred + residual vs true torque
  05_cross_split_nrmse.png         — NRMSE comparison across train / val / test splits

Usage
-----
    python ML/training/emg_torque_diagnostics.py --subject HM
    python ML/training/emg_torque_diagnostics.py --subject HM --splits all
    python ML/training/emg_torque_diagnostics.py --subject HM --plantarflexion-pos 0.195
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ML.config import (SUBJECTS, DATA_DIR, MODEL_DIR, PLOT_DIR,
                        POSITION_FEATURE_INDEX)
from ML.training import load_subject, evaluate_model
from ML.evaluation import (
    compute_metrics, compute_per_position_metrics, snap_to_operating_points,
    plot_nrmse_and_residual_boxplots,
    plot_emg_torque_scatter_by_position,
    plot_emg_torque_linearity_by_position,
    plot_plantarflexion_residual_detail,
)


def parse_args():
    p = argparse.ArgumentParser(
        description='EMG-torque relationship diagnostics for one subject.')
    p.add_argument('--subject', choices=list(SUBJECTS.keys()), default=None,
                   help='Subject short name (e.g. HM, EG)')
    p.add_argument('--flb', default=None,
                   help='Override path to .flb file')
    p.add_argument('--subject-id', default=None,
                   help='Override subject ID for channel ordering')
    p.add_argument('--output-dir', default=None,
                   help='Override output directory')
    p.add_argument('--test-trials', nargs='+', type=int, default=None)
    p.add_argument('--retest-trials', nargs='+', type=int, default=None)
    p.add_argument('--splits', choices=['train', 'val', 'test', 'all'],
                   default='test',
                   help='Data split to use for diagnostics (default: test)')
    p.add_argument('--plantarflexion-pos', type=float, default=None,
                   help='Target position for deep-dive plot (default: most plantarflexed)')
    return p.parse_args()


def _load_model(subject_key):
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError('TensorFlow required. Install with: pip install tensorflow')
    ckpt_path = os.path.join(MODEL_DIR, subject_key, 'best_lstm.keras')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f'No checkpoint at {ckpt_path}. Run train.py --subject {subject_key} first.')
    model = tf.keras.models.load_model(ckpt_path)
    print(f'Loaded checkpoint: {ckpt_path}')
    return model


def _lin_r2(x, y):
    """R² of a degree-1 polynomial fit of x → y."""
    if len(x) < 2:
        return float('nan')
    coeffs = np.polyfit(x, y, 1)
    fitted = np.polyval(coeffs, x)
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def _print_summary(X, y_true, y_pred, operating_positions,
                   subject_key, split_label):
    """Print per-position metrics table with flags."""
    positions   = X[:, -1, POSITION_FEATURE_INDEX]
    pos_snapped, valid_pos = snap_to_operating_points(positions, operating_positions)

    sep = '=' * 78
    print(f'\n{sep}')
    print(f'EMG-TORQUE DIAGNOSTIC SUMMARY  —  subject: {subject_key}  '
          f'split: {split_label}')
    print(f'{sep}')
    header = (f'  {"Position":>10}  {"n":>6}  {"R²":>7}  {"RMSE":>8}  '
              f'{"NRMSE%":>7}  {"MAE":>7}  {"Bias":>7}  {"Std":>7}  '
              f'{"IQR":>6}  {"LinR²":>6}')
    print(header)
    print(f'  {"-"*10}  {"-"*6}  {"-"*7}  {"-"*8}  {"-"*7}  {"-"*7}  '
          f'{"-"*7}  {"-"*7}  {"-"*6}  {"-"*6}')

    per_pos_stats = {}
    for p in valid_pos:
        mask = pos_snapped == p
        yt, yp = y_true[mask], y_pred[mask]
        res    = yp - yt
        m      = compute_metrics(yt, yp)
        iqr    = float(np.percentile(yt, 75) - np.percentile(yt, 25))
        bias   = float(res.mean())
        std    = float(res.std())
        esum   = X[mask, -1, 0:4].sum(axis=1)
        lr2    = _lin_r2(esum, yt)
        per_pos_stats[p] = dict(n=int(mask.sum()), iqr=iqr, bias=bias,
                                std=std, lin_r2=lr2, **m)
        print(f'  {p:>+10.3f}  {mask.sum():>6d}  {m["r2"]:>7.4f}  '
              f'{m["rmse"]:>8.4f}  {m["nrmse"]*100:>7.2f}  {m["mae"]:>7.4f}  '
              f'{bias:>+7.3f}  {std:>7.3f}  {iqr:>6.2f}  {lr2:>6.3f}')

    # Pooled
    overall = compute_metrics(y_true, y_pred)
    print(f'  {"Pooled":>10}  {len(y_true):>6d}  {overall["r2"]:>7.4f}  '
          f'{overall["rmse"]:>8.4f}  {"—":>7}  {overall["mae"]:>7.4f}')

    # ── Flags ──────────────────────────────────────────────────────────────
    flags = []

    dorf_pos  = valid_pos[0]   # most dorsiflexed (smallest value)
    plant_pos = valid_pos[-1]  # most plantarflexed (largest value)

    # Dorsiflexion: small-signal artifact check
    ds = per_pos_stats[dorf_pos]
    if ds['iqr'] < 2.0:
        flags.append(
            f'[INFO] {dorf_pos:+.3f} rad: IQR={ds["iqr"]:.2f} Nm < 2 Nm — '
            f'R²={ds["r2"]:.3f} is likely a small-signal metric artifact. '
            f'NRMSE={ds["nrmse"]*100:.1f}% is a fairer measure.')

    # Plantarflexion: systematic bias check
    ps = per_pos_stats[plant_pos]
    if ps['rmse'] > 0 and abs(ps['bias']) > 0.3 * ps['rmse']:
        direction = 'over' if ps['bias'] > 0 else 'under'
        flags.append(
            f'[WARNING] {plant_pos:+.3f} rad: |bias|={abs(ps["bias"]):.3f} Nm = '
            f'{abs(ps["bias"])/ps["rmse"]*100:.0f}% of RMSE — '
            f'model systematically {direction}-predicts.')

    # Low linear R² at any position
    for p, st in per_pos_stats.items():
        if not np.isnan(st['lin_r2']) and st['lin_r2'] < 0.85:
            flags.append(
                f'[WARNING] {p:+.3f} rad: linear EMG-torque R²={st["lin_r2"]:.3f} < 0.85 '
                f'— intrinsically nonlinear relationship there.')

    # EMG saturation check
    for p in valid_pos:
        mask = pos_snapped == p
        emg_max_per_window = X[mask, -1, 0:4].max(axis=1)
        frac_sat = float((emg_max_per_window > 0.90).mean())
        if frac_sat > 0.10:
            flags.append(
                f'[WARNING] {p:+.3f} rad: {frac_sat*100:.0f}% of windows have '
                f'EMG channel > 0.90 — possible saturation.')

    print()
    if flags:
        print(f'  FLAGS ({len(flags)}):')
        for f in flags:
            print(f'    {f}')
    else:
        print('  No critical flags raised.')
    print(sep)


def _plot_cross_split_nrmse(splits_data, operating_positions, out_path,
                             subject_key):
    """Grouped bar chart: NRMSE per position for train / val / test."""
    split_labels  = [s['label'] for s in splits_data]
    split_colors  = ['steelblue', 'darkorange', 'seagreen']
    valid_pos     = np.array(sorted(operating_positions))
    n_pos         = len(valid_pos)
    n_splits      = len(splits_data)
    bar_w         = 0.8 / n_splits
    x             = np.arange(n_pos)

    fig, ax = plt.subplots(figsize=(14, 5))

    for i, split in enumerate(splits_data):
        X_s, y_s, yp_s = split['X'], split['y_true'], split['y_pred']
        positions = X_s[:, -1, POSITION_FEATURE_INDEX]
        pos_snapped, _ = snap_to_operating_points(positions, operating_positions)
        nrmses = []
        for p in valid_pos:
            mask = pos_snapped == p
            if mask.sum() < 2:
                nrmses.append(float('nan'))
                continue
            m = compute_metrics(y_s[mask], yp_s[mask])
            nrmses.append(m['nrmse'] * 100)
        offset = (i - n_splits / 2 + 0.5) * bar_w
        ax.bar(x + offset, nrmses, width=bar_w,
               color=split_colors[i % len(split_colors)], alpha=0.85,
               label=split['label'])

    ax.set_xticks(x)
    ax.set_xticklabels([f'{p:+.3f}' for p in valid_pos], rotation=45, fontsize=8)
    ax.set_ylabel('NRMSE (%)', fontsize=9)
    ax.set_title(f'NRMSE per Position — Train / Val / Test Comparison ({subject_key})\n'
                 '(similar bars across splits = no overfitting at that position)',
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', linewidth=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Cross-split NRMSE saved → {out_path}')


def main():
    args = parse_args()
    subject_key = args.subject or next(iter(SUBJECTS))

    out_dir = args.output_dir or os.path.join(
        PLOT_DIR, subject_key, 'emg_torque_diagnostics')
    os.makedirs(out_dir, exist_ok=True)

    def path(fname):
        return os.path.join(out_dir, fname)

    print(f'\n{"─"*60}')
    print(f'EMG-torque diagnostics — subject: {subject_key}')
    print(f'{"─"*60}')

    # ── Load data and model ───────────────────────────────────────────────────
    data = load_subject(
        subject_key,
        test_trial_indices=args.test_trials,
        retest_trial_indices=args.retest_trials,
        flb_path=args.flb,
        subject_id=args.subject_id,
    )
    model        = _load_model(subject_key)
    known_pos    = data['operating_positions']
    mvc_tq_test  = data['mvc_tq_test']

    # ── Generate predictions on all splits ───────────────────────────────────
    y_pred_train = model.predict(data['X_train'], verbose=0).flatten()
    y_pred_val   = model.predict(data['X_val'],   verbose=0).flatten()

    # Test split: apply MVC denormalization if used
    result_test  = evaluate_model(model, data['X_test'], data['y_test'],
                                  known_pos, mvc_tq_test=mvc_tq_test)
    y_pred_test  = result_test['y_pred']
    y_test       = result_test['y_test']

    # ── Select diagnostic split ───────────────────────────────────────────────
    split_map = {
        'train': (data['X_train'], data['y_train'], y_pred_train, 'Train'),
        'val':   (data['X_val'],   data['y_val'],   y_pred_val,   'Validation'),
        'test':  (data['X_test'],  y_test,           y_pred_test,  'Test (retest)'),
        'all':   (
            np.concatenate([data['X_train'], data['X_val'], data['X_test']]),
            np.concatenate([data['y_train'], data['y_val'], y_test]),
            np.concatenate([y_pred_train,    y_pred_val,    y_pred_test]),
            'All splits',
        ),
    }
    X_diag, y_diag, yp_diag, split_label = split_map[args.splits]

    # Target position for deep-dive
    target_pos = args.plantarflexion_pos
    if target_pos is None:
        target_pos = float(max(known_pos))   # default: most plantarflexed

    # ── Plot 01: NRMSE + residual overview ────────────────────────────────────
    plot_nrmse_and_residual_boxplots(
        X_diag, y_diag, yp_diag, known_pos,
        path('01_nrmse_residual_overview.png'),
        title=f'Error Overview — {subject_key} ({split_label})')

    # ── Plot 02: EMG-torque scatter per position ──────────────────────────────
    plot_emg_torque_scatter_by_position(
        X_diag, y_diag, known_pos,
        path('02_emg_torque_scatter.png'),
        title=f'EMG-Torque Scatter — {subject_key} ({split_label})')

    # ── Plot 03: EMG-torque linearity ─────────────────────────────────────────
    plot_emg_torque_linearity_by_position(
        X_diag, y_diag, known_pos,
        path('03_emg_torque_linearity.png'),
        title=f'EMG-Torque Linearity — {subject_key} ({split_label})')

    # ── Plot 04: Deep dive at target position ─────────────────────────────────
    plot_plantarflexion_residual_detail(
        X_diag, y_diag, yp_diag, known_pos, target_pos,
        path('04_detail_plantarflexion.png'),
        title=f'Deep Dive — {subject_key} ({target_pos:+.3f} rad, {split_label})')

    # ── Plot 05: Cross-split NRMSE comparison ─────────────────────────────────
    _plot_cross_split_nrmse(
        [{'label': 'Train',      'X': data['X_train'], 'y_true': data['y_train'], 'y_pred': y_pred_train},
         {'label': 'Validation', 'X': data['X_val'],   'y_true': data['y_val'],   'y_pred': y_pred_val},
         {'label': 'Test',       'X': data['X_test'],  'y_true': y_test,           'y_pred': y_pred_test}],
        known_pos,
        path('05_cross_split_nrmse.png'),
        subject_key=subject_key)

    # ── Console summary ───────────────────────────────────────────────────────
    _print_summary(X_diag, y_diag, yp_diag, known_pos, subject_key, split_label)

    print(f'\nAll plots saved → {out_dir}/')


if __name__ == '__main__':
    main()
