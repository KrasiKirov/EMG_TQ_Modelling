"""
Cross-subject transferability testing for sEMG-to-torque LSTM.

Evaluates all cross-subject (train_subject, test_subject) pairs:
    HM→EG, EG→HM

For each pair, a fresh model is trained on the training subject's
test-session trials and evaluated on the test subject's retest trials.

Position is min-max normalized to [0, 1] per subject so that
"most dorsiflexed" = 0 and "most plantarflexed" = 1, removing the
effect of different absolute ankle angles between subjects.

EMG is normalized per subject using per-position MVC peaks
(standard practice — avoids cross-subject amplitude bias).
"""

import os
import sys
import numpy as np

# Allow `python ML/training/cross_subject.py` from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ML.config import SUBJECTS, MODEL_DIR, PLOT_DIR
from ML.training import load_subject, train_model, evaluate_model
from ML.evaluation import (plot_full_trials, plot_per_position_metrics,
                            plot_r2_summary, plot_r2_overlay)


def train_and_evaluate(train_data, test_data, label):
    """Train a fresh model on train_data, evaluate on test_data."""
    print(f'\n{"="*60}')
    print(f'  {label}')
    print(f'{"="*60}')
    print(f'  Train: {train_data["X_train"].shape[0]} windows  |  '
          f'Val: {train_data["X_val"].shape[0]}  |  '
          f'Test: {test_data["X_test"].shape[0]}')

    ckpt_path = os.path.join(MODEL_DIR,
                              f'cross_subj_{label.replace("→", "_")}.keras')
    model, _ = train_model(
        train_data['X_train'], train_data['y_train'],
        train_data['X_val'],   train_data['y_val'],
        label=label, save_path=ckpt_path,
    )

    result = evaluate_model(
        model, test_data['X_test'], test_data['y_test'],
        test_data['operating_positions'],
        mvc_tq_test=test_data['mvc_tq_test'],
    )

    overall = result['overall']
    per_pos = result['per_position']

    print(f'\n  Results ({label}):')
    print(f'    Pooled R² = {overall["r2"]:.4f}   RMSE = {overall["rmse"]:.4f} Nm   '
          f'MAE = {overall["mae"]:.4f} Nm')
    print(f'    Per-position metrics:')
    print(f'      {"pos":>7s}  {"R²":>8s}  {"RMSE":>8s}  {"NRMSE":>8s}  {"MAE":>8s}')
    for pos in sorted(per_pos):
        m = per_pos[pos]
        print(f'      {pos:7.3f}  {m["r2"]:8.4f}  {m["rmse"]:8.4f}  {m["nrmse"]:8.4f}  {m["mae"]:8.4f}')

    return {
        'label': label,
        'overall': overall,
        'per_position': per_pos,
        'X_test': test_data['X_test'],
        'y_test': result['y_test'],
        'y_pred': result['y_pred'],
        'known_positions': test_data['operating_positions'],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR,  exist_ok=True)

    # Load both subjects
    data = {}
    for subj in SUBJECTS:
        data[subj] = load_subject(subj, normalize_position=True,
                                  normalize_mvc=True)

    # Cross-subject pairs only (within-subject baselines are in train.py)
    subject_keys = list(SUBJECTS.keys())
    pairs = []
    for train_s in subject_keys:
        for test_s in subject_keys:
            if train_s != test_s:
                pairs.append((train_s, test_s, f'{train_s}→{test_s}'))

    results = []
    for train_subj, test_subj, label in pairs:
        # Pool source's test + retest sessions for training
        src = data[train_subj]
        train_data = {
            'X_train': np.concatenate([src['X_train'], src['X_test']]),
            'y_train': np.concatenate([src['y_train'], src['y_test']]),
            'X_val':   src['X_val'],
            'y_val':   src['y_val'],
        }
        res = train_and_evaluate(train_data, data[test_subj], label)
        results.append(res)

    # Summary table
    print(f'\n\n{"="*60}')
    print(f'CROSS-SUBJECT SUMMARY')
    print(f'{"="*60}')
    print(f'  {"Scenario":<10}  {"R²":>7}  {"RMSE (Nm)":>10}  {"MAE (Nm)":>9}')
    print(f'  {"-"*10}  {"-"*7}  {"-"*10}  {"-"*9}')
    for r in results:
        o = r['overall']
        print(f'  {r["label"]:<10}  {o["r2"]:7.4f}  {o["rmse"]:10.4f}  {o["mae"]:9.4f}')

    # ── Per-scenario subfolders ────────────────────────────────────────────────
    for res in results:
        label    = res['label']
        folder   = label.replace('→', '_')
        scen_dir = os.path.join(PLOT_DIR, folder)
        os.makedirs(scen_dir, exist_ok=True)

        X_test    = res['X_test']
        y_test    = res['y_test']
        y_pred    = res['y_pred']
        known_pos = res['known_positions']
        pooled_r2 = res['overall']['r2']

        plot_full_trials(
            X_test, y_test, y_pred,
            os.path.join(scen_dir, 'full_trial_inspection.png'),
            known_positions=known_pos,
            title=f'{label} — Full Trial Inspection (Retest Session)')

        pos_r2 = plot_per_position_metrics(
            X_test, y_test, y_pred,
            os.path.join(scen_dir, 'per_position_metrics.png'),
            known_positions=known_pos, pooled_r2=pooled_r2,
            title=f'{label} — Per-Position Metrics (Retest Session)')

        plot_r2_summary(
            pos_r2, pooled_r2,
            os.path.join(scen_dir, 'r2_vs_position.png'),
            title=f'{label} — R\u00b2 vs Position (Retest Session)')

    # ── Cross-subject comparison overlay ──────────────────────────────────────
    plot_r2_overlay(results,
                    os.path.join(PLOT_DIR, 'cross_subject_r2_overlay.png'))

    print(f'\nDone. All plots saved to {PLOT_DIR}/')


if __name__ == '__main__':
    main()
