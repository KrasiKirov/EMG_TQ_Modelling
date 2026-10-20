"""
Transfer learning via fine-tuning for cross-subject sEMG-to-torque prediction.

Workflow for each transfer direction (e.g. HM→EG):
  1. Train a base model on the SOURCE subject (same as cross_subject.py)
  2. Fine-tune on K calibration positions from the TARGET subject's test session
  3. Evaluate on the TARGET subject's held-out retest session

Sweep:
  K ∈ {0, 1, 2, 4, 8}  calibration positions
  Strategies:
    dense_only  — freeze both LSTM layers, retrain only the Dense output
    lstm2_dense — freeze first LSTM, retrain second LSTM + Dense
    all_layers  — retrain everything at a lower learning rate

Calibration data comes exclusively from the target's TEST session (train+val
windows), so the RETEST session remains a truly held-out evaluation set.

Calibration positions are selected by uniform spacing across the 8 operating
positions (e.g. K=2 → p2 + p6, K=4 → p1 + p3 + p5 + p7).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Allow `python ML/training/fine_tune.py` from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping

from ML.config import (SUBJECTS, MODEL_DIR, PLOT_DIR,
                        N_STEPS, N_FEATURES, POSITION_FEATURE_INDEX,
                        BATCH_SIZE)
from ML.training import load_subject, train_model, evaluate_model
from ML.models.lstm import build_model
from ML.evaluation import (snap_to_operating_points, plot_full_trials,
                            plot_per_position_metrics, plot_r2_summary)

# ── Fine-tuning hyperparameters ───────────────────────────────────────────────

FINETUNE_LR       = 1e-4   # much lower than base LR to avoid catastrophic forgetting
FINETUNE_EPOCHS   = 200
FINETUNE_PATIENCE = 15     # early stopping on fine-tune val loss

K_VALUES   = [0, 1, 2, 4, 8]
STRATEGIES = ['dense_only', 'lstm2_dense', 'all_layers']
STRATEGY_LABELS = {
    'dense_only':   'Dense only (freeze both LSTMs)',
    'lstm2_dense':  'LSTM2 + Dense (freeze LSTM1)',
    'all_layers':   'All layers (low LR)',
}
STRATEGY_COLORS = {
    'dense_only':  'steelblue',
    'lstm2_dense': 'darkorange',
    'all_layers':  'seagreen',
}


# ── Calibration data extraction ───────────────────────────────────────────────

def _select_cal_positions(operating_positions, k):
    """Pick K uniformly-spaced positions from the sorted operating set."""
    sorted_pos = np.array(sorted(operating_positions))
    if k == 0:
        return []
    if k >= len(sorted_pos):
        return list(sorted_pos)
    indices = np.round(np.linspace(0, len(sorted_pos) - 1, k)).astype(int)
    return list(sorted_pos[indices])


def get_calibration_data(target_data, k):
    """Return (X_cal, y_cal) from the target's test session for K positions.

    The pool is X_train + X_val (test session only — not retest).
    Returns (None, None) when k == 0.
    """
    if k == 0:
        return None, None

    X_pool = np.concatenate([target_data['X_train'], target_data['X_val']])
    y_pool = np.concatenate([target_data['y_train'], target_data['y_val']])

    cal_pos = _select_cal_positions(target_data['operating_positions'], k)
    pos_vals = X_pool[:, -1, POSITION_FEATURE_INDEX]
    pos_snapped, _ = snap_to_operating_points(pos_vals,
                                              target_data['operating_positions'])
    mask = np.isin(pos_snapped, cal_pos)
    return X_pool[mask], y_pool[mask]


# ── Layer freezing ────────────────────────────────────────────────────────────

def _clone_and_freeze(base_model, strategy):
    """Clone base_model and set layer trainability per strategy."""
    cloned = build_model(n_steps=N_STEPS, n_features=N_FEATURES)
    cloned.set_weights(base_model.get_weights())

    lstm_seen = 0
    for layer in cloned.layers:
        if isinstance(layer, LSTM):
            lstm_seen += 1
            if strategy == 'dense_only':
                layer.trainable = False
            elif strategy == 'lstm2_dense':
                layer.trainable = (lstm_seen > 1)   # only second LSTM trains
            else:                                    # all_layers
                layer.trainable = True
        elif isinstance(layer, Dense):
            layer.trainable = True

    cloned.compile(optimizer=Nadam(learning_rate=FINETUNE_LR),
                   loss='mse', metrics=['mae'])
    return cloned


# ── Fine-tuning ───────────────────────────────────────────────────────────────

def _fine_tune(cloned_model, X_cal, y_cal):
    """Fine-tune cloned_model on calibration data with early stopping."""
    rng  = np.random.default_rng(42)
    perm = rng.permutation(len(X_cal))
    X_s, y_s = X_cal[perm], y_cal[perm]
    n_val = max(1, int(len(X_s) * 0.1))
    X_tr, X_vl = X_s[n_val:], X_s[:n_val]
    y_tr, y_vl = y_s[n_val:], y_s[:n_val]

    monitor = 'val_loss' if len(X_vl) >= 4 else 'loss'
    val_data = (X_vl, y_vl) if monitor == 'val_loss' else None

    callbacks = [EarlyStopping(monitor=monitor, patience=FINETUNE_PATIENCE,
                               restore_best_weights=True, verbose=0)]

    fit_kwargs = dict(epochs=FINETUNE_EPOCHS, batch_size=BATCH_SIZE,
                      callbacks=callbacks, shuffle=True, verbose=0)
    if val_data:
        cloned_model.fit(X_tr, y_tr, validation_data=val_data, **fit_kwargs)
    else:
        cloned_model.fit(X_tr, y_tr, **fit_kwargs)

    return cloned_model


# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_transfer_direction(source_key, target_key, source_data, target_data):
    """Train base model on source, sweep fine-tuning on target.

    Returns a results dict:
      {
        'direction'          : 'HM→EG',
        'r2'                 : {strategy: {k: float}},
        'per_position'       : {strategy: {k: per_pos_dict}},
        'within_r2'          : float,     # upper bound
        'within_per_pos'     : per_pos_dict,
        'operating_positions': list,
      }
    """
    direction = f'{source_key}→{target_key}'
    print(f'\n{"="*60}')
    print(f'  Fine-tune sweep: {direction}')
    print(f'{"="*60}')

    # ── 1. Base model (trained on source) ─────────────────────────────────────
    print(f'\n  [1/3] Base model (source = {source_key})')
    base_model, _ = train_model(
        source_data['X_train'], source_data['y_train'],
        source_data['X_val'],   source_data['y_val'],
        label=f'base {source_key}',
        save_path=os.path.join(MODEL_DIR,
                               f'finetune_base_{source_key}.keras'))

    # ── 2. Within-subject upper bound (trained on target from scratch) ────────
    print(f'\n  [2/3] Within-subject upper bound (target = {target_key})')
    within_model, _ = train_model(
        target_data['X_train'], target_data['y_train'],
        target_data['X_val'],   target_data['y_val'],
        label=f'within {target_key}')
    within_result = evaluate_model(
        within_model, target_data['X_test'], target_data['y_test'],
        target_data['operating_positions'],
        mvc_tq_test=target_data['mvc_tq_test'])
    print(f'  Within-subject R² = {within_result["overall"]["r2"]:.4f}')

    # ── 3. Fine-tuning sweep ──────────────────────────────────────────────────
    print(f'\n  [3/3] Fine-tuning sweep (K × strategy)')

    results = {
        'direction':           direction,
        'r2':                  {s: {} for s in STRATEGIES},
        'per_position':        {s: {} for s in STRATEGIES},
        'within_r2':           within_result['overall']['r2'],
        'within_per_pos':      within_result['per_position'],
        'operating_positions': target_data['operating_positions'],
    }

    # K=0: evaluate base model as-is (no fine-tuning) — same for all strategies
    base_result = evaluate_model(
        base_model, target_data['X_test'], target_data['y_test'],
        target_data['operating_positions'],
        mvc_tq_test=target_data['mvc_tq_test'])
    for strategy in STRATEGIES:
        results['r2'][strategy][0]         = base_result['overall']['r2']
        results['per_position'][strategy][0] = base_result['per_position']
    results['X_test']  = target_data['X_test']
    results['y_test']  = base_result['y_test']
    results['y_pred']  = base_result['y_pred']
    print(f'\n  K=0 (no fine-tune): R²={base_result["overall"]["r2"]:.4f}')

    # K > 0: fine-tune
    for k in [v for v in K_VALUES if v > 0]:
        X_cal, y_cal = get_calibration_data(target_data, k)
        cal_pos = _select_cal_positions(target_data['operating_positions'], k)
        print(f'\n  K={k}  |  {X_cal.shape[0]} calibration windows  '
              f'|  positions: {[f"{p:.2f}" for p in cal_pos]}')

        for strategy in STRATEGIES:
            cloned   = _clone_and_freeze(base_model, strategy)
            ft_model = _fine_tune(cloned, X_cal, y_cal)
            ft_result = evaluate_model(
                ft_model, target_data['X_test'], target_data['y_test'],
                target_data['operating_positions'],
                mvc_tq_test=target_data['mvc_tq_test'])
            results['r2'][strategy][k]         = ft_result['overall']['r2']
            results['per_position'][strategy][k] = ft_result['per_position']
            print(f'    {strategy:<15}: R²={ft_result["overall"]["r2"]:.4f}  '
                  f'RMSE={ft_result["overall"]["rmse"]:.3f} Nm')

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_adaptation_curve(results, out_path):
    """R² vs K for each strategy, with within-subject reference."""
    direction = results['direction']
    fig, ax = plt.subplots(figsize=(8, 5))

    for strategy in STRATEGIES:
        r2_vals = [results['r2'][strategy][k] for k in K_VALUES]
        ax.plot(K_VALUES, r2_vals,
                'o-', color=STRATEGY_COLORS[strategy], linewidth=1.8, markersize=6,
                label=STRATEGY_LABELS[strategy])

    ax.axhline(results['within_r2'], color='black', linestyle='--', linewidth=1.2,
               label=f'Within-subject upper bound (R²={results["within_r2"]:.3f})')

    ax.set_xlabel('Calibration positions K', fontsize=10)
    ax.set_ylabel('R\u00b2 (pooled, retest session)', fontsize=10)
    ax.set_title(f'{direction} — Adaptation Curve', fontsize=11)
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.4)
    ax.set_ylim(min(-0.1, min(results['r2'][s][0] for s in STRATEGIES) - 0.05), 1.05)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Adaptation curve saved → {out_path}')


def plot_per_position_recovery(results, strategy, out_path):
    """R² per position at K=0, K=4, and within-subject for a given strategy."""
    known_pos  = np.array(sorted(results['operating_positions']))
    x          = np.arange(1, len(known_pos) + 1)

    def _r2_vals(per_pos_dict):
        return [per_pos_dict[p]['r2'] for p in known_pos]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(x, _r2_vals(results['per_position'][strategy][0]),
            'o--', color='tomato', linewidth=1.5, markersize=6,
            label='K=0 (no fine-tune)')

    if 4 in results['per_position'][strategy]:
        ax.plot(x, _r2_vals(results['per_position'][strategy][4]),
                'o-', color=STRATEGY_COLORS[strategy], linewidth=1.8, markersize=6,
                label=f'K=4  [{STRATEGY_LABELS[strategy]}]')

    ax.plot(x, _r2_vals(results['within_per_pos']),
            'o--', color='black', linewidth=1.2, markersize=6, alpha=0.6,
            label='Within-subject upper bound')

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Position index (dorsiflexion \u2192 plantarflexion)', fontsize=10)
    ax.set_ylabel('R\u00b2', fontsize=10)
    ax.set_title(f'{results["direction"]} — Per-Position Recovery (strategy: {strategy})',
                 fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f'p{i}' for i in x])
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.4)
    ax.set_ylim(-0.1, 1.05)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Per-position recovery saved → {out_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR,  exist_ok=True)

    data = {}
    for subj in SUBJECTS:
        data[subj] = load_subject(subj, normalize_position=True,
                                  normalize_mvc=True)

    all_results = []
    for source_key, target_key in [('HM', 'EG'), ('EG', 'HM')]:
        result = run_transfer_direction(
            source_key, target_key, data[source_key], data[target_key])
        all_results.append(result)

        direction   = result['direction'].replace('\u2192', '_')
        plot_subdir = os.path.join(PLOT_DIR, f'finetune_{direction}')
        os.makedirs(plot_subdir, exist_ok=True)

        plot_adaptation_curve(
            result,
            os.path.join(plot_subdir, 'adaptation_curve.png'))

        plot_per_position_recovery(
            result, strategy='lstm2_dense',
            out_path=os.path.join(plot_subdir, 'per_position_recovery.png'))

        # Cross-subject style plots (K=0 baseline on retest session)
        known_pos = result['operating_positions']
        pooled_r2 = result['r2']['dense_only'][0]
        dir_label = result['direction']

        plot_full_trials(
            result['X_test'], result['y_test'], result['y_pred'],
            os.path.join(plot_subdir, 'full_trial_inspection.png'),
            known_positions=known_pos,
            title=f'{dir_label} — K=0 Baseline (Retest Session)')

        pos_r2 = plot_per_position_metrics(
            result['X_test'], result['y_test'], result['y_pred'],
            os.path.join(plot_subdir, 'per_position_metrics.png'),
            known_positions=known_pos, pooled_r2=pooled_r2,
            title=f'{dir_label} — K=0 Per-Position Metrics')

        plot_r2_summary(
            pos_r2, pooled_r2,
            os.path.join(plot_subdir, 'r2_vs_position.png'),
            title=f'{dir_label} — K=0 R² vs Position')

    # Summary table
    print(f'\n\n{"="*70}')
    print(f'FINE-TUNING SUMMARY')
    print(f'{"="*70}')
    header = f'  {"Direction":<10}  {"Strategy":<17}  ' + \
             '  '.join(f'K={k:>1}' for k in K_VALUES) + \
             f'  {"within":>7}'
    print(header)
    print(f'  {"-"*10}  {"-"*17}  ' + '  '.join('-----' for _ in K_VALUES) + '  -------')
    for result in all_results:
        direction = result['direction']
        for strategy in STRATEGIES:
            r2_row = '  '.join(f'{result["r2"][strategy][k]:5.3f}'
                                for k in K_VALUES)
            ws = result['within_r2']
            print(f'  {direction:<10}  {strategy:<17}  {r2_row}  {ws:7.3f}')

    print(f'\nDone.')


if __name__ == '__main__':
    main()
