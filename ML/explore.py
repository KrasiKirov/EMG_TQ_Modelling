"""
explore.py
----------
Exploratory CLI: reads a REKLAB .flb file, exports every trial to CSV, and
generates signal plots.

Usage
-----
    python -m ML.explore <path/to/file.flb> [options]

Options
-------
    --output_dir DIR    Root output directory (default: <flb_name>_output/)
    --subject_id ID     Subject ID; use 'IES01' for that channel order
                        (default: default)
    --downsample N      Downsample factor for the overlay plot (default: 10)
    --no_csv            Skip CSV export
    --no_plots          Skip plot generation

Output structure
----------------
    <output_dir>/
        csv/
            <flb_name>_trial001.csv
            <flb_name>_trial002.csv
            ...
        plots/
            trial01_all_channels.png   — all 6 channels for Trial 1
            all_trials_overlay.png     — all trials overlaid per channel
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend; works without a display
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ML.preprocessing.flb_reader import read_flb
from ML.preprocessing.dataset_builder import build_dataset, classify_trials


# ── Plotting helpers ──────────────────────────────────────────────────────

# Per-channel display names and units (two separate strings for two-line y-label)
CHANNEL_META = {
    'position': ('position',  'rad'),
    'torque':   ('torque',    'Nm'),
    'gm':       ('gm_emg',    'mV'),
    'gl':       ('gl_emg',    'mV'),
    'sol':      ('sol_emg',   'mV'),
    'ta':       ('ta_emg',    'mV'),
}

# One distinct colour per channel (matches reference image)
CHANNEL_COLORS = {
    'position': '#1f77b4',   # blue
    'torque':   '#d62728',   # red
    'gm':       '#2ca02c',   # dark green
    'gl':       '#ff7f0e',   # orange
    'sol':      '#9467bd',   # purple
    'ta':       '#8c564b',   # brown
}

# Aliases for backward compatibility with old short names
_NAME_ALIASES = {'pos': 'position', 'tq': 'torque'}


def _ch_label(col: str):
    """Return (name, unit) for y-axis label."""
    key = _NAME_ALIASES.get(col.lower(), col.lower())
    return CHANNEL_META.get(key, (col, ''))


# Ordered fallback palette — applied positionally when name doesn't match
_COLOR_PALETTE = list(CHANNEL_COLORS.values())


def _ch_color(col: str, idx: int = 0) -> str:
    """Return the hex colour for a channel by name, then by index."""
    key = _NAME_ALIASES.get(col.lower(), col.lower())
    c = CHANNEL_COLORS.get(key)
    if c:
        return c
    return _COLOR_PALETTE[idx % len(_COLOR_PALETTE)]


def plot_trial(df, trial_num: int, output_path: str) -> None:
    """
    Save a figure showing all signal channels for a single trial,
    styled to match the reference image.

    Parameters
    ----------
    df : pd.DataFrame
        Trial DataFrame with a 'time' column plus one column per channel.
    trial_num : int
        1-based trial index (used in the figure title).
    output_path : str
        Full path for the saved PNG file.
    """
    signal_cols = [c for c in df.columns if c != 'time']
    n_ch  = len(signal_cols)
    n_smp = len(df)
    fs    = round(1.0 / df.attrs.get('domainIncr', 0.001))
    dur   = n_smp / fs
    subj  = os.path.splitext(df.attrs.get('source_file', ''))[0]
    comment = df.attrs.get('comment', '')

    # Build two-line title
    trial_label  = f'{subj} — Trial {trial_num} (first test trial)' if trial_num == 1 \
                   else f'{subj} — Trial {trial_num}'
    stats_label  = f'Comment: "{comment}"  |  {n_smp} samples @ {fs} Hz = {dur:.1f} s'

    fig, axes = plt.subplots(n_ch, 1, figsize=(14, 2.5 * n_ch),
                             sharex=True, facecolor='white')
    if n_ch == 1:
        axes = [axes]

    fig.suptitle(f'{trial_label}\n{stats_label}',
                 fontsize=10, fontweight='bold', y=1.01,
                 fontfamily='monospace')

    for idx, (ax, col) in enumerate(zip(axes, signal_cols)):
        name, unit = _ch_label(col)
        color = _ch_color(col, idx)

        ax.plot(df['time'], df[col], color=color, linewidth=0.55)
        ax.set_facecolor('white')

        # Two-line y-label: name on top, (unit) below
        ax.set_ylabel(f'{name}\n({unit})', fontsize=8, labelpad=6,
                      rotation=90, va='center')

        # Tight y-limits with a small margin
        ydata = df[col].values
        ymin, ymax = ydata.min(), ydata.max()
        margin = (ymax - ymin) * 0.08 if ymax != ymin else 0.1
        ax.set_ylim(ymin - margin, ymax + margin)

        ax.grid(True, linewidth=0.4, color='#cccccc', linestyle='-')
        ax.tick_params(labelsize=7.5)
        ax.spines[['top', 'right']].set_visible(False)

    axes[-1].set_xlabel('Time (s)', fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_emg_envelopes(df, output_path: str, title: str = '') -> None:
    """
    Plot raw EMG (blue) with the extracted envelope overlaid (red).

    Expects the DataFrame to already have ``*_env`` columns added by
    ``process_trials``.  One subplot per EMG channel.

    Parameters
    ----------
    df : pd.DataFrame
        Trial DataFrame with ``*_env`` columns present.
    output_path : str
        Full path for the saved PNG file.
    title : str
        Figure suptitle (e.g. "Ankle Position = -0.5 rad").
    """
    # Identify channel names by finding *_env columns
    env_cols  = [c for c in df.columns if c.endswith('_env') and not c.endswith('_env_norm')]
    base_cols = [c[:-4] for c in env_cols]   # strip '_env'
    # Keep only channels that also have the rectified intermediate
    triples = [(b, f'{b}_rect', f'{b}_env')
               for b in base_cols if f'{b}_rect' in df.columns]

    if not triples:
        print("No rectified+envelope column pairs found — run process_trials first.")
        return

    n_ch = len(triples)
    fig, axes = plt.subplots(n_ch, 1, figsize=(12, 2.8 * n_ch),
                             sharex=True, facecolor='white')
    if n_ch == 1:
        axes = [axes]

    if title:
        fig.suptitle(title, fontsize=10, fontweight='bold')

    for ax, (base_col, rect_col, env_col) in zip(axes, triples):
        t = df['time'].values
        rect = df[rect_col].values   # bandpass-filtered + rectified
        env  = df[env_col].values

        ax.plot(t, rect, color='#4c9be8', linewidth=0.4, alpha=0.55, label='|Bandpass filtered|')
        ax.plot(t, env,  color='#d62728', linewidth=1.8,  label='Envelope')

        name, unit = _ch_label(base_col)
        ax.set_ylabel(f'{name.upper()}\n({unit})', fontsize=8, labelpad=6,
                      rotation=90, va='center')
        ax.set_facecolor('white')
        ax.grid(True, linewidth=0.35, color='#cccccc')
        ax.tick_params(labelsize=7.5)
        ax.spines[['top', 'right']].set_visible(False)

    axes[-1].set_xlabel('Time (s)', fontsize=9)
    axes[0].legend(fontsize=7, loc='upper right', framealpha=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved envelope plot → '{output_path}'")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Read a REKLAB .flb file, export CSVs, and generate plots.')
    parser.add_argument('flb_file',
                        help='Path to the .flb file.')
    parser.add_argument('--output_dir', default=None,
                        help='Root output directory '
                             '(default: <flb_name>_output/ next to the .flb file).')
    parser.add_argument('--subject_id', default='default',
                        help="Subject ID. Use 'IES01' for that subject's channel order.")
    parser.add_argument('--downsample', type=int, default=10,
                        help='Downsample factor for the overlay plot (default: 10).')
    parser.add_argument('--trial', type=int, default=1,
                        help='Trial number to plot (1-based, default: 1).')
    parser.add_argument('--no_csv', action='store_true',
                        help='Skip CSV export.')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip plot generation.')
    parser.add_argument('--test-trials', nargs='+', type=int, default=None,
                        help='1-indexed test trial numbers (default: auto-detect)')
    parser.add_argument('--retest-trials', nargs='+', type=int, default=None,
                        help='1-indexed retest trial numbers (default: auto-detect)')
    args = parser.parse_args()

    flb_path = os.path.abspath(args.flb_file)
    if not os.path.isfile(flb_path):
        raise FileNotFoundError(f"File not found: {flb_path}")

    base_name = os.path.splitext(os.path.basename(flb_path))[0]

    # Resolve output directory
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(os.path.dirname(flb_path), f'{base_name}_output')

    csv_dir   = os.path.join(output_dir, 'csv')
    plots_dir = os.path.join(output_dir, 'plots')

    # ── Read all trials ───────────────────────────────────────────────────
    trials = read_flb(flb_path, subject_id=args.subject_id)

    if not trials:
        print("No trials found in the file. Exiting.")
        return

    # ── Export CSVs ───────────────────────────────────────────────────────
    if not args.no_csv:
        os.makedirs(csv_dir, exist_ok=True)
        csv_paths = []
        for i, df in enumerate(trials):
            csv_name = f"{base_name}_trial{i+1:03d}.csv"
            csv_path = os.path.join(csv_dir, csv_name)
            df.to_csv(csv_path, index=False)
            csv_paths.append(csv_path)
        print(f"Saved {len(csv_paths)} CSV files → '{csv_dir}'")

        # Quick preview of trial 1
        import pandas as pd
        print("\nTrial 1 preview (first 5 rows):")
        print(pd.read_csv(csv_paths[0]).head().to_string(index=False))

    # ── Generate raw signal plots (before build_dataset adds envelope columns) ─
    if not args.no_plots:
        os.makedirs(plots_dir, exist_ok=True)

        # Single-trial plot (--trial N, default 1)
        t_idx = args.trial - 1
        if not (0 <= t_idx < len(trials)):
            print(f"Warning: --trial {args.trial} out of range "
                  f"(file has {len(trials)} trials). Plotting trial 1.")
            t_idx = 0
        trial_path = os.path.join(plots_dir, f'trial{t_idx+1:02d}_all_channels.png')
        plot_trial(trials[t_idx], trial_num=t_idx + 1, output_path=trial_path)
        print(f"Saved trial-{t_idx+1} plot     → '{trial_path}'")

    # ── Build dataset (classify → envelopes → passive subtract → windows) ─
    (X_train, y_train, X_val, y_val, X_test, y_test,
     emg_max, passive_entries, _operating_pos, _pos_range) = \
        build_dataset(trials,
                      test_trial_indices=args.test_trials,
                      retest_trial_indices=args.retest_trials)

    print("\n── Dataset quality checks ──────────────────────────────────────")
    print(f"  NaN in inputs : {np.isnan(X_train).any()}")
    print(f"  EMG range     : [{X_train[:, :, :4].min():.4f},  {X_train[:, :, :4].max():.4f}]  (expect ≈ [0, 1])")
    print(f"  Torque range  : [{y_train.min():.2f},  {y_train.max():.2f}] Nm")

    # ── Envelope check plot (after build_dataset adds *_env/*_rect columns) ─
    if not args.no_plots:
        _, _, active_trials = classify_trials(trials)
        if active_trials:
            first_active = active_trials[0]
            comment = first_active.attrs.get('comment', '')
            env_path = os.path.join(plots_dir, 'envelope_check.png')
            plot_emg_envelopes(first_active, env_path,
                               title=f'EMG Envelope Check — {comment}')

    print("\nDone.")


if __name__ == '__main__':
    main()
