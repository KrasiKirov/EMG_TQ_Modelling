"""
Exploratory CLI: read a REKLAB .flb file, export every trial to CSV and plot the signals.

    python -m scripts.explore_flb <file.flb> [--output_dir DIR] [--subject_id IES01] [--trial N] [--no_csv] [--no_plots]

Writes <output_dir>/csv/<flb_name>_trialNNN.csv and plots/trialNN_all_channels.png.
"""

import os
import argparse
import matplotlib
matplotlib.use('Agg')  # non-interactive backend; works without a display
import matplotlib.pyplot as plt

from ML.preprocessing.flb_reader import read_flb


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
    """Save a figure of every channel of one trial (``trial_num`` is 1-based)."""
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
    parser.add_argument('--trial', type=int, default=1,
                        help='Trial number to plot (1-based, default: 1).')
    parser.add_argument('--no_csv', action='store_true',
                        help='Skip CSV export.')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip plot generation.')
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

    # Read all trials
    trials = read_flb(flb_path, subject_id=args.subject_id)

    if not trials:
        print("No trials found in the file. Exiting.")
        return

    # Export CSVs
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

    # Raw signal plot
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

    print("\nDone.")


if __name__ == '__main__':
    main()
