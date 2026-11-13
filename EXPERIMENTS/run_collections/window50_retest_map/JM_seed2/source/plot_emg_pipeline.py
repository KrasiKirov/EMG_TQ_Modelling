"""
Plot one active isometric trial through the production EMG envelope steps.

Usage
-----
    python -m ML.plot_emg_pipeline --subject YES --trial 11 --muscle gm
    python -m ML.plot_emg_pipeline --subject YES

Normalization uses the train/val global max for that muscle (no leakage
from retest, not this trial's own peak).
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ML.config import SUBJECTS, DATA_DIR, PLOT_DIR
from ML.preprocessing.flb_reader import read_flb
from ML.preprocessing.dataset_builder import (
    classify_trials, TORQUE_COL, _ACTIVE_STD_THRESHOLD,
)
from ML.preprocessing.emg_envelope import (
    extract_envelope, detect_emg_columns, process_trials,
)


MUSCLE_ALIASES = {
    '1': 'gm', 'muscle1': 'gm', 'muscle_1': 'gm', 'mg': 'gm', 'gm': 'gm',
    '2': 'gl', 'muscle2': 'gl', 'muscle_2': 'gl', 'lg': 'gl', 'gl': 'gl',
    '3': 'sol', 'muscle3': 'sol', 'muscle_3': 'sol', 'sol': 'sol',
    '4': 'ta', 'muscle4': 'ta', 'muscle_4': 'ta', 'ta': 'ta',
}

MUSCLE_LABELS = {
    'gm': 'GM (medial gastrocnemius)',
    'gl': 'GL (lateral gastrocnemius)',
    'sol': 'SOL (soleus)',
    'ta': 'TA (tibialis anterior)',
}


def resolve_muscle(name):
    key = str(name).strip().lower().replace(' ', '').replace('-', '_')
    if key.startswith('muscle') and len(key) > 6:
        key = key.replace('muscle', '', 1)
        if key.startswith('_'):
            key = key[1:]
    if key not in MUSCLE_ALIASES:
        raise ValueError(
            f"Unknown muscle '{name}'. Use gm/gl/sol/ta or muscle 1–4.")
    return MUSCLE_ALIASES[key]


def isometric_active_trials(trials):
    """Active trials that pass the isometric torque-std filter."""
    _, _, active = classify_trials(trials)
    return [df for df in active if df[TORQUE_COL].std() <= _ACTIVE_STD_THRESHOLD]


def split_trainval_retest(active_trials):
    """Split isometric active trials using comment session tags."""
    trainval, retest, unlabelled = [], [], []
    for df in active_trials:
        session = df.attrs.get('session')
        if session == 'retest':
            retest.append(df)
        elif session == 'test':
            trainval.append(df)
        else:
            unlabelled.append(df)
    if not trainval and retest and unlabelled:
        trainval = unlabelled
        unlabelled = []
    if not trainval:
        trainval = unlabelled
    return trainval, retest


def pick_pipeline_trial(active_trials, trial_number=None):
    """Pick a trial by 1-indexed number, else first test-session isometric."""
    if not active_trials:
        raise ValueError('No isometric active trials found.')
    if trial_number is not None:
        for df in active_trials:
            if df.attrs.get('trial_index') == trial_number - 1:
                return df
        available = sorted(df.attrs.get('trial_index') + 1 for df in active_trials)
        raise ValueError(
            f'Trial {trial_number} is not an isometric active trial. '
            f'Available: {available}')
    trainval, _ = split_trainval_retest(active_trials)
    pool = trainval if trainval else active_trials
    return pool[0]


def plot_pipeline_figure(raw, rectified, envelope, normalized, time,
                         title, out_path):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title, fontsize=16, fontweight='bold')

    axes[0].plot(time, raw, color='gray', linewidth=0.5)
    axes[0].set_title('1. Raw sEMG Signal', fontsize=12)
    axes[0].set_ylabel('Amplitude (mV)')

    axes[1].plot(time, rectified, color='steelblue', linewidth=0.5)
    axes[1].set_title('2. Bandpass Filtered (30-300Hz) & Rectified |x(t)|', fontsize=12)
    axes[1].set_ylabel('Amplitude (mV)')

    axes[2].plot(time, envelope, color='crimson', linewidth=1.5)
    axes[2].set_title('3. Low-pass Filtered (2Hz) Linear Envelope', fontsize=12)
    axes[2].set_ylabel('Amplitude (mV)')

    axes[3].plot(time, normalized, color='darkorange', linewidth=1.5)
    axes[3].set_title('4. Normalized [0, 1]  (train/val global max)', fontsize=12)
    axes[3].set_ylabel('Normalized Activation')
    axes[3].set_xlabel('Time (seconds)', fontsize=12)
    ymax = float(np.nanmax(normalized)) if len(normalized) else 1.0
    axes[3].set_ylim([-0.05, max(1.05, ymax * 1.05)])

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description='Plot production EMG envelope steps for one active trial.')
    p.add_argument('--subject', choices=list(SUBJECTS.keys()), default='YES',
                   help='Subject short name (default: YES)')
    p.add_argument('--flb', default=None, help='Override path to .flb file')
    p.add_argument('--subject-id', default=None,
                   help='Override subject ID for channel ordering')
    p.add_argument('--trial', type=int, default=None,
                   help='1-indexed trial number (default: first test-session isometric)')
    p.add_argument('--muscle', default='gm',
                   help='Muscle: gm/gl/sol/ta or muscle 1–4 (default: gm)')
    p.add_argument('--output', default=None, help='Output PNG path')
    return p.parse_args()


def main():
    args = parse_args()
    muscle = resolve_muscle(args.muscle)
    flb_name, subj_id = SUBJECTS[args.subject]
    flb_path = args.flb or os.path.join(DATA_DIR, flb_name)
    subject_id = args.subject_id or subj_id

    if not os.path.isfile(flb_path):
        raise FileNotFoundError(f'FLB not found: {flb_path}')

    trials = read_flb(flb_path, subject_id=subject_id)
    active = isometric_active_trials(trials)
    trainval, _ = split_trainval_retest(active)
    if not trainval:
        raise ValueError('No test-session isometric trials to compute EMG max.')

    emg_columns = detect_emg_columns(trainval[0])
    if muscle not in emg_columns:
        raise ValueError(f"Muscle '{muscle}' not in {emg_columns}")

    _, emg_max = process_trials(trainval, emg_columns=emg_columns)
    chosen = pick_pipeline_trial(active, trial_number=args.trial)
    raw = chosen[muscle].values.astype(np.float64)
    envelope, stages = extract_envelope(raw, return_intermediates=True)
    scale = emg_max[muscle] if emg_max[muscle] > 0 else 1.0
    normalized = envelope / scale

    trial_num = chosen.attrs.get('trial_index', 0) + 1
    comment = chosen.attrs.get('comment', '')
    muscle_label = MUSCLE_LABELS.get(muscle, muscle.upper())
    title = (f'EMG Data Processing Pipeline '
             f'({args.subject} — Trial {trial_num} — {muscle_label})')
    if comment:
        title += f'\n"{comment}"'

    out_path = args.output or os.path.join(
        PLOT_DIR, args.subject,
        f'emg_pipeline_trial{trial_num:02d}_{muscle}.png')

    time = chosen['time'].values if 'time' in chosen.columns \
        else np.arange(len(raw)) / 1000.0
    plot_pipeline_figure(
        raw, stages['rectified'], envelope, normalized, time, title, out_path)

    print(f'Trial {trial_num}  comment="{comment}"')
    print(f'Muscle {muscle}  train/val max={scale:.6f}  '
          f'trial peak / max={envelope.max() / scale:.3f}')
    print(f'Saved → {out_path}')


if __name__ == '__main__':
    main()
