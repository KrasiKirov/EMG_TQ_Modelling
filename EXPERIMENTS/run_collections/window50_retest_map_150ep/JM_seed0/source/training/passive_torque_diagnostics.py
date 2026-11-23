"""
Passive torque quality diagnostics.

Investigates whether poor model performance at extreme ankle positions is
caused by inaccurate passive torque subtraction.  Six plots are generated,
each targeting a distinct failure mode:

  01_passive_coverage.png          — segment counts per cluster (accepted vs rejected)
  02_interpolation_range.png       — which operating positions are extrapolated
  03_passive_variability.png       — within-cluster reproducibility
  04_passive_curve.png             — physiological plausibility (monotonicity, slope)
  05_residual_trainval.png         — active torque distribution per position (train+val)
  05_residual_test.png             — same, for held-out retest session
  06_extreme_vs_middle_summary.png — all metrics side-by-side, extremes highlighted

Usage
-----
    python ML/training/passive_torque_diagnostics.py --subject HM
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ML.config import SUBJECTS, DATA_DIR, PLOT_DIR
from ML.preprocessing.flb_reader import read_flb
from ML.preprocessing.dataset_builder import (
    classify_trials,
    get_passive_torque_map,
    _split_passive_segments,
    _PASSIVE_STD_THRESHOLD,
    _PASSIVE_TQ_BOUND,
    POSITION_COL,
    TORQUE_COL,
    build_dataset,
)
from ML.evaluation import (
    plot_passive_coverage,
    plot_passive_interpolation_range,
    plot_passive_variability,
    plot_passive_torque_curve,
    plot_passive_residual_by_position,
    plot_extreme_vs_middle_summary,
)


def parse_args():
    p = argparse.ArgumentParser(
        description='Passive torque quality diagnostics for one subject.')
    p.add_argument('--subject', choices=list(SUBJECTS.keys()), default=None,
                   help='Subject short name (e.g. HM, EG)')
    p.add_argument('--flb', default=None,
                   help='Override path to .flb file')
    p.add_argument('--subject-id', default=None,
                   help='Override subject ID for channel ordering')
    p.add_argument('--output-dir', default=None,
                   help='Output directory (default: ML/plots/{subject}/passive_diagnostics/)')
    p.add_argument('--test-trials', nargs='+', type=int, default=None,
                   help='1-indexed test trial numbers (default: auto-detect)')
    p.add_argument('--retest-trials', nargs='+', type=int, default=None,
                   help='1-indexed retest trial numbers (default: auto-detect)')
    p.add_argument('--std-threshold', type=float, default=_PASSIVE_STD_THRESHOLD,
                   help=f'Passive segment rejection threshold in Nm '
                        f'(default: {_PASSIVE_STD_THRESHOLD})')
    p.add_argument('--variability-flag-threshold', type=float, default=2.0,
                   help='Within-cluster std flag threshold in Nm (default: 2.0)')
    p.add_argument('--passive-pos', nargs='+', type=float, default=None,
                   metavar='RAD',
                   help='Ankle positions (rad) for manual passive torque map. '
                        'Must be paired with --passive-tq in the same order.')
    p.add_argument('--passive-tq', nargs='+', type=float, default=None,
                   metavar='NM',
                   help='Passive torque values (Nm) for manual passive torque map. '
                        'Must be paired with --passive-pos in the same order.')
    return p.parse_args()


def _collect_raw_segments(passive_trials, std_threshold, pos_tolerance):
    """Re-collect pre-merge passive segment data with pass/fail labels.

    Mirrors the logic inside get_passive_torque_map but preserves the
    per-segment details that the production function discards.

    Returns
    -------
    list of dict with keys: pos, tq, std, n_samples, passed
    """
    raw_segments = []
    for df in passive_trials:
        for seg in _split_passive_segments(df, POSITION_COL):
            tq_std   = float(seg[TORQUE_COL].std())
            mean_pos = float(seg[POSITION_COL].mean())
            mean_tq  = float(seg[TORQUE_COL].mean())
            passed   = (tq_std <= std_threshold) and (abs(mean_tq) <= _PASSIVE_TQ_BOUND)
            raw_segments.append({
                'pos':      mean_pos,
                'tq':       mean_tq,
                'std':      tq_std,
                'n_samples': len(seg),
                'passed':   passed,
            })
    return raw_segments


def _print_summary(passive_entries, raw_segments, operating_positions,
                   std_threshold, variability_flag_threshold):
    """Print a structured console summary with flagged issues."""
    from collections import Counter, defaultdict

    passive_entries = sorted(passive_entries, key=lambda x: x[0])
    cluster_pos = [p for p, _ in passive_entries]
    pos_min, pos_max = cluster_pos[0], cluster_pos[-1]

    accepted = [s for s in raw_segments if s['passed']]
    rejected = [s for s in raw_segments if not s['passed']]

    def _snap(pos):
        return min(cluster_pos, key=lambda cp: abs(cp - pos))

    from collections import Counter, defaultdict
    acc_counts = Counter(_snap(s['pos']) for s in accepted)
    groups = defaultdict(list)
    for s in accepted:
        groups[_snap(s['pos'])].append(s['tq'])

    sep = '=' * 70
    print(f'\n{sep}')
    print(f'PASSIVE TORQUE DIAGNOSTIC SUMMARY')
    print(f'{sep}')
    print(f'  Passive clusters found : {len(passive_entries)}')
    print(f'  Raw segments — accepted: {len(accepted)}  rejected: {len(rejected)}')
    print(f'  Passive range          : [{pos_min:+.3f}, {pos_max:+.3f}] rad')
    print(f'  Operating range        : [{min(operating_positions):+.3f}, '
          f'{max(operating_positions):+.3f}] rad')
    print()
    print(f'  {"pos (rad)":>10}  {"count":>6}  {"merged_tq (Nm)":>15}  '
          f'{"within_std (Nm)":>16}  {"status":>10}  {"extrap_dist":>12}')
    print(f'  {"-"*10}  {"-"*6}  {"-"*15}  {"-"*16}  {"-"*10}  {"-"*12}')

    for p, tq in passive_entries:
        count = acc_counts.get(p, 0)
        vals  = groups.get(p, [])
        std   = float(np.std(vals)) if len(vals) > 1 else float('nan')
        std_s = f'{std:.3f}' if not np.isnan(std) else 'N/A'
        extrap = p < pos_min or p > pos_max  # never true for clusters themselves
        print(f'  {p:>+10.3f}  {count:>6}  {tq:>+15.4f}  {std_s:>16}  '
              f'{"OK":>10}  {"0.000":>12}')

    print(f'\n  Operating positions vs passive range:')
    flags = []
    for op in sorted(operating_positions):
        if op < pos_min:
            dist  = pos_min - op
            label = 'EXTRAP'
            flags.append(f'[CRITICAL] Position {op:+.3f} rad: extrapolated '
                         f'({dist:.4f} rad outside measured range)')
        elif op > pos_max:
            dist  = op - pos_max
            label = 'EXTRAP'
            flags.append(f'[CRITICAL] Position {op:+.3f} rad: extrapolated '
                         f'({dist:.4f} rad outside measured range)')
        else:
            label = 'interp'
        print(f'    {op:+.3f} rad  →  {label}')

    # Single-segment clusters at operating positions
    for op in operating_positions:
        snap = _snap(op)
        cnt  = acc_counts.get(snap, 0)
        if cnt == 1:
            flags.append(f'[WARNING] Position {op:+.3f} rad: nearest cluster '
                         f'({snap:+.3f}) has only 1 accepted segment')
        if cnt == 0:
            flags.append(f'[CRITICAL] Position {op:+.3f} rad: NO accepted '
                         f'passive segments — using extrapolation with zero support')

    # Within-cluster variability
    for p, _ in passive_entries:
        vals = groups.get(p, [])
        if len(vals) > 1:
            std = float(np.std(vals))
            if std > variability_flag_threshold:
                flags.append(f'[WARNING] Cluster {p:+.3f} rad: within-cluster '
                             f'std = {std:.3f} Nm > {variability_flag_threshold} Nm threshold')

    # Monotonicity
    tqs = [tq for _, tq in passive_entries]
    diffs = np.diff(tqs)
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        flags.append('[CRITICAL] Passive torque curve is NON-MONOTONIC — '
                     'check for mislabeled passive trials')

    print()
    if flags:
        print(f'  FLAGS ({len(flags)}):')
        for f in flags:
            print(f'    {f}')
    else:
        print('  No flags raised — passive torque map looks clean.')
    print(sep)


def main():
    args = parse_args()
    subject_key = args.subject or next(iter(SUBJECTS))

    flb_name, subj_id = SUBJECTS[subject_key]
    flb_path   = args.flb or os.path.join(DATA_DIR, flb_name)
    subject_id = args.subject_id or subj_id

    out_dir = args.output_dir or os.path.join(
        PLOT_DIR, subject_key, 'passive_diagnostics')
    os.makedirs(out_dir, exist_ok=True)

    print(f'\n{"─"*60}')
    print(f'Passive torque diagnostics — subject: {subject_key}')
    print(f'{"─"*60}')

    # ── 1. Load raw trials ────────────────────────────────────────────────────
    trials = read_flb(flb_path, subject_id=subject_id)
    _, passive_trials, _ = classify_trials(trials)

    # ── 2. Collect raw segment data (pre-merge) ───────────────────────────────
    raw_segments = _collect_raw_segments(
        passive_trials,
        std_threshold=args.std_threshold,
        pos_tolerance=0.05,
    )
    print(f'\nRaw passive segments: {len(raw_segments)}  '
          f'(accepted: {sum(s["passed"] for s in raw_segments)}  '
          f'rejected: {sum(not s["passed"] for s in raw_segments)})')

    # ── 3. Build the final passive map ────────────────────────────────────────
    if args.passive_pos and args.passive_tq:
        passive_entries = sorted(zip(args.passive_pos, args.passive_tq), key=lambda x: x[0])
        print(f'\nPassive map (manually overridden — {len(passive_entries)} entries):')
        for pos, tq in passive_entries:
            print(f'  pos ≈ {pos:+.3f} rad → passive torque = {tq:+.3f} Nm')
    else:
        passive_entries = get_passive_torque_map(passive_trials)

    # ── 4. Full dataset pipeline (for operating_positions + subtracted targets) ─
    (X_train, y_train, X_val, y_val, X_test, y_test,
     _, _, operating_positions, _, _) = build_dataset(
        trials,
        test_trial_indices=args.test_trials,
        retest_trial_indices=args.retest_trials,
        subtract_passive=True,
        passive_entries_override=passive_entries if (args.passive_pos and args.passive_tq) else None,
    )

    X_trainval = np.concatenate([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    # ── 5. Generate plots ─────────────────────────────────────────────────────
    def path(fname):
        return os.path.join(out_dir, fname)

    plot_passive_coverage(
        raw_segments, passive_entries, operating_positions,
        path('01_passive_coverage.png'))

    plot_passive_interpolation_range(
        passive_entries, operating_positions,
        path('02_interpolation_range.png'))

    if raw_segments:
        plot_passive_variability(
            raw_segments, passive_entries,
            path('03_passive_variability.png'),
            flag_threshold=args.variability_flag_threshold)
    else:
        print('Skipping plot 03 (passive variability) — no raw segments from FLB '
              '(passive map was manually overridden).')

    plot_passive_torque_curve(
        passive_entries, operating_positions,
        path('04_passive_curve.png'))

    plot_passive_residual_by_position(
        X_trainval, y_trainval, operating_positions,
        path('05_residual_trainval.png'),
        split_label='Train + Validation')

    plot_passive_residual_by_position(
        X_test, y_test, operating_positions,
        path('05_residual_test.png'),
        split_label='Test (retest)')

    if raw_segments:
        plot_extreme_vs_middle_summary(
            passive_entries, raw_segments, operating_positions,
            X_trainval, y_trainval,
            path('06_extreme_vs_middle_summary.png'))
    else:
        print('Skipping plot 06 (extreme vs middle summary) — no raw segments from FLB.')

    # ── 6. Console summary ────────────────────────────────────────────────────
    _print_summary(
        passive_entries, raw_segments, operating_positions,
        std_threshold=args.std_threshold,
        variability_flag_threshold=args.variability_flag_threshold)

    print(f'\nAll plots saved → {out_dir}/')


if __name__ == '__main__':
    main()
