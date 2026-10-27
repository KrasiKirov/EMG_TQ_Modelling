"""Run the 500 ms window + retest passive map benchmark as parallel single-thread jobs.

Each (subject, seed) is its own ``ML.training.benchmark`` run in a separate process
limited to one CPU thread, so several jobs share the machine without oversubscribing
it. Jobs that already wrote their metrics are skipped, so an interrupted collection
can be resumed with the same command.

Example (from the repository root):
    python -m ML.training.parallel_benchmark --output ../runs/window50_retest_map \\
        --overrides ML/reviewed_overrides.json --data-dir ../data --workers 8
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from ML.config import DATA_DIR, SUBJECTS
from ML.training.benchmark import MIN_SEEDS

REPO = Path(__file__).resolve().parents[2]
# The method's settings, spelled out instead of relying on benchmark defaults.
# The epoch cap and early-stopping patience come from --epochs / --patience.
METHOD_ARGS = ['--protocol', 'within', '--models', 'lstm', '--batch', '64', '--window', '50',
               '--train-stride', '5', '--session-passive-calibration', '--evaluate-retest']


def worker(argv):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    from ML.training import benchmark
    sys.argv = ['ML.training.benchmark', *argv]
    benchmark.main()


def run_job(subject, seed, args):
    name = f'{subject}_seed{seed}'
    folder = args.output / name
    if (folder / f'{subject}_lstm_seed{seed}' / 'metrics.json').exists():
        return name, 'skipped', 0.
    if folder.exists():
        shutil.rmtree(folder)  # left behind by an interrupted attempt
    # One seed per job; the collection-level minimum is checked in main().
    command = [sys.executable, '-m', 'ML.training.parallel_benchmark', '--worker', *METHOD_ARGS, '--allow-few-seeds',
               '--epochs', str(args.epochs), '--patience', str(args.patience),
               '--subjects', subject, '--seeds', str(seed), '--output', str(folder),
               '--data-dir', str(args.data_dir)]
    if args.overrides:
        command += ['--overrides', str(args.overrides)]
    env = dict(os.environ, OMP_NUM_THREADS='1', TF_CPP_MIN_LOG_LEVEL='2', PYTHONIOENCODING='utf-8')
    start = time.perf_counter()
    with open(args.output / 'logs' / f'{name}.log', 'w', encoding='utf-8') as log:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    return name, 'done' if result.returncode == 0 else f'FAILED (exit {result.returncode})', time.perf_counter() - start


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        return worker(sys.argv[2:])
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--subjects', nargs='+', choices=list(SUBJECTS), default=list(SUBJECTS))
    p.add_argument('--seeds', nargs='+', type=int, default=list(range(10)))
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--epochs', type=int, default=60, help='epoch cap')
    p.add_argument('--patience', type=int, default=12, help='early-stopping patience in epochs')
    p.add_argument('--data-dir', type=Path, default=Path(DATA_DIR))
    p.add_argument('--overrides', type=Path)
    p.add_argument('--allow-few-seeds', action='store_true',
                   help=f'Allow fewer than {MIN_SEEDS} seeds (smoke tests only; such results are not comparable)')
    args = p.parse_args()
    if len(set(args.seeds)) < MIN_SEEDS and not args.allow_few_seeds:
        p.error(f'use at least {MIN_SEEDS} seeds so run-to-run variance can be judged (--allow-few-seeds for smoke tests)')
    args.output = args.output.resolve()
    args.data_dir = args.data_dir.resolve()
    args.overrides = args.overrides.resolve() if args.overrides else None
    (args.output / 'logs').mkdir(parents=True, exist_ok=True)
    jobs = [(s, seed) for seed in args.seeds for s in args.subjects]
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job, s, seed, args) for s, seed in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            name, status, seconds = future.result()
            failed += status.startswith('FAILED')
            print(f'[{i}/{len(jobs)}] {name}: {status} in {seconds / 60:.1f} min', flush=True)
    print(f'ALL DONE: {len(jobs) - failed} ok, {failed} failed', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
