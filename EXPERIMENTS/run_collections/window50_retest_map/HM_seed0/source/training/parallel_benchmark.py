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

REPO = Path(__file__).resolve().parents[2]
# The method's settings, spelled out instead of relying on benchmark defaults.
METHOD_ARGS = ['--protocol', 'within', '--models', 'lstm', '--epochs', '60', '--patience', '12',
               '--batch', '64', '--window', '50', '--train-stride', '5',
               '--session-passive-calibration', '--evaluate-retest']


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
    command = [sys.executable, '-m', 'ML.training.parallel_benchmark', '--worker', *METHOD_ARGS,
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
    p.add_argument('--data-dir', type=Path, default=Path(DATA_DIR))
    p.add_argument('--overrides', type=Path)
    args = p.parse_args()
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
