"""
Run many jobs in parallel on the CPU.

    python -m ML.research.queue --study screen --subjects YES JM EG \
        --protocols S1toS2 S2toS1 --configs E01_old E03_final --seeds 0 1 2 \
        --workers 7 --threads 2 [--train-protocol fast] [--plot-seed 0]

* One job = one (subject, protocol, config, seed) -> ``python -m ML.research.run``.
* Jobs are ordered config-first (the order given), so the most important
  configurations finish first.
* A job whose result JSON already exists is skipped -> the queue can be
  stopped and relaunched safely.
* While a job runs, a ``<stem>.running`` marker sits next to its result; a
  second queue skips jobs with a marker younger than ``STALE_S`` (so two
  queues never train the same job twice).
* Each job's console output goes to RESULTS_DIR/_logs/<study>__....log.
* ``--plot-seed k`` saves full-trial inspection plots for seed k only.
"""

import os
import sys
import time
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ML.research import config as C


STALE_S = 4 * 3600      # a .running marker older than this is ignored


def job_paths(study, subject, protocol, config, seed, tp):
    d = os.path.join(C.RESULTS_DIR, study, subject, protocol, config)
    return os.path.join(d, f'{tp}_seed{seed}.json')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--study', required=True)
    ap.add_argument('--subjects', nargs='+', required=True)
    ap.add_argument('--protocols', nargs='+', required=True)
    ap.add_argument('--configs', nargs='+', required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--train-protocol', default='fast')
    ap.add_argument('--workers', type=int, default=7)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--plot-seed', type=int, default=0)
    a = ap.parse_args()

    jobs = [(s, p, c, sd) for c in a.configs for sd in a.seeds for s in a.subjects for p in a.protocols]
    os.makedirs(C.LOG_DIR, exist_ok=True)
    print(f'{len(jobs)} jobs, {a.workers} workers x {a.threads} threads', flush=True)

    def run(job):
        s, p, c, sd = job
        jp = job_paths(a.study, s, p, c, sd, a.train_protocol)
        if os.path.exists(jp):
            return 'skip'
        marker = jp[:-len('.json')] + '.running'
        if os.path.exists(marker) and time.time() - os.path.getmtime(marker) < STALE_S:
            return 'skip (running elsewhere)'
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        with open(marker, 'w') as fh:
            fh.write('queue pid %d, %s' % (os.getpid(), time.ctime()))
        log = os.path.join(C.LOG_DIR, f'{a.study}__{s}__{p}__{c}__{a.train_protocol}_s{sd}.log')
        cmd = [sys.executable, '-m', 'ML.research.run', '--study', a.study, '--subject', s, '--protocol', p,
               '--config', c, '--seed', str(sd), '--train-protocol', a.train_protocol, '--threads', str(a.threads)]
        if sd == a.plot_seed:
            cmd.append('--plot')
        t0 = time.time()
        try:
            with open(log, 'w', encoding='utf-8') as fh:
                rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
        finally:
            if os.path.exists(marker):
                os.remove(marker)
        msg = f"{'done' if rc == 0 else 'FAILED rc=%d' % rc} {s}/{p}/{c}/s{sd} {time.time() - t0:.0f}s"
        print(msg, flush=True)
        return msg

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, jobs))
    print('queue finished', flush=True)


if __name__ == '__main__':
    main()
