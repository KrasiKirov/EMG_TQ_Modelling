"""Run a list of experiments with a bounded worker pool.

jobs.json: [{"subject": "JM", "name": "rezero_const", "opts": {...},
             "model": "lstm8", "protocol": "fast", "seeds": [0,1,2],
             "sample_weight": null, "loss": "mse", "model_kw": {}}, ...]

Each job runs as `python -m ML.experiments.run ...` (which also takes a CPU
slot lock, so several queues can coexist without oversubscribing).  Jobs whose
result JSON already exists are skipped, so a queue can be re-launched safely.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ML.experiments.harness import RESULTS_DIR


def job_cmd(j, threads):
    py = sys.executable
    if j.get('runner') == 'pipeline':
        conf = j['config'] if isinstance(j['config'], str) else json.dumps(j['config'])
        return [py, '-m', 'ML.experiments.run_pipeline', '--subject', j['subject'],
                '--name', j['name'], '--config', conf,
                '--protocol', j.get('protocol', 'orig'), '--threads', str(threads),
                '--seeds', *[str(s) for s in j.get('seeds', [0, 1, 2])]]
    cmd = [py, '-m', 'ML.experiments.run', '--subject', j['subject'],
           '--name', j['name'], '--protocol', j.get('protocol', 'fast'),
           '--threads', str(threads), '--opts', json.dumps(j.get('opts', {})),
           '--model', j.get('model', 'lstm8'),
           '--model-kw', json.dumps(j.get('model_kw', {})),
           '--loss', j.get('loss', 'mse'),
           '--seeds', *[str(s) for s in j.get('seeds', [0, 1, 2])]]
    if j.get('sample_weight'):
        cmd += ['--sample-weight', j['sample_weight']]
    if j.get('out_dir'):
        cmd += ['--out-dir', j['out_dir']]
    return cmd


def result_path(j):
    if j.get('runner') == 'pipeline':
        return os.path.join(RESULTS_DIR, 'pipeline',
                            f"{j['subject']}__{j['name']}__{j.get('protocol', 'orig')}.json")
    d = j.get('out_dir') or RESULTS_DIR
    return os.path.join(d, f"{j['subject']}__{j['name']}__{j.get('protocol', 'fast')}.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('jobs')
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    with open(a.jobs) as f:
        jobs = json.load(f)
    log_dir = os.environ.get('EMG_LOG_DIR', '.')
    os.makedirs(log_dir, exist_ok=True)

    def run_one(j):
        if os.path.exists(result_path(j)):
            print(f"skip (exists) {j['subject']}/{j['name']}", flush=True)
            return
        pre = 'PIPE_' if j.get('runner') == 'pipeline' else ''
        log = os.path.join(log_dir, f"{pre}{j['subject']}_{j['name']}_{j.get('protocol', 'fast')}.log")
        t0 = time.time()
        with open(log, 'w', encoding='utf-8') as fh:
            rc = subprocess.call(job_cmd(j, a.threads), stdout=fh, stderr=subprocess.STDOUT)
        print(f"{'done' if rc == 0 else 'FAILED rc=%d' % rc} {j['subject']}/{j['name']} "
              f"in {time.time() - t0:.0f}s  log={log}", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run_one, jobs))
    print('queue finished', flush=True)


if __name__ == '__main__':
    main()
