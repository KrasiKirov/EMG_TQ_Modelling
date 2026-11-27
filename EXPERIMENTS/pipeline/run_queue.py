"""
Run many jobs in parallel on the CPU (one process per job, 1 thread each).
A job whose result JSON already exists is skipped only if the stored config equals the spec's current config;
if the spec changed since, the queue stops with ConfigMismatch instead of silently keeping old results.
The queue can be stopped and relaunched safely. Each job's console output goes to EXPERIMENTS/logs/.

    python -m EXPERIMENTS.pipeline.run_queue --exps FE_A0 FE_A1 \
        --subjects YES JM --seeds 0 1 --workers 13
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from EXPERIMENTS.paths import LOGS, ROOT
from EXPERIMENTS.pipeline.run import results_root
from EXPERIMENTS.pipeline.settings import BASE, DIRECTION as D, load_config, strip_retired
from EXPERIMENTS.pipeline.spec import SpecError


class ConfigMismatch(RuntimeError):
    """A saved result was produced with a different config than the spec now asks for."""


def is_done(json_path, cfg):
    """True if json_path exists and was produced with `cfg`; raise ConfigMismatch if it exists with another config.
    Keys missing from the stored config are taken from BASE (results saved before a default was added); retired
    overlap settings at their full-overlap values are ignored, other values raise ConfigMismatch."""
    json_path = str(json_path)
    if not os.path.exists(json_path):
        return False
    with open(json_path, encoding='utf-8') as f:
        stored = dict(BASE, **(json.load(f).get('config') or {}))
    try:
        stored = strip_retired(stored)
    except SpecError as e:
        raise ConfigMismatch(f'{json_path}: {e}') from e
    want = json.loads(json.dumps(cfg))
    stored = json.loads(json.dumps(stored))
    if stored != want:
        diff = sorted(k for k in set(stored) | set(want) if stored.get(k) != want.get(k))
        raise ConfigMismatch(f'{json_path}: stored config differs from the spec in {diff}; '
                             'move the old results away or give the changed spec a new id')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exps', nargs='*', default=[])
    ap.add_argument('--subjects', nargs='+', required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=[0])
    ap.add_argument('--workers', type=int, default=13)
    ap.add_argument('--jobs-file', default=None,
                    help='JSON list of [exp, subject, direction, seed] in priority order (added before --exps jobs); direction must be S1toS2')
    ap.add_argument('--skip-file', default=None,
                    help='JSON list of [exp, subject, direction, seed] currently running elsewhere (not launched again)')
    a = ap.parse_args()
    py = sys.executable
    jobs = []
    if a.jobs_file:
        for e, s_, d, seed in json.load(open(a.jobs_file)):
            if d != D:
                raise SystemExit(f'{a.jobs_file}: {e} {s_} seed {seed} has direction {d}; every run is {D}')
            res = os.path.join(results_root(e), e, s_, d, f'seed{seed}.json')
            cmd = [py, '-m', 'EXPERIMENTS.pipeline.run', '--exp', e, '--subject', s_, '--seed', str(seed)]
            jobs.append((res, cmd, f'{e}__{s_}__{d}__s{seed}', e))
    for e in a.exps:
        for seed in a.seeds:
            for s in a.subjects:
                res = os.path.join(results_root(e), e, s, D, f'seed{seed}.json')
                cmd = [py, '-m', 'EXPERIMENTS.pipeline.run', '--exp', e, '--subject', s, '--seed', str(seed)]
                jobs.append((res, cmd, f'{e}__{s}__{D}__s{seed}', e))
    skip = set()
    if a.skip_file:
        skip = {f'{e}__{s_}__{d}__s{seed}' for e, s_, d, seed in json.load(open(a.skip_file))}
    seen, todo = set(), []
    cfgs = {}
    for j in jobs:
        res, _, name, exp = j
        if exp not in cfgs:
            cfgs[exp] = load_config(exp)[1]
        done = is_done(res, cfgs[exp])
        if not done and name not in skip and name not in seen:
            seen.add(name)
            todo.append(j)
    print(f'{len(jobs)} jobs, {len(todo)} to run, {a.workers} workers', flush=True)
    os.makedirs(LOGS, exist_ok=True)

    def run(job):
        res, cmd, name, _ = job
        with open(LOGS / f'{name}.log', 'w') as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
        print(('done ' if r.returncode == 0 else f'FAILED({r.returncode}) ') + name, flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, todo))
    print('QUEUE_DONE', flush=True)


if __name__ == '__main__':
    main()
