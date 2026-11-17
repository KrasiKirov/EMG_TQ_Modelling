"""
Run many jobs in parallel on the CPU (one process per job, 1 thread each).
A job whose result JSON already exists is skipped only if the stored config equals the spec's current config;
if the spec changed since, the queue stops with ConfigMismatch instead of silently keeping old results.
The queue can be stopped and relaunched safely. Each job's console output goes to EXPERIMENTS/logs/.

    python -m EXPERIMENTS.pipeline.run_queue --exps A0_active_trainmax N0_total_trainmax \
        --subjects YES JM --directions S1toS2 S2toS1 --seeds 0 1 --workers 13
    python -m EXPERIMENTS.pipeline.run_queue --original-baseline --subjects YES JM EG IES CH HM --seeds 0 1 2
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(EXP))

from EXPERIMENTS.pipeline.run import BASE, load_config   # noqa: E402


class ConfigMismatch(RuntimeError):
    """A saved result was produced with a different config than the spec now asks for."""


def is_done(json_path, cfg):
    """True if json_path exists and was produced with `cfg`; raise ConfigMismatch if it exists with another config.
    Keys missing from the stored config are taken from BASE (results saved before a default was added)."""
    json_path = str(json_path)
    if not os.path.exists(json_path):
        return False
    with open(json_path, encoding='utf-8') as f:
        stored = dict(BASE, **(json.load(f).get('config') or {}))
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
    ap.add_argument('--original-baseline', action='store_true')
    ap.add_argument('--subjects', nargs='+', required=True)
    ap.add_argument('--directions', nargs='+', default=['S1toS2'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0])
    ap.add_argument('--workers', type=int, default=13)
    ap.add_argument('--jobs-file', default=None,
                    help='JSON list of [exp, subject, direction, seed] in priority order (added before --exps jobs)')
    ap.add_argument('--skip-file', default=None,
                    help='JSON list of [exp, subject, direction, seed] currently running elsewhere (not launched again)')
    a = ap.parse_args()
    py = sys.executable
    jobs = []
    if a.original_baseline:
        for seed in a.seeds:
            for s in a.subjects:
                res = os.path.join(EXP, 'results', 'X00_baseline', s, 'S1toS2', f'seed{seed}.json')
                cmd = [py, os.path.join(EXP, 'code', 'run_original_baseline.py'), '--subject', s, '--seed', str(seed)]
                jobs.append((res, cmd, f'X00_baseline__{s}__S1toS2__s{seed}', None))
    if a.jobs_file:
        for e, s_, d, seed in json.load(open(a.jobs_file)):
            res = os.path.join(EXP, 'results', e, s_, d, f'seed{seed}.json')
            cmd = [py, os.path.join(HERE, 'run.py'), '--exp', e, '--subject', s_, '--direction', d, '--seed', str(seed)]
            jobs.append((res, cmd, f'{e}__{s_}__{d}__s{seed}', e))
    for e in a.exps:
        for seed in a.seeds:
            for d in a.directions:
                for s in a.subjects:
                    res = os.path.join(EXP, 'results', e, s, d, f'seed{seed}.json')
                    cmd = [py, os.path.join(HERE, 'run.py'), '--exp', e, '--subject', s,
                           '--direction', d, '--seed', str(seed)]
                    jobs.append((res, cmd, f'{e}__{s}__{d}__s{seed}', e))
    skip = set()
    if a.skip_file:
        skip = {f'{e}__{s_}__{d}__s{seed}' for e, s_, d, seed in json.load(open(a.skip_file))}
    seen, todo = set(), []
    cfgs = {}
    for j in jobs:
        res, _, name, exp = j
        if exp is not None and exp not in cfgs:
            cfgs[exp] = load_config(exp)[1]
        done = is_done(res, cfgs[exp]) if exp is not None else os.path.exists(res)
        if not done and name not in skip and name not in seen:
            seen.add(j[2])
            todo.append(j)
    print(f'{len(jobs)} jobs, {len(todo)} to run, {a.workers} workers', flush=True)
    os.makedirs(os.path.join(EXP, 'logs'), exist_ok=True)
    env = dict(os.environ)
    if a.original_baseline:
        if not a.exps:
            env.pop('PYTHONPATH', None)

    def run(job):
        res, cmd, name, _ = job
        e2 = dict(env)
        if 'run_original_baseline.py' in cmd[1]:
            e2.pop('PYTHONPATH', None)                 # the frozen original code must not see the worktree package
        with open(os.path.join(EXP, 'logs', name + '.log'), 'w') as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=e2)
        print(('done ' if r.returncode == 0 else f'FAILED({r.returncode}) ') + name, flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, todo))
    print('QUEUE_DONE', flush=True)


if __name__ == '__main__':
    main()
