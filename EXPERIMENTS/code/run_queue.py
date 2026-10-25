"""
Run many jobs in parallel on the CPU (one process per job, 1 thread each).
Jobs whose result JSON already exists are skipped, so the queue can be stopped
and relaunched safely.  Each job's console output goes to EXPERIMENTS/logs/.

    python EXPERIMENTS/code/run_queue.py --exps X00f_baseline_framework X01_clean_passive \
        --subjects YES JM --directions S1toS2 S2toS1 --seeds 0 1 --workers 13
    python EXPERIMENTS/code/run_queue.py --original-baseline --subjects YES JM EG IES CH HM --seeds 0 1 2
"""
import os
import sys
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)


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
                cmd = [py, os.path.join(HERE, 'run_original_baseline.py'), '--subject', s, '--seed', str(seed)]
                jobs.append((res, cmd, f'X00_baseline__{s}__S1toS2__s{seed}'))
    import json
    if a.jobs_file:
        for e, s_, d, seed in json.load(open(a.jobs_file)):
            res = os.path.join(EXP, 'results', e, s_, d, f'seed{seed}.json')
            cmd = [py, os.path.join(HERE, 'run_experiment.py'), '--exp', e, '--subject', s_, '--direction', d, '--seed', str(seed)]
            jobs.append((res, cmd, f'{e}__{s_}__{d}__s{seed}'))
    for e in a.exps:
        for seed in a.seeds:
            for d in a.directions:
                for s in a.subjects:
                    res = os.path.join(EXP, 'results', e, s, d, f'seed{seed}.json')
                    cmd = [py, os.path.join(HERE, 'run_experiment.py'), '--exp', e, '--subject', s,
                           '--direction', d, '--seed', str(seed)]
                    jobs.append((res, cmd, f'{e}__{s}__{d}__s{seed}'))
    skip = set()
    if a.skip_file:
        skip = {f'{e}__{s_}__{d}__s{seed}' for e, s_, d, seed in json.load(open(a.skip_file))}
    seen, todo = set(), []
    for j in jobs:
        if not os.path.exists(j[0]) and j[2] not in skip and j[2] not in seen:
            seen.add(j[2])
            todo.append(j)
    print(f'{len(jobs)} jobs, {len(todo)} to run, {a.workers} workers', flush=True)
    os.makedirs(os.path.join(EXP, 'logs'), exist_ok=True)
    env = dict(os.environ)
    if a.original_baseline:
        env.pop('PYTHONPATH', None) if not a.exps else None

    def run(job):
        res, cmd, name = job
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
