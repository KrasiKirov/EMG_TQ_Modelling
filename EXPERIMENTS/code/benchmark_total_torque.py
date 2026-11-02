"""
Benchmark this computer for the total-torque study before launching it.

    python EXPERIMENTS/code/benchmark_total_torque.py --subject YES --out EXPERIMENTS/results/total_torque_benchmark.json

Measures (all on one representative subject, S1 -> S2, TT02 = total torque,
200 ms windows, balanced positions, fixed training-max EMG scale):
 * preprocessing, separately from training: reading + filtering the .flb file
   into the cached subject bundle (cold), and building the windows from the
   cache (warm, what every job pays);
 * seconds per epoch with 1, 2 and 4 TensorFlow threads (one job alone), from
   two short runs (E1 and E2 epochs; slope = epoch time, intercept = model
   build + prediction overhead);
 * throughput with K concurrent single-thread jobs (K in --concurrency), and
   the peak memory of each job.
Short runs write to a scratch folder, never to EXPERIMENTS/results/<exp>.
"""
import os
import sys
import json
import time
import shutil
import argparse
import platform
import tempfile
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXP_ID = 'TT02_total_w200_bal_trainmax'


def run_jobs(n, threads, epochs, subject, scratch, seed0=0):
    """Start n run_experiment.py jobs at once; return their result JSONs and the wall time."""
    out = os.path.join(scratch, f'k{n}_t{threads}_e{epochs}')
    procs, t0 = [], time.time()
    for i in range(n):
        cmd = [sys.executable, os.path.join(HERE, 'run_experiment.py'), '--exp', EXP_ID, '--subject', subject,
               '--direction', 'S1toS2', '--seed', str(seed0 + i), '--threads', str(threads),
               '--max-epochs', str(epochs), '--out-root', out]
        procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    codes = [p.wait() for p in procs]
    wall = time.time() - t0
    if any(codes):
        raise RuntimeError(f'benchmark job failed: {codes}')
    res = []
    for i in range(n):
        with open(os.path.join(out, EXP_ID, subject, 'S1toS2', f'seed{seed0 + i}.json')) as f:
            res.append(json.load(f))
    return res, wall


def hardware():
    import psutil
    info = {'cpu': platform.processor(), 'physical_cores': psutil.cpu_count(logical=False),
            'logical_cpus': psutil.cpu_count(logical=True), 'ram_gb': round(psutil.virtual_memory().total / 2 ** 30, 1),
            'os': platform.platform(), 'python': sys.version.split()[0]}
    try:
        info['cpu'] = subprocess.run(['powershell', '-NoProfile', '-Command',
                                      '(Get-CimInstance Win32_Processor).Name'],
                                     capture_output=True, text=True).stdout.strip() or info['cpu']
        info['gpu'] = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                                     capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    import tensorflow as tf
    info['tensorflow'] = tf.__version__
    info['tf_built_with_cuda'] = bool(tf.test.is_built_with_cuda())
    info['tf_visible_gpus'] = [d.name for d in tf.config.list_physical_devices('GPU')]
    info['tf_can_use_gpu'] = bool(info['tf_visible_gpus'])
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subject', default='YES')
    ap.add_argument('--epochs', nargs=2, type=int, default=[2, 6])
    ap.add_argument('--threads', nargs='+', type=int, default=[1, 2, 4])
    ap.add_argument('--concurrency', nargs='+', type=int, default=[8, 16, 24])
    ap.add_argument('--out', default=os.path.join(ROOT, 'EXPERIMENTS', 'results', 'total_torque_benchmark.json'))
    a = ap.parse_args()
    sys.path.insert(0, ROOT)
    rep = {'subject': a.subject, 'experiment': EXP_ID, 'hardware': hardware()}
    print(json.dumps(rep['hardware']), flush=True)

    # preprocessing: cold bundle (own empty cache) and warm window building
    cold = tempfile.mkdtemp(prefix='ttcache_')
    code = ('import sys,time,json; sys.path.insert(0, sys.argv[1]); t=time.time();'
            'from ML.research.data import load_subject; load_subject(sys.argv[2]); t1=time.time()-t;'
            'from ML.research.run import prepare; t=time.time();'
            'prepare(sys.argv[2], "S1toS2", dict(passive="none", emg="trainmax_tr", window=20)); t2=time.time()-t;'
            't=time.time(); prepare(sys.argv[2], "S1toS2", dict(passive="none", emg="trainmax_tr", window=20));'
            'print(json.dumps(dict(bundle_cold_s=t1, prepare_first_s=t2, prepare_warm_s=time.time()-t)))')
    env = dict(os.environ, RESEARCH_CACHE_DIR=cold)
    r = subprocess.run([sys.executable, '-c', code, ROOT, a.subject], env=env, capture_output=True, text=True)
    rep['preprocessing'] = json.loads(r.stdout.strip().splitlines()[-1])
    shutil.rmtree(cold, ignore_errors=True)
    print('preprocessing', rep['preprocessing'], flush=True)

    scratch = tempfile.mkdtemp(prefix='ttbench_')
    e1, e2 = a.epochs
    rep['single_job'] = {}
    for th in a.threads:
        (r1,), w1 = run_jobs(1, th, e1, a.subject, scratch)
        (r2,), w2 = run_jobs(1, th, e2, a.subject, scratch)
        per_epoch = (r2['train_s'] - r1['train_s']) / (e2 - e1)
        rep['single_job'][th] = {'s_per_epoch': per_epoch, 'overhead_s': r1['train_s'] - e1 * per_epoch,
                                 'prep_s': r1['prep_s'], 'wall_s': [w1, w2], 'peak_memory_mb': r2['peak_memory_mb']}
        print('threads', th, rep['single_job'][th], flush=True)

    rep['concurrent'] = {}
    base = rep['single_job'][1]
    for k in a.concurrency:
        res, wall = run_jobs(k, 1, e2, a.subject, scratch, seed0=100)
        pe = [(x['train_s'] - base['overhead_s']) / e2 for x in res]
        rep['concurrent'][k] = {'s_per_epoch_median': sorted(pe)[len(pe) // 2], 's_per_epoch_max': max(pe),
                                'wall_s': wall, 'epochs_per_hour_total': k * e2 / wall * 3600,
                                'peak_memory_mb_max': max(x['peak_memory_mb'] or 0 for x in res)}
        print('concurrent', k, rep['concurrent'][k], flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    with open(a.out, 'w') as f:
        json.dump(rep, f, indent=1)
    print('written', a.out)


if __name__ == '__main__':
    main()
