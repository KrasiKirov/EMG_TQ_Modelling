"""CLI for one experiment.

Example
-------
python -m ML.experiments.run --subject JM --name baseline --protocol fast \
    --seeds 0 1 2 --threads 3 --opts '{"target": "passive"}'
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

from ML.experiments import harness

N_SLOTS = int(os.environ.get('EMG_SLOTS', '5'))


def acquire_slot():
    """Block until one of N_SLOTS CPU slots is free (OS file lock → released
    automatically if the process dies).  Keeps parallel experiments from
    oversubscribing the 16 hardware threads."""
    import time
    slot_dir = os.path.join(os.environ.get('EMG_LOG_DIR', harness.CACHE_DIR), 'slots')
    os.makedirs(slot_dir, exist_ok=True)
    if os.name == 'nt':
        import msvcrt
        def try_lock(fh):
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                return True
            except OSError:
                return False
    else:
        import fcntl
        def try_lock(fh):
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except OSError:
                return False
    waited = 0
    while True:
        for i in range(N_SLOTS):
            fh = open(os.path.join(slot_dir, f'slot{i}.lock'), 'a+')
            fh.seek(0)
            if try_lock(fh):
                print(f'[slot {i}] acquired after {waited}s', flush=True)
                return fh
            fh.close()
        time.sleep(5)
        waited += 5


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subject', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--opts', default='{}', help='JSON dict merged into DEFAULT_OPTS')
    p.add_argument('--model', default='lstm8',
                   help="registry name or 'module.path:builder'")
    p.add_argument('--model-kw', default='{}')
    p.add_argument('--protocol', default='fast', choices=list(harness.PROTOCOLS))
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--threads', type=int, default=3)
    p.add_argument('--loss', default='mse')
    p.add_argument('--sample-weight', default=None,
                   help="'module.path:fn(data) -> weights'")
    p.add_argument('--out-dir', default=None)
    p.add_argument('--no-slot', action='store_true')
    a = p.parse_args()
    _slot = None if a.no_slot else acquire_slot()
    harness.run(a.subject, a.name, json.loads(a.opts), model=a.model,
                protocol=a.protocol, seeds=a.seeds, threads=a.threads,
                loss=a.loss, model_kw=json.loads(a.model_kw),
                sample_weight=a.sample_weight, out_dir=a.out_dir)


if __name__ == '__main__':
    main()
