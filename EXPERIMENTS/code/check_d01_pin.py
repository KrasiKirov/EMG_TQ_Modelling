"""
Check that this branch's ML/research builds exactly the same data as the
branch the D01 results were trained on (experimentation-dorsiflexion).

    git worktree add --detach <ref> origin/experimentation-dorsiflexion
    python EXPERIMENTS/code/check_d01_pin.py --reference-root <ref> --subjects YES JM

For each subject and direction it builds D01's windowed arrays (passive
'static', EMG 'agc_auto', window 20) with both code trees, each with its own
empty processed-data cache, and compares every array bit for bit.
Exit code 0 = identical.
"""
import os
import sys
import json
import argparse
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

DUMP = r'''
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
from ML.research.run import prepare
out = {}
for d in ('S1toS2', 'S2toS1'):
    data, _ = prepare(sys.argv[3], d, dict(passive='static', emg='agc_auto', window=20))
    for k, v in data.items():
        if k.startswith('meta_'):
            for kk, vv in v.items():
                out[f'{d}/{k}/{kk}'] = np.asarray(vv)
        elif k != 'train_angles':
            out[f'{d}/{k}'] = np.asarray(v)
np.savez(sys.argv[2], **out)
'''


def build(root, subject, out, cache):
    env = dict(os.environ, RESEARCH_CACHE_DIR=cache, PYTHONPATH=root, TF_CPP_MIN_LOG_LEVEL='2')
    subprocess.run([sys.executable, '-c', DUMP, root, out, subject], check=True, env=env, cwd=root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference-root', required=True)
    ap.add_argument('--subjects', nargs='+', default=['YES', 'JM'])
    a = ap.parse_args()
    import numpy as np
    ok, report = True, {}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for s in a.subjects:
            files = {}
            for name, root in (('this', ROOT), ('reference', a.reference_root)):
                files[name] = os.path.join(tmp, f'{s}_{name}.npz')
                build(root, s, files[name], os.path.join(tmp, f'cache_{name}'))
            with np.load(files["this"]) as za, np.load(files["reference"]) as zb:
                A, B = dict(za), dict(zb)
            diff = sorted(k for k in set(A) | set(B)
                          if k not in A or k not in B or A[k].shape != B[k].shape
                          or not np.array_equal(A[k], B[k]))
            report[s] = {'arrays': len(A), 'different': diff}
            ok &= not diff
            print(f'{s}: {len(A)} arrays compared, {len(diff)} different {diff[:5]}', flush=True)
    print(json.dumps({'identical': ok, 'subjects': report}))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
