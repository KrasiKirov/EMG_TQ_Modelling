"""Would the experiment pipeline still build the same data if `ML/` came from main?

    python EXPERIMENTS/analysis/compare_pipelines.py --subject EG [--ref origin/main]

The experiments (EXPERIMENTS/pipeline/run.py) build their arrays with ML.research.run.prepare, which
imports ML.config, ML.models.lstm and ML.preprocessing.* . This script builds the default experiment's arrays
(A0 settings: window 20, each session's own relaxed-hold passive table, EMG / session-1 maximum) twice:

  A  with this checkout's ML package
  B  with ML from <ref> (git archive), plus this checkout's ML/research copied into it
  C  as B, plus the few names B lacks copied from this checkout (config overrides, missing SUBJECTS, N_EMG, active_torque.py,
     emg_agc.py): what a merge that keeps <ref>'s versions of the shared files would run

Each variant runs in its own process with its own processed-data cache. It reports import errors, array shapes
and the largest absolute differences. It does not train anything.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CFG = {'passive': 'static', 'emg': 'trainmax', 'window': 20}

CHILD = r'''
import json, sys, numpy as np
sys.stdout = sys.stderr   # keep loader chatter out of the report
sys.path.insert(0, sys.argv[1])
out = sys.argv[4]
try:
    from ML.research.run import prepare
    data, details = prepare(sys.argv[2], sys.argv[3], json.loads(sys.argv[5]))
except Exception as e:  # report, do not hide
    json.dump({'error': f'{type(e).__name__}: {e}'}, open(out + '.json', 'w')); sys.exit(0)
np.savez(out + '.npz', **{k: v for k, v in data.items() if isinstance(v, np.ndarray)})
json.dump({'shapes': {k: list(v.shape) for k, v in data.items() if isinstance(v, np.ndarray)},
           'details': details}, open(out + '.json', 'w'), default=str)
'''


def patch(ml):
    """Copy into `ml` only the names this checkout's ML/research imports and `ml` lacks."""
    import ast
    here = (ROOT / 'ML/config.py').read_text(encoding='utf-8')
    cfg = (ml / 'config.py').read_text(encoding='utf-8')
    for node in ast.parse(here).body:
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', '') in ('TRIAL_OVERRIDES', 'PASSIVE_TORQUE_OVERRIDES')
                                                for t in node.targets) and t_missing(cfg, node):
            cfg += '\n' + ast.get_source_segment(here, node) + '\n'
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', '') == 'SUBJECTS' for t in node.targets):
            for key, value in ast.literal_eval(ast.get_source_segment(here, node.value)).items():
                cfg += f'\nSUBJECTS.setdefault({key!r}, {value!r})'
    (ml / 'config.py').write_text(cfg + '\n', encoding='utf-8')
    lstm = (ml / 'models/lstm.py').read_text(encoding='utf-8')
    if 'N_EMG' not in lstm:
        (ml / 'models/lstm.py').write_text(lstm + '\nN_EMG = 4\n', encoding='utf-8')
    for f in ('preprocessing/active_torque.py', 'preprocessing/emg_agc.py'):
        if not (ml / f).exists():
            shutil.copy(ROOT / 'ML' / f, ml / f)


def t_missing(src, node):
    return all(f'{t.id} =' not in src for t in node.targets)


def build(ml_root, subject, direction, work, tag):
    env = dict(os.environ, EMG_DATA_DIR=str(ROOT / 'ML/subjects'), RESEARCH_CACHE_DIR=str(work / f'cache_{tag}'),
               TF_CPP_MIN_LOG_LEVEL='3', PYTHONDONTWRITEBYTECODE='1')
    out = work / tag
    subprocess.run([sys.executable, '-c', CHILD, str(ml_root), subject, direction, str(out), json.dumps(CFG)],
                   env=env, check=True, cwd=work)
    info = json.loads((out.with_suffix('.json')).read_text())
    arrays = dict(np.load(out.with_suffix('.npz'))) if 'error' not in info else {}
    return info, arrays


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subject', default='EG')
    ap.add_argument('--ref', default='origin/main')
    a = ap.parse_args()
    a.direction = 'S1toS2'   # the only direction
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        other = work / 'other'
        other.mkdir()
        archive = subprocess.run(['git', 'archive', a.ref, 'ML'], cwd=ROOT, capture_output=True, check=True).stdout
        subprocess.run(['tar', '-x', '-C', str(other)], input=archive, check=True)
        shutil.rmtree(other / 'ML/research', ignore_errors=True)
        shutil.copytree(ROOT / 'ML/research', other / 'ML/research',
                        ignore=shutil.ignore_patterns('results', '_cache', '__pycache__'))
        patched = work / 'patched'
        shutil.copytree(other, patched)
        ia, A = build(ROOT, a.subject, a.direction, work, 'A')
        ib, B = build(other, a.subject, a.direction, work, 'B')
        patch(patched / 'ML')
        ic, Cc = build(patched, a.subject, a.direction, work, 'C')
    report = {'settings': CFG, 'subject': a.subject, 'direction': a.direction, 'ref': a.ref,
              'A_this_checkout': ia.get('error') or ia['shapes'], 'B_ML_from_ref': ib.get('error') or ib['shapes'],
              'C_ref_plus_missing_names': ic.get('error') or ic['shapes']}
    for name, other_info, O in (('B', ib, B), ('C', ic, Cc)):
        if A and O:
            report[f'max_abs_diff_A_vs_{name}'] = {
                k: (float(np.nanmax(np.abs(A[k].astype(float) - O[k].astype(float)))) if A[k].shape == O[k].shape
                    else f'shape {A[k].shape} vs {O[k].shape}') for k in sorted(set(A) & set(O))}
            report[f'passive_{name}'] = other_info['details'].get('passive')
    if A:
        report['passive_A'] = ia['details'].get('passive')
    print(json.dumps(report, indent=1, default=str))


if __name__ == '__main__':
    import numpy as np
    main()
