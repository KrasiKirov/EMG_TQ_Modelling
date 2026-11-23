"""Golden fingerprints: hashes of what the live pipeline produces, taken before a refactor and checked after it.

    .venv/bin/python -m scripts.fingerprint capture [--subjects YES JM]
    .venv/bin/python -m scripts.fingerprint check   [--section bundles|prepare|outputs]

Three sections, written to tests/golden/fingerprints.json:
  bundles  every distinct (envelope, trim_end_s, lp_order) the specs use, built from the .flb with the cache bypassed
  prepare  ML.research.run.prepare for every distinct data setting of the FE_* specs, one carried-state TS2_ spec,
           one spec with input_step 5, the default target, and each EMG scaling the specs use
  outputs  the study v2 dashboard data (confirm, export, waves) rebuilt into a temporary folder

Only hashes are stored, never participant data. The processed-data cache is redirected to a fresh temporary folder
for the whole run, so a stale cache can never hide a change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / 'tests' / 'golden' / 'fingerprints.json'
FRESH_CACHE = tempfile.mkdtemp(prefix='fingerprint_cache_')
os.environ['RESEARCH_CACHE_DIR'] = FRESH_CACHE          # for anything that imports ML later
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np  # noqa: E402

SUBJECTS = ('CH', 'EG', 'HM', 'IES', 'JM', 'YES')
RCFG_KEYS = ('passive', 'emg', 'window', 'envelope', 'trim_end_s', 'input_step', 'train_stride', 'state', 'lp_order',
             'val_gap', 'metadata_passive', 'scale_part')
# keys that change between two runs of the same code; none were found on 23 Nov 2026 (runs were byte-identical)
VOLATILE: tuple[str, ...] = ()
OUTPUT_MODULES = ('confirm', 'study_v2_export', 'study_v2_waves')


def digest(obj) -> str:
    """sha256 of any nesting of dicts, lists, arrays, DataFrames and scalars (dict order ignored)."""
    h = hashlib.sha256()
    _feed(h, obj)
    return h.hexdigest()


def _feed(h, obj):
    import pandas as pd
    if isinstance(obj, np.ndarray):
        a = np.ascontiguousarray(obj)
        h.update(f'nd{a.dtype.str}{a.shape}'.encode())
        if a.dtype == object:
            for x in a.ravel():
                _feed(h, x)
        else:
            h.update(a.tobytes())
    elif isinstance(obj, pd.DataFrame):
        h.update(b'df')
        _feed(h, list(map(str, obj.columns)))
        _feed(h, obj.to_numpy())
    elif isinstance(obj, dict):
        h.update(b'{')
        for k in sorted(obj, key=repr):
            h.update(repr(k).encode())
            _feed(h, obj[k])
        h.update(b'}')
    elif isinstance(obj, (list, tuple)):
        h.update(b'[')
        for x in obj:
            _feed(h, x)
        h.update(b']')
    elif isinstance(obj, np.generic):
        _feed(h, np.asarray(obj))
    else:
        h.update(f'{type(obj).__name__}:{obj!r}'.encode())


def _split(d: dict, nested=('active',)) -> dict:
    """One hash per top-level key; for list-of-record keys, one hash per record field (easier to tell what moved)."""
    out = {}
    for k, v in d.items():
        if k in nested and isinstance(v, list) and v and isinstance(v[0], dict):
            for f in sorted(v[0]):
                out[f'{k}.{f}'] = digest([r.get(f) for r in v])
        else:
            out[str(k)] = digest(v)
    return out


def bundle_settings() -> list[tuple]:
    from EXPERIMENTS.pipeline.run import load_config
    found = set()
    for p in sorted((ROOT / 'EXPERIMENTS' / 'configs').glob('*.json')):
        cfg = load_config(p.stem)[1]
        found.add((cfg['envelope'], float(cfg['trim_end_s']), int(cfg['lp_order'])))
    return sorted(found)


def prepare_settings() -> list[dict]:
    from EXPERIMENTS.pipeline.run import BASE, load_config
    configs = sorted((ROOT / 'EXPERIMENTS' / 'configs').glob('*.json'))
    seen, out = set(), []

    def add(cfg):
        r = {k: cfg[k] for k in RCFG_KEYS}
        key = json.dumps(r, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(r)

    for p in configs:
        if p.stem.startswith('FE_'):
            add(load_config(p.stem)[1])
    carry = [p.stem for p in configs if p.stem.startswith('TS2_') and load_config(p.stem)[1]['state'] == 'carry']
    add(load_config(carry[0])[1])
    step5 = [p.stem for p in configs if load_config(p.stem)[1]['input_step'] == 5]
    add(load_config(step5[0])[1])
    add(dict(BASE))                                            # legacy target, trainmax, zero-phase envelope
    for emg in ('agc_auto', 'train_p99', 'session_p99', 'mvc_global'):
        add(dict(BASE, emg=emg))
    return out


def fingerprint_bundles(subjects) -> dict:
    from ML.research.data import load_subject
    out = {}
    for env, trim, lp in bundle_settings():
        for s in subjects:
            b = load_subject(s, use_cache=False, envelope=env, trim_end_s=trim, lp_order=lp)
            out[f'{s}|{env}|{trim:g}|{lp}'] = _split(b)
    return out


def fingerprint_prepare(subjects) -> dict:
    from ML.research.run import prepare
    out = {}
    for rcfg in prepare_settings():
        key = json.dumps(rcfg, sort_keys=True)
        for s in subjects:
            data, details = prepare(s, 'S1toS2', rcfg)
            fp = _split(data, nested=('train_recs',))
            fp['details'] = digest(json.loads(json.dumps(details, default=float)))
            out[f'{s}|{key}'] = fp
    return out


def _canonical(path: Path) -> str:
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k not in VOLATILE}
        if isinstance(o, list):
            return [strip(v) for v in o]
        return o
    data = strip(json.loads(path.read_text(encoding='utf-8')))
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def fingerprint_outputs() -> dict:
    import subprocess
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for m in OUTPUT_MODULES:
            dest = Path(tmp) / f'{m}.json'
            subprocess.run([sys.executable, '-m', f'EXPERIMENTS.analysis.{m}', '--out', str(dest)], cwd=ROOT,
                           check=True, stdout=subprocess.DEVNULL)
            out[m] = _canonical(dest)
    return out


SECTIONS = {'bundles': fingerprint_bundles, 'prepare': fingerprint_prepare, 'outputs': lambda subjects: fingerprint_outputs()}


def _fresh_cache():
    """Point the processed-data cache at this run's empty folder, even if ML.research.config was imported first."""
    from ML.research import config as RC
    RC.CACHE_DIR = FRESH_CACHE


def fingerprints(subjects=SUBJECTS, sections=tuple(SECTIONS)) -> dict:
    _fresh_cache()
    return {s: SECTIONS[s](subjects) for s in sections}


def check(section: str, golden: Path = GOLDEN) -> list[str]:
    """Recompute one section for the subjects in the golden file; return the differences (empty when identical)."""
    _fresh_cache()
    want = json.loads(golden.read_text())[section]
    subjects = sorted({k.split('|')[0] for k in want}) if section != 'outputs' else SUBJECTS
    got = SECTIONS[section](subjects)
    diffs = []
    for k in sorted(set(want) | set(got)):
        if k not in got or k not in want:
            diffs.append(f'{section}: {k} {"missing now" if k not in got else "new"}')
        elif want[k] != got[k]:
            if isinstance(want[k], dict):
                bad = [f for f in sorted(set(want[k]) | set(got[k])) if want[k].get(f) != got[k].get(f)]
                diffs.append(f'{section}: {k}: {", ".join(bad)}')
            else:
                diffs.append(f'{section}: {k}')
    return diffs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=('capture', 'check'))
    ap.add_argument('--subjects', nargs='+', default=list(SUBJECTS))
    ap.add_argument('--section', choices=tuple(SECTIONS), action='append')
    a = ap.parse_args(argv)
    sections = a.section or list(SECTIONS)
    if a.action == 'capture':
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(fingerprints(a.subjects, sections), indent=1, sort_keys=True) + '\n')
        print(f'wrote {GOLDEN.relative_to(ROOT)}')
        return
    diffs = [d for s in sections for d in check(s)]
    for d in diffs:
        print(d)
    print('fingerprints identical' if not diffs else f'{len(diffs)} fingerprint(s) changed')
    raise SystemExit(1 if diffs else 0)


if __name__ == '__main__':
    main()
