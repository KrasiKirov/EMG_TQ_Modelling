"""Check the transferred handoff artifacts using only the Python standard library.

    python EXPERIMENTS/code/verify_handoff.py
    python EXPERIMENTS/code/verify_handoff.py --quick

After deliberately updating the included artifacts, create a new inventory with
--write-manifest and review its diff before committing. This inventories the files
present; it does not establish scientific validity or completeness of an experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'EXPERIMENTS/handoff_manifest.json'
ARTIFACT_ROOTS = (
    'ML/subjects', 'ML/experiments/results', 'ML/research/results',
    'EXPERIMENTS/results', 'EXPERIMENTS/run_collections',
    'EXPERIMENTS/baseline/retest_map_code',
)
SNAPSHOTS = (
    'EXPERIMENTS/dashboard_react/src/data/results.json',
    'EXPERIMENTS/dashboard_react/src/data/catalog.json',
    'EXPERIMENTS/dashboard_react/src/data/presentation-catalog.json',
    'EXPERIMENTS/dashboard_react/dist/index.html',
)
TEXT_SUFFIXES = {'.json', '.py', '.md', '.log', '.diff', '.txt', '.csv'}


def digest(path: Path, normalization: str = 'none') -> str:
    if normalization == 'lf':
        return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def size(path: Path, normalization: str) -> int:
    return len(path.read_bytes().replace(b'\r\n', b'\n')) if normalization == 'lf' else path.stat().st_size


def inventory() -> dict:
    paths = set()
    for name in ARTIFACT_ROOTS:
        directory = ROOT / name
        if not directory.is_dir():
            raise ValueError(f'Missing artifact directory: {name}')
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix != '.pyc' and not any(
                part in {'__pycache__', '_cache', 'local'}
                for part in path.relative_to(directory).parts
            ):
                paths.add(path)
    paths.update(ROOT / name for name in SNAPSHOTS)
    files = []
    for path in sorted(paths):
        normalization = 'lf' if path.suffix in TEXT_SUFFIXES else 'none'
        files.append(dict(path=path.relative_to(ROOT).as_posix(),
                          normalization=normalization, bytes=size(path, normalization),
                          sha256=digest(path, normalization)))
    return dict(schema_version=1, description='Teammate artifact transfer snapshot; see EXPERIMENTS/HANDOFF.md',
                files=files, total_bytes=sum(item['bytes'] for item in files))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='Check paths and sizes only; skip SHA-256')
    parser.add_argument('--write-manifest', action='store_true', help='Replace the inventory after intentional artifact updates')
    args = parser.parse_args()
    if args.quick and args.write_manifest:
        parser.error('--quick and --write-manifest cannot be combined')
    if args.write_manifest:
        data = inventory()
        MANIFEST.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
        print(f'Inventoried {len(data["files"])} files ({data["total_bytes"] / 2**20:.1f} MiB)')
        return 0
    data = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if data['schema_version'] != 1:
        raise ValueError('Unsupported manifest schema')
    errors, seen = [], set()
    for item in data['files']:
        relative = Path(item['path'])
        path = (ROOT / relative).resolve()
        normalization = item['normalization']
        if normalization not in {'none', 'lf'}:
            raise ValueError('Unsupported content normalization')
        if relative.is_absolute() or not path.is_relative_to(ROOT) or item['path'] in seen:
            errors.append(f'Unsafe or duplicate path: {item["path"]}')
            continue
        seen.add(item['path'])
        if not path.is_file():
            errors.append(f'Missing: {item["path"]}')
        elif size(path, normalization) != item['bytes']:
            errors.append(f'Size differs: {item["path"]}')
        elif not args.quick and digest(path, normalization) != item['sha256']:
            errors.append(f'SHA-256 differs: {item["path"]}')
    if errors:
        for error in errors[:20]:
            print(error)
        print(f'FAILED: {len(errors)} issue(s)')
        return 1
    counts = Counter(Path(item['path']).suffix for item in data['files'])
    mode = 'paths and sizes' if args.quick else 'paths, sizes and SHA-256'
    print(f'PASS: {len(data["files"])} files checked ({mode})')
    print(f'Prediction archives: {counts[".npz"]}; recordings: {counts[".flb"]}; '
          f'saved models: {counts[".keras"] + counts[".h5"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
