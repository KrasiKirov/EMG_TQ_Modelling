"""What produced a run: code commit, local edits, recording hashes, versions, threads, host, protocol.

Every run stores ``collect(...)`` as ``res['provenance']`` and ``config_hash(cfg)`` as ``res['config_hash']``.
Study v2 runs (ids TS2_ / FE_) refuse to start when the code they depend on has uncommitted edits.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys

from EXPERIMENTS.paths import ROOT as _ROOT

ROOT = str(_ROOT)
# Paths whose uncommitted edits make a run irreproducible (untracked files are ignored, except the run's own spec).
CODE_PATHS = ('ML', 'EXPERIMENTS/paths.py', 'EXPERIMENTS/pipeline', 'EXPERIMENTS/protocol', 'EXPERIMENTS/tuning')
_SHA_CACHE: dict[str, str] = {}


def config_hash(cfg: dict) -> str:
    """SHA-256 of the full config (sorted keys), first 16 hex characters."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=float).encode()).hexdigest()[:16]


def sha256_file(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    if path not in _SHA_CACHE:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                h.update(block)
        _SHA_CACHE[path] = h.hexdigest()
    return _SHA_CACHE[path]


def _git(*args) -> str:
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def git_state(spec_path: str | None = None) -> tuple[str, bool]:
    """(HEAD commit, dirty): dirty when a tracked file under CODE_PATHS has uncommitted edits, or when the run's
    own spec file is modified or untracked."""
    commit = _git('rev-parse', 'HEAD').strip()
    dirty = bool(_git('status', '--porcelain', '--untracked-files=no', '--', *CODE_PATHS).strip())
    if spec_path is not None:
        rel = os.path.relpath(spec_path, ROOT)
        dirty = dirty or bool(_git('status', '--porcelain', '--', rel).strip())
    return commit, dirty


def data_files(subject: str) -> dict[str, str]:
    """{file name: path} of the recordings a run of ``subject`` reads."""
    from ML.config import SUBJECTS, DATA_DIR
    if subject not in SUBJECTS:
        return {}
    fn = SUBJECTS[subject][0]
    return {fn: os.path.join(DATA_DIR, fn)}


def collect(*, threads: int, protocol_hash: str | None, subject: str, spec_path: str | None = None) -> dict:
    import numpy as np
    commit, dirty = git_state(spec_path)
    versions = {'python': sys.version.split()[0], 'numpy': np.__version__}
    try:
        import tensorflow as tf
        versions['tensorflow'] = tf.__version__
    except ImportError:            # pragma: no cover
        pass
    return {'git_commit': commit, 'git_dirty': dirty,
            'data_sha256': {fn: sha256_file(p) for fn, p in data_files(subject).items()},
            'versions': versions, 'threads': threads, 'host': platform.node(), 'protocol_hash': protocol_hash}
