"""Experiment specifications: one JSON file per experiment, EXPERIMENTS/configs/<id>.json.

Fields
  id, title                 required; the file name must equal the id
  description               optional (falls back to 'hypothesis')
  tags                      optional list of strings
  changes                   required; overrides of the training baseline (read by pipeline/run.py)
  results_from              optional; reuse another experiment's saved predictions (scoring-only variants)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'
ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.+-]*$')


class SpecError(ValueError):
    """A spec file is missing a field or has an invalid value."""


@dataclass
class Spec:
    id: str
    title: str
    description: str
    tags: list
    changes: dict
    results_from: str
    raw: dict = field(repr=False)


def parse(raw: dict, stem: str | None = None) -> Spec:
    for key in ('id', 'title', 'changes'):
        if key not in raw:
            raise SpecError(f'missing field {key!r}')
    if 'calibration' in raw:
        raise SpecError('calibration was removed: predictions are always scored as saved, on the whole trial')
    if 'dashboard' in raw:
        raise SpecError('the dashboard block was removed with the v1 dashboard')
    if not ID_RE.match(raw['id']):
        raise SpecError(f'invalid id {raw["id"]!r}')
    if stem is not None and stem != raw['id']:
        raise SpecError(f'file name {stem!r} does not match id {raw["id"]!r}')
    if not isinstance(raw['changes'], dict):
        raise SpecError('changes must be an object')
    tags = raw.get('tags', [])
    if not (isinstance(tags, list) and all(isinstance(t, str) for t in tags)):
        raise SpecError('tags must be a list of strings')
    return Spec(id=raw['id'], title=raw['title'], description=raw.get('description') or raw.get('hypothesis') or '',
                tags=list(tags), changes=dict(raw['changes']),
                results_from=raw.get('results_from', raw['id']), raw=raw)


def load(exp_id: str, configs: Path = None) -> Spec:
    path = Path(configs or CONFIGS) / f'{exp_id}.json'
    if not path.exists():
        raise SpecError(f'no spec file {path}')
    return parse(json.loads(path.read_text(encoding='utf-8')), path.stem)


def load_all(configs: Path = None) -> list:
    configs = Path(configs or CONFIGS)
    return [load(p.stem, configs) for p in sorted(configs.glob('*.json'))]

