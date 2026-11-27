"""Experiment specifications: one JSON file per experiment, EXPERIMENTS/configs/<id>.json.

Fields
  id, title                 required; the file name must equal the id
  description               optional (falls back to 'hypothesis')
  tags                      optional list of strings
  changes                   required; overrides of the training baseline (read by pipeline/run.py)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from EXPERIMENTS.paths import CONFIGS
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
    raw: dict = field(repr=False)


def parse(raw: dict, stem: str | None = None) -> Spec:
    for key in ('id', 'title', 'changes'):
        if key not in raw:
            raise SpecError(f'missing field {key!r}')
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
                tags=list(tags), changes=dict(raw['changes']), raw=raw)


def load(exp_id: str, configs: Path = None) -> Spec:
    path = Path(configs or CONFIGS) / f'{exp_id}.json'
    if not path.exists():
        raise SpecError(f'no spec file {path}')
    return parse(json.loads(path.read_text(encoding='utf-8')), path.stem)


