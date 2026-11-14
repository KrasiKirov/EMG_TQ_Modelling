"""Experiment specifications: one JSON file per experiment, EXPERIMENTS/configs/<id>.json.

Fields
  id, title                 required; the file name must equal the id
  description               optional (falls back to 'hypothesis')
  tags                      optional list of strings
  changes                   required; overrides of the training baseline (read by pipeline/run.py)
  calibration               optional, scoring only. Default {"type": "none"}.
                            {"type": "gain_offset", "seconds": 10, "score_from_s": 30}
  results_from              optional; reuse another experiment's saved predictions (scoring-only variants)
  dashboard                 optional; present = shown in the dashboard
                            {"group", "label", "id"?, "name"?, "order"?, "blurb"?, "text"?}
                            text may hold purpose/target/protocol/overview/params and report (a
                            repo-relative Markdown file shown on the experiment page)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'
ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.+-]*$')
NO_CALIBRATION = {'type': 'none'}


class SpecError(ValueError):
    """A spec file is missing a field or has an invalid value."""


@dataclass
class DashboardEntry:
    id: str
    group: str
    label: str
    name: str
    order: int = 100
    blurb: str = ''
    text: dict = field(default_factory=dict)   # optional verbatim purpose/target/protocol/overview/params


@dataclass
class Spec:
    id: str
    title: str
    description: str
    tags: list
    changes: dict
    calibration: dict
    results_from: str
    dashboard: DashboardEntry | None
    raw: dict = field(repr=False)


def _number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _calibration(raw):
    cal = raw.get('calibration', NO_CALIBRATION)
    kind = cal.get('type') if isinstance(cal, dict) else None
    if kind == 'none':
        return dict(NO_CALIBRATION)
    if kind == 'gain_offset':
        sec, start = cal.get('seconds'), cal.get('score_from_s')
        if not (_number(sec) and sec > 0):
            raise SpecError('calibration.seconds must be a positive number')
        if not (_number(start) and start >= sec):
            raise SpecError('calibration.score_from_s must be >= calibration.seconds')
        return {'type': 'gain_offset', 'seconds': float(sec), 'score_from_s': float(start)}
    raise SpecError(f"calibration.type must be 'none' or 'gain_offset', got {kind!r}")


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
    entry = None
    dash = raw.get('dashboard')
    if dash is not None:
        for key in ('group', 'label'):
            if not dash.get(key):
                raise SpecError(f'dashboard.{key} is required')
        entry = DashboardEntry(id=dash.get('id', raw['id']), group=dash['group'], label=dash['label'],
                               name=dash.get('name', dash['label']), order=int(dash.get('order', 100)),
                               blurb=dash.get('blurb', ''), text=dict(dash.get('text', {})))
    return Spec(id=raw['id'], title=raw['title'], description=raw.get('description') or raw.get('hypothesis') or '',
                tags=list(tags), changes=dict(raw['changes']), calibration=_calibration(raw),
                results_from=raw.get('results_from', raw['id']), dashboard=entry, raw=raw)


def load(exp_id: str, configs: Path = None) -> Spec:
    path = Path(configs or CONFIGS) / f'{exp_id}.json'
    if not path.exists():
        raise SpecError(f'no spec file {path}')
    return parse(json.loads(path.read_text(encoding='utf-8')), path.stem)


def load_all(configs: Path = None) -> list:
    configs = Path(configs or CONFIGS)
    return [load(p.stem, configs) for p in sorted(configs.glob('*.json'))]


def tag(exp_id, *, label, group, name=None, description=None, tags=None, order=None, blurb=None,
        configs: Path = None, dry_run=False) -> Spec:
    """Add or update a spec's dashboard entry (and optionally description/tags); validate before writing.
    Tags given here replace the spec's tags. dry_run=True validates and returns the result without writing."""
    path = Path(configs or CONFIGS) / f'{exp_id}.json'
    raw = json.loads(path.read_text(encoding='utf-8'))
    dash = dict(raw.get('dashboard') or {})
    dash.update(group=group, label=label)
    for key, value in (('name', name), ('order', order), ('blurb', blurb)):
        if value is not None:
            dash[key] = value
    raw['dashboard'] = dash
    if description is not None:
        raw['description'] = description
    if tags is not None:
        raw['tags'] = sorted(set(tags))
    spec = parse(raw, path.stem)
    if dry_run:
        return spec
    path.write_text(json.dumps(raw, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    return spec
