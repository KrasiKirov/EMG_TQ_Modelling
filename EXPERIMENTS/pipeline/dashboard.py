"""Update the results dashboard from tagged experiments.

    python -m EXPERIMENTS.pipeline.dashboard list
    python -m EXPERIMENTS.pipeline.dashboard tag TEAM01_active_dropout --label "Dropout 0.2" --group Models \\
        --description "A0 with dropout 0.2" --tag dropout --tag active
    python -m EXPERIMENTS.pipeline.dashboard export [EXP ...]     # rescore into presentation-catalog.json
    python -m EXPERIMENTS.pipeline.dashboard build [EXP ...]      # export + rebuild dist/index.html
    python -m EXPERIMENTS.pipeline.dashboard remove EXP
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import spec as S
from .export import _drop, export

EXP = Path(__file__).resolve().parents[1]
DASH = EXP / 'dashboard_react'
CATALOG = DASH / 'src/data/presentation-catalog.json'
RESULTS = EXP / 'results'
GROUPS = EXP / 'dashboard_groups.json'


def _coverage(spec):
    runs = list(RESULTS.glob(f'{spec.results_from}/*/*/seed*.npz'))
    people = {p.parts[-3] for p in runs}
    return f'{len(people)} people, {len(runs)} runs' if runs else 'no results'


def cmd_list(_):
    for s in S.load_all():
        d = s.dashboard
        place = f'[dashboard] {d.group} / {d.label}' if d else '-'
        print(f"{s.id:34s} {place:70s} "
              f"tags={','.join(s.tags) or '-':30s} {_coverage(s)}")


def cmd_tag(a):
    groups = {g['label'] for g in json.loads(GROUPS.read_text(encoding='utf-8'))['groups']}
    if a.group not in groups:
        sys.exit(f'unknown group {a.group!r}; add it to {GROUPS} first. Known: {sorted(groups)}')
    s = S.tag(a.exp, label=a.label, group=a.group, name=a.name, description=a.description,
              tags=a.tag, order=a.order, blurb=a.blurb, dry_run=a.dry_run)
    if a.dry_run:
        shown = {k: s.raw[k] for k in ('description', 'tags', 'dashboard') if k in s.raw}
        print(json.dumps(shown, indent=1, ensure_ascii=False))
        print('(dry run: nothing written)')
        return
    print(f'tagged {s.id} -> {s.dashboard.group} / {s.dashboard.label}')


def _ready(specs):
    """Specs that can appear on the dashboard: untagged ones, and tagged ones with at least one result."""
    return [s for s in specs if not (s.dashboard and _coverage(s) == 'no results')]


def cmd_export(a):
    every = S.load_all()
    if a.exp:
        chosen = [S.load(i) for i in a.exp]
    else:                                   # everything tagged; experiments still waiting for results are skipped
        chosen = [s for s in every if s.dashboard]
        waiting = [s.id for s in chosen if _coverage(s) == 'no results']
        if waiting:
            print(f'skipped (no results yet): {", ".join(waiting)}', file=sys.stderr)
        chosen = [s for s in chosen if s.id not in waiting]
    print(json.dumps(export(chosen, CATALOG, RESULTS, GROUPS, all_specs=_ready(every)), indent=1))


def cmd_build(a):
    cmd_export(a)
    npm = 'npm.cmd' if os.name == 'nt' else 'npm'
    subprocess.run([npm, 'run', 'build'], cwd=DASH, check=True)
    print(f'built {DASH / "dist/index.html"}')


def cmd_remove(a):
    path = Path(S.CONFIGS) / f'{a.exp}.json'
    raw = json.loads(path.read_text(encoding='utf-8'))
    ident = (raw.pop('dashboard', None) or {}).get('id', a.exp)
    path.write_text(json.dumps(raw, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    data = json.loads(CATALOG.read_text(encoding='utf-8'))
    _drop(data, ident)
    CATALOG.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    export([], CATALOG, RESULTS, GROUPS, all_specs=_ready(S.load_all()))   # refresh the sidebar groups
    print(f'removed {ident}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list', help='every spec, its dashboard place, tags and result coverage').set_defaults(fn=cmd_list)
    t = sub.add_parser('tag', help='name, describe, tag and place an experiment on the dashboard')
    t.set_defaults(fn=cmd_tag)
    t.add_argument('exp')
    t.add_argument('--label', required=True, help='short sidebar label')
    t.add_argument('--group', required=True, help='a group from EXPERIMENTS/dashboard_groups.json')
    t.add_argument('--name', help='long display name (default: the label)')
    t.add_argument('--description')
    t.add_argument('--tag', action='append', help='repeat for several tags; replaces the existing tags')
    t.add_argument('--order', type=int, help='position within the group (lower first)')
    t.add_argument('--blurb', help='one sentence under the title')
    t.add_argument('--dry-run', action='store_true', help='show the result without writing')
    for name, fn, help_ in (('export', cmd_export, 'rescore tagged experiments into the catalogue'),
                            ('build', cmd_build, 'export, then rebuild dist/index.html')):
        p = sub.add_parser(name, help=help_)
        p.add_argument('exp', nargs='*')
        p.set_defaults(fn=fn)
    r = sub.add_parser('remove', help='take an experiment off the dashboard (keeps its results)')
    r.add_argument('exp')
    r.set_defaults(fn=cmd_remove)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == '__main__':
    main()
