"""Study v2 compute budget (deviation of 18 Nov): the study must end within the protocol's budget.days of wall time.

    python -m EXPERIMENTS.tuning.budget --check

When the R1 runs of every cell but 75 % overlap end, and again when R1 ends, ``check`` projects the total wall time from measured run times only
(seedN.json prep_s + train_s, never a score or a prediction) and adds the protocol's fallback steps, in order, until
the projection fits. Steps are only added, never removed. Writes results_v2/budget.json, which make_specs, confirm and
scripts/run_study_v2.sh read:
  drop_o75       the 75 % overlap cells are dropped (their R1 runs come after the first check)
  skip_2s_check  the 2 s check is not run
  fine_2         the fine round has 2 draws per cell (the first 2 of the same 4)
Cells not measured yet are estimated with the plan's run multipliers; A0 and A3 with the slowest measured cell.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

from EXPERIMENTS.protocol import protocol as PT

_P = PT.load()
_B = _P['budget']
FILE = 'budget.json'
_RUN = re.compile(r'^TS2_(R\d)_(.+)_[^_]+$')
_SEED = re.compile(r'^seed(\d+)\.json$')


def _rd(results_dir=None) -> Path:
    if results_dir is None:
        from EXPERIMENTS.pipeline.run import RESULTS_V2
        results_dir = RESULTS_V2
    return Path(results_dir)


def applied(results_dir=None) -> list[str]:
    f = _rd(results_dir) / FILE
    return list(json.loads(f.read_text()).get('applied', [])) if f.exists() else []


def fine_n(results_dir=None) -> int:
    return _B['fine_reduced_n'] if 'fine_2' in applied(results_dir) else _P['fine']['n']


def multiplier(cell: str) -> float:
    v = _P['cells'][cell]
    return _B['multipliers']['carry' if v.get('state') == 'carry' else f'w{v["window"]}']


def measured(results_dir=None) -> tuple[dict, float | None]:
    """({(round, cell, kind): [seconds]} with kind 'first' (seed 0) or 'more' (other seeds), earliest run start)."""
    out, start = {}, None
    for f in _rd(results_dir).glob('TS2_*/*/*/seed*.json'):
        m, s = _RUN.match(f.parts[-4]), _SEED.match(f.name)
        if not (m and s) or m.group(2) not in _P['cells']:
            continue
        js = json.loads(f.read_text())
        sec = float(js.get('prep_s', 0.0)) + float(js.get('train_s', 0.0))
        kind = 'first' if int(s.group(1)) == 0 else 'more'
        out.setdefault((m.group(1), m.group(2), kind), []).append(sec)
        t0 = f.stat().st_mtime - sec
        start = t0 if start is None else min(start, t0)
    return out, start


def project(steps: list[str], results_dir=None, now=None) -> dict:
    """Projected wall hours: elapsed since the first run started, plus every remaining run at 6 workers."""
    from EXPERIMENTS.tuning import make_specs as MS
    rd, now = _rd(results_dir), time.time() if now is None else now
    runs, start = measured(rd)
    people = len(_P['people'])
    per_cell = {}
    for (_, c, _), v in runs.items():
        per_cell.setdefault(c, []).extend(v)
    unit = (statistics.median(s / multiplier(c) for c, v in per_cell.items() for s in v) if per_cell else None)

    def t(cell):
        if cell in per_cell:
            return statistics.mean(per_cell[cell])
        return (unit or 0.0) * multiplier(cell)

    def done(rnd, cell, kind='first'):
        return len(runs.get((rnd, cell, kind), []))

    cells = MS.active_cells(rd, steps=steps)
    n1 = _P['n_draws'] + 1
    nf = _B['fine_reduced_n'] if 'fine_2' in steps else _P['fine']['n']
    r3 = _P['rounds']['R3']['keep'] * len(_P['rounds']['R3']['new_seeds'])
    fs = _P['final_seeds']
    phases = {
        'R1': sum(max(0, n1 * people - done('R1', c)) * t(c) for c in cells),
        '2s_check': 0.0 if ('skip_2s_check' in steps or (rd / 'gate_w200_full.json').exists())
                    else max(0, n1 * people - done('R1', 'w200_full')) * t('w200_full'),
        'R2': sum(max(0, nf * people - done('R2', c)) * t(c) for c in cells),
        'R3': sum(max(0, r3 * people - done('R1', c, 'more') - done('R2', c, 'more')) * t(c) for c in cells),
    }
    worst = max(t(c) for c in cells)
    phases['final'] = (sum(len(fs['cell']) * people * t(c) for c in cells)
                       + (len(fs['A0']) - len(fs['cell'])) * people * worst
                       + len(fs['A2']) * people * t('w20_full') + len(fs['A3']) * people * worst)
    hours = {k: v / _B['workers'] / 3600 for k, v in phases.items()}
    elapsed = 0.0 if start is None else max(0.0, now - start) / 3600
    total = elapsed + sum(hours.values())
    return {'steps': list(steps), 'elapsed_h': round(elapsed, 1), 'remaining_h': {k: round(v, 1) for k, v in hours.items()},
            'total_h': round(total, 1), 'projected_days': round(total / 24, 2), 'fits': total <= _B['days'] * 24,
            'unit_s': unit, 'n_runs_measured': sum(len(v) for v in runs.values())}


def check(results_dir=None, now=None) -> dict:
    """Add fallback steps in order until the projection fits; write and return budget.json."""
    rd = _rd(results_dir)
    steps = applied(rd)
    proj = project(steps, rd, now)
    for s in _B['fallback']:
        if proj['fits']:
            break
        if s not in steps:
            steps.append(s)
            proj = project(steps, rd, now)
    f = rd / FILE
    old = json.loads(f.read_text()) if f.exists() else {}
    out = {'applied': steps, 'fits': proj['fits'], 'projected_days': proj['projected_days'], 'budget_days': _B['days'],
           'rule': _B['rule'], 'checks': old.get('checks', []) + [dict(proj, checked_at=time.time() if now is None
                                                                         else now)]}
    rd.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, indent=1) + '\n')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', required=True)
    ap.parse_args(argv)
    out = check()
    last = out['checks'][-1]
    print(json.dumps({k: last[k] for k in ('elapsed_h', 'remaining_h', 'projected_days', 'n_runs_measured')}, indent=1))
    print(f"applied: {out['applied'] or 'none'}; fits {out['budget_days']} days: {out['fits']}")
    if not out['fits']:
        print('still over budget with every fallback step: stop and ask the user before R2')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
