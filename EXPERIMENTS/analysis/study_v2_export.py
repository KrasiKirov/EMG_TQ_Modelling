"""One dashboard data file for study v2: EXPERIMENTS/dashboard_react/src/data/study_v2.json.

    python -m EXPERIMENTS.analysis.study_v2_export

  tuning   cell -> round -> config -> session-1 validation nMAE per person (full block and D24 targets); usable
           before A0 is chosen (reads seedN_val.npz only)
  confirm  the confirmatory analysis (analysis/confirm.py) once A0 is chosen; None before
  runs     every study run: spec, subject, direction, seed, config hash, provenance, the command that reproduces
           it, and per-position session-2 nMAE once A0 is chosen (read through protocol/lock.py)
  legacy   the v1 window study (EXPERIMENTS/results/W*), seeds 0-4 only, flagged exploratory
Size budget: 12 MB.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from EXPERIMENTS.pipeline.run import RESULTS, RESULTS_V2
from EXPERIMENTS.pipeline.scoring import person_scale
from EXPERIMENTS.protocol import lock as LK, protocol as PT
from EXPERIMENTS.tuning import select as SEL

OUT = Path(__file__).resolve().parents[1] / 'dashboard_react' / 'src' / 'data' / 'study_v2.json'
SIZE_BUDGET = 12_000_000
LEGACY_RE = re.compile(r'^W[A-Z0-9]')          # the v1 window study ids (W00_, WG_, WL1_, WO4_, WC2_, WS1_, WT1_)
LEGACY_SEEDS = range(5)
SEED_RE = re.compile(r'^seed(\d+)\.json$')


def _seeds(d: Path, val=False):
    pat = re.compile(r'^seed(\d+)_val\.npz$') if val else SEED_RE
    return sorted(int(m.group(1)) for f in d.iterdir() if (m := pat.match(f.name)))


def _positions(npz, scale):
    out = {}
    for tr in np.unique(npz['trial']):
        m = npz['trial'] == tr
        a = float(npz['angle'][m][0]) if 'angle' in npz else float(tr)
        out[f'{a:.3f}'] = float(np.mean(np.abs(npz['y_pred'][m] - npz['y_own_target'][m])) / scale)
    return out


def tuning(rd: Path) -> dict:
    out = {}
    for d in sorted(rd.glob('TS2_*')):
        _, rnd, rest = d.name.split('_', 2)
        cell, name = rest.rsplit('_', 1)
        people = PT.load()['people']
        if not any((d / p / SEL.DIRECTION).is_dir() for p in people):
            continue
        seeds = sorted({x for p in people if (d / p / SEL.DIRECTION).is_dir()
                        for x in _seeds(d / p / SEL.DIRECTION, val=True)})
        if not seeds:
            continue
        try:
            tables, cov = SEL.load_val([d.name], seeds, people, rd)
        except FileNotFoundError:                       # still running: shown as unfinished
            out.setdefault(cell, {}).setdefault(rnd, {})[name] = {'id': d.name, 'seeds': seeds, 'incomplete': True}
            continue
        out.setdefault(cell, {}).setdefault(rnd, {})[name] = {
            'id': d.name, 'seeds': seeds, 'val_nmae': tables['full'][d.name],
            'val_nmae_common': tables['common'][d.name], 'n_common': cov[d.name]}
    return out


def runs(rd: Path) -> dict:
    open_ = LK.test_open(rd)
    out = {}
    for f in sorted(rd.glob('*/*/*/seed*.json')):
        e, p, d = f.parts[-4:-1]
        s = int(SEED_RE.match(f.name).group(1))
        js = json.loads(f.read_text())
        r = {'spec': e, 'subject': p, 'direction': d, 'seed': s, 'config_hash': js.get('config_hash'),
             # the computer's network name stays out of the (public) export
             'provenance': {k: v for k, v in (js.get('provenance') or {}).items() if k != 'host'},
             'command': f'python -m EXPERIMENTS.pipeline.run --exp {e} --subject {p} --direction {d} --seed {s}'}
        if open_ and f.with_suffix('.npz').exists():
            z = LK.load_test(f.with_suffix('.npz'), rd)
            r['positions'] = _positions(z, person_scale(z['y_own_target']))
        out[f'{e}/{p}/{d}/{s}'] = r
    return out


def legacy(v1: Path) -> dict:
    exps = {}
    for f in sorted(Path(v1).glob(f'*/*/{SEL.DIRECTION}/seed*.npz')) if Path(v1).exists() else []:
        e, p = f.parts[-4], f.parts[-3]
        m = re.match(r'^seed(\d+)\.npz$', f.name)
        if not (m and LEGACY_RE.match(e) and int(m.group(1)) in LEGACY_SEEDS):
            continue
        z = np.load(f)
        pos = _positions(z, person_scale(z['y_own_target']))
        exps.setdefault(e, {}).setdefault(p, {})[m.group(1)] = float(np.mean(list(pos.values())))
    return {'note': 'v1 window study, exploratory: settings were chosen on session 1 -> 2 test results; '
                    'seeds 0-4 only', 'experiments': exps}


def build(rd=None, v1=None, with_confirm=True) -> dict:
    rd = Path(rd or RESULTS_V2)
    conf = None
    if with_confirm and LK.test_open(rd):
        from EXPERIMENTS.analysis import confirm as CF
        try:
            conf = CF.run(rd)
        except CF.Refused as e:
            conf = {'refused': str(e)}
    return {'tuning': tuning(rd) if rd.exists() else {}, 'confirm': conf,
            'runs': runs(rd) if rd.exists() else {}, 'legacy': legacy(Path(v1 or RESULTS))}


def write(data: dict, out=OUT) -> int:
    text = json.dumps(data, separators=(',', ':'), default=float)
    if len(text) > SIZE_BUDGET:
        raise ValueError(f'study_v2.json would be {len(text) / 1e6:.1f} MB, over the {SIZE_BUDGET / 1e6:.0f} MB budget')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text + '\n')
    return len(text)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(OUT))
    a = ap.parse_args(argv)
    n = write(build(), a.out)
    print(f'wrote {a.out} ({n / 1e6:.2f} MB)')


if __name__ == '__main__':
    main()
