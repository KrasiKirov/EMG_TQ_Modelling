"""One dashboard data file for study v2: EXPERIMENTS/dashboard_react/src/data/study_v2.json.

    python -m EXPERIMENTS.analysis.study_v2_export

  tuning        cell -> round -> config -> session-1 validation nMAE per person (full block and D24 targets) and the
                config's network parameters; usable before A0 is chosen (reads seedN_val.npz only)
  confirm       the confirmatory analysis (analysis/confirm.py) once A0 is chosen; None before
  settings      the page's 18 final settings (A0, the other cell winners, A1, A2, A3) and what defines each
  final         per setting and person, per position (ascending angle) x seed: MAE (Nm), nMAE and R² on session 2,
                scored on confirm's own samples (from the shared start ``score_from_s``), the peak |measured
                target| there, and each seed's training time. A0's seeds 10-12 are its cell winner's runs, so that
                cell is not a separate setting.
  runs          every study run: spec, subject, direction, seed, config hash, provenance (no host name) and the
                command that reproduces it
  machine, versions  the computer the runs trained on and the software versions they reported
Size budget: 12 MB.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from EXPERIMENTS.analysis import confirm as CF, study_scores as SS
from EXPERIMENTS.pipeline import spec as SPEC
from EXPERIMENTS.pipeline.run import RESULTS_V2
from EXPERIMENTS.protocol import lock as LK, protocol as PT
from EXPERIMENTS.tuning import select as SEL

OUT = Path(__file__).resolve().parents[1] / 'dashboard_react' / 'src' / 'data' / 'study_v2.json'
SIZE_BUDGET = 12_000_000
SEED_RE = re.compile(r'^seed(\d+)\.json$')
NET = ('units', 'layers', 'dropout', 'lr', 'batch')
THREADS = 1                                      # threads per run (every run's provenance)


def _seeds(d: Path, val=False):
    pat = re.compile(r'^seed(\d+)_val\.npz$') if val else SEED_RE
    return sorted(int(m.group(1)) for f in d.iterdir() if (m := pat.match(f.name)))


def _changes(exp: str) -> dict | None:
    try:
        return SPEC.load(exp).changes
    except SPEC.SpecError:
        return None


def _params(exp: str) -> dict | None:
    ch = _changes(exp)
    return {k: ch.get(k) for k in NET} if ch is not None else None


def tuning(rd: Path) -> dict:
    out = {}
    people = PT.load()['people']
    for d in sorted(rd.glob('TS2_*')):
        _, rnd, rest = d.name.split('_', 2)
        cell, name = rest.rsplit('_', 1)
        if not any((d / p / SEL.DIRECTION).is_dir() for p in people):
            continue
        seeds = sorted({x for p in people if (d / p / SEL.DIRECTION).is_dir()
                        for x in _seeds(d / p / SEL.DIRECTION, val=True)})
        if not seeds:
            continue
        try:
            tables, cov = SEL.load_val([d.name], seeds, people, rd)
        except FileNotFoundError:                       # still running: shown as unfinished
            out.setdefault(cell, {}).setdefault(rnd, {})[name] = {'id': d.name, 'seeds': seeds, 'incomplete': True,
                                                                  'params': _params(d.name)}
            continue
        out.setdefault(cell, {}).setdefault(rnd, {})[name] = {
            'id': d.name, 'seeds': seeds, 'val_nmae': tables['full'][d.name],
            'val_nmae_common': tables['common'][d.name], 'n_common': cov[d.name], 'params': _params(d.name)}
    return out


def _title(kind: str, st: dict) -> str:
    if kind == 'A1':
        return 'A1: lagged ridge'
    if kind == 'A2':
        return "A2: last semester's network (8×2)"
    if kind == 'A3':
        return 'A3: A0 trained on total torque'
    if st['state'] == 'carry':
        # trained on whole chunks: the spec's window only aligns the scored samples, so it is not part of the name
        what = f"carried state, {st['chunk_s']:g} s chunks"
    elif st['overlap'] == 'full':
        what = f"{st['window_s']:g} s window, full overlap"
    else:
        o = re.search(r'_o(\d+)$', st['cell'] or '')
        what = f"{st['window_s']:g} s window, {int(o.group(1)) if o else round(st['overlap'] * 100)} % overlap"
    return f'A0 (chosen): {what}' if kind == 'A0' else what


def settings(rd: Path, protocol: dict, choice: dict) -> dict:
    """The page's final settings, A0 first: what defines each (from its spec and the protocol cell)."""
    cell0 = CF.a0_cell(choice)
    exps = list(CF._expected(rd, protocol))
    ids = ['FE_A0'] + [e for e in exps if e.startswith('FE_cell_') and e != f'FE_cell_{cell0}'] + \
          [e for e in ('FE_A1', 'FE_A2', 'FE_A3') if e in exps]
    out = {}
    for e in ids:
        ch = _changes(e)
        if ch is None:
            continue
        kind = 'cell' if e.startswith('FE_cell_') else e[3:]
        cell = e[len('FE_cell_'):] if kind == 'cell' else cell0 if kind in ('A0', 'A3') else None
        ridge = ch.get('model', 'lstm') != 'lstm'
        carry = ch.get('state') == 'carry'
        pc = protocol['cells'].get(cell, {}) if cell else {}
        stride = int(ch.get('train_stride', 1))
        overlap = (None if ridge or carry else pc['overlap'] if 'overlap' in pc
                   else 'full' if stride == 1 else round(1 - stride / int(ch['window']), 2))
        st = {'kind': kind, 'cell': cell, 'model': ch.get('model', 'lstm'),
              'window_s': round(int(ch['window']) * int(ch.get('input_step', 1)) / 100, 3), 'overlap': overlap,
              'state': None if ridge else 'carry' if carry else 'reset',
              'chunk_s': round(int(ch['chunk']) / 100, 3) if carry else None,
              'target': 'total' if ch.get('passive') == 'none' else 'active', **{k: ch.get(k) for k in NET}}
        out[e] = {'title': _title(kind, st), **st}
    return out


def trial_order(z) -> list[tuple[float, int]]:
    """A run's positions as (angle rounded to 3 decimals, trial), in the page's order: ascending angle, then trial
    (two trials at one angle stay apart)."""
    tr = np.asarray(z['trial'])
    ang = np.asarray(z['angle'], float) if 'angle' in z else tr.astype(float)
    return sorted((round(float(ang[tr == i][0]), 3), int(i)) for i in np.unique(tr))


def final_scores(rd: Path, loaded=None) -> tuple[dict, dict]:
    """({page id: {person: positions, seeds, mae/nmae/r2 [position][seed], peak [position], train_s {seed},
    train {seed: best_epoch, epochs, n_params}}},
    the shared scoring start) on confirm's own samples. ``loaded`` is ``confirm.final_runs``'s result."""
    rd = Path(rd)
    if not LK.test_open(rd):
        return {}, {}
    runs, (start, _, scale), sfrom = loaded or CF.final_runs(rd)
    cell0 = CF.a0_cell(json.loads((rd / 'choice.json').read_text()))
    d1 = PT.load()['primary_direction']
    out = {}
    for e, p, d, s in sorted(runs):
        if d != d1 or e == f'FE_cell_{cell0}':          # A0's seeds 10-12 are these runs (read as FE_A0)
            continue
        js, z = runs[e, p, d, s]
        y, yp, t, tr = (np.asarray(z[k], float) for k in ('y_own_target', 'y_pred', 't', 'trial'))
        scored = CF.scored_mask(t, tr, start, p, d, sfrom[p][d])
        rows = []
        for a, i in trial_order(z):
            m = scored & (tr == i)
            err, ym = yp[m] - y[m], y[m]
            sst = float(np.sum((ym - ym.mean()) ** 2))
            mae = float(np.mean(np.abs(err)))
            rows.append((a, i, mae, mae / scale[p, d],
                         1 - float(np.sum(err ** 2)) / sst if sst > 0 else None, float(np.max(np.abs(ym)))))
        f = out.setdefault(e, {}).setdefault(p, {'positions': [r[0] for r in rows], '_trials': [r[1] for r in rows],
                                                 'seeds': [], 'mae': [[] for _ in rows], 'nmae': [[] for _ in rows],
                                                 'r2': [[] for _ in rows], 'peak': [r[5] for r in rows],
                                                 'train_s': {}, 'train': {}})
        if f['_trials'] != [r[1] for r in rows]:
            raise ValueError(f'{e} {p} seed {s}: positions differ from the other seeds')
        f['seeds'].append(s)
        for k, r in enumerate(rows):
            f['mae'][k].append(r[2])
            f['nmae'][k].append(r[3])
            f['r2'][k].append(r[4])
        f['train_s'][str(s)] = js.get('train_s')
        ti = js.get('train_info') or {}
        f['train'][str(s)] = {k: ti.get(k) for k in ('best_epoch', 'epochs', 'n_params')}
    for per in out.values():
        for f in per.values():
            del f['_trials']
    return out, sfrom


def _versions(runs) -> list:
    seen = []
    for js, _ in runs.values():
        v = js.get('versions') or (js.get('provenance') or {}).get('versions')
        if v and v not in seen:
            seen.append(v)
    return seen


def runs(rd: Path) -> dict:
    out = {}
    for f in sorted(rd.glob('*/*/*/seed*.json')):
        e, p, d = f.parts[-4:-1]
        s = int(SEED_RE.match(f.name).group(1))
        js = json.loads(f.read_text())
        out[f'{e}/{p}/{d}/{s}'] = {
            'spec': e, 'subject': p, 'direction': d, 'seed': s, 'config_hash': js.get('config_hash'),
            # the computer's network name stays out of the (public) export
            'provenance': {k: v for k, v in (js.get('provenance') or {}).items() if k != 'host'},
            'command': f'python -m EXPERIMENTS.pipeline.run --exp {e} --subject {p} --seed {s}'}
    return out


def build(rd=None, with_confirm=True, machine_raw=None) -> dict:
    """``machine_raw``: study_scores.read_machine() of the computer the runs trained on (main passes it)."""
    rd = Path(rd or RESULTS_V2)
    out = {'tuning': tuning(rd) if rd.exists() else {}, 'confirm': None, 'settings': {}, 'final': {},
           'score_from_s': {}, 'runs': runs(rd) if rd.exists() else {}, 'versions': [],
           'machine': SS.machine(machine_raw, PT.load()['budget']['workers'], THREADS) if machine_raw else None}
    if not LK.test_open(rd):
        return out
    protocol = PT.load()
    out['settings'] = settings(rd, protocol, json.loads((rd / 'choice.json').read_text()))
    try:
        loaded = CF.final_runs(rd, protocol)
    except CF.Refused as e:
        loaded, out['confirm'] = None, {'refused': str(e)}
    if loaded:
        out['final'], out['score_from_s'] = final_scores(rd, loaded)
        out['versions'] = _versions(loaded[0])
        if with_confirm:
            try:
                out['confirm'] = CF.run(rd)
            except CF.Refused as e:
                out['confirm'] = {'refused': str(e)}
    return out


def write(data: dict, out=OUT) -> int:
    text = json.dumps(data, separators=(',', ':'), default=float, allow_nan=False)
    if len(text) > SIZE_BUDGET:
        raise ValueError(f'study_v2.json would be {len(text) / 1e6:.1f} MB, over the {SIZE_BUDGET / 1e6:.0f} MB budget')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text + '\n')
    return len(text)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(OUT))
    a = ap.parse_args(argv)
    n = write(build(machine_raw=SS.read_machine()), a.out)
    print(f'wrote {a.out} ({n / 1e6:.2f} MB)')


if __name__ == '__main__':
    main()
