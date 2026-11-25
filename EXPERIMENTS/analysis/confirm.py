"""Confirmatory analysis of study v2 (decisions D5, D6, D9): the pre-registered families on session-2 predictions.

    python -m EXPERIMENTS.analysis.confirm      # writes EXPERIMENTS/dashboard_react/src/data/confirm.json

Refuses (``Refused``) when any final run is missing, was made with another protocol or on uncommitted code, has a
stored config that does not match its own hash or differs from its spec's current config (``config_check``), or when
two runs read different versions of the same recording. The retired overlap cells (2026-11-24, full overlap only)
have no spec on main (tag study-v2-final keeps them): their runs are checked against their own hash and the protocol
cell's overlap, and F2 is still reported from them. Opens session-2 predictions only through ``lock.load_test`` (so only after A0 was chosen).

Score of a run: per trial (position), MAE divided by the person's scale (99th percentile of |measured active
target| of the tested session, from FE_A0, also used for A3: D6); the run's nMAE is the mean over positions.
Each comparison is scored on the samples both of its arms predict: per person, from the protocol's ``score_from_s``
after the trial's start, or later if one of the two arms first predicts later (a w-step window first predicts
(w - 1) x 10 ms in); each comparison reports its start as ``score_from_s``. The descriptive scores and D2b put every
final arm on one shared start, from max(score_from_s, the latest first prediction of any final arm), so they line up
as one curve; the top-level ``score_from_s`` {person: {direction: s}} is that shared start. The trial start is the first prediction time minus the run's
window span. FE_A0's session 1 -> 2 seeds that its cell winner already ran (10-12) are read from FE_cell_<cell>.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from EXPERIMENTS.analysis import stats as ST
from EXPERIMENTS.pipeline import provenance as PV
from EXPERIMENTS.paths import DASH_DATA, RESULTS_V2
from EXPERIMENTS.pipeline.settings import load_config, strip_retired
from EXPERIMENTS.pipeline.spec import SpecError
from EXPERIMENTS.pipeline.scoring import person_scale
from EXPERIMENTS.protocol import lock as LK, protocol as PT
from EXPERIMENTS.tuning import budget as BG
from ML.research import config as MC

OUT = DASH_DATA / 'confirm.json'
ARM = {'A0': 'FE_A0', 'A1': 'FE_A1', 'A2': 'FE_A2', 'A3': 'FE_A3'}


class Refused(RuntimeError):
    """The final runs are incomplete or inconsistent; nothing is reported."""


def _expected(rd: Path, protocol: dict) -> dict[str, tuple[list[str], list[int]]]:
    fs, s1 = protocol['final_seeds'], protocol['primary_direction']
    out = {}
    dropped = 'drop_o75' in BG.applied(rd)
    for c, v in protocol['cells'].items():
        if dropped and c.endswith('_o75'):
            continue
        if not v['gated'] or (rd / f'FE_cell_{c}').exists():
            out[f'FE_cell_{c}'] = ([s1], fs['cell'])
    out['FE_A0'] = ([s1], fs['A0'])
    out['FE_A1'] = ([s1], [fs['A0'][0]])
    out['FE_A2'] = ([s1], fs['A2'])
    out['FE_A3'] = ([s1], fs['A3'])
    return out


def a0_cell(choice: dict) -> str | None:
    return next((c for c, i in choice.get('cell_winners', {}).items() if i == choice.get('A0')), None)


def _load_runs(rd, protocol):
    """{(arm, person, direction, seed): (json, npz)}; refuses on a missing run."""
    runs, missing = {}, []
    cell = a0_cell(json.loads((rd / 'choice.json').read_text()))
    for e, (dirs, seeds) in _expected(rd, protocol).items():
        for p in protocol['people']:
            for d in dirs:
                for s in seeds:
                    stem = rd / e / p / d / f'seed{s}'
                    shared = (e == 'FE_A0' and d == protocol['primary_direction'] and cell
                              and s in protocol['final_seeds']['cell'])
                    if shared and not stem.with_suffix('.json').exists():
                        stem = rd / f'FE_cell_{cell}' / p / d / f'seed{s}'
                    if not stem.with_suffix('.json').exists() or not stem.with_suffix('.npz').exists():
                        missing.append(f'{e} {p} {d} seed {s}')
                        continue
                    runs[e, p, d, s] = (json.loads(stem.with_suffix('.json').read_text()),
                                        LK.load_test(stem.with_suffix('.npz'), rd))
    if missing:
        raise Refused(f'{len(missing)} final runs missing: ' + '; '.join(missing[:20]))
    return runs


def config_check(exp: str, js: dict, protocol: dict, name: str | None = None) -> str:
    """How one run's stored config was verified: 'spec' (it equals its spec's current config, retired overlap
    settings at their full-overlap values aside) or 'retired cell' (an overlap cell with no spec: its stored window and
    train_stride equal the protocol cell's). The stored config must match the run's own config_hash first."""
    name = name or exp
    cfg = js.get('config') or {}
    if PV.config_hash(cfg) != js.get('config_hash'):
        raise Refused(f'{name}: stored config does not match its hash {js.get("config_hash")}')
    cell = protocol['cells'].get(exp[len('FE_cell_'):]) if exp.startswith('FE_cell_') else None
    try:
        spec = load_config(exp)[1]
    except SpecError:
        spec = None
    if spec is not None:
        try:
            same = strip_retired(cfg) == spec
        except SpecError:
            same = False
        if not same:
            diff = sorted(k for k in set(cfg) | set(spec) if cfg.get(k) != spec.get(k))
            raise Refused(f'{name}: stored config differs from spec {exp} in {diff}')
        return 'spec'
    if cell is not None and PT.retired_cell(cell):
        got = {k: cfg.get(k) for k in ('window', 'train_stride')}
        if got != {k: cell[k] for k in ('window', 'train_stride')}:
            raise Refused(f'{name}: stored {got} differ from the retired protocol cell {exp[len("FE_cell_"):]}')
        return 'retired cell'
    raise Refused(f'{name}: no spec {exp}')


def provenance_check(runs, protocol) -> dict:
    checks, data, commits = {}, {}, set()
    for (e, p, d, s), (js, _) in runs.items():
        pv, name = js.get('provenance') or {}, f'{e} {p} {d} seed {s}'
        if pv.get('protocol_hash') != protocol['hash']:
            raise Refused(f'{name}: made with another protocol ({pv.get("protocol_hash")})')
        if pv.get('git_dirty'):
            raise Refused(f'{name}: made on uncommitted code')
        checks[e] = config_check(e, js, protocol, name)
        for f, h in (pv.get('data_sha256') or {}).items():
            data.setdefault(f, {}).setdefault(h, []).append(name)
        commits.add(pv.get('git_commit'))
    for f, by_hash in data.items():
        if len(by_hash) > 1:
            raise Refused(f'recording {f} was read with different contents: ' +
                          '; '.join(f'{h[:12]}: {", ".join(n[:3])}' for h, n in by_hash.items()))
    return {'n_runs': len(runs), 'commits': sorted(c for c in commits if c), 'protocol_hash': protocol['hash'],
            'config_checks': dict(sorted(checks.items()))}


def _span_s(js):
    c = js.get('config') or {}
    return (int(c.get('window', 20)) - 1) * int(c.get('input_step', 1)) / MC.FS


def _timing(runs):
    """Trial starts {(person, direction, trial): s}, each arm's first-prediction delay after the start
    {(arm, person, direction, trial): s} (the latest over its seeds) and the person's scale {(person, direction)}."""
    start, first, scale = {}, {}, {}
    for (e, p, d, s), (js, z) in runs.items():
        t, tr = np.asarray(z['t'], float), np.asarray(z['trial'])
        for i in np.unique(tr):
            f = float(t[tr == i].min())
            start[p, d, int(i)] = min(start.get((p, d, int(i)), np.inf), f - _span_s(js))
            first.setdefault((e, p, d, int(i)), []).append(f)
        if e == 'FE_A0' and (p, d) not in scale:
            scale[p, d] = person_scale(z['y_own_target'])
    delay = {(e, p, d, i): max(fs) - start[p, d, i] for (e, p, d, i), fs in first.items()}
    return start, delay, scale


def scored_mask(t, trial, start, person, direction, sfrom) -> np.ndarray:
    """The samples a run is scored on: at least ``sfrom`` seconds after their trial's start
    (``start`` is ``_timing``'s {(person, direction, trial): s})."""
    t, trial = np.asarray(t, float), np.asarray(trial)
    t0 = np.empty(len(t))
    for i in np.unique(trial):
        t0[trial == i] = start[person, direction, int(i)]
    return t - t0 >= sfrom - 1e-6


def final_runs(results_dir=None, protocol=None):
    """Every final run with its timing and the shared scoring start: (runs, timing, {person: {direction: s}}).
    Opens session 2 only once A0 is chosen."""
    rd, protocol = Path(results_dir or RESULTS_V2), protocol or PT.load()
    if not LK.test_open(rd):
        raise LK.TestLocked(f'no {rd / "choice.json"}: A0 is not chosen yet')
    runs = _load_runs(rd, protocol)
    timing = _timing(runs)
    sfrom = _scores(runs, protocol, {e for e, _, _, _ in runs}, timing)[2]
    return runs, timing, sfrom


def _scores(runs, protocol, arms, timing):
    """Score the runs of ``arms`` on the samples every one of them predicts: per person and direction, from
    max(score_from_s, the latest first prediction of these arms after the trial start). Returns
    ({(arm, person, direction): {seed: nmae}}, per-position nMAE {(arm, person, direction): {trial: {angle, nmae}}},
    the scoring start {person: {direction: s}})."""
    start, delay, scale = timing
    arms = set(arms)
    sfrom = {}
    for (e, p, d, i), f in delay.items():
        if e in arms:
            sfrom.setdefault(p, {})[d] = max(sfrom.get(p, {}).get(d, protocol['score_from_s']), round(f, 2))
    nm, pos = {}, {}
    for (e, p, d, s), (_, z) in runs.items():
        if e not in arms:
            continue
        y, yp, t, tr = (np.asarray(z[k], float) for k in ('y_own_target', 'y_pred', 't', 'trial'))
        per = {}
        scored = scored_mask(t, tr, start, p, d, sfrom[p][d])
        for i in np.unique(tr):
            m = (tr == i) & scored
            ang = float(np.asarray(z['angle'])[tr == i][0]) if 'angle' in z else float('nan')
            per[str(int(i))] = (ang, float(np.mean(np.abs(yp[m] - y[m])) / scale[p, d]))
        nm.setdefault((e, p, d), {})[s] = float(np.mean([v for _, v in per.values()]))
        for i, (a, v) in per.items():
            pos.setdefault((e, p, d), {}).setdefault(i, (a, []))[1].append(v)
    pos = {k: {i: {'angle': round(a, 3), 'nmae': float(np.median(v))} for i, (a, v) in dd.items()}
           for k, dd in pos.items()}
    return nm, pos, sfrom


def _compare(runs, timing, arm, ref, people, d, protocol):
    """One pre-registered comparison, scored on the samples both arms predict."""
    nm, pos, sfrom = _scores(runs, protocol, {arm, ref}, timing)
    a = {p: nm[arm, p, d] for p in people}
    r = {p: nm[ref, p, d] for p in people}
    eff = ST.person_effects(a, r)
    m, lo, hi = ST.t_interval(eff)
    return {'arm': arm, 'comparator': ref, 'per_person': a, 'comparator_seeds': {p: sorted(r[p]) for p in people},
            'effects': eff, 'effect': float(np.median(list(eff.values()))),
            'ci': [m, lo, hi], 'decision': ST.decide(eff, protocol['min_agree'], protocol['min_effect']),
            'score_from_s': {p: sfrom[p][d] for p in people},
            'per_position': {p: pos[arm, p, d] for p in people}}


def _not_run(rd, cell, protocol, steps) -> dict:
    """F1 entry of a history length with no final runs: stopped at its gate, passed but not tuned within the
    budget, or not run (budget: not screened, or the 2 s check skipped)."""
    g = rd / f'gate_{cell}.json'
    if g.exists():
        gate = json.loads(g.read_text())
        return {'decision': 'not tuned (budget)' if gate['continues'] else 'stopped', 'gate': gate}
    order = protocol['gates']['order']
    if cell in order:
        for s in order[:order.index(cell)]:
            f = rd / f'gate_{s}.json'
            if f.exists() and not json.loads(f.read_text())['continues']:
                return {'decision': 'stopped', 'gate': f'{s} stopped'}
    budget = 'skip_2s_check' in steps or cell not in protocol['budget']['screens']
    return {'decision': 'not run (budget)' if budget else 'not run'}


def run(results_dir=None, protocol=None) -> dict:
    rd, protocol = Path(results_dir or RESULTS_V2), protocol or PT.load()
    runs, timing, _ = final_runs(rd, protocol)
    prov = provenance_check(runs, protocol)
    people, d1, fam = protocol['people'], protocol['primary_direction'], protocol['families']
    have = {e for e, _, _, _ in runs}
    nm, _, sfrom = _scores(runs, protocol, have, timing)  # descriptive and D2b: every arm on one shared start
    out = {'families': {'F1': {}, 'F2': {}, 'F3': {}}, 'score_from_s': sfrom}

    steps = BG.applied(rd)
    out['budget'] = {'applied': steps}
    f1 = fam['F1']
    for c in f1['arms']:
        if f'FE_cell_{c}' in have:
            out['families']['F1'][c] = _compare(runs, timing, f'FE_cell_{c}', f'FE_cell_{f1["comparator"]}', people,
                                                 d1, protocol)
        else:
            out['families']['F1'][c] = _not_run(rd, c, protocol, steps)

    overlaps = [o for o in fam['F2']['arms'] if not (o == 'o75' and 'drop_o75' in steps)]
    lengths = sorted({c.split('_')[0] for c, v in protocol['cells'].items() if 'state' not in v
                      and all(f'FE_cell_{c.split("_")[0]}_{o}' in have for o in ['full'] + overlaps)},
                     key=lambda w: int(w[1:]))
    for o in fam['F2']['arms']:
        if o not in overlaps:
            out['families']['F2'][o] = {'decision': 'dropped (budget)'}
            continue
        per_len = {}
        for w in lengths:
            arm, ref = f'FE_cell_{w}_{o}', f'FE_cell_{w}_full'
            pn = _scores(runs, protocol, {arm, ref}, timing)[0]
            per_len[w] = ST.person_effects({p: pn[arm, p, d1] for p in people}, {p: pn[ref, p, d1] for p in people})
        eff = {p: float(np.mean([per_len[w][p] for w in lengths])) for p in people}
        m, lo, hi = ST.t_interval(eff)
        out['families']['F2'][o] = {'lengths': lengths, 'per_length': per_len, 'effects': eff,
                                    'effect': float(np.median(list(eff.values()))), 'ci': [m, lo, hi],
                                    'decision': ST.decide(eff, protocol['min_agree'], protocol['min_effect'])}

    for a, b in fam['F3']['pairs']:
        out['families']['F3'][f'{a} vs {b}'] = _compare(runs, timing, ARM[a], ARM[b], people, d1, protocol)

    out['descriptive'] = {f'{e} {d}': {p: nm[e, p, d] for p in people} for e, d in sorted({(e, d) for e, _, d in nm})}
    out['val_test_agreement'] = _agreement(rd, nm, people, d1, [e for e in have if e.startswith('FE_cell_')])
    out['provenance_check'] = prov
    out['deviations'] = protocol.get('deviations', [])
    out['limitations'] = protocol.get('limitations', [])
    out['choice'] = json.loads((rd / 'choice.json').read_text())
    return out


def _agreement(rd, nm, people, d, arms):
    """D2b: per person, Spearman rho between session-1 validation nMAE and session-2 nMAE over the cell winners."""
    from scipy.stats import spearmanr
    out = {}
    for p in people:
        val, test = [], []
        for e in arms:
            vs = []
            for f in (rd / e / p / d).glob('seed*_val.npz'):
                z = np.load(f)
                vs.append(np.mean(np.abs(z['y_pred'] - z['y'])))
            val.append(float(np.median(vs)))
            test.append(float(np.median(list(nm[e, p, d].values()))))
        out[p] = float(spearmanr(val, test)[0]) if len(arms) > 2 else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(OUT))
    a = ap.parse_args(argv)
    res = run()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float) + '\n')
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
