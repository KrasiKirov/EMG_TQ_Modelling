"""Confirmatory analysis of study v2 (decisions D5, D6, D9): the pre-registered families on session-2 predictions.

    python -m EXPERIMENTS.analysis.confirm      # writes EXPERIMENTS/dashboard_react/src/data/confirm.json

Refuses (``Refused``) when any final run is missing, was made with another protocol or on uncommitted code, has a
config hash that differs from its spec's current config, or when two runs read different versions of the same
recording. Opens session-2 predictions only through ``lock.load_test`` (so only after A0 was chosen).

Score of a run: per trial (position), MAE from ``score_from_s`` after the trial's start, divided by the person's
scale (99th percentile of |measured active target| of the tested session, from FE_A0, also used for A3: D6); the
run's nMAE is the mean over positions. The trial start is the earliest prediction time of that trial over every
final run of the person (carried-state and 0.2 s runs predict from the trial's first samples).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from EXPERIMENTS.analysis import stats as ST
from EXPERIMENTS.pipeline import provenance as PV
from EXPERIMENTS.pipeline.run import RESULTS_V2, load_config
from EXPERIMENTS.pipeline.scoring import person_scale
from EXPERIMENTS.protocol import lock as LK, protocol as PT

OUT = Path(__file__).resolve().parents[1] / 'dashboard_react' / 'src' / 'data' / 'confirm.json'
ARM = {'A0': 'FE_A0', 'A1': 'FE_A1', 'A2': 'FE_A2', 'A3': 'FE_A3'}


class Refused(RuntimeError):
    """The final runs are incomplete or inconsistent; nothing is reported."""


def _expected(rd: Path, protocol: dict) -> dict[str, tuple[list[str], list[int]]]:
    fs, s1 = protocol['final_seeds'], protocol['primary_direction']
    out = {}
    for c, v in protocol['cells'].items():
        if not v['gated'] or (rd / f'FE_cell_{c}').exists():
            out[f'FE_cell_{c}'] = ([s1], fs['cell'])
    out['FE_A0'] = ([s1, protocol['supporting_direction']], fs['A0'])
    out['FE_A1'] = ([s1], [fs['A0'][0]])
    out['FE_A2'] = ([s1], fs['A2'])
    out['FE_A3'] = ([s1], fs['A3'])
    return out


def _load_runs(rd, protocol):
    """{(arm, person, direction, seed): (json, npz)}; refuses on a missing run."""
    runs, missing = {}, []
    for e, (dirs, seeds) in _expected(rd, protocol).items():
        for p in protocol['people']:
            for d in dirs:
                for s in seeds:
                    stem = rd / e / p / d / f'seed{s}'
                    if not stem.with_suffix('.json').exists() or not stem.with_suffix('.npz').exists():
                        missing.append(f'{e} {p} {d} seed {s}')
                        continue
                    runs[e, p, d, s] = (json.loads(stem.with_suffix('.json').read_text()),
                                        LK.load_test(stem.with_suffix('.npz'), rd))
    if missing:
        raise Refused(f'{len(missing)} final runs missing: ' + '; '.join(missing[:20]))
    return runs


def provenance_check(runs, protocol) -> dict:
    spec_hash, data, commits = {}, {}, set()
    for (e, p, d, s), (js, _) in runs.items():
        pv, name = js.get('provenance') or {}, f'{e} {p} {d} seed {s}'
        if pv.get('protocol_hash') != protocol['hash']:
            raise Refused(f'{name}: made with another protocol ({pv.get("protocol_hash")})')
        if pv.get('git_dirty'):
            raise Refused(f'{name}: made on uncommitted code')
        if e not in spec_hash:
            spec_hash[e] = PV.config_hash(load_config(e)[1])
        if js.get('config_hash') != spec_hash[e]:
            raise Refused(f'{name}: config hash {js.get("config_hash")} differs from spec {e} ({spec_hash[e]})')
        for f, h in (pv.get('data_sha256') or {}).items():
            data.setdefault(f, {}).setdefault(h, []).append(name)
        commits.add(pv.get('git_commit'))
    for f, by_hash in data.items():
        if len(by_hash) > 1:
            raise Refused(f'recording {f} was read with different contents: ' +
                          '; '.join(f'{h[:12]}: {", ".join(n[:3])}' for h, n in by_hash.items()))
    return {'n_runs': len(runs), 'commits': sorted(c for c in commits if c), 'protocol_hash': protocol['hash']}


def _scores(runs, protocol):
    """{(arm, person, direction): {seed: nmae}} and per-position nMAE {(arm, person, direction): {angle: median}}."""
    sf = protocol['score_from_s']
    t0, scale = {}, {}
    for (e, p, d, s), (_, z) in runs.items():
        for tr in np.unique(z['trial']):
            k = (p, d, int(tr))
            t0[k] = min(t0.get(k, np.inf), float(z['t'][z['trial'] == tr].min()))
        if e == 'FE_A0' and (p, d) not in scale:
            scale[p, d] = person_scale(z['y_own_target'])
    nm, pos = {}, {}
    for (e, p, d, s), (_, z) in runs.items():
        y, yp, t, tr = (np.asarray(z[k], float) for k in ('y_own_target', 'y_pred', 't', 'trial'))
        per = {}
        for i in np.unique(tr):
            m = (tr == i) & (t - t0[p, d, int(i)] >= sf)
            ang = float(np.asarray(z['angle'])[tr == i][0]) if 'angle' in z else float(i)
            per[round(ang, 3)] = float(np.mean(np.abs(yp[m] - y[m])) / scale[p, d])
        nm.setdefault((e, p, d), {})[s] = float(np.mean(list(per.values())))
        for a, v in per.items():
            pos.setdefault((e, p, d), {}).setdefault(a, []).append(v)
    pos = {k: {a: float(np.median(v)) for a, v in d.items()} for k, d in pos.items()}
    return nm, pos


def _compare(nm, pos, arm, ref, people, d, protocol):
    a = {p: nm[arm, p, d] for p in people}
    r = {p: nm[ref, p, d] for p in people}
    eff = ST.person_effects(a, r)
    m, lo, hi = ST.t_interval(eff)
    return {'arm': arm, 'comparator': ref, 'per_person': a, 'effects': eff, 'effect': float(np.median(list(eff.values()))),
            'ci': [m, lo, hi], 'decision': ST.decide(eff, protocol['min_agree'], protocol['min_effect']),
            'per_position': {p: pos[arm, p, d] for p in people}}


def run(results_dir=None, protocol=None) -> dict:
    rd, protocol = Path(results_dir or RESULTS_V2), protocol or PT.load()
    if not LK.test_open(rd):
        raise LK.TestLocked(f'no {rd / "choice.json"}: A0 is not chosen yet')
    runs = _load_runs(rd, protocol)
    prov = provenance_check(runs, protocol)
    nm, pos = _scores(runs, protocol)
    people, d1, fam = protocol['people'], protocol['primary_direction'], protocol['families']
    have = {e for e, _, _ in nm}
    out = {'families': {'F1': {}, 'F2': {}, 'F3': {}}}

    f1 = fam['F1']
    for c in f1['arms']:
        if f'FE_cell_{c}' in have:
            out['families']['F1'][c] = _compare(nm, pos, f'FE_cell_{c}', f'FE_cell_{f1["comparator"]}', people, d1, protocol)
        else:
            g = rd / f'gate_{c}.json'
            out['families']['F1'][c] = {'decision': 'stopped',
                                        'gate': json.loads(g.read_text()) if g.exists() else 'a shorter length stopped'}

    lengths = sorted({c.split('_')[0] for c, v in protocol['cells'].items() if 'state' not in v
                      and all(f'FE_cell_{c.split("_")[0]}_{o}' in have for o in ('full', 'o75', 'o50', 'o0'))},
                     key=lambda w: int(w[1:]))
    for o in fam['F2']['arms']:
        per_len = {w: ST.person_effects({p: nm[f'FE_cell_{w}_{o}', p, d1] for p in people},
                                        {p: nm[f'FE_cell_{w}_full', p, d1] for p in people}) for w in lengths}
        eff = {p: float(np.mean([per_len[w][p] for w in lengths])) for p in people}
        m, lo, hi = ST.t_interval(eff)
        out['families']['F2'][o] = {'lengths': lengths, 'per_length': per_len, 'effects': eff,
                                    'effect': float(np.median(list(eff.values()))), 'ci': [m, lo, hi],
                                    'decision': ST.decide(eff, protocol['min_agree'], protocol['min_effect'])}

    for a, b in fam['F3']['pairs']:
        out['families']['F3'][f'{a} vs {b}'] = _compare(nm, pos, ARM[a], ARM[b], people, d1, protocol)

    out['descriptive'] = {f'{e} {d}': {p: nm[e, p, d] for p in people} for e, d in sorted({(e, d) for e, _, d in nm})}
    out['val_test_agreement'] = _agreement(rd, nm, people, d1, [e for e in have if e.startswith('FE_cell_')])
    out['provenance_check'] = prov
    out['deviations'] = protocol.get('deviations', [])
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
