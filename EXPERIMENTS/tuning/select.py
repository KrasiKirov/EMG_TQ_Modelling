"""Selection on session-1 validation nMAE (decision D4): the paired 1-SE rule picks each cell's winner, the best
score picks A0 among the cell winners; writes results_v2/choice.json.

Reads only ``seedN.json`` and ``seedN_val.npz``; session-2 predictions (``seedN.npz``) are never opened here.

Tables: {config: {person: nMAE}}, the median over seeds. nMAE = MAE / scale, where the scale of a person is the
99th percentile of |validation target| on the D24 common rows (the same rows for every window length, so the
scale does not depend on the config). Two tables are built: 'full' (every validation target of the run) for the
cuts inside a cell, and 'common' (D24 targets only) for comparisons across cells (gates, A0).

    python -m EXPERIMENTS.tuning.select --round R1 --cell w20_full          # best coarse config (fine centre)
    python -m EXPERIMENTS.tuning.select --round R2 --cell w20_full          # the 2 candidates of R1 + R2
    python -m EXPERIMENTS.tuning.select --round R3 --cell w20_full          # the cell's winner
    python -m EXPERIMENTS.tuning.select --gate w200_full                    # D23 gate of a long window
    python -m EXPERIMENTS.tuning.select --a0                                # A0 over every finished cell
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from EXPERIMENTS.analysis.stats import person_scale
from EXPERIMENTS.paths import RESULTS_V2
from EXPERIMENTS.pipeline.settings import DIRECTION
from EXPERIMENTS.protocol import protocol as PT
from EXPERIMENTS.tuning import halving as HV


def _run_dir(results_dir, exp, person, direction=DIRECTION):
    return Path(results_dir or RESULTS_V2) / exp / person / direction


def load_val(ids, seeds, people, results_dir=None, direction=DIRECTION):
    """({'full': table, 'common': table}, coverage {config: {person: n common rows}}). Missing runs raise."""
    full, common, cov = {}, {}, {}
    for e in ids:
        full[e], common[e], cov[e] = {}, {}, {}
        for p in people:
            f, c = [], []
            for s in seeds:
                path = _run_dir(results_dir, e, p, direction) / f'seed{s}_val.npz'
                if not path.exists():
                    raise FileNotFoundError(f'missing validation predictions {path}')
                v = np.load(path)
                y, yp, m = v['y'], v['y_pred'], v['common'].astype(bool)
                scale = person_scale(y[m])
                f.append(np.mean(np.abs(yp - y)) / scale)
                c.append(np.mean(np.abs(yp[m] - y[m])) / scale)
                cov[e][p] = int(m.sum())
            full[e][p], common[e][p] = float(np.median(f)), float(np.median(c))
    return {'full': full, 'common': common}, cov


def n_params(ids, people, results_dir=None, direction=DIRECTION, seed=0):
    """config → number of trainable parameters (read from the first person's seedN.json)."""
    out = {}
    for e in ids:
        info = json.loads((_run_dir(results_dir, e, people[0], direction) / f'seed{seed}.json').read_text())
        out[e] = int(info['train_info']['n_params'])
    return out


def cut(table, keep: int) -> list[str]:
    return HV.keep(table, keep)


def within_one_se(table) -> list[str]:
    """Configs whose mean per-person log difference to the best config is at most its standard error (paired)."""
    s = HV.score(table)
    b = min(s, key=lambda c: (s[c], c))
    out = []
    for c, v in table.items():
        d = np.array([np.log(v[p]) - np.log(table[b][p]) for p in table[b]])
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        if d.mean() <= se:
            out.append(c)
    return out


def winner(table, params: dict[str, int]) -> str:
    """Paired 1-SE rule: among configs within 1 SE of the best, the fewest parameters; ties go to the lower score."""
    s = HV.score(table)
    return min(within_one_se(table), key=lambda c: (params[c], s[c], c))


def choose_a0(tables, out=None, cell_winners=None, ridge_alpha=None) -> str:
    """A0 = the cell winner with the best score on D24 common targets (no 1-SE rule: parameter count does not
    depend on window length, so it says nothing about the setting being chosen). Writes choice.json, which opens
    session 2."""
    a0 = HV.keep(tables['common'], 1)[0]
    out = Path(out or Path(RESULTS_V2) / 'choice.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'A0': a0, 'cell_winners': cell_winners or {}, 'ridge_alpha': ridge_alpha or {},
                               'protocol_hash': PT.load()['hash'],
                               'common_nmae': tables['common']}, indent=1) + '\n')
    return a0


def round_ids(rnd, cell, configs_dir=None):
    from EXPERIMENTS.pipeline import spec as SPEC
    return sorted(p.stem for p in Path(configs_dir or SPEC.CONFIGS).glob(f'TS2_{rnd}_{cell}_*.json'))


def main(argv=None):
    P = PT.load()
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', choices=list(P['rounds']))
    ap.add_argument('--cell')
    ap.add_argument('--gate')
    ap.add_argument('--a0', action='store_true')
    ap.add_argument('--json', help='also write the answer to this file')
    a = ap.parse_args(argv)
    people = P['people']
    if a.round:
        r = P['rounds'][a.round]
        if a.round == 'R3':
            ids = r3_ids(a.cell, people)
            tables, _ = load_val(ids, r['seeds'], people)
            res = {'winner': winner(tables['full'], n_params(ids, people))}
        elif a.round == 'R2':
            res = {'keep': r3_ids(a.cell, people)}
        else:
            res = {'best': best_r1(a.cell, people)}
    elif a.gate:
        order = P['gates']['order']
        shorter = [c for c, v in P['cells'].items() if v.get('overlap') == 'full'
                   and v['window'] < P['cells'][a.gate]['window'] and (not v['gated'] or c in order)]
        r1 = P['rounds']['R1']['seeds']
        screen = load_val(round_ids('R1', a.gate), r1, people)[0]['common']
        missing = [c for c in shorter if not round_ids('R1', c)]
        if missing:
            raise FileNotFoundError(f'the gate needs R1 of every shorter length first; missing {missing}')
        short = {c: load_val(round_ids('R1', c), r1, people)[0]['common'] for c in shorter}
        cont, ref, ratios = HV.gate(screen, short)
        res = {'continues': cont, 'reference': ref, 'ratios': ratios}
    elif a.a0:
        winners = {c: _r3_winner(c, people) for c in P['cells'] if round_ids('R2', c)}
        ids = list(winners.values())
        tables, _ = load_val(ids, P['rounds']['R3']['seeds'], people)
        res = {'A0': choose_a0(tables, cell_winners=winners)}
    else:
        ap.error('give --round with --cell, --gate or --a0')
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1) + '\n')


def best_r1(cell, people):
    """The best coarse (R1) config of a cell: the centre of its fine draws."""
    r = PT.load()['rounds']
    return cut(load_val(round_ids('R1', cell), r['R1']['seeds'], people)[0]['full'], 1)[0]


def r3_ids(cell, people):
    """R3 candidates: the best 2 of R1 + R2 on seed 0 (same specs, seed 1 added)."""
    r = PT.load()['rounds']
    ids = round_ids('R1', cell) + round_ids('R2', cell)
    return cut(load_val(ids, r['R1']['seeds'], people)[0]['full'], r['R3']['keep'])


def _r3_winner(cell, people):
    ids = r3_ids(cell, people)
    tables, _ = load_val(ids, PT.load()['rounds']['R3']['seeds'], people)
    return winner(tables['full'], n_params(ids, people))


if __name__ == '__main__':
    main()
