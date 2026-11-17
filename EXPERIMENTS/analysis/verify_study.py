"""Check every window-study run and what the dashboard and the guide say about it.

    python -m EXPERIMENTS.analysis.verify_study [--results EXPERIMENTS/results]

Checks, for every current-round setting (study_scores.question(id) is not None), session 1 -> 2:
  1. completeness: the same seeds for every person, and a .json next to every .npz;
  2. the run used its config: the saved config equals configs/<id>.json applied to BASE, key by key;
  3. classification: the window length, step, overlap, target, state, shuffling and epoch rule implied by the id
     (W00 reference, WO overlap, WL length, WG grid, WC carried state, ...) and by the title match the saved config;
  4. scores: MAE and RMSE recomputed here from the saved predictions (plain numpy, per position) equal study.json;
  5. targets: every active-torque setting predicts the same measured torque as the reference at every position and
     time stamp they share (total torque: the same as WT1_total), and the positions are the same angles;
  6. peaks: the peak torque per position in study.json is the largest |measured torque| of any run of that target;
  7. IES (no relaxed-hold trials): its active torque is its total torque minus the experiment-sheet passive table
     (median over each position within 0.1 Nm of the table at the recorded angle).
Prints one line per problem and a summary; exit status 1 if anything failed.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from EXPERIMENTS.analysis import study_scores as SS
from EXPERIMENTS.pipeline.run import BASE

ROOT = Path(__file__).resolve().parents[2]
LENGTH = {'0500ms': 0.5, '1s': 1.0, '2s': 2.0, '5s': 5.0, '10s': 10.0}


def expected(exp):
    """Facets implied by a setting id (None = not a window-study id). Overlap allows ±2 % (a stride must be a whole
    number of 10 ms samples, so 75 % of 500 ms becomes 76 %)."""
    tot = exp.endswith('_tot') or exp.startswith('WT1_total')
    base = exp[:-4] if exp.endswith('_tot') else exp
    e = {'target': 'total' if tot else 'active', 'state': 'reset', 'shuffle': False, 'epochs': 150}
    if base in ('W00_reference', 'WT1_total', 'WT1_total_400'):
        e.update(length_s=0.2, step_ms=10, overlap_pct=95)
        if base == 'WT1_total_400':
            e['epochs'] = 400
    elif m := re.fullmatch(r'WO\d_(\d\d)(_eq)?', base):
        e.update(length_s=0.2, step_ms=10, overlap_pct=int(m.group(1)))
    elif m := re.fullmatch(r'WL\d_(0500ms|1s|2s|5s|10s)', base):
        L = LENGTH[m.group(1)]
        e.update(length_s=L, step_ms=50, overlap_pct=round(100 * (1 - 0.01 / L)))
    elif base == 'WL2c_1s_10ms':
        e.update(length_s=1.0, step_ms=10, overlap_pct=99)
    elif m := re.fullmatch(r'WG_(0500ms|1s|2s|5s|10s)_o(\d\d)', base):
        e.update(length_s=LENGTH[m.group(1)], step_ms=50, overlap_pct=int(m.group(2)))
    elif base == 'WS1_shuffled':
        e.update(length_s=0.2, step_ms=10, overlap_pct=95, shuffle=True)
    elif m := re.fullmatch(r'WC\d_carry_(500ms|1s|2s|5s|10s)', base):
        e.update(state='carry', chunk_s={'500ms': 0.5, '1s': 1.0, '2s': 2.0, '5s': 5.0, '10s': 10.0}[m.group(1)])
        if m.group(1) == '10s':                       # 8 updates per epoch: run with the 400-epoch limit on purpose
            e['epochs'] = 400
    else:
        return None
    return e


def title_claims(title):
    """Facets a title states outright (window length, overlap %, 'total', 'shuffled', '10 ms steps')."""
    c = {}
    if m := re.search(r'(\d+(?:\.\d+)?)\s*(ms|s)\b(?!\s*chunks)', title):
        v = float(m.group(1))
        c['length_s'] = v / 1000 if m.group(2) == 'ms' else v
    if m := re.search(r'(\d+)\s*%', title):
        c['overlap_pct'] = int(m.group(1))
    if 'total' in title.lower():
        c['target'] = 'total'
    if 'shuffled' in title.lower():
        c['shuffle'] = True
    if '10 ms steps' in title:
        c['step_ms'] = 10
    return c


def check_facets(exp, f, title):
    """Problems with one setting's facets (f, from its saved config) against its id and its title."""
    out = []
    e = expected(exp)
    if e is None:
        return [f'{exp}: id not recognised by the verifier']
    for k, v in e.items():
        tol = 2 if k == 'overlap_pct' else 1e-9
        if k not in f or (abs(f[k] - v) > tol if isinstance(v, (int, float)) and not isinstance(v, bool) else f[k] != v):
            out.append(f'{exp}: {k} is {f.get(k)!r}, the id says {v!r}')
    for k, v in title_claims(title).items():
        if k == 'length_s' and f.get('state') == 'carry':
            continue
        tol = 2 if k == 'overlap_pct' else 1e-9
        if isinstance(v, bool) or isinstance(v, str):
            bad = f.get(k) != v
        else:
            bad = abs(f.get(k, np.nan) - v) > tol
        if bad and not (k == 'overlap_pct' and exp.startswith('WL')):
            out.append(f'{exp}: title "{title}" says {k} = {v!r}, the run has {f.get(k)!r}')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default=str(ROOT / 'EXPERIMENTS/results'))
    ap.add_argument('--study', default=str(ROOT / 'EXPERIMENTS/dashboard_react/src/data/study.json'))
    a = ap.parse_args(argv)
    results, study = Path(a.results), json.loads(Path(a.study).read_text())
    titles = SS.spec_titles()
    problems, n_runs, n_pos = [], 0, 0
    targets = {}                                      # (target kind, person, angle) -> (t, y) from the first setting
    peaks = {}                                        # (target kind, person, position index) -> max |torque|

    exps = sorted(d.name for d in results.iterdir() if d.is_dir() and SS.question(d.name) is not None)
    missing_in_study = [e for e in exps if e not in study['settings']]
    problems += [f'{e}: has runs but is not in study.json' for e in missing_in_study]
    for exp in exps:
        spec = json.loads((ROOT / 'EXPERIMENTS/configs' / f'{exp}.json').read_text())
        want = dict(BASE, **spec['changes'])
        seeds_by_person = {}
        for npz in sorted((results / exp).glob(f'*/{SS.DIRECTION}/seed*.npz')):
            subj, seed = npz.parts[-3], npz.stem[4:]
            js = npz.with_suffix('.json')
            if not js.exists():
                problems.append(f'{exp} {subj} seed {seed}: .npz without .json')
                continue
            seeds_by_person.setdefault(subj, set()).add(seed)
            info = json.loads(js.read_text())
            saved = info['config']
            diff = [k for k in want if k in saved and saved[k] != want[k]]
            if diff:
                problems.append(f'{exp} {subj} seed {seed}: saved config differs from the spec in {diff}')
            z = np.load(npz)
            y, p, ang, t = (z[k].astype(float) for k in ('y_own_target', 'y_pred', 'angle', 't'))
            rows = study['runs'].get(exp, {}).get(subj, {}).get(seed)
            angles = sorted(np.unique(ang))
            if rows is None:
                problems.append(f'{exp} {subj} seed {seed}: run missing from study.json')
                continue
            if [round(float(x), 3) for x in angles] != study['positions'][subj]:
                problems.append(f'{exp} {subj} seed {seed}: positions differ from study.json')
            for k, x in enumerate(angles):
                m = ang == x
                mae, rmse = np.mean(np.abs(p[m] - y[m])), np.sqrt(np.mean((p[m] - y[m]) ** 2))
                if abs(mae - rows[k][0]) > 2e-5 or abs(rmse - rows[k][1]) > 2e-5:
                    problems.append(f'{exp} {subj} seed {seed} p{k + 1}: MAE/RMSE {mae:.5f}/{rmse:.5f}, study.json {rows[k][0]}/{rows[k][1]}')
                key = (SS.facets(dict(BASE, **saved))['target'], subj, round(float(x), 3))
                pk = (key[0], subj, k)
                peaks[pk] = max(peaks.get(pk, 0.0), float(np.max(np.abs(y[m]))))
                if key not in targets:
                    targets[key] = (exp, t[m], y[m])
                else:
                    e0, t0, y0 = targets[key]
                    common, i0, i1 = np.intersect1d(np.round(t0, 2), np.round(t[m], 2), return_indices=True)
                    d = np.max(np.abs(y0[i0] - y[m][i1])) if len(common) else np.inf
                    if d > 1e-3:
                        problems.append(f'{exp} {subj} p{k + 1}: measured torque differs from {e0} by up to {d:.3f} Nm')
                n_pos += 1
            n_runs += 1
        sets = {frozenset(s) for s in seeds_by_person.values()}
        if len(sets) > 1 or len(seeds_by_person) != len(SS.PEOPLE):
            problems.append(f'{exp}: seeds differ between people or a person is missing: '
                            + ', '.join(f'{p} {sorted(map(int, s))}' for p, s in sorted(seeds_by_person.items())))
        f = study['settings'].get(exp)
        if f:
            problems += check_facets(exp, f, f['title'])
            if f['title'] != titles.get(exp, exp):
                problems.append(f'{exp}: study.json title "{f["title"]}" is not the spec label "{titles.get(exp)}"')

    # active and total torque must differ (otherwise the passive subtraction did nothing)
    for (kind, subj, x), (e0, t0, y0) in targets.items():
        if kind == 'active' and ('total', subj, x) in targets:
            _, t1, y1 = targets[('total', subj, x)]
            common, i0, i1 = np.intersect1d(np.round(t0, 2), np.round(t1, 2), return_indices=True)
            if len(common) and np.max(np.abs(y0[i0] - y1[i1])) < 1e-3 and subj != 'IES':
                problems.append(f'{subj} {x}: active and total torque are identical')

    for (kind, subj, k), v in sorted(peaks.items()):
        got = study.get('peaks', {}).get(kind, {}).get(subj, [None] * (k + 1))[k]
        if got is None or abs(got - v) > 0.006:
            problems.append(f'{subj} p{k + 1}: {kind} peak in study.json is {got}, the runs give {v:.2f} Nm')

    # IES has no relaxed-hold trials: its passive torque is the experiment-sheet table, interpolated at the measured
    # angle sample by sample (so total - active follows the angle a little); compare the median at each position
    from ML.config import METADATA_PASSIVE
    from ML.research.passive import map_lookup
    for subj, table in METADATA_PASSIVE.items():
        angles = sorted(x for kind, s_, x in targets if kind == 'active' and s_ == subj)
        compared = 0
        for x in angles:
            a, tt = targets.get(('active', subj, x)), targets.get(('total', subj, x))
            if not (a and tt):
                continue
            common, i0, i1 = np.intersect1d(np.round(a[1], 2), np.round(tt[1], 2), return_indices=True)
            d = np.median(tt[2][i1] - a[2][i0]) if len(common) else np.inf
            want = float(map_lookup(sorted(table), [x])[0])
            compared += 1
            if abs(d - want) > 0.1:
                problems.append(f'{subj} {x}: total - active is {d:.2f} Nm, the passive table gives {want:.2f} Nm')
        if compared != len(table):
            problems.append(f'{subj}: {compared} positions compared against its {len(table)}-row passive table')

    for line in problems:
        print('FAIL', line)
    print(f'{len(exps)} settings, {n_runs} runs, {n_pos} run-positions checked; {len(problems)} problems')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
