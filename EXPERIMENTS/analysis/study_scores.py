"""Scores of every window-study run, per position, for the dashboard's study page (no waveforms).

    python -m EXPERIMENTS.analysis.study_scores [--out EXPERIMENTS/dashboard_react/src/data/study.json]

For each current-round window-study setting (EXPERIMENTS/configs/W*.json, not the earlier rounds) and each saved
session 1 -> 2 run, every position (one trial each) is scored with pipeline.scoring.score_trial, without
calibration: MAE, RMSE, R² and demeaned R² are computed within that position, so R² measures how well the torque is
tracked at a fixed angle, not how well the torque level of each angle is guessed. The setting's facets (window
length, step, overlap, target, state, epoch limit) are read from the runs' own saved config.
"""
import argparse
import json
import re
from datetime import date
from pathlib import Path

import numpy as np

from EXPERIMENTS.pipeline.run import BASE
from EXPERIMENTS.pipeline.scoring import score_trial

ROOT = Path(__file__).resolve().parents[2]
METRICS = ('mae', 'rmse', 'r2', 'r2_demeaned', 'bias')
PEOPLE = ('YES', 'JM', 'EG', 'IES', 'CH', 'HM')
DIRECTION = 'S1toS2'


def facets(cfg):
    """Window geometry and training choices of one run config (BASE already filled in)."""
    span_ms = cfg['window'] * cfg['input_step'] * 10
    return {'length_s': span_ms / 1000, 'step_ms': cfg['input_step'] * 10,
            'overlap_pct': int(round(100 * (1 - cfg['train_stride'] * 10 / span_ms))),
            'target': 'total' if cfg['passive'] == 'none' else 'active', 'state': cfg['state'],
            'chunk_s': cfg['chunk'] * 10 / 1000 if cfg.get('chunk') else None, 'shuffle': bool(cfg['shuffle']),
            'epochs': 400 if cfg['train_protocol'] == 'fast400' else 150}


def question(exp):
    """(study question, paired setting) of a setting id; None for settings outside the current round.
    The pair of a total-torque setting is its active-torque version; the 400-epoch check pairs with WT1_total."""
    if exp == 'W00_reference':
        return 'reference', None
    if exp == 'WT1_total':
        return 'reference', 'W00_reference'
    if exp == 'WT1_total_400':
        return 'epochs', 'WT1_total'
    base = exp[:-4] if exp.endswith('_tot') else exp
    pair = base if base != exp else None
    rules = [(r'WO\d_\d\d', 'overlap'), (r'WO\d_\d\d_eq', 'overlap_eq'), (r'WL2c_.*', 'step'), (r'WL\d_[0-9a-z]+', 'length'),
             (r'WG_[0-9a-z]+_o\d\d', 'grid'), (r'WS1_shuffled', 'shuffle'), (r'WC[234]_carry_[0-9a-z]+', 'carry')]
    for pattern, q in rules:
        if re.fullmatch(pattern, base):
            return q, pair
    return None


def position_scores(y, p, angle, t):
    """One row of METRICS per position, positions in increasing angle (most toes-down first)."""
    rows = []
    for a in sorted(np.unique(angle)):
        m = angle == a
        r, _ = score_trial(y[m], p[m], t[m], {'type': 'none'})
        rows.append([round(float(r[k]), 5) for k in METRICS])
    return rows


def build(results, titles):
    results = Path(results)
    out = {'built': date.today().isoformat(), 'direction': DIRECTION, 'metrics': list(METRICS), 'people': [],
           'positions': {}, 'settings': {}, 'runs': {}, 'epochs': {}, 'windows': {}}
    for d in sorted(p for p in results.iterdir() if p.is_dir()):
        exp = d.name
        q = question(exp)
        if q is None:
            continue
        runs, epochs, windows, first = {}, {}, {}, None
        for f in sorted(d.glob(f'*/{DIRECTION}/seed*.npz')):
            subj, seed = f.parts[-3], f.stem[4:]
            if not f.with_suffix('.json').exists():          # a run still being written
                continue
            info = json.loads(f.with_suffix('.json').read_text(encoding='utf-8'))
            cfg = dict(BASE, **(info.get('config') or {}))
            first = first or cfg
            z = np.load(f, allow_pickle=False)
            y, pr = z['y_own_target'].astype(float), z['y_pred'].astype(float)
            angle, t = z['angle'].astype(float), z['t'].astype(float)
            runs.setdefault(subj, {})[seed] = position_scores(y, pr, angle, t)
            epochs.setdefault(subj, {})[seed] = info.get('train_info', {}).get('epochs')
            nw = info.get('n_windows')
            if subj not in windows and nw is not None:
                windows[subj] = nw.get('train') if isinstance(nw, dict) else nw
            if subj not in out['positions']:
                out['positions'][subj] = [round(float(a), 3) for a in sorted(np.unique(angle))]
        if not runs:
            continue
        out['settings'][exp] = dict(facets(first), id=exp, question=q[0], pair=q[1], title=titles.get(exp, exp))
        out['runs'][exp], out['epochs'][exp], out['windows'][exp] = runs, epochs, windows
    out['people'] = [p for p in PEOPLE if p in out['positions']] + sorted(set(out['positions']) - set(PEOPLE))
    return out


def spec_titles(configs=ROOT / 'EXPERIMENTS/configs'):
    titles = {}
    for p in Path(configs).glob('W*.json'):
        raw = json.loads(p.read_text(encoding='utf-8'))
        titles[raw['id']] = (raw.get('dashboard') or {}).get('label') or raw.get('title', raw['id'])
    return titles


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default=str(ROOT / 'EXPERIMENTS/results'))
    ap.add_argument('--out', default=str(ROOT / 'EXPERIMENTS/dashboard_react/src/data/study.json'))
    a = ap.parse_args()
    out = build(a.results, spec_titles())
    Path(a.out).write_text(json.dumps(out, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    n = sum(len(s) for r in out['runs'].values() for s in r.values())
    print(f"{len(out['settings'])} settings, {n} runs -> {a.out}")


if __name__ == '__main__':
    main()
