"""Scores of every window-study run, per position, for the dashboard's study page, plus the waveforms of the
reference, the best and the worst setting (study-waves.json).

    python -m EXPERIMENTS.analysis.study_scores [--out EXPERIMENTS/dashboard_react/src/data/study.json]

For each current-round window-study setting (EXPERIMENTS/configs/W*.json, not the earlier rounds) and each saved
session 1 -> 2 run, every position (one trial each) is scored with pipeline.scoring.score_trial, without
calibration: MAE, RMSE, R² and demeaned R² are computed within that position, so R² measures how well the torque is
tracked at a fixed angle, not how well the torque level of each angle is guessed. The setting's facets (window
length, step, overlap, target, state, epoch limit) are read from the runs' own saved config. Also exported: the peak
measured torque (max |torque|) at every position, for active and total torque; the software versions the runs
reported; and the computer they ran on (read when exporting, since the run files record software versions only).
"""
import argparse
import importlib.metadata
import json
import platform
import re
import subprocess
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
    """One row of METRICS per position, positions in increasing angle (most plantarflexed first)."""
    rows = []
    for a in sorted(np.unique(angle)):
        m = angle == a
        r, _ = score_trial(y[m], p[m], t[m])
        rows.append([round(float(r[k]), 5) for k in METRICS])
    return rows


def build(results, titles):
    results = Path(results)
    out = {'built': date.today().isoformat(), 'direction': DIRECTION, 'metrics': list(METRICS), 'people': [],
           'positions': {}, 'settings': {}, 'runs': {}, 'epochs': {}, 'windows': {}, 'times': {},
           'peaks': {'active': {}, 'total': {}}, 'versions': []}
    for d in sorted(p for p in results.iterdir() if p.is_dir()):
        exp = d.name
        q = question(exp)
        if q is None:
            continue
        runs, epochs, windows, times, first = {}, {}, {}, {}, None
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
            peaks = out['peaks'][facets(cfg)['target']].setdefault(subj, {})
            for k, a in enumerate(sorted(np.unique(angle))):
                peaks[k] = max(peaks.get(k, 0.0), round(float(np.max(np.abs(y[angle == a]))), 2))
            if info.get('versions') and info['versions'] not in out['versions']:
                out['versions'].append(info['versions'])
            epochs.setdefault(subj, {})[seed] = info.get('train_info', {}).get('epochs')
            if info.get('train_s') is not None:
                times.setdefault(subj, {})[seed] = round(float(info['train_s']), 1)
            nw = info.get('n_windows')
            if subj not in windows and nw is not None:
                windows[subj] = nw.get('train') if isinstance(nw, dict) else nw
            if subj not in out['positions']:
                out['positions'][subj] = [round(float(a), 3) for a in sorted(np.unique(angle))]
        if not runs:
            continue
        out['settings'][exp] = dict(facets(first), id=exp, question=q[0], pair=q[1], title=titles.get(exp, exp))
        out['runs'][exp], out['epochs'][exp], out['windows'][exp], out['times'][exp] = runs, epochs, windows, times
    out['people'] = [p for p in PEOPLE if p in out['positions']] + sorted(set(out['positions']) - set(PEOPLE))
    out['peaks'] = {kind: {subj: [v[k] for k in sorted(v)] for subj, v in per.items()} for kind, per in out['peaks'].items()}
    return out


def run_versions(results):
    """Software every results folder's runs reported ({exp: [distinct versions]}, [] if none were saved), for all
    folders, so the experiment pages can tell which runs share the window study's environment."""
    out = {}
    for d in sorted(p for p in Path(results).iterdir() if p.is_dir()):
        seen = []
        for f in sorted(d.glob('*/*/seed*.json')):
            v = json.loads(f.read_text(encoding='utf-8')).get('versions')
            if v and v not in seen:
                seen.append(v)
        out[d.name] = seen
    return out


def machine(raw, workers, threads):
    """The computer the runs trained on, from raw system values (see read_machine), and how the runs shared it."""
    return {'computer': raw.get('machine_name'), 'processor': raw.get('chip_type'),
            'cores': int(raw['hw.physicalcpu']), 'performance_cores': int(raw['hw.perflevel0.physicalcpu']),
            'efficiency_cores': int(raw['hw.perflevel1.physicalcpu']),
            'ram_gb': int(round(int(raw['hw.memsize']) / 2 ** 30)), 'os': f"macOS {raw['os']}",
            'accelerator': ', '.join(raw['gpus']) if raw.get('gpus') else 'none (CPU only)',
            'parallel_runs': workers, 'threads_per_run': threads}


def read_machine():
    """Raw values for machine() on this Mac: model name and chip (system_profiler, no serial numbers), core counts
    and memory (sysctl), macOS version, and any GPU plug-in TensorFlow could use (tensorflow-metal)."""
    hw = json.loads(subprocess.run(['system_profiler', 'SPHardwareDataType', '-json'], capture_output=True,
                                   text=True, check=True).stdout)['SPHardwareDataType'][0]
    raw = {k: hw.get(k) for k in ('machine_name', 'chip_type')}
    for key in ('hw.physicalcpu', 'hw.perflevel0.physicalcpu', 'hw.perflevel1.physicalcpu', 'hw.memsize'):
        raw[key] = subprocess.run(['sysctl', '-n', key], capture_output=True, text=True, check=True).stdout.strip()
    raw['os'] = platform.mac_ver()[0]
    raw['gpus'] = []
    for pkg in ('tensorflow-metal',):
        try:
            raw['gpus'].append(f'{pkg} {importlib.metadata.version(pkg)}')
        except importlib.metadata.PackageNotFoundError:
            pass
    return raw


def _default(path, pattern, fallback):
    """A default read from a script (e.g. WORKERS=${WORKERS:-6}), so the export states what the runs used."""
    m = re.search(pattern, (ROOT / path).read_text(encoding='utf-8')) if (ROOT / path).exists() else None
    return int(m.group(1)) if m else fallback


REFERENCE = 'W00_reference'


def mae_ratios(out, exp, ref=REFERENCE):
    """Per person: median over matched seeds of the run MAE (mean over positions) of exp, divided by that of ref,
    the study page's rule. Returns {person: (ratio, matched seeds)}."""
    col = METRICS.index('mae')
    run_mae = lambda rows: float(np.mean([r[col] for r in rows]))
    res = {}
    for subj, seeds in out['runs'][exp].items():
        base = out['runs'][ref].get(subj, {})
        k = [s for s in seeds if s in base]
        if k:
            res[subj] = (np.median([run_mae(seeds[s]) for s in k]) / np.median([run_mae(base[s]) for s in k]), len(k))
    return res


def extremes(out, min_seeds=5, band=0.05):
    """Best and worst active-torque setting by the median (over people) MAE ratio against the reference, among the
    settings with at least min_seeds matched seeds for every person; and the settings the study's rule calls
    clearly better (ratio below 1 - band and all people but at most one better)."""
    scored = {}
    for exp, s in out['settings'].items():
        if exp == REFERENCE or s['target'] != 'active':
            continue
        r = mae_ratios(out, exp)
        if r and min(n for _, n in r.values()) >= min_seeds:
            scored[exp] = (float(np.median([v for v, _ in r.values()])), sum(v < 1 for v, _ in r.values()), len(r))
    if not scored:
        return {'best': None, 'worst': None, 'clearly_better': []}
    return {'best': min(scored, key=lambda e: scored[e][0]), 'worst': max(scored, key=lambda e: scored[e][0]),
            'clearly_better': sorted(e for e, (m, b, n) in scored.items() if m < 1 - band and b >= max(1, n - 1))}


def _blocks(t, y, hz):
    """Block means at hz (from 100 Hz samples): (time of the first block, values rounded to 0.01 Nm)."""
    k = max(1, int(round(100 / hz)))
    n = len(y) // k
    return round(float(t[:n * k].reshape(n, k).mean(1)[0]), 3), np.round(y[:n * k].reshape(n, k).mean(1), 2).tolist()


def waves(results, out, ext, hz=20):
    """Measured torque and the predictions of the reference, the best and the worst setting (and any clearly better
    one) at every position, block-averaged to hz. Each prediction is the setting's typical run at that position: the
    lower-middle seed when its seeds are sorted by MAE there. The target comes from the setting that starts earliest
    (a longer window gives its first prediction later)."""
    results = Path(results)
    exps = [e for e in dict.fromkeys([REFERENCE, ext['best'], ext['worst'], *ext['clearly_better']]) if e in out['runs']]
    col = METRICS.index('mae')
    people = {}
    for subj, positions in out['positions'].items():
        entry = {'positions': positions, 'target': [None] * len(positions), 'pred': {}}
        for exp in exps:
            seeds = out['runs'][exp].get(subj)
            if not seeds:
                continue
            loaded = {}
            per_pos = []
            for k, a in enumerate(positions):
                order = sorted(seeds, key=lambda s: (seeds[s][k][col], int(s)))
                seed = order[(len(order) - 1) // 2]
                if seed not in loaded:
                    loaded[seed] = np.load(results / exp / subj / DIRECTION / f'seed{seed}.npz', allow_pickle=False)
                z = loaded[seed]
                m = np.isclose(z['angle'].astype(float), a, atol=5e-4)
                t = z['t'].astype(float)[m]
                t0, y = _blocks(t, z['y_pred'].astype(float)[m], hz)
                per_pos.append({'seed': int(seed), 't0': t0, 'y': y})
                old = entry['target'][k]
                if old is None or t0 < old['t0']:
                    tt0, ty = _blocks(t, z['y_own_target'].astype(float)[m], hz)
                    entry['target'][k] = {'t0': tt0, 'y': ty}
            entry['pred'][exp] = per_pos
        people[subj] = entry
    return {'hz': hz, 'reference': REFERENCE, 'best': ext['best'], 'worst': ext['worst'],
            'clearly_better': ext['clearly_better'], 'settings': exps, 'people': people}


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
    ap.add_argument('--waves-out', default=str(ROOT / 'EXPERIMENTS/dashboard_react/src/data/study-waves.json'))
    ap.add_argument('--workers', type=int, default=_default('scripts/run_window_study.sh', r'WORKERS:-(\d+)', 6),
                    help='runs trained at once (default: what scripts/run_window_study.sh uses)')
    ap.add_argument('--threads', type=int, default=_default('EXPERIMENTS/pipeline/run.py', r"'--threads', type=int, default=(\d+)", 1),
                    help='TensorFlow threads per run (default: run.py\'s)')
    a = ap.parse_args()
    out = build(a.results, spec_titles())
    out['machine'] = machine(read_machine(), a.workers, a.threads)
    out['run_versions'] = run_versions(a.results)
    Path(a.out).write_text(json.dumps(out, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    n = sum(len(s) for r in out['runs'].values() for s in r.values())
    print(f"{len(out['settings'])} settings, {n} runs -> {a.out}")
    ext = extremes(out)
    w = waves(a.results, out, ext)
    Path(a.waves_out).write_text(json.dumps(w, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    print(f"best {ext['best']}, worst {ext['worst']}, clearly better {ext['clearly_better']} -> {a.waves_out}")


if __name__ == '__main__':
    main()
