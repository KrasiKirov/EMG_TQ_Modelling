"""
Aggregate results, select a configuration without peeking, and test it.

    python -m ML.research.analyze --study screen [--out <dir>]

Analysis plan
-------------
Unit             one test trial = one (subject, protocol fold, angle).
Truth            PRIMARY: the common reference truth of truth.py
                 (measured torque minus the passive baseline through the
                 trial's own EMG-rest periods), the SAME for every
                 configuration.  SECONDARY: each configuration's own target
                 (measured minus its own passive model), as in the earlier
                 report.  Why the change: with own targets, configurations are
                 scored against different truths.  Decided after the
                 literature review, with only the E01 baseline results known.
Primary metric   mean over the 8 angles of per-angle R², each floored at -1
                 (one catastrophic angle cannot dominate), averaged over
                 seeds (per-angle R² = mean over seeds, then floored).
Secondary        weak-angle R² (mean of the 2 most plantarflexed angles of
                 each test session, floored at -1), worst angle, RMSE (Nm),
                 RMSE in % of the angle's MVC torque, R² on active windows only,
                 false torque at rest (mean |prediction| in rest windows,
                 Nm), seed spread.
Selection        CROSS-FITTED: the configuration reported on fold S2toS1 is the
                 best on fold S1toS2 (mean over subjects, 3 seeds), and vice
                 versa, among configurations complete on both folds.  This
                 removes the optimism of choosing and scoring on the same
                 test windows, but NOT the optimism from sharing the same 3
                 subjects and session pairs.  A leave-one-subject-out
                 selection (choose on 2 subjects, score the third) is also
                 reported.  Transductive configurations (they use unlabelled
                 test-session EMG) are excluded from the causal selection and
                 reported separately.
Statistics       Selected configuration: CROSS-FITTED paired test (units of
                 the fold it was NOT selected on), Wilcoxon signed-rank +
                 bootstrap 95 % CI of the mean difference.  All other
                 configurations: the same statistics on both folds,
                 DESCRIPTIVE, with Holm-adjusted p-values for the number of
                 comparisons.  With 3 subjects these describe THESE subjects;
                 they are not population claims.
Extras           seed ensemble (average of the seeds' predictions, a real
                 deployable model, not a selection); per-trial linear
                 recalibration on the first K s (K = 5, 10, 20, 30) of each
                 test trial, scored on t >= 30.2 s only (same samples for
                 every K); pooled-session vs cross-session models on
                 identical windows.
"""

import os
import re
import sys
import glob
import json
import argparse
import warnings

import numpy as np

warnings.filterwarnings('ignore', message='Mean of empty slice')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ML.research import config as C
from ML.research import truth as T
from ML.research import experiments
from ML.research.evaluate import metrics

FLOOR = -1.0
FOLDS = ('S1toS2', 'S2toS1')
CALIB_K = (5, 10, 20, 30)
MIN_SEEDS = 3
TRUTH_CACHE_VERSION = 3


def transductive(config):
    """True if the configuration uses unlabelled test-session data."""
    return config in experiments.CONFIGS and experiments.CONFIGS[config].get('emg') == 'session_p99'


# ── loading ──────────────────────────────────────────────────────────────────

def run_files(study, train_protocol='fast'):
    """Result JSONs of real runs (not the *_truth.json caches)."""
    pat = re.compile(rf'^{re.escape(train_protocol)}_seed\d+\.json$')
    return sorted(f for f in glob.glob(os.path.join(C.RESULTS_DIR, study, '*', '*', '*', f'{train_protocol}_seed*.json'))
                  if pat.match(os.path.basename(f)))


_MVC = {}


def mvc_at(subject, angle):
    """Peak MVC torque (Nm, |min - start|, single-sample peak, so approximate)
    at ``angle``, linearly interpolated between the MVC trials."""
    if subject not in _MVC:
        from ML.research.data import load_subject
        m = sorted(load_subject(subject)['mvc'], key=lambda q: q['angle'])
        _MVC[subject] = (np.array([q['angle'] for q in m]), np.array([q['tq_peak'] for q in m]))
    a, v = _MVC[subject]
    return float(np.interp(angle, a, v))


def angle_of(key):
    return float(key.split(':')[1])


def truth_scores(json_path):
    """Reference-truth metrics of one run (cached next to it as *_truth.json)."""
    cache = json_path[:-5] + '_truth.json'
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(json_path):
        with open(cache) as f:
            res = json.load(f)
        if res.get('version') == TRUTH_CACHE_VERSION:
            return res
    with open(json_path) as f:
        r = json.load(f)
    res = T.rescore(r['subject'], json_path[:-5] + '.npz')
    res['version'] = TRUTH_CACHE_VERSION
    for k, m in res['per_angle'].items():
        m['rmse_pct_mvc'] = 100.0 * m['rmse'] / mvc_at(r['subject'], angle_of(k))
    if r['protocol'] != 'pooled':
        res['calibration'] = calibration_curve(r['subject'], json_path[:-5] + '.npz')
    with open(cache, 'w') as f:
        json.dump(res, f, default=float)
    return res


def calibration_curve(subject, npz_path, ks=CALIB_K, eval_from_s=30.2):
    """Per test trial: fit truth ≈ a·pred + b on samples with trial time < K s,
    score on trial time >= eval_from_s (the same samples for every K and every
    configuration; trial time counted from the trial's first kept sample).
    Returns {K: mean floored R² over trials, 'raw': no calibration}."""
    z = np.load(npz_path)
    tr = T.subject_truth(subject)
    trial, t, p = z['trial'].astype(int), z['t'].astype(np.float64), z['y_pred'].astype(np.float64)
    out = {str(k): [] for k in ks}
    out['raw'] = []
    for i in np.unique(trial):
        m = trial == i
        tt = t[m] - tr[i]['t0']
        y = tr[i]['truth'][np.rint(tt * C.FS).astype(int)]
        ev = tt >= eval_from_s
        if ev.sum() < 100:
            continue
        out['raw'].append(max(metrics(y[ev], p[m][ev])['r2'], FLOOR))
        for k in ks:
            cal = tt < k
            A = np.column_stack([p[m][cal], np.ones(cal.sum())])
            a, b = np.linalg.lstsq(A, y[cal], rcond=None)[0]
            out[str(k)].append(max(metrics(y[ev], a * p[m][ev] + b)['r2'], FLOOR))
    return {k: float(np.mean(v)) for k, v in out.items() if v}


def load(study, train_protocol='fast'):
    rows = []
    for f in run_files(study, train_protocol):
        with open(f) as fh:
            r = json.load(fh)
        r['truth'] = truth_scores(f)
        r['_path'] = f
        rows.append(r)
    return rows


# ── tables ───────────────────────────────────────────────────────────────────

METRIC_KEYS = ('r2', 'rmse', 'r2_active', 'false_rest', 'bias', 'rmse_pct_mvc')


def unit_table(rows, which='truth'):
    """{(subject, protocol, config): {angle_key: {metric: [per seed]}}}"""
    out = {}
    for r in rows:
        pa = r['truth']['per_angle'] if which == 'truth' else r['per_angle']
        d = out.setdefault((r['subject'], r['protocol'], r['config']), {})
        for ak, m in pa.items():
            e = d.setdefault(ak, {k: [] for k in METRIC_KEYS + ('seed',)})
            for k in METRIC_KEYS:
                e[k].append(m.get(k, np.nan))
            e['seed'].append(r['seed'])
    return out


def weak_index(keys, n_weak=2):
    """Indices of the ``n_weak`` most plantarflexed angles of EACH session."""
    out = []
    for sess in sorted({k.split(':')[0] for k in keys}):
        ks = [i for i, k in enumerate(keys) if k.split(':')[0] == sess]
        out += sorted(ks, key=lambda i: angle_of(keys[i]))[:n_weak]
    return np.array(out)


def summarise(units, subject, protocol, config):
    d = units.get((subject, protocol, config))
    if not d:
        return None
    keys = sorted(d, key=lambda k: (k[:2], angle_of(k)))
    n = {len(d[k]['r2']) for k in keys}
    if len(n) != 1:
        return None
    per_seed = np.array([d[k]['r2'] for k in keys], float)            # angles x seeds
    r2 = per_seed.mean(1)
    mean = lambda k: np.array([np.nanmean(d[a][k]) for a in keys])
    prim_seed = np.mean(np.maximum(per_seed, FLOOR), axis=0)
    wi = weak_index(keys)
    return {'primary': float(np.mean(np.maximum(r2, FLOOR))), 'primary_seed_sd': float(np.std(prim_seed)),
            'primary_per_seed': prim_seed.tolist(), 'n_seeds': int(per_seed.shape[1]),
            'weak2': float(np.mean(np.maximum(r2[wi], FLOOR))), 'weak2_raw': float(np.mean(r2[wi])),
            'weak_angles': [keys[i] for i in wi], 'worst': float(r2.min()),
            'mean_rmse': float(mean('rmse').mean()), 'weak2_rmse': float(mean('rmse')[wi].mean()),
            'r2_active': float(np.nanmean(np.maximum(mean('r2_active'), FLOOR))),
            'false_rest': float(np.nanmean(mean('false_rest'))),
            'mean_rmse_pct_mvc': float(np.nanmean(mean('rmse_pct_mvc'))),
            'max_angle_seed_sd': float(per_seed.std(1).max()),
            'angles': keys, 'r2': r2.tolist(), 'r2_sd': per_seed.std(1).tolist(), 'rmse': mean('rmse').tolist(),
            'r2_active_angles': mean('r2_active').tolist(), 'false_rest_angles': mean('false_rest').tolist(),
            'bias': mean('bias').tolist(), 'rmse_pct_mvc': mean('rmse_pct_mvc').tolist()}


def complete(units, configs, subjects, protocols=FOLDS):
    """Configurations with >= MIN_SEEDS seeds for every subject and protocol."""
    ok = []
    for c in configs:
        s = [summarise(units, sub, p, c) for sub in subjects for p in protocols]
        if all(x is not None and x['n_seeds'] >= MIN_SEEDS for x in s):
            ok.append(c)
    return ok


def fold_score(units, configs, subjects, protocol, key='primary'):
    return {c: float(np.mean([summarise(units, s, protocol, c)[key] for s in subjects])) for c in configs}


def paired_diffs(units, subjects, a, b, protocols=FOLDS, weak_only=False):
    """Paired differences (b - a) of floored per-angle seed-mean R²."""
    diffs = []
    for sub in subjects:
        for p in protocols:
            sa, sb = summarise(units, sub, p, a), summarise(units, sub, p, b)
            if sa is None or sb is None or sa['angles'] != sb['angles']:
                continue
            if min(sa['n_seeds'], sb['n_seeds']) < MIN_SEEDS:
                continue
            ra, rb = np.maximum(sa['r2'], FLOOR), np.maximum(sb['r2'], FLOOR)
            idx = weak_index(sa['angles']) if weak_only else range(len(ra))
            diffs += [rb[i] - ra[i] for i in idx]
    return np.array(diffs)


def test_diffs(diffs, n_boot=10000, seed=0):
    from scipy.stats import wilcoxon
    if len(diffs) < 3:
        return None
    rng = np.random.default_rng(seed)
    boots = rng.choice(diffs, (n_boot, len(diffs)), replace=True).mean(1)
    try:
        p = float(wilcoxon(diffs).pvalue)
    except ValueError:
        p = float('nan')
    return {'n_units': int(len(diffs)), 'mean_diff': float(diffs.mean()), 'median_diff': float(np.median(diffs)),
            'ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            'wilcoxon_p': p, 'n_better': int((diffs > 0).sum()), 'n_worse': int((diffs < 0).sum())}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (dict -> dict)."""
    items = sorted((p, k) for k, p in pvals.items() if np.isfinite(p))
    m, out, running = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def selection(units, configs, subjects):
    """Cross-fitted and leave-one-subject-out selection + tests."""
    sel = {}
    causal = [c for c in complete(units, configs, subjects) if not transductive(c)]
    if not causal:
        return sel
    f1, f2 = fold_score(units, causal, subjects, 'S1toS2'), fold_score(units, causal, subjects, 'S2toS1')
    p1, p2 = max(causal, key=f1.get), max(causal, key=f2.get)
    sel.update({'candidates': causal, 'fold_scores_S1toS2': f1, 'fold_scores_S2toS1': f2,
                'selected_on_S1toS2': p1, 'crossfit_score_on_S2toS1': f2[p1],
                'selected_on_S2toS1': p2, 'crossfit_score_on_S1toS2': f1[p2],
                'crossfit_mean': float(np.mean([f2[p1], f1[p2]])),
                'optimistic_best_mean': float(max((f1[c] + f2[c]) / 2 for c in causal)),
                'optimistic_best_config': max(causal, key=lambda c: f1[c] + f2[c])})
    # leave-one-subject-out: choose on the other subjects (both folds), score the held-out one
    loso = {}
    for h in subjects:
        others = [s for s in subjects if s != h]
        sc = {c: np.mean([summarise(units, s, p, c)['primary'] for s in others for p in FOLDS]) for c in causal}
        pick = max(causal, key=sc.get)
        loso[h] = {'selected': pick, 'score_on_heldout': float(np.mean(
            [summarise(units, h, p, pick)['primary'] for p in FOLDS]))}
    sel['loso'] = loso
    sel['loso_mean'] = float(np.mean([v['score_on_heldout'] for v in loso.values()]))
    # cross-fitted paired test of the selected configuration(s)
    for ref in ('E01_old', 'E03_final'):
        if ref not in causal or (p1 == ref and p2 == ref):
            continue
        for weak in (False, True):
            d = np.concatenate([paired_diffs(units, subjects, ref, p1, ('S2toS1',), weak),
                                paired_diffs(units, subjects, ref, p2, ('S1toS2',), weak)])
            sel[f'crossfit_test_vs_{ref}' + ('_weak' if weak else '')] = test_diffs(d)
    # descriptive table: every configuration vs the references, both folds, Holm-adjusted
    desc = {}
    for ref in ('E01_old', 'E03_final'):
        if ref not in configs:
            continue
        for weak in (False, True):
            tag = f'vs_{ref}' + ('_weak' if weak else '')
            res = {c: test_diffs(paired_diffs(units, subjects, ref, c, FOLDS, weak)) for c in configs if c != ref}
            res = {c: v for c, v in res.items() if v}
            adj = holm({c: v['wilcoxon_p'] for c, v in res.items()})
            for c, v in res.items():
                v['p_holm'] = adj.get(c, float('nan'))
            desc[tag] = res
    sel['descriptive_paired'] = desc
    return sel


def seed_ensemble(rows):
    """Average the seeds' predictions per (subject, protocol, config) and score
    against the reference truth."""
    groups = {}
    for r in rows:
        groups.setdefault((r['subject'], r['protocol'], r['config']), []).append(r['_path'][:-5] + '.npz')
    out = {}
    for (sub, p, c), files in groups.items():
        if len(files) < MIN_SEEDS:
            continue
        zs = [dict(np.load(f)) for f in files]
        ref = zs[0]
        if any(len(z['y_pred']) != len(ref['y_pred']) or not np.array_equal(z['trial'], ref['trial'])
               or not np.allclose(z['t'], ref['t']) for z in zs):
            continue
        z = dict(ref)
        z['y_pred'] = np.mean([q['y_pred'].astype(np.float64) for q in zs], axis=0)
        res = T.rescore(sub, z)
        keys = sorted(res['per_angle'], key=lambda k: (k[:2], angle_of(k)))
        r2 = np.array([res['per_angle'][k]['r2'] for k in keys])
        out[f'{sub}|{p}|{c}'] = {'n_seeds': len(files), 'primary': float(np.mean(np.maximum(r2, FLOOR))),
                                 'weak2': float(np.mean(np.maximum(r2[weak_index(keys)], FLOOR))),
                                 'r2': r2.tolist(), 'angles': keys}
    return out


def last20_comparison(rows):
    """Pooled-session vs cross-session training on IDENTICAL windows: the
    cross-session predictions are cut down to exactly the (trial, time)
    pairs of the pooled run's test windows, then both are scored against
    the reference truth.  {config: {'pooled', 'cross', 'pairs'}} (floored
    per-angle R², mean over subjects, trials and seeds)."""
    groups = {}
    for r in rows:
        groups.setdefault((r['subject'], r['config'], r['seed']), {})[r['protocol']] = r['_path'][:-5] + '.npz'
    acc = {}
    for (sub, cfg, seed), d in groups.items():
        if 'pooled' not in d:
            continue
        zp = dict(np.load(d['pooled']))
        res_p = T.rescore(sub, zp)['per_angle']
        pkeys = set(zip(zp['trial'].astype(int), np.rint(zp['t'] * C.FS).astype(int)))
        for prot in FOLDS:
            if prot not in d:
                continue
            z = dict(np.load(d[prot]))
            keep = np.array([(int(i), int(k)) in pkeys for i, k in zip(z['trial'], np.rint(z['t'] * C.FS))])
            res_c = T.rescore(sub, {k: v[keep] for k, v in z.items()})['per_angle']
            for k, m in res_c.items():
                if k in res_p:
                    e = acc.setdefault(cfg, {'pooled': [], 'cross': []})
                    e['pooled'].append(max(res_p[k]['r2'], FLOOR)); e['cross'].append(max(m['r2'], FLOOR))
    return {c: {'pooled': float(np.mean(v['pooled'])), 'cross': float(np.mean(v['cross'])), 'pairs': len(v['pooled'])}
            for c, v in acc.items()}


def calibration_summary(rows, configs_complete):
    """Mean calibration curve per complete configuration (seed mean within
    each (subject, fold), then mean over the units)."""
    acc = {}
    for r in rows:
        if r['protocol'] == 'pooled' or r['config'] not in configs_complete:
            continue
        for k, v in r['truth'].get('calibration', {}).items():
            acc.setdefault(r['config'], {}).setdefault((r['subject'], r['protocol']), {}).setdefault(k, []).append(v)
    out = {}
    for c, units in acc.items():
        ks = set.intersection(*[set(u) for u in units.values()])
        out[c] = {k: float(np.mean([np.mean(u[k]) for u in units.values()])) for k in ks}
        out[c]['n_units'] = len(units)
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def md_table(tab, subjects, configs):
    head = ('| config | ' + ' | '.join(f'{s} {p}' for s in subjects for p in FOLDS) +
            ' | mean | weak-2 R² | weak-2 RMSE (Nm) | false rest (Nm) |')
    lines = [head, '|---' * (2 * len(subjects) + 5) + '|']
    for c in configs:
        cells, vals, weak, wr, fr = [], [], [], [], []
        for s in subjects:
            for p in FOLDS:
                x = tab.get(f'{s}|{p}|{c}')
                cells.append((f"{x['primary']:+.3f}" + ('' if x['n_seeds'] >= MIN_SEEDS else f" (n={x['n_seeds']})"))
                             if x else '—')
                if x:
                    vals.append(x['primary']); weak.append(x['weak2']); wr.append(x['weak2_rmse'])
                    fr.append(x['false_rest'])
        m = lambda v, f='{:+.3f}': f.format(np.nanmean(v)) if v else '—'
        name = c + (' (transductive)' if transductive(c) else '')
        lines.append(f'| {name} | ' + ' | '.join(cells) +
                     f' | {m(vals)} | {m(weak)} | {m(wr, "{:.2f}")} | {m(fr, "{:.2f}")} |')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--study', default='screen')
    ap.add_argument('--train-protocol', default='fast')
    ap.add_argument('--out', default=None)
    ap.add_argument('--no-stats', action='store_true')
    a = ap.parse_args()
    rows = load(a.study, a.train_protocol)
    subjects = sorted({r['subject'] for r in rows})
    configs = sorted({r['config'] for r in rows})
    out_dir = a.out or os.path.join(C.RESULTS_DIR, a.study, '_summary')
    os.makedirs(out_dir, exist_ok=True)

    report = {'n_runs': len(rows), 'subjects': subjects, 'configs': configs,
              'transductive': [c for c in configs if transductive(c)]}
    for which in ('truth', 'own'):
        units = unit_table(rows, which)
        report[which] = {f'{s}|{p}|{c}': summarise(units, s, p, c)
                         for s in subjects for p in FOLDS + ('pooled',) for c in configs
                         if summarise(units, s, p, c)}
        if not a.no_stats:
            report[f'selection_{which}'] = selection(units, configs, subjects)
    units = unit_table(rows, 'truth')
    report['complete_configs'] = complete(units, configs, subjects)
    report['seed_ensemble_truth'] = seed_ensemble(rows)
    report['calibration_truth'] = calibration_summary(rows, report['complete_configs'])
    report['pooled_vs_cross_last20'] = last20_comparison(rows)

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(report, f, indent=1, default=float)
    txt = (f'# {a.study}: {len(rows)} runs, subjects {subjects}\n\n'
           f'Cells: mean per-angle R² (each angle floored at -1, seed mean). "weak-2" = 2 most plantarflexed angles.\n\n'
           f'## Primary: common reference truth\n\n' + md_table(report['truth'], subjects, configs) +
           f'\n\n## Secondary: each configuration\'s own target\n\n' + md_table(report['own'], subjects, configs) + '\n')
    with open(os.path.join(out_dir, 'tables.md'), 'w', encoding='utf-8') as f:
        f.write(txt)
    print(txt)
    for which in ('truth', 'own'):
        s = report.get(f'selection_{which}', {})
        print(which, json.dumps({k: v for k, v in s.items() if k not in ('descriptive_paired', 'candidates')
                                 and not k.startswith('fold_scores')}, indent=1))


if __name__ == '__main__':
    main()
