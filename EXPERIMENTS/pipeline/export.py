"""Export saved experiment predictions into the dashboard catalogue.

Reads EXPERIMENTS/results/<results_from>/<subject>/<direction>/seed<k>.{npz,json} for each spec with a
'dashboard' block; scores every position with pipeline.scoring; writes rows, packed waveforms, timings and the
experiment entry (title, description, tags, parameters) to presentation-catalog.json. Entries for the
exported ids are replaced; everything else in the catalogue is kept. Also writes the sidebar groups and landing page
from EXPERIMENTS/dashboard_groups.json.
"""
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from .run import BASE
from .scoring import score_trial

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ('positions', 'targets', 'predictions', 'means', 'times')


def packed(a):
    return base64.b64encode(np.asarray(a, dtype='<f4').tobytes()).decode()


def _finite(x):
    """Timing value for JSON: a float, or None when missing or non-finite."""
    return float(x) if isinstance(x, (int, float)) and np.isfinite(x) else None


def _sample(grid, ts, values):
    out = np.interp(grid, ts, values)
    out[(grid < ts[0]) | (grid > ts[-1])] = np.nan
    right = np.clip(np.searchsorted(ts, grid), 1, len(ts) - 1)
    out[(ts[right] - ts[right - 1]) > .015] = np.nan            # keep gaps as gaps
    return out


def _drop(data, ident):
    data['rows'] = [r for r in data['rows'] if r['exp'] != ident]
    for f in FIELDS:
        data[f] = {k: v for k, v in data[f].items() if k != ident and not k.startswith(ident + '|')}
    data['experiments'].pop(ident, None)


def _entry(spec, cfg, coverage, old=None):
    d = spec.dashboard
    seeds = sorted({s for v in coverage.values() for s in v})
    text = d.text
    settings = text.get('overview', {}).get('settings') or [[k, v if isinstance(v, str) else json.dumps(v)]
                                                           for k, v in sorted(cfg.items())]
    return dict(id=d.id, title=d.label, name=d.name, group=d.group, kind='spec', native=spec.id,
                model=cfg.get('model', 'lstm'), config=cfg, description=spec.description, tags=spec.tags,
                blurb=d.blurb,
                purpose=text.get('purpose', spec.description), target=text.get('target', ''),
                protocol=text.get('protocol', f'People: {", ".join(sorted(coverage))}. Seeds: {", ".join(map(str, seeds))}.'),
                overview=dict(settings=settings, steps=text.get('overview', {}).get('steps', [])),
                params=text.get('params', {}),
                **({'reportText': _report(text['report'], old)} if text.get('report') else {}))


def _report(path, old):
    """The report's text; the Markdown reports stay out of git, so a clone without the file keeps the saved copy."""
    f = ROOT / path
    if f.exists():
        return f.read_text(encoding='utf-8')
    if old and old.get('reportText'):
        return old['reportText']
    raise FileNotFoundError(f'{path} (the report named in the spec) is missing and the catalogue has no saved copy')


def _groups(specs_all, groups_cfg):
    items = {g['label']: [] for g in groups_cfg['groups']}
    for s in specs_all:
        if s.dashboard:
            items.setdefault(s.dashboard.group, []).append((s.dashboard.order, s.dashboard.id, s.dashboard.label))
    for x in groups_cfg.get('extra_items', []):
        items.setdefault(x['group'], []).append((x.get('order', 100), x['id'], x['label']))
    ids = [i for v in items.values() for _, i, _ in v]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f'dashboard ids used more than once: {dupes}')
    unknown = set(items) - {g['label'] for g in groups_cfg['groups']}
    if unknown:
        raise ValueError(f'groups missing from dashboard_groups.json: {sorted(unknown)}')
    return [dict(g, items=[{'id': i, 'label': lab} for _, i, lab in sorted(items[g['label']])])
            for g in groups_cfg['groups'] if items[g['label']]]          # empty groups stay hidden


def export(specs, catalog, results, groups, all_specs=None):
    """Score and write `specs` into `catalog`; returns {dashboard_id: {subject: n_runs}}."""
    catalog, results = Path(catalog), Path(results)
    groups_cfg = json.loads(Path(groups).read_text(encoding='utf-8'))
    sidebar = _groups(all_specs or specs, groups_cfg)        # validate groups before touching the catalogue
    data = json.loads(catalog.read_text(encoding='utf-8'))
    t0, dt, n = data['grid']
    grid = t0 + dt * np.arange(n)
    summary = {}
    for spec in specs:
        if not spec.dashboard:
            raise ValueError(f'{spec.id} has no dashboard block; tag it first')
        ident = spec.dashboard.id
        files = sorted(results.glob(f'{spec.results_from}/*/*/seed*.npz'))
        if not files:
            raise FileNotFoundError(f'no results for {spec.results_from} under {results}')
        old = data['experiments'].get(ident)
        _drop(data, ident)
        coverage, cfg, first = {}, None, None
        for f in files:
            subject, direction, seed = f.parts[-3], f.parts[-2], int(f.stem[4:])
            info = json.loads(f.with_suffix('.json').read_text(encoding='utf-8'))
            full = json.loads(json.dumps(dict(BASE, **(info.get('config') or {}))))
            if first is None:
                cfg, first = info.get('config', {}), (f, full)
            elif full != first[1]:
                diff = sorted(k for k in set(full) | set(first[1]) if full.get(k) != first[1].get(k))
                raise ValueError(f'{f}: run config differs from {first[0]} in {diff}; '
                                 'every run of one experiment must use the same config')
            z = np.load(f, allow_pickle=False)
            y, p = z['y_own_target'].astype(float), z['y_pred'].astype(float)
            ang, tt, trial = z['angle'].astype(float), z['t'].astype(float), z['trial']
            if not (len(y) == len(p) == len(tt) and np.isfinite(y).all() and np.isfinite(p).all()):
                raise ValueError(f'{f}: non-finite or misaligned arrays')
            poses = sorted(np.unique(ang))
            key = f'{ident}|{subject}|{direction}|{seed}'
            data['positions'][f'{ident}|{subject}|{direction}'] = [round(float(a), 3) for a in poses]
            targets, predictions = [], []
            for k, a in enumerate(poses):
                m = ang == a
                if len(np.unique(trial[m])) != 1:
                    raise ValueError(f'{f}: position {a} spans several trials')
                yy, pp, ts = y[m], p[m], tt[m]
                row = score_trial(yy, pp, ts)
                row.update(exp=ident, subj=subject, dir=direction, seed=seed, pos=round(float(a), 3), toes=int(k < 2))
                data['rows'].append(row)
                data['means'][f'{key}|{round(float(a), 3)}'] = [float(yy.mean()), float(pp.mean())]
                targets.append(_sample(grid, ts, yy))
                predictions.append(_sample(grid, ts, pp))
            blob = packed(np.concatenate(targets))
            target_id = hashlib.sha256(blob.encode()).hexdigest()[:16]
            data['targetPool'][target_id] = blob
            data['targets'][key] = target_id
            data['predictions'][key] = packed(np.concatenate(predictions))
            data['times'][key] = dict(seconds=_finite(info.get('train_s')), prep=_finite(info.get('prep_s')),
                                      definition='fit + prediction (pipeline/run.py); data preparation separate')
            coverage.setdefault(subject, []).append(seed)
        data['experiments'][ident] = _entry(spec, cfg, coverage, old)
        summary[ident] = {s: len(v) for s, v in coverage.items()}
    data['dashboard'] = dict(landing=groups_cfg['landing'], groups=sidebar)
    data.pop('calibration', None)   # old home-page calibration summary; replaced by the per-row error breakdown
    used = set(data['targets'].values())
    data['targetPool'] = {k: v for k, v in data['targetPool'].items() if k in used}
    catalog.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    return summary
