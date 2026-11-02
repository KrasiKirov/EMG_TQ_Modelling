"""Import the normalization study (total torque, no auto-gain, 0.2 s) into the presentation catalogue.

Reads the runs written by ``EXPERIMENTS/code/run_experiment.py`` in this repository
(``EXPERIMENTS/results/N*``). Only arrays and JSON are read; no model is loaded. Every
position score is computed here from the saved predictions, in float64, against the total
measured torque. The plain-English report (``EXPERIMENTS/normalization/REPORT.md``) is attached
to every entry.

Usage (from the repository root): python EXPERIMENTS/dashboard_react/scripts/import_normalization.py
"""
from pathlib import Path
import argparse, base64, hashlib, json
import numpy as np

DASH = Path(__file__).resolve().parents[1]
ROOT = DASH.parents[1]
PREFIX = 'norm/'
GROUP = 'Normalization study'
ARMS = {
    'N0_total_trainmax': ('N0 · session-1 maximum (control)', 'max',
        'Each muscle is divided by the largest value it reached anywhere in session 1: one single sample of the '
        '1 kHz envelope. Session 2 is divided by the same numbers. This is the control: the same scaling as the baseline.'),
    'N1_total_trainp99': ('N1 · session-1 99th percentile', 'p99 S1',
        'Each muscle is divided by its 99th percentile over session 1’s training part: the level it stays below 99 % '
        'of the time. One spike can no longer set the scale. Session 2 is divided by the same numbers.'),
    'N2_total_mvc': ('N2 · maximum-effort (MVC) trial', 'MVC',
        'Each muscle is divided by its peak during the maximum-effort trials, recorded once at the start of the '
        'visit. The EMG becomes a percentage of maximal activation. Session 2 is divided by the same numbers.'),
    'N3_total_sessionp99': ('N3 · each session’s own 99th percentile', 'p99 per session',
        'Each session is divided by its own 99th percentile per muscle. Session 2’s scale comes from all of its '
        'active-trial EMG (no torque). It needs the whole of session 2 before it can predict.'),
}
REFS = {'NR' + k[1:]: k for k in ARMS}
TARGET = 'Total measured torque, nothing subtracted. The passive table plays no part in training or scoring.'


def metrics(y, p):
    e = p-y
    ss = np.sum((y-y.mean())**2)
    ed = (p-p.mean())-(y-y.mean())
    assert ss > 0
    return dict(r2=float(1-np.sum(e**2)/ss), mae=float(np.mean(abs(e))),
                rmse=float(np.sqrt(np.mean(e**2))), bias=float(np.mean(e)),
                r2_demeaned=float(1-np.sum(ed**2)/ss), mae_demeaned=float(np.mean(abs(ed))),
                peak=float(np.max(abs(y))))


def packed(a):
    return base64.b64encode(np.asarray(a, dtype='<f4').tobytes()).decode()


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    results = ROOT/'EXPERIMENTS'/'results'
    report = ROOT/'EXPERIMENTS'/'normalization'/'REPORT.md'
    report_text = report.read_text(encoding='utf-8') if report.exists() else ''
    path = DASH/'src/data/presentation-catalog.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    # Idempotent: replace this study only.
    data['rows'] = [r for r in data['rows'] if not r['exp'].startswith(PREFIX)]
    for field in ['positions', 'targets', 'predictions', 'means', 'times']:
        data[field] = {k: v for k, v in data[field].items() if not k.startswith(PREFIX)}
    data['experiments'] = {k: v for k, v in data['experiments'].items() if not k.startswith(PREFIX)}
    t0, dt, n = data['grid']; grid = t0+dt*np.arange(n)
    summary = {}
    for native in [*ARMS, *REFS]:
        ridge = native.startswith('NR')
        arm = REFS.get(native, native)
        title, emg, how = ARMS[arm]
        ident = PREFIX+native.split('_')[0]
        files = sorted(results.glob(f'{native}/*/S1toS2/seed*.npz'))
        if not files:
            continue
        cfg = json.loads(files[0].with_suffix('.json').read_text(encoding='utf-8'))['config']
        coverage = {}
        for f in files:
            subject, seed = f.parts[-3], int(f.stem[4:])
            info = json.loads(f.with_suffix('.json').read_text(encoding='utf-8'))
            z = np.load(f)
            y, p = z['y_own_target'].astype(float), z['y_pred'].astype(float)
            ang, tt, trial = z['angle'].astype(float), z['t'].astype(float), z['trial']
            assert np.isfinite(y).all() and np.isfinite(p).all()
            poses = sorted(np.unique(ang))
            key = f'{ident}|{subject}|S1toS2|{seed}'
            data['positions'][f'{ident}|{subject}|S1toS2'] = [round(float(a), 3) for a in poses]
            targets, predictions = [], []
            for k, a in enumerate(poses):
                m = ang == a
                assert len(np.unique(trial[m])) == 1, 'one trial per position expected'
                yy, pp, ts = y[m], p[m], tt[m]
                assert np.all(np.diff(ts) > 0)
                row = metrics(yy, pp)
                row.update(exp=ident, subj=subject, dir='S1toS2', seed=seed, pos=round(float(a), 3), toes=int(k < 2))
                data['rows'].append(row)
                data['means'][f'{key}|{round(float(a), 3)}'] = [float(yy.mean()), float(pp.mean())]

                def sample(values):
                    out = np.interp(grid, ts, values)
                    out[(grid < ts[0]) | (grid > ts[-1])] = np.nan
                    right = np.clip(np.searchsorted(ts, grid), 1, len(ts)-1)
                    out[(ts[right]-ts[right-1]) > .015] = np.nan   # keep gaps as gaps
                    return out
                targets.append(sample(yy)); predictions.append(sample(pp))
            blob = packed(np.concatenate(targets)); target_id = hashlib.sha256(blob.encode()).hexdigest()[:16]
            data['targetPool'][target_id] = blob; data['targets'][key] = target_id
            data['predictions'][key] = packed(np.concatenate(predictions))
            data['times'][key] = dict(seconds=float(info['train_s']), prep=float(info['prep_s']),
                                      definition='fit + prediction (run_experiment.py); data preparation separate')
            coverage.setdefault(subject, []).append(seed)
        seeds = sorted({s for v in coverage.values() for s in v})
        data['experiments'][ident] = dict(
            id=ident, title=(f'Ridge reference · {title.split(" · ")[1]}' if ridge else title), group=GROUP,
            kind='normalization', native=native, model='ridge' if ridge else 'lstm', config=cfg, target=TARGET,
            reportText=report_text,
            purpose=(('Reference model, never chosen: per-position ridge regression with the scaling of '
                      + arm.split('_')[0] + '. ') if ridge else '') + how,
            protocol=('Train on session 1, test on session 2. Total torque, 0.2 s window (20 samples at 100 Hz) of four '
                      'muscle envelopes plus angle, no auto-gain. ' + (
                          'Ridge regression with its own coefficients per position, no seeds.' if ridge else
                          'Balanced positions (each position’s target ÷ its standard deviation). LSTM 8×2, dropout 0.3, '
                          'batch 8, at most 200 epochs, early stopping after 25.')
                      + f' People: {", ".join(sorted(coverage))}. Seeds: {", ".join(map(str, seeds))}.'),
            params=dict(code='norm', passive='none', emg=emg, weighting='none' if ridge else 'scale', window=0.2,
                        dropout=0.0 if ridge else 0.3, lr=0.003, extra='ridge' if ridge else 'none'))
        summary[ident] = {s: len(v) for s, v in coverage.items()}
    used = set(data['targets'].values()); data['targetPool'] = {k: v for k, v in data['targetPool'].items() if k in used}
    path.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
