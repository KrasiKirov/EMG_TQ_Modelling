"""Import the normalization study (total torque, 0.2 s) into the presentation catalogue.

Reads the runs written by ``EXPERIMENTS/code/run_experiment.py`` in this repository
(``EXPERIMENTS/results/N*``). Only arrays and JSON are read; no model is loaded. Every
position score is computed here from the saved predictions, in float64, against the total
measured torque (active torque for A0). Calibrated entries (N0 + cal, A0 + cal) apply the 10 s
gain+offset rule of ``EXPERIMENTS/normalization/calibration_test.py`` per trial and are scored from 30 s
to the end of each trial only. The plain-English report (``EXPERIMENTS/normalization/REPORT.md``) is attached
to every entry.

Usage (from the repository root): python EXPERIMENTS/dashboard_react/scripts/import_normalization.py
"""
from pathlib import Path
import argparse, base64, hashlib, json
import sys
import numpy as np

DASH = Path(__file__).resolve().parents[1]
ROOT = DASH.parents[1]
sys.path.insert(0, str(ROOT/'EXPERIMENTS'/'normalization'))
from calibration_test import correct, SCORE_FROM_S   # noqa: E402
CAL_S = 10
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
ARMS['A0_active_trainmax'] = ('A0 · balanced positions, active torque', 'max',
    'Exactly N0 (session-1 maximum, balanced positions, 0.2 s), but the network learns active torque: '
    'total torque minus the clean passive table built from the relaxed holds of each session (IES has no holds, '
    'so the old passive rule is used for IES).')
REFS = {'NR' + k[1:]: k for k in ARMS if k.startswith('N')}
CALIBRATED = {k + '+cal': k for k in ('N0_total_trainmax', 'A0_active_trainmax')}
SCALING = {
    'N0': ('÷ session-1 maximum', 'Each muscle envelope is divided by the largest value it reached anywhere in session 1 '
           '(one sample of the 1 kHz envelope). Session 2 is divided by the same four numbers.'),
    'N1': ('÷ session-1 99th percentile', 'Each muscle envelope is divided by its 99th percentile over session 1’s '
           'training part. Session 2 is divided by the same four numbers.'),
    'N2': ('÷ maximum-effort (MVC) peak', 'Each muscle envelope is divided by its peak during the maximum-effort trials. '
           'Session 2 is divided by the same four numbers.'),
    'N3': ('÷ each session’s own 99th percentile', 'Each session is divided by its own 99th percentile per muscle, '
           'taken over all of that session’s active trials (EMG only, no torque).'),
}
SCALING['A0'] = SCALING['N0']


def overview(arm, ridge, active, cal):
    """Short parameter list and computation steps shown at the top of the experiment page."""
    code = arm.split('_')[0]
    scale_short, scale_how = SCALING['N' + code[1:] if code.startswith('NR') else code]
    target = ('Active torque: total minus that session’s clean passive table (relaxed holds)' if active
              else 'Total measured torque, nothing subtracted')
    settings = [
        ['Train → test', 'Session 1 → session 2 · 8 angles · one ~90 s trial per angle'],
        ['Inputs', '4 EMG envelopes (GM, GL, SOL, TA) + ankle angle, 100 Hz'],
        ['Window', '0.2 s = 20 samples → torque at the window’s last sample'],
        ['Target', target],
        ['EMG scaling', scale_short],
        ['Model', 'Ridge regression per position (reference)' if ridge else 'LSTM 8 × 2 layers, dropout 0.3'],
        ['Balanced positions', 'No' if ridge else 'Yes: each position’s target ÷ its standard deviation'],
        ['Seeds', 'None (deterministic)' if ridge else '3 (0, 1, 2), median shown'],
        ['Calibration', '10 s gain + offset per angle, scored from 30 s' if cal else 'None'],
    ]
    steps = [
        'EMG: band-pass 30–300 Hz, rectify, 2 Hz low-pass → envelope (1 kHz), then down-sampled to 100 Hz.',
        'Scale: ' + scale_how,
        'Standardise the 5 inputs (mean 0, sd 1) with session 1’s training statistics.',
        'Cut 0.2 s windows sliding by one sample. Each session-1 trial: first 80 % to fit, last 20 % to stop training.',
    ]
    if ridge:
        steps.append('Fit one ridge regression per position on EMG, EMG² and angle features; predict session 2.')
    else:
        steps.append('Train the LSTM on the target ÷ each position’s standard deviation, then multiply its predictions '
                     'back (Nadam, lr 0.003, batch 8, ≤ 200 epochs, early stopping after 25).')
        steps.append('Predict every window of session 2, which the model never saw.')
    if cal:
        steps.append('Calibrate, without retraining: on the first 10 s of each session-2 trial, fit torque ≈ a × '
                     'prediction + b (a between 0.5 and 2; offset only if the torque barely moves) and apply it to the '
                     'whole trial.')
    steps.append('Score each position on every 100 Hz sample' + (' from 30 s to the end' if cal else '') +
                 ': R² = 1 − Σ(y − ŷ)² / Σ(y − ȳ)², MAE = mean |y − ŷ|. Median over seeds, then median across '
                 'positions; R² below 0 is shown as 0.')
    return dict(settings=settings, steps=steps)


TARGET = 'Total measured torque, nothing subtracted. The passive table plays no part in training or scoring.'
TARGET_ACTIVE = ('Active torque: total measured torque minus the clean passive table of that session (relaxed '
                 'holds). Scored against active torque.')
CAL_HOW = (' Then a 10 s calibration, no retraining: on the first 10 s of every session-2 trial (torque measured), '
           'fit torque ≈ a × prediction + b (a kept between 0.5 and 2; offset only if the person barely pushed) and '
           'apply it to the rest of that trial. Scores use only 30 s to the end of each trial, so seconds 0–30 '
           '(the calibration block and a gap) are never scored. The trace shows the whole corrected trial.')


def metrics(y, p):
    e = p-y
    ss = np.sum((y-y.mean())**2)
    ed = (p-p.mean())-(y-y.mean())
    assert ss > 0
    return dict(r2=float(1-np.sum(e**2)/ss), mae=float(np.mean(abs(e))),
                rmse=float(np.sqrt(np.mean(e**2))), bias=float(np.mean(e)),
                r2_demeaned=float(1-np.sum(ed**2)/ss), mae_demeaned=float(np.mean(abs(ed))),
                peak=float(np.max(abs(y))))


def calibration_summary(norm_dir):
    """Home-page numbers: N0 / A0 with and without the 10 s calibration, all scored from 30 s to the end of each
    trial (the calibration test's own rows, so the four variants are judged on the same stretch)."""
    import pandas as pd
    out = {'people': [], 'variants': {}}
    for code, name in (('N0', 'calibration_runs.csv'), ('A0', 'calibration_runs_A0_active_trainmax.csv')):
        f = norm_dir/name
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = d[(d.seconds == 0) | ((d.variant == 'gain+offset') & (d.seconds == 10))]
        for cal, part in ((False, d[d.seconds == 0]), (True, d[d.seconds == 10])):
            cells = part.groupby(['subject', 'position_rad', 'toes_down'])[['mae', 'r2', 'vaf']].median().reset_index()
            cells['r2'] = cells.r2.clip(lower=0)
            cells['rank'] = cells.groupby('subject').position_rad.rank(method='first').astype(int)
            v = {'group': {}, 'person': {}, 'angle': {}}
            for t, key in ((1, 'toes'), (0, 'other')):
                g = cells[cells.toes_down == t]
                v['group'][key] = {m: float(g[m].median()) for m in ('mae', 'r2', 'vaf')}
                v['person'][key] = {k: float(x) for k, x in g.groupby('subject').mae.median().items()}
            v['angle'] = {int(k): float(x) for k, x in cells.groupby('rank').mae.median().items()}
            out['variants'][code + ('cal' if cal else '')] = v
            out['people'] = sorted(set(out['people']) | set(cells.subject))
    return out


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
    for entry in [*ARMS, *REFS, *CALIBRATED]:
        cal = entry in CALIBRATED
        native = CALIBRATED.get(entry, entry)
        ridge = native.startswith('NR')
        arm = REFS.get(native, native)
        title, emg, how = ARMS[arm]
        if cal:
            title, how = title.split(' · ')[0] + ' + 10 s calibration', how + CAL_HOW
        active = arm.startswith('A')
        ident = PREFIX+native.split('_')[0]+('cal' if cal else '')
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
                if cal:
                    rel = ts-ts[0]
                    pp = correct(pp[rel < CAL_S], yy[rel < CAL_S], pp, 'gain+offset')
                    scored = rel >= SCORE_FROM_S
                    row = metrics(yy[scored], pp[scored])
                else:
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
            kind='normalization', native=native, model='ridge' if ridge else 'lstm', config=cfg,
            target=TARGET_ACTIVE if active else TARGET,
            overview=overview(arm, ridge, active, cal),
            purpose=(('Reference model, never chosen: per-position ridge regression with the scaling of '
                      + arm.split('_')[0] + '. ') if ridge else '') + how,
            protocol=('Train on session 1, test on session 2. ' + ('Active torque' if active else 'Total torque') + ', 0.2 s window (20 samples at 100 Hz) of four '
                      'muscle envelopes plus angle. ' + (
                          'Ridge regression with its own coefficients per position, no seeds.' if ridge else
                          'Balanced positions (each position’s target ÷ its standard deviation). LSTM 8×2, dropout 0.3, '
                          'batch 8, at most 200 epochs, early stopping after 25.')
                      + f' People: {", ".join(sorted(coverage))}. Seeds: {", ".join(map(str, seeds))}.'),
            params=dict(code='norm', passive='clean' if active else 'none', emg=emg, weighting='none' if ridge else 'scale', window=0.2,
                        dropout=0.0 if ridge else 0.3, lr=0.003, extra='ridge' if ridge else 'none'))
        summary[ident] = {s: len(v) for s, v in coverage.items()}
    data['calibration'] = calibration_summary(ROOT/'EXPERIMENTS'/'normalization')
    used = set(data['targets'].values()); data['targetPool'] = {k: v for k, v in data['targetPool'].items() if k in used}
    path.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
