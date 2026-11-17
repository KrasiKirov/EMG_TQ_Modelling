"""Every processing stage of one session-1 trial per person, for the dashboard's pipeline page (no model).

    python -m EXPERIMENTS.analysis.pipeline_stages [--out EXPERIMENTS/dashboard_react/src/data/pipeline.json]

The stages follow the reference setting (EXPERIMENTS/configs/W00_reference.json): raw EMG at 1 kHz -> offset removed
(mean of the first 1 s) -> causal 4th-order band-pass 30-300 Hz -> rectified -> causal 2nd-order 2 Hz low-pass
(the envelope) -> every 10th sample (100 Hz) after the 1 s start-up trim -> divided by the session-1 maximum
-> 20-step windows with the ankle angle as the fifth input. The script checks that its own stage chain reproduces the
inputs the reference model is trained on (ML.research.run.prepare) and refuses to write otherwise.

Per person it keeps one session-1 trial at a middle position:
  excerpt  2 s at 1 kHz around the strongest rise in EMG: raw, band-passed and envelope per channel, plus the
           zero-phase 4th-order envelope of earlier rounds (to show the causal filter's lag); rectified = |band-passed|
  inputs   the same 2 s at 100 Hz: the four scaled envelopes, the angle and the active-torque label (model inputs)
  trial    the whole trial at 20 Hz: scaled envelopes, angle, measured and active torque
  psd      power spectrum of the raw EMG (offset removed) over the whole trial, per channel
  passive  the passive torque subtracted at every position of both sessions
"""
import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfreqz, welch

FS = 1000.0
EXCERPT_S = 2.0
LEAD_S = 0.6            # excerpt starts this long before the strongest rise
PEOPLE = ['YES', 'JM', 'EG', 'IES', 'CH', 'HM']
CHANNELS = ['GM', 'GL', 'SOL', 'TA']
OUT = Path('EXPERIMENTS/dashboard_react/src/data/pipeline.json')


def stages_1k(x, lp_order=2, fs=FS, settle_s=1.0, bp=(30.0, 300.0), lp=2.0, bp_order=4):
    """The causal envelope of ML.preprocessing.emg_envelope.extract_envelope_causal, keeping every stage."""
    x = np.asarray(x, np.float64)
    demeaned = x - np.mean(x[:max(1, int(settle_s * fs))])
    nyq = 0.5 * fs
    band = sosfilt(butter(bp_order, [bp[0] / nyq, bp[1] / nyq], btype='band', output='sos'), demeaned)
    rect = np.abs(band)
    sos_lp = butter(lp_order, lp / nyq, btype='low', output='sos')
    zi = sosfilt_zi(sos_lp) * np.mean(rect[:max(1, int(0.2 * fs))])
    env = sosfilt(sos_lp, rect, zi=zi)[0]
    return {'demeaned': demeaned, 'band': band, 'rect': rect, 'env': env}


def reference_cfg():
    from EXPERIMENTS.pipeline.run import load_config
    _, cfg = load_config('W00_reference')
    return cfg, {k: cfg[k] for k in ('passive', 'emg', 'window', 'envelope', 'trim_end_s', 'input_step',
                                     'train_stride', 'state', 'lp_order', 'val_gap', 'metadata_passive')}


def model_inputs(subject, rcfg):
    """recs, scaled EMG features {idx: (T, 4)}, active-torque targets {idx: (T,)} and the window arrays exactly as
    the reference model gets them (captured from the call that builds its windows)."""
    from ML.research import run as RR
    cap = {}
    build = RR.splits.build

    def spy(protocol, recs, feats, targets, **kw):
        cap.update(recs=recs, feats=feats, targets=targets)
        return build(protocol, recs, feats, targets, **kw)

    RR.splits.build = spy
    try:
        data, _ = RR.prepare(subject, 'S1toS2', rcfg)
    finally:
        RR.splits.build = build
    from EXPERIMENTS.pipeline.run import drop_last_val_window
    return cap['recs'], cap['feats'], cap['targets'], drop_last_val_window(data)


def raw_trial(subject, idx):
    """the raw 1 kHz DataFrame of trial ``idx`` and its EMG column names (canonical order GM, GL, SOL, TA)."""
    import os
    from ML.config import SUBJECTS, DATA_DIR
    from ML.preprocessing.flb_reader import read_flb
    from ML.preprocessing.emg_envelope import detect_emg_columns
    fn, sid = SUBJECTS[subject]
    trials = read_flb(os.path.join(DATA_DIR, fn), subject_id=sid)
    df = next(d for d in trials if d.attrs.get('trial_index') == idx)
    return df, detect_emg_columns(df)


def split_structure(recs, data, window, step=1, batch=64):
    """Where the training, validation and test windows of every trial start and end (times in s since the start of
    the recording; a window is labelled at its last sample). From the reference model's own window arrays."""
    span = (window - 1) * step
    by = {}
    for part in ('train', 'val', 'test'):
        m = data[f'meta_{part}']
        for i in np.unique(m['trial']):
            sel = m['trial'] == i
            by.setdefault(int(i), {})[part] = (np.asarray(m['last'])[sel], np.asarray(m['t'])[sel])
    trials = []
    for r in sorted(recs, key=lambda r: (r['session'], r['angle'])):
        t = np.asarray(r['t'], np.float64)
        e = {'session': r['session'], 'angle': round(float(r['angle']), 4), 'start': round(float(t[0]), 2),
             'end': round(float(t[-1]), 2), 'samples': int(len(t))}
        for part, (last, tl) in by.get(r['idx'], {}).items():
            e[part] = {'n': int(len(last)), 'first_input': round(float(t[last[0] - span]), 2),
                       'first_label': round(float(tl[0]), 2), 'last_label': round(float(tl[-1]), 2)}
        trials.append(e)
    n_tr = int(len(data['y_train']))
    return {'span_samples': span, 'window': window, 'step': step, 'batch': batch,
            'n_train': n_tr, 'n_val': int(len(data['y_val'])), 'n_test': int(len(data['y_test'])),
            'updates_per_epoch': int(-(-n_tr // batch)), 'trials': trials}


def pick_trial(recs):
    """the session-1 trial at the middle position (4th of 8 from most plantarflexed)."""
    s1 = sorted([r for r in recs if r['session'] == 'S1'], key=lambda r: r['angle'])
    return s1[(len(s1) - 1) // 2]


def strongest_rise(feats, fs=100):
    """100 Hz sample where the summed scaled EMG rises most over the next second (not in the first/last 2 s)."""
    s = feats.sum(axis=1)
    d = s[fs:] - s[:-fs]
    lo, hi = 2 * fs, len(d) - 2 * fs
    return lo + int(np.argmax(d[lo:hi]))


def sig(a, n=4):
    """round to n significant digits for a compact JSON."""
    return [float(f'{v:.{n}g}') for v in np.asarray(a, np.float64)]


def person(subject, cfg, rcfg):
    from ML.research import config as C
    from ML.research.config import PROTOCOLS
    from ML.preprocessing.emg_envelope import extract_envelope, extract_envelope_causal
    recs, feats, targets, data = model_inputs(subject, rcfg)
    rec = pick_trial(recs)
    idx = rec['idx']
    df, cols = raw_trial(subject, idx)
    st = [stages_1k(df[c].values, lp_order=cfg['lp_order']) for c in cols]
    for k, c in enumerate(cols):                       # same as the library's causal envelope
        assert np.allclose(st[k]['env'], extract_envelope_causal(df[c].values, lp_order=cfg['lp_order']))

    s0, step = C.TRIM_START, C.DOWNSAMPLE
    f = np.asarray(feats[idx], np.float64)
    n = len(f)
    env100 = np.column_stack([st[k]['env'][s0:s0 + n * step:step] for k in range(len(cols))])
    scale = env100 / f                                  # the session-1 maximum per channel
    trainmax = np.median(scale, axis=0)
    if not np.allclose(scale, trainmax, rtol=1e-6):
        raise AssertionError(f'{subject}: stage chain does not reproduce the model inputs')

    j0 = max(0, strongest_rise(f) - int(LEAD_S * 100))
    j1 = j0 + int(EXCERPT_S * 100)
    a0 = s0 + j0 * step                                 # 1 kHz index of the excerpt start
    a1 = a0 + int(EXCERPT_S * FS)
    zp = [extract_envelope(df[c].values) for c in cols]
    t_rec = np.asarray(rec['t'])
    y = np.asarray(targets[idx], np.float64)
    passive = rec['tq'] - y

    by = lambda k, key: sig(st[k][key][a0:a1])
    sub = slice(None, None, 5)                          # 100 Hz -> 20 Hz for the whole-trial view
    nper = 1000
    fr, pxx = welch(np.column_stack([s['demeaned'] for s in st]), fs=FS, nperseg=nper, axis=0)

    sessions = {}
    for r in recs:
        p = np.asarray(r['tq'] - np.asarray(targets[r['idx']]), np.float64)
        sessions.setdefault(r['session'], []).append({'angle': round(float(r['angle']), 4),
                                                      'passive': round(float(np.median(p)), 3),
                                                      'spread': round(float(np.ptp(p)), 4)})
    for v in sessions.values():
        v.sort(key=lambda e: e['angle'])

    return {
        'subject': subject, 'trial': int(idx), 'session': rec['session'], 'angle': round(float(rec['angle']), 4),
        'position': f"{sorted(r['angle'] for r in recs if r['session'] == 'S1').index(rec['angle']) + 1} of "
                    f"{sum(r['session'] == 'S1' for r in recs)}",
        'channels_file': list(cols), 'trainmax': sig(trainmax, 5),
        'raw_unit': 'V',
        'excerpt': {
            't0': round(a0 / FS, 3), 'fs': FS,
            'raw': [sig(df[c].values[a0:a1]) for c in cols],
            'demeaned_offset': sig([np.mean(df[c].values[:int(FS)]) for c in cols], 5),
            'band': [by(k, 'band') for k in range(len(cols))],
            'env': [by(k, 'env') for k in range(len(cols))],
            'env_zerophase4': [sig(z[a0:a1]) for z in zp],
        },
        'inputs': {
            't0': round(float(t_rec[j0]), 3), 'fs': 100,
            'emg': [sig(f[j0:j1, k]) for k in range(f.shape[1])],
            'angle': sig(rec['pos'][j0:j1], 5),
            'label': sig(y[j0:j1]),
        },
        'trial_view': {
            't0': round(float(t_rec[0]), 3), 'fs': 20,
            'emg': [sig(f[sub, k], 3) for k in range(f.shape[1])],
            'angle': sig(rec['pos'][sub], 4),
            'measured': sig(rec['tq'][sub]),
            'active': sig(y[sub]),
            'passive': round(float(np.median(passive)), 3),
        },
        'psd': {'f_step': float(fr[1] - fr[0]), 'db': [sig(10 * np.log10(pxx[:, k] / pxx[:, k].max() + 1e-12), 3)
                                                      for k in range(pxx.shape[1])]},
        'passive_by_session': sessions,
        'splits': split_structure(recs, data, cfg['window'], cfg['input_step'], PROTOCOLS[cfg['train_protocol']]['batch']),
    }


STUDY_SETTINGS = ('W00_reference', 'WL1_0500ms', 'WL2_1s', 'WL2c_1s_10ms', 'WL3_2s', 'WL4_5s', 'WL5_10s',
                  'WO1_90', 'WO2_75', 'WO3_50', 'WO4_00', 'WO2_75_eq', 'WO3_50_eq', 'WO4_00_eq', 'WS1_shuffled', 'WC2_carry_500ms', 'WC3_carry_1s',
                  'WC4_carry_10s')


def window_settings(samples, n_trials=8, results=Path('EXPERIMENTS/results')):
    """Window layout of every active-torque window-study setting (not the length x overlap grid): steps, span,
    training stride, purge gap, windows and weight updates per epoch, median epochs actually trained.
    ``samples`` = samples per trial after trimming (the same for every trial and person)."""
    import glob
    from EXPERIMENTS.pipeline.run import BASE
    from EXPERIMENTS.pipeline.spec import load_all
    from ML.research.config import PROTOCOLS, TRAIN_FRAC
    out = []
    for spec in sorted(load_all(), key=lambda sp: STUDY_SETTINGS.index(sp.id) if sp.id in STUDY_SETTINGS else 99):
        if spec.id not in STUDY_SETTINGS:
            continue
        c = dict(BASE, **spec.changes)
        W, st, k = c['window'], c['input_step'], c['train_stride']
        span = (W - 1) * st
        n = samples - span
        n_tr = int(n * TRAIN_FRAC)
        per_trial = len(range(0, n_tr, k))
        proto = PROTOCOLS[c['train_protocol']]
        passes = max(1, round(c['epoch_windows'] / (per_trial * n_trials))) if c['epoch_windows'] else 1
        if c['state'] == 'carry':
            chunk = c['chunk'] or W * st
            updates = -(-(span + n_tr) // chunk)
        else:
            chunk, updates = None, -(-(per_trial * n_trials * passes) // proto['batch'])
        ep = [json.load(open(f))['train_info']['epochs'] for f in glob.glob(str(results / spec.id / '*' / 'S1toS2' / 'seed*.json'))]
        out.append({'id': spec.id, 'title': spec.raw.get('title', spec.id), 'steps': W, 'step_ms': st * 10,
                    'span_s': round(span / 100, 2), 'stride_ms': k * 10, 'overlap_pct': round(100 * max(0, 1 - k / (W * st)), 1),
                    'train_windows': per_trial * n_trials, 'passes_per_epoch': passes, 'val_gap_s': round(span / 100, 2) if c['val_gap'] else 0,
                    'state': c['state'], 'chunk_s': chunk / 100 if chunk else None, 'batch': proto['batch'],
                    'shuffle': bool(c['shuffle']), 'updates_per_epoch': int(updates), 'epoch_limit': proto['epochs'],
                    'epochs_median': float(np.median(ep)) if ep else None, 'runs': len(ep)})
    return out


def filter_responses(lp_order=2):
    nyq = 0.5 * FS
    f = np.logspace(-1, np.log10(490), 240)
    out = {'f': sig(f, 4)}
    for key, sos in (('band', butter(4, [30 / nyq, 300 / nyq], btype='band', output='sos')),
                     ('lp2', butter(2, 2 / nyq, btype='low', output='sos')),
                     ('lp4', butter(4, 2 / nyq, btype='low', output='sos'))):
        _, h = sosfreqz(sos, worN=f, fs=FS)
        out[key] = sig(20 * np.log10(np.abs(h) + 1e-12), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--people', nargs='*', default=PEOPLE)
    a = ap.parse_args()
    cfg, rcfg = reference_cfg()
    data = {'built': date.today().isoformat(), 'setting': 'W00_reference', 'channels': CHANNELS,
            'lp_order': cfg['lp_order'], 'trim_start_s': 1.0, 'trim_end_s': cfg['trim_end_s'], 'window': cfg['window'],
            'filters': filter_responses(cfg['lp_order']), 'people': {}}
    for s in a.people:
        data['people'][s] = person(s, cfg, rcfg)
        print(s, 'trial', data['people'][s]['trial'], 'angle', data['people'][s]['angle'])
    data['settings'] = window_settings(data['people'][a.people[0]]['splits']['trials'][0]['samples'])
    Path(a.out).write_text(json.dumps(data, separators=(',', ':')))
    print('wrote', a.out, round(Path(a.out).stat().st_size / 1e6, 2), 'MB')


if __name__ == '__main__':
    main()
