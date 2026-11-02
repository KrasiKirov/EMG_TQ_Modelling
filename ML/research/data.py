"""
Load one subject into a "bundle" of per-trial records at 100 Hz.

A bundle contains
-----------------
active : list of dicts, one per isometric active trial, both sessions
    idx         0-based trial index in the .flb file
    comment     the trial comment
    session     'S1' (lab name "test")  or  'S2' (lab name "retest")
    angle       mean ankle angle (rad)
    t           (T,) time since the start of the recording (s), after the
                1 s start-up trim  ->  t[0] = 1.0 s
    env_raw     (T, 4) EMG envelopes [GM, GL, SOL, TA] (band-pass 30-300 Hz,
                rectify, 2 Hz low-pass; computed at 1 kHz, trimmed, decimated)
    env_max1k   (4,) max of the 1 kHz envelope over the whole trial (used to
                reproduce the pipeline's "training max" normalisation exactly)
    pos, tq     (T,) ankle angle (rad), measured torque (Nm)
    hold        the passive static hold recorded right before this trial at
                the same angle (dict, see below) or None
mvc : list of dicts (angle, env_peak (4,), tq_peak) — MVC trials
holds : list of all static holds found (dict: idx, angle, session, mean,
    start, end, slope, dur, series (100 Hz torque))
maps : passive maps per session, as the existing pipeline builds them
    'legacy' -> old rule (20 Nm std / |20 Nm| bounds), 'static' -> static
    holds only, 'worksheet' -> hand-entered values (YES only, else None)

How trials are found
--------------------
* Active trials, their session and the YES index overrides follow
  ML.preprocessing.dataset_builder.build_dataset exactly (same rescue rule,
  same 30 Nm std filter).
* Static holds are searched in EVERY non-active, non-MVC trial (some YES files
  are mis-labelled): split at device steps, keep segments with angle std
  < 5 mrad, torque std < 1 Nm and >= 5 s.  A hold is attached to the next
  active trial in recording order if their angles match within 0.05 rad.

Processing is cached per subject in config.CACHE_DIR (pickle).
"""

import os
import pickle
import warnings

import numpy as np
from scipy.signal import decimate

from .d01_preprocessing.config import SUBJECTS, DATA_DIR, TRIAL_OVERRIDES, PASSIVE_TORQUE_OVERRIDES
from .d01_preprocessing.flb_reader import read_flb
from .d01_preprocessing import dataset_builder as db
from .d01_preprocessing import active_torque as at
from .d01_preprocessing.emg_envelope import extract_envelope, detect_emg_columns

from . import config as C

CACHE_VERSION = 1
HOLD_MAX_POS_STD = 0.005   # rad
HOLD_MAX_TQ_STD = 1.0      # Nm
HOLD_MIN_DUR_S = 5.0       # s
HOLD_ANGLE_TOL = 0.05      # rad, hold <-> next active trial angle match


def _decim(x):
    return decimate(np.asarray(x, np.float64), C.DOWNSAMPLE, ftype='fir', zero_phase=True)


def _split_trials(trials, subject):
    """Active trials per session — same logic as build_dataset()."""
    ov = TRIAL_OVERRIDES.get(subject, {})
    test_idx, retest_idx = ov.get('test'), ov.get('retest')
    mvc, passive, active = db.classify_trials(trials)
    if test_idx is not None or retest_idx is not None:
        forced = {i - 1 for i in (test_idx or []) + (retest_idx or [])}
        rescued = [df for df in passive if df.attrs.get('trial_index') in forced]
        passive = [df for df in passive if df.attrs.get('trial_index') not in forced]
        active = active + rescued
    active = [df for df in active if df[db.TORQUE_COL].std() <= db._ACTIVE_STD_THRESHOLD]
    if test_idx is not None and retest_idx is not None:
        s1 = {i - 1 for i in test_idx}
        s2 = {i - 1 for i in retest_idx}
    else:
        s1, s2, unl = set(), set(), set()
        for df in active:
            i, sess = df.attrs.get('trial_index'), df.attrs.get('session')
            (s2 if sess == 'retest' else s1 if sess == 'test' else unl).add(i)
        if not s1 and s2 and unl:
            s1 = unl
    return mvc, passive, active, s1, s2


def _process_active(df, emg_cols, session):
    env1k = np.column_stack([extract_envelope(df[c].values) for c in emg_cols])
    pos = df[db.POSITION_COL].values
    tq = df[db.TORQUE_COL].values
    s = C.TRIM_START
    env = np.column_stack([_decim(env1k[s:, k]) for k in range(env1k.shape[1])])
    p, q = _decim(pos[s:]), _decim(tq[s:])
    n = min(len(env), len(p), len(q))
    t = (s + np.arange(n) * C.DOWNSAMPLE) / C.FS_RAW
    return {
        'idx': int(df.attrs.get('trial_index')), 'comment': df.attrs.get('comment', ''),
        'session': session, 'angle': float(np.mean(p[:n])), 't': t,
        'env_raw': env[:n], 'env_max1k': env1k.max(axis=0), 'pos': p[:n], 'tq': q[:n],
        'hold': None,
    }


def _find_holds(trials, exclude_idx):
    """Static passive holds in every non-active, non-MVC trial."""
    holds = []
    for df in trials:
        i = int(df.attrs.get('trial_index'))
        if i in exclude_idx or df.attrs.get('comment', '').lower().strip().startswith('mvc'):
            continue
        for seg in db._split_passive_segments(df):
            p = seg[db.POSITION_COL].values
            q = seg[db.TORQUE_COL].values
            if len(q) < HOLD_MIN_DUR_S * C.FS_RAW or p.std() > HOLD_MAX_POS_STD or q.std() > HOLD_MAX_TQ_STD:
                continue
            q100 = _decim(q)
            tt = np.arange(len(q100)) / C.FS
            slope = float(np.polyfit(tt, q100, 1)[0]) if len(q100) > 10 else 0.0
            holds.append({
                'idx': i, 'comment': df.attrs.get('comment', ''), 'angle': float(p.mean()),
                'mean': float(q.mean()), 'start': float(q[:2000].mean()), 'end': float(q[-5000:].mean()),
                'slope': slope, 'dur': len(q) / C.FS_RAW, 'series': q100.astype(np.float32),
            })
    return holds


def _passive_maps(subject, passive_trials):
    """Session maps exactly as the pipeline builds them (legacy / static)."""
    out = {'legacy': {}, 'static': {}, 'worksheet': None}
    ov = PASSIVE_TORQUE_OVERRIDES.get(subject)
    if ov:
        out['worksheet'] = {'S1': sorted(ov['test']), 'S2': sorted(ov.get('retest', ov['test']))}
    p1 = [df for df in passive_trials if df.attrs.get('session') != 'retest']
    p2 = [df for df in passive_trials if df.attrs.get('session') == 'retest']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for name, fn in (('legacy', lambda tr: db.get_passive_torque_map(tr)),
                         ('static', lambda tr: (lambda e: e if len(e) >= 2 else db.get_passive_torque_map(tr))(
                             at.static_hold_passive_map(tr, db._split_passive_segments, db.POSITION_COL,
                                                        db.TORQUE_COL, db._POSITION_TOLERANCE)))):
            m1 = fn(p1)
            m2 = fn(p2) if p2 else m1
            out[name] = {'S1': m1, 'S2': m2}
    if ov:   # the pipeline uses the worksheet for both rules when overrides exist
        out['legacy'] = dict(out['worksheet'])
        out['static'] = dict(out['worksheet'])
    return out


def load_subject(subject, use_cache=True):
    """Return the processed bundle for ``subject`` (see module docstring)."""
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    path = os.path.join(C.CACHE_DIR, f'{subject}_bundle_v{CACHE_VERSION}.pkl')
    if use_cache and os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    fn, sid = SUBJECTS[subject]
    trials = read_flb(os.path.join(DATA_DIR, fn), subject_id=sid)
    mvc, passive, active, s1, s2 = _split_trials(trials, subject)
    emg_cols = detect_emg_columns(active[0])

    recs = []
    for df in active:
        i = df.attrs.get('trial_index')
        if i in s1 or i in s2:
            recs.append(_process_active(df, emg_cols, 'S1' if i in s1 else 'S2'))
    recs.sort(key=lambda r: r['idx'])

    holds = _find_holds(trials, exclude_idx={r['idx'] for r in recs})
    for h in holds:
        nxt = [r for r in recs if r['idx'] > h['idx']]
        h['session'] = nxt[0]['session'] if nxt else 'S2'
        if nxt and abs(nxt[0]['angle'] - h['angle']) <= HOLD_ANGLE_TOL and nxt[0]['hold'] is None:
            nxt[0]['hold'] = {k: h[k] for k in ('idx', 'angle', 'mean', 'start', 'end', 'slope', 'dur')}

    mvc_recs = []
    for df in mvc:
        env1k = np.column_stack([extract_envelope(df[c].values) for c in emg_cols])
        q = df[db.TORQUE_COL].values
        base = float(np.median(q[:1000]))
        mvc_recs.append({'idx': int(df.attrs.get('trial_index')), 'angle': float(df[db.POSITION_COL].mean()),
                         'env_peak': env1k.max(axis=0), 'tq_peak': float(abs(q.min() - base))})
    mvc_recs.sort(key=lambda m: m['angle'])

    bundle = {'subject': subject, 'emg_cols': emg_cols, 'active': recs, 'holds': holds,
              'mvc': mvc_recs, 'maps': _passive_maps(subject, passive),
              'sessions': {'S1': sorted(s1), 'S2': sorted(s2)}}
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return bundle


def session_records(bundle, session):
    """Active records of one session ('S1' / 'S2'), sorted by angle."""
    return sorted([r for r in bundle['active'] if r['session'] == session], key=lambda r: r['angle'])
