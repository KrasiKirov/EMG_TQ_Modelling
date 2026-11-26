"""
Load one subject into a "bundle" of per-trial records at 100 Hz (cached per subject in config.CACHE_DIR).

active : one dict per active trial, both sessions: idx (0-based trial index), comment, session ('S1' = lab "test",
         'S2' = "retest"), angle (rad), t (s, t[0] = 1.0 after the start-up trim), env_raw (T, 4) envelopes
         [GM, GL, SOL, TA], env_max1k (4,) maximum of the 1 kHz envelope, pos, tq (Nm), hold (the static hold
         recorded just before at the same angle, or None)
mvc    : MVC trials (angle, env_peak (4,), tq_peak)
holds  : every static hold found (idx, angle, session, mean, start, end, slope, dur, series)
maps   : passive maps per session: 'legacy' (old rule), 'static' (static holds only), 'worksheet' (YES only)

Active trials and their sessions follow the original pipeline (trial comments, ML.config.TRIAL_OVERRIDES). Static holds
are searched in every non-active, non-MVC trial (some YES trials are mislabelled) and attached to the next active trial
when the angles match within 0.05 rad.
"""

import os
import pickle
import warnings

import numpy as np
from scipy.signal import decimate

from ML.config import DATA_DIR, DOWNSAMPLE, FS, FS_RAW, PASSIVE_TORQUE_OVERRIDES, SUBJECTS, TRIAL_OVERRIDES
from ML.preprocessing.flb_reader import read_flb
from ML.preprocessing.trials import (ACTIVE_STD_THRESHOLD, POSITION_COL, TORQUE_COL, classify_trials,
                                     split_passive_segments)
from ML.preprocessing.passive_map import get_passive_torque_map, is_static_hold, static_hold_passive_map
from ML.preprocessing.emg_envelope import extract_envelope, extract_envelope_causal, detect_emg_columns

from .config import CACHE_DIR, TRIM_START

CACHE_VERSION = 2          # 2: the torque target is down-sampled before trimming (non-legacy bundles)
HOLD_ANGLE_TOL = 0.05      # rad, hold <-> next active trial angle match
HOLD_START_S = 2.0         # s, hold 'start' = mean torque of its first 2 s
HOLD_END_S = 5.0           # s, hold 'end' = mean torque of its last 5 s
MVC_BASELINE_S = 1.0       # s, MVC baseline = median torque of the first 1 s


def _decim(x):
    """1 kHz -> 100 Hz (zero-phase FIR decimation)."""
    return decimate(np.asarray(x, np.float64), DOWNSAMPLE, ftype='fir', zero_phase=True)


def _split_trials(trials, subject):
    """(mvc, passive, active trials, S1 trial indices, S2 trial indices), as the original pipeline splits them."""
    ov = TRIAL_OVERRIDES.get(subject, {})
    test_idx, retest_idx = ov.get('test'), ov.get('retest')
    mvc, passive, active = classify_trials(trials)
    if test_idx is not None or retest_idx is not None:
        forced = {i - 1 for i in (test_idx or []) + (retest_idx or [])}
        rescued = [df for df in passive if df.attrs.get('trial_index') in forced]
        passive = [df for df in passive if df.attrs.get('trial_index') not in forced]
        active = active + rescued
    active = [df for df in active if df[TORQUE_COL].std() <= ACTIVE_STD_THRESHOLD]
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


ENVELOPES = ('zerophase', 'causal')


def _envelope_fn(envelope, lp_order=4):
    if envelope not in ENVELOPES:
        raise ValueError(f'envelope must be one of {ENVELOPES}, got {envelope!r}')
    if envelope == 'zerophase':
        return extract_envelope if lp_order == 4 else (lambda x: extract_envelope(x, lp_order=lp_order))
    return lambda x: extract_envelope_causal(x, lp_order=lp_order)


def _process_active(df, emg_cols, session, envelope='zerophase', trim_end_s=0.0, lp_order=4):
    """One active trial at 100 Hz.

    envelope   'zerophase' (forward-backward filters) or 'causal' (forward-only; inputs are plain 1-in-10 samples,
               so no input uses later data; the torque target keeps the zero-phase decimation)
               Outside the legacy setting the target is down-sampled over the whole trial and then trimmed, so every
               kept target value has real torque on both sides of its ~0.1 s zero-phase filter (deviation 2026-11-23).
               The legacy setting (zero-phase envelope, no end trim) keeps the original order, for reproducibility.
    trim_end_s seconds dropped at the end, besides the first TRIM_START samples
    lp_order   order of the 2 Hz envelope low-pass (both envelopes; the band-pass stays 4th order)
    With the defaults env_max1k is the whole-trial maximum (as the original pipeline), otherwise over kept samples.
    """
    env1k = np.column_stack([_envelope_fn(envelope, lp_order)(df[c].values) for c in emg_cols])
    pos = df[POSITION_COL].values
    tq = df[TORQUE_COL].values
    s = TRIM_START
    e = len(tq) - int(round(trim_end_s * FS_RAW))
    if envelope == 'causal':
        env, p = env1k[s:e:DOWNSAMPLE], pos[s:e:DOWNSAMPLE].astype(np.float64)
    else:
        env = np.column_stack([_decim(env1k[s:e, k]) for k in range(env1k.shape[1])])
        p = _decim(pos[s:e])
    legacy = envelope == 'zerophase' and not trim_end_s
    if legacy:
        q = _decim(tq[s:e])                      # the original pipeline: trim, then down-sample
    else:
        assert s % DOWNSAMPLE == 0
        q = _decim(tq)[s // DOWNSAMPLE:]        # down-sample, then trim: no target sits at a filter edge
    n = min(len(env), len(p), len(q))
    t = (s + np.arange(n) * DOWNSAMPLE) / FS_RAW
    return {
        'idx': int(df.attrs.get('trial_index')), 'comment': df.attrs.get('comment', ''),
        'session': session, 'angle': float(np.mean(p[:n])), 't': t,
        'env_raw': env[:n], 'env_max1k': env1k.max(axis=0) if legacy else env1k[s:e].max(axis=0),
        'pos': p[:n], 'tq': q[:n], 'hold': None, 'envelope': envelope, 'trim_end_s': float(trim_end_s),
        'lp_order': int(lp_order),
    }


def _find_holds(trials, exclude_idx):
    """Static passive holds in every non-active, non-MVC trial."""
    holds = []
    for df in trials:
        i = int(df.attrs.get('trial_index'))
        if i in exclude_idx or df.attrs.get('comment', '').lower().strip().startswith('mvc'):
            continue
        for seg in split_passive_segments(df):
            if not is_static_hold(seg, fs=FS_RAW):
                continue
            p, q = seg[POSITION_COL].values, seg[TORQUE_COL].values
            q100 = _decim(q)
            tt = np.arange(len(q100)) / FS
            slope = float(np.polyfit(tt, q100, 1)[0]) if len(q100) > 10 else 0.0
            holds.append({
                'idx': i, 'comment': df.attrs.get('comment', ''), 'angle': float(p.mean()),
                'mean': float(q.mean()), 'start': float(q[:int(HOLD_START_S * FS_RAW)].mean()),
                'end': float(q[-int(HOLD_END_S * FS_RAW):].mean()),
                'slope': slope, 'dur': len(q) / FS_RAW, 'series': q100.astype(np.float32),
            })
    return holds


def _static_map(trials):
    """Static-hold map, or the legacy map when fewer than 2 static holds are found."""
    entries = static_hold_passive_map(trials)
    return entries if len(entries) >= 2 else get_passive_torque_map(trials)


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
        for name, fn in (('legacy', get_passive_torque_map), ('static', _static_map)):
            m1 = fn(p1)
            m2 = fn(p2) if p2 else m1
            out[name] = {'S1': m1, 'S2': m2}
    if ov:   # the pipeline uses the worksheet for both rules when overrides exist
        out['legacy'] = dict(out['worksheet'])
        out['static'] = dict(out['worksheet'])
    return out


def load_subject(subject, use_cache=True, envelope='zerophase', trim_end_s=0.0, lp_order=4):
    """The processed bundle for ``subject`` (see the module docstring and ``_process_active``)."""
    _envelope_fn(envelope, lp_order)
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag = '' if (envelope == 'zerophase' and not trim_end_s) else f'_{envelope}_end{trim_end_s:g}'
    tag += '' if lp_order == 4 else f'_lp{lp_order}'
    path = os.path.join(CACHE_DIR, f'{subject}_bundle_v{CACHE_VERSION}{tag}.pkl')
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
            recs.append(_process_active(df, emg_cols, 'S1' if i in s1 else 'S2', envelope, trim_end_s, lp_order))
    recs.sort(key=lambda r: r['idx'])

    holds = _find_holds(trials, exclude_idx={r['idx'] for r in recs})
    for h in holds:
        nxt = [r for r in recs if r['idx'] > h['idx']]
        h['session'] = nxt[0]['session'] if nxt else 'S2'
        if nxt and abs(nxt[0]['angle'] - h['angle']) <= HOLD_ANGLE_TOL and nxt[0]['hold'] is None:
            nxt[0]['hold'] = {k: h[k] for k in ('idx', 'angle', 'mean', 'start', 'end', 'slope', 'dur')}

    mvc_recs = []
    for df in mvc:
        env1k = np.column_stack([_envelope_fn(envelope, lp_order)(df[c].values) for c in emg_cols])
        q = df[TORQUE_COL].values
        base = float(np.median(q[:int(MVC_BASELINE_S * FS_RAW)]))
        mvc_recs.append({'idx': int(df.attrs.get('trial_index')), 'angle': float(df[POSITION_COL].mean()),
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
