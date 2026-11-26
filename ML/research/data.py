"""
Load one subject into a "bundle" of per-trial records at 100 Hz (cached per subject and envelope in ML.config.CACHE_DIR).

active : one dict per active trial, both sessions: idx (0-based trial index), comment, session ('S1' = lab "test",
         'S2' = "retest"), angle (rad), t (s, t[0] = 1.0 after the start-up trim), env_raw (T, 4) envelopes
         [GM, GL, SOL, TA], env_max1k (4,) maximum of the 1 kHz envelope over the kept samples, pos (rad), tq (Nm),
         envelope
maps   : {'static': {'S1': map, 'S2': map}}: each session's passive map from its static holds (YES: the worksheet
         values in ML.config.PASSIVE_TORQUE_OVERRIDES)

Active trials and their sessions follow the original pipeline (trial comments, ML.config.TRIAL_OVERRIDES).
"""

import os
import pickle
import warnings

import numpy as np
from scipy.signal import decimate

from ML.config import (CACHE_DIR, DATA_DIR, DOWNSAMPLE, FS_RAW, PASSIVE_TORQUE_OVERRIDES, SUBJECTS,
                       TRIAL_OVERRIDES, TRIM_END_S, TRIM_START)
from ML.preprocessing.flb_reader import read_flb
from ML.preprocessing.trials import ACTIVE_STD_THRESHOLD, POSITION_COL, TORQUE_COL, classify_trials
from ML.preprocessing.passive_map import get_passive_torque_map, static_hold_passive_map
from ML.preprocessing.emg_envelope import LP_ORDER, extract_envelope, extract_envelope_causal, detect_emg_columns

CACHE_VERSION = 2          # 2: the torque target is down-sampled before trimming
ENVELOPES = ('zerophase', 'causal')


def _decim(x):
    """1 kHz -> 100 Hz (zero-phase FIR decimation)."""
    return decimate(np.asarray(x, np.float64), DOWNSAMPLE, ftype='fir', zero_phase=True)


def _split_trials(trials, subject):
    """(passive trials, active trials, S1 trial indices, S2 trial indices), as the original pipeline splits them."""
    ov = TRIAL_OVERRIDES.get(subject, {})
    test_idx, retest_idx = ov.get('test'), ov.get('retest')
    _, passive, active = classify_trials(trials)
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
    return passive, active, s1, s2


def _process_active(df, emg_cols, session, envelope):
    """One active trial at 100 Hz, without the first TRIM_START samples and the last TRIM_END_S seconds.

    envelope   'zerophase' (forward-backward filters) or 'causal' (forward-only; inputs are plain 1-in-10 samples,
               so no input uses later data; the torque target keeps the zero-phase decimation)
    The target is down-sampled over the whole trial and then trimmed, so every kept target value has real torque on
    both sides of its ~0.1 s zero-phase filter.
    """
    fn = extract_envelope_causal if envelope == 'causal' else extract_envelope
    env1k = np.column_stack([fn(df[c].values) for c in emg_cols])
    pos = df[POSITION_COL].values
    tq = df[TORQUE_COL].values
    s = TRIM_START
    e = len(tq) - int(round(TRIM_END_S * FS_RAW))
    if envelope == 'causal':
        env, p = env1k[s:e:DOWNSAMPLE], pos[s:e:DOWNSAMPLE].astype(np.float64)
    else:
        env = np.column_stack([_decim(env1k[s:e, k]) for k in range(env1k.shape[1])])
        p = _decim(pos[s:e])
    assert s % DOWNSAMPLE == 0
    q = _decim(tq)[s // DOWNSAMPLE:]            # down-sample, then trim: no target sits at a filter edge
    n = min(len(env), len(p), len(q))
    t = (s + np.arange(n) * DOWNSAMPLE) / FS_RAW
    return {
        'idx': int(df.attrs.get('trial_index')), 'comment': df.attrs.get('comment', ''),
        'session': session, 'angle': float(np.mean(p[:n])), 't': t,
        'env_raw': env[:n], 'env_max1k': env1k[s:e].max(axis=0),
        'pos': p[:n], 'tq': q[:n], 'envelope': envelope,
    }


def _static_map(trials):
    """Static-hold map, or the original pipeline's map when fewer than 2 static holds are found."""
    entries = static_hold_passive_map(trials)
    return entries if len(entries) >= 2 else get_passive_torque_map(trials)


def _passive_maps(subject, passive_trials):
    """{'static': {'S1': map, 'S2': map}}; a subject with worksheet values (PASSIVE_TORQUE_OVERRIDES) uses those."""
    ov = PASSIVE_TORQUE_OVERRIDES.get(subject)
    if ov:
        return {'static': {'S1': sorted(ov['test']), 'S2': sorted(ov.get('retest', ov['test']))}}
    p1 = [df for df in passive_trials if df.attrs.get('session') != 'retest']
    p2 = [df for df in passive_trials if df.attrs.get('session') == 'retest']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m1 = _static_map(p1)
        m2 = _static_map(p2) if p2 else m1
    return {'static': {'S1': m1, 'S2': m2}}


def load_subject(subject, envelope='causal', use_cache=True):
    """The processed bundle for ``subject`` (see the module docstring and ``_process_active``)."""
    if envelope not in ENVELOPES:
        raise ValueError(f'envelope must be one of {ENVELOPES}, got {envelope!r}')
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag = f'_{envelope}_end{TRIM_END_S:g}_lp{LP_ORDER}'
    path = os.path.join(CACHE_DIR, f'{subject}_bundle_v{CACHE_VERSION}{tag}.pkl')
    if use_cache and os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    fn, sid = SUBJECTS[subject]
    trials = read_flb(os.path.join(DATA_DIR, fn), subject_id=sid)
    passive, active, s1, s2 = _split_trials(trials, subject)
    emg_cols = detect_emg_columns(active[0])

    recs = []
    for df in active:
        i = df.attrs.get('trial_index')
        if i in s1 or i in s2:
            recs.append(_process_active(df, emg_cols, 'S1' if i in s1 else 'S2', envelope))
    recs.sort(key=lambda r: r['idx'])

    bundle = {'subject': subject, 'active': recs, 'maps': _passive_maps(subject, passive)}
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return bundle
