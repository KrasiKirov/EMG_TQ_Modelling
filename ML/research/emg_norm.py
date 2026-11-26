"""
EMG normalisation options; each returns {trial idx: (T, 4) features}.

Statistics come from the training session only (mvc_global: the MVC trials). The per-channel maximum uses the
training and validation parts of the training trials, as the original pipeline; the auto-gain priors use the
training part only.

trainmax     ÷ largest 1 kHz envelope value per channel in the training-session trials (original pipeline)
mvc_global   ÷ largest MVC envelope (any angle)
agc_auto     trainmax, then causal adaptive gain control (ML.preprocessing.emg_agc, rmin = 0.35 x prior range),
             unless the training session pushes both ways (> +2 Nm and < -2 Nm, see push_directions): 'trainmax'
train_p99    ÷ 99th percentile over the training part of the training-session trials
session_p99  each session ÷ its own 99th percentile (test session: all its EMG, no torque). Transductive, after
             AdaBN (Du et al. 2017, Sensors 17:458)
"""

import numpy as np

from ML.preprocessing import emg_agc

from .config import TRAIN_FRAC, WINDOW

AUTO_DIR_NM = 2.0   # Nm, push-direction threshold of the 'agc_auto' rule (as in report 1)


def _train_part(rec, train_frac=TRAIN_FRAC, window=WINDOW):
    n_win = len(rec['t']) - window + 1
    return int(n_win * train_frac) + window - 1


KINDS = ('trainmax', 'agc_auto', 'train_p99', 'session_p99', 'mvc_global')


def normalise(kind, bundle, train_recs, all_recs, train_frac=TRAIN_FRAC, window=WINDOW, scale_part='train_val'):
    """Normalised envelopes for every record in ``all_recs``; statistics from ``train_recs``.

    scale_part  : 'train_val' (original) = whole-trial 1 kHz maximum of each training trial; 'train' = maximum of the
                  100 Hz envelope over the training part of each training trial only (before the validation block),
                  so validation never shapes the scale.
    ``kind`` 'agc' (adaptive gain control, always on) is the branch agc_auto takes; it is not a run setting.
    """
    if scale_part == 'train':
        emg_max = np.max(np.stack([r['env_raw'][:_train_part(r, train_frac, window)].max(axis=0)
                                   for r in train_recs]), axis=0)
    elif scale_part != 'train_val':
        raise ValueError(f'scale_part must be train or train_val, got {scale_part!r}')
    else:
        emg_max = np.max(np.stack([r['env_max1k'] for r in train_recs]), axis=0)
    emg_max = np.where(emg_max > 0, emg_max, 1.0)
    base = {r['idx']: r['env_raw'] / emg_max for r in all_recs}
    info = {'kind': kind, 'train_max': emg_max.tolist(), 'scale_part': scale_part}

    if kind == 'trainmax':
        return base, info
    if kind == 'agc_auto':
        dirs = push_directions(bundle, train_recs, base, train_frac, window)
        bidirectional = max(dirs) > AUTO_DIR_NM and min(dirs) < -AUTO_DIR_NM
        out, sub = normalise('trainmax' if bidirectional else 'agc', bundle, train_recs, all_recs,
                             train_frac, window, scale_part)
        sub.update(kind='agc_auto', auto_decision='AGC OFF (pushes in both directions)' if bidirectional
                   else 'AGC ON', push_directions=dirs)
        return out, sub
    if kind == 'agc':
        priors = emg_agc.agc_priors([base[r['idx']][:_train_part(r, train_frac, window)] for r in train_recs])
        info.update(rmin_frac=emg_agc.RMIN_FRAC, prior_lo=priors[0].tolist(), prior_hi=priors[1].tolist())
        return {i: emg_agc.apply_agc(e, priors[0], priors[1])[0] for i, e in base.items()}, info
    if kind == 'train_p99':
        src = np.concatenate([r['env_raw'][:_train_part(r, train_frac, window)] for r in train_recs])
        sc = np.maximum(np.percentile(src, 99, axis=0), 1e-9)
        info['train_p99'] = sc.tolist()
        return {r['idx']: r['env_raw'] / sc for r in all_recs}, info
    if kind == 'session_p99':
        train_ids = {r['idx'] for r in train_recs}
        out, scales = {}, {}
        for sess in sorted({r['session'] for r in all_recs}):
            rs = [r for r in all_recs if r['session'] == sess]
            if any(r['idx'] in train_ids for r in rs):
                src = [r['env_raw'][:_train_part(r, train_frac, window)] for r in rs if r['idx'] in train_ids]
            else:
                src = [r['env_raw'] for r in rs]
            sc = np.maximum(np.percentile(np.concatenate(src), 99, axis=0), 1e-9)
            scales[sess] = sc.tolist()
            for r in rs:
                out[r['idx']] = r['env_raw'] / sc
        info.update(transductive=True, session_scale=scales)
        return out, info
    if kind == 'mvc_global':
        peak = np.max(np.stack([m['env_peak'] for m in bundle['mvc']]), axis=0)
        info['mvc_peak'] = peak.tolist()
        return {r['idx']: r['env_raw'] / peak for r in all_recs}, info
    raise ValueError(kind)


def push_directions(bundle, train_recs, env_norm, train_frac=TRAIN_FRAC, window=WINDOW):
    """Push direction per training trial (Nm): torque in high-EMG phases minus torque in low-EMG phases."""
    dirs = []
    for r in train_recs:
        cut = _train_part(r, train_frac, window)
        act = env_norm[r['idx']][:cut].sum(axis=1)
        tq = r['tq'][:cut]
        hi, lo = act >= np.percentile(act, 80), act <= np.percentile(act, 20)
        dirs.append(float(tq[hi].mean() - tq[lo].mean()))
    return dirs
