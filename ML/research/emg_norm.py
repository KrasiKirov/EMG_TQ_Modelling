"""
EMG normalisation options.  Each returns {trial idx: (T, 4) features}.

All statistics come from the TRAINING session only, except the MVC options
which use the MVC calibration trials (recorded once, at the start of the
experiment, separate from every active trial that is evaluated).

Which part of a training trial: the per-channel maximum ('trainmax', also
the starting scale of the auto-gain options) follows the original pipeline
and uses the training AND validation parts of the training trials (cross-
session protocols: the whole trial).  In the pooled protocol, whose last
20 % of every trial is test data, it uses only the first ``val_end_frac``
(= 80 %) of each trial.  The auto-gain priors use the training part only.

kinds
-----
trainmax     divide each channel by its largest 1 kHz envelope value in the
             training-session trials (the original pipeline).
mvc_global   divide each channel by its largest MVC envelope (any angle):
             EMG in "% of maximal voluntary activation".
mvc_angle    divide each channel by the MVC envelope peak at THIS trial's
             angle (linear interpolation between MVC angles).
agc          trainmax, then causal adaptive gain control (ML.preprocessing.
             emg_agc): per channel (x - floor_t) / max(top_t - floor_t, rmin),
             floor = running 10th percentile of the past 20 s, top = running
             95th percentile of the past 45 s, rmin = 0.35 x prior range.
agc_auto     report 1's 'auto' rule: 'agc' unless the training session shows
             pushes in BOTH directions (some trial's push direction > +2 Nm
             and another's < -2 Nm, see push_directions), then 'trainmax'.
             Auto-gain removes absolute EMG amplitude, which a subject who
             pushes both ways (CH) needs to tell a push from co-contraction.
             Decided from the training part of the training session only.
agc_soft     same with rmin = 0.7 x prior range: amplifies a near-silent
             channel at most ~1.4x instead of ~2.9x (less noise blow-up at
             weak angles).
session_p99  every session is divided by ITS OWN per-channel 99th percentile
             of the envelope (training session: training part only; test
             session: all of its active-trial EMG, without any torque).
             TRANSDUCTIVE: it uses unlabelled test-session EMG, i.e. the
             model needs the whole test session before it can predict.
             The unsupervised re-normalisation idea follows AdaBN (Du et
             al. 2017, Sensors 17:458), which re-estimates normalisation
             statistics on the new session's unlabelled data.

Total-torque study (EXPERIMENTS/configs/TT*.json).  Fixed, non-adaptive
amplitude scales, one per muscle, pooled across positions:
raw          no amplitude normalisation: the processed envelopes in their
             original units (mV) are fed to the network.
trainmax_tr  divide each channel by the maximum of the 100 Hz envelope over
             the TRAINING part only (first 80 % of the windows of every
             source-session trial; the validation part is excluded, unlike
             'trainmax').  Applied unchanged to validation and to the other
             session.
trainp99     the same with the 99th percentile instead of the maximum.
session_p99  (above) is the study's per-session variant: offline,
             unlabelled target-session adaptation.
A scale that is not finite or not above MIN_SCALE is replaced by 1.0 (the
channel is then left in its original units) and listed in
'guarded_channels'; with real recordings this has never happened.
"""

import numpy as np

from .d01_preprocessing import emg_agc

from . import config as C

AUTO_DIR_NM = 2.0   # Nm, push-direction threshold of the 'agc_auto' rule (as in report 1)
MIN_SCALE = 1e-9    # envelope units; smaller fixed scales are treated as zero (see 'guarded_channels')
CHANNELS = ('GM', 'GL', 'SOL', 'TA')


def _safe_scale(sc):
    """Replace non-finite or ~zero channel scales by 1.0; return (scale, guarded channel names)."""
    sc = np.asarray(sc, np.float64)
    bad = ~np.isfinite(sc) | (sc <= MIN_SCALE)
    return np.where(bad, 1.0, sc), [CHANNELS[k] for k in np.flatnonzero(bad)]


def _train_part(rec, train_frac=C.TRAIN_FRAC, window=C.WINDOW):
    n_win = len(rec['t']) - window + 1
    return int(n_win * train_frac) + window - 1


def _agc(env, prior_lo, prior_hi, rmin_frac):
    """emg_agc.apply_agc with a configurable minimum range (same algorithm)."""
    rmin = rmin_frac * np.maximum(prior_hi - prior_lo, 1e-3)
    lo, hi = emg_agc._running_lo_hi(env, prior_lo, prior_hi)
    clean, n_bad = env.copy(), 0
    for k in range(env.shape[1]):
        clean[:, k], n = emg_agc._despike(env[:, k], lo[:, k], np.maximum(hi[:, k] - lo[:, k], rmin[k]))
        n_bad += n
    if n_bad:
        lo, hi = emg_agc._running_lo_hi(clean, prior_lo, prior_hi)
    return np.clip((clean - lo) / np.maximum(hi - lo, rmin[None, :]), 0.0, emg_agc.CLIP_HI)


def normalise(kind, bundle, train_recs, all_recs, train_frac=C.TRAIN_FRAC, val_end_frac=1.0,
              window=C.WINDOW):
    """Normalised envelopes for every record in ``all_recs``.

    train_recs   : the records of the training session (statistics source)
    train_frac   : fraction of each training record that is training data
                   (auto-gain priors use only that part)
    val_end_frac : end of the validation part (pooled protocol: 0.8); the
                   maximum is taken before it.  1.0 = whole trial (1 kHz
                   envelope maximum, exactly as the original pipeline)
    window       : window length (steps) of the configuration
    """
    if val_end_frac >= 1.0:
        emg_max = np.max(np.stack([r['env_max1k'] for r in train_recs]), axis=0)
    else:
        emg_max = np.max(np.stack([r['env_raw'][:int(len(r['env_raw']) * val_end_frac)].max(axis=0)
                                   for r in train_recs]), axis=0)
    emg_max = np.where(emg_max > 0, emg_max, 1.0)
    base = {r['idx']: r['env_raw'] / emg_max for r in all_recs}
    info = {'kind': kind, 'train_max': emg_max.tolist()}

    if kind == 'trainmax':
        return base, info
    if kind == 'raw':
        return {r['idx']: r['env_raw'].copy() for r in all_recs}, {
            'kind': kind, 'scale': [1.0] * len(CHANNELS), 'scale_source': 'none (original envelope units)',
            'adaptive': False, 'transductive': False}
    if kind in ('trainmax_tr', 'trainp99'):
        src = np.concatenate([r['env_raw'][:_train_part(r, train_frac, window)] for r in train_recs])
        sc = src.max(axis=0) if kind == 'trainmax_tr' else np.percentile(src, 99, axis=0)
        sc, guarded = _safe_scale(sc)
        return {r['idx']: r['env_raw'] / sc for r in all_recs}, {
            'kind': kind, 'scale': sc.tolist(), 'guarded_channels': guarded, 'adaptive': False,
            'transductive': False, 'n_samples_scale': int(len(src)),
            'scale_source': f"{'max' if kind == 'trainmax_tr' else '99th percentile'} of the 100 Hz envelope "
                            f"over the training part of every source-session trial (positions pooled)"}
    if kind == 'agc_auto':
        dirs = push_directions(bundle, train_recs, base, train_frac, window)
        bidirectional = max(dirs) > AUTO_DIR_NM and min(dirs) < -AUTO_DIR_NM
        out, sub = normalise('trainmax' if bidirectional else 'agc', bundle, train_recs, all_recs,
                             train_frac, val_end_frac, window)
        sub.update(kind='agc_auto', auto_decision='AGC OFF (pushes in both directions)' if bidirectional
                   else 'AGC ON', push_directions=dirs)
        return out, sub
    if kind in ('agc', 'agc_soft'):
        rmin_frac = 0.35 if kind == 'agc' else 0.7
        priors = emg_agc.agc_priors([base[r['idx']][:_train_part(r, train_frac, window)] for r in train_recs])
        info.update(rmin_frac=rmin_frac, prior_lo=priors[0].tolist(), prior_hi=priors[1].tolist())
        return {i: _agc(e, priors[0], priors[1], rmin_frac) for i, e in base.items()}, info
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
        info.update(transductive=True, adaptive=False, session_scale=scales,
                    scale_source='99th percentile of the 100 Hz envelope per session, positions pooled: source '
                                 'session = training part only, target session = all of its active-trial EMG '
                                 '(no torque); offline, unlabelled target-session adaptation')
        return out, info
    if kind == 'mvc_global':
        peak = np.max(np.stack([m['env_peak'] for m in bundle['mvc']]), axis=0)
        info['mvc_peak'] = peak.tolist()
        return {r['idx']: r['env_raw'] / peak for r in all_recs}, info
    if kind == 'mvc_angle':
        ang = np.array([m['angle'] for m in bundle['mvc']])
        pk = np.stack([m['env_peak'] for m in bundle['mvc']])
        order = np.argsort(ang)
        ang, pk = ang[order], pk[order]
        out = {}
        for r in all_recs:
            p = np.array([np.interp(r['angle'], ang, pk[:, k]) for k in range(pk.shape[1])])
            out[r['idx']] = r['env_raw'] / np.maximum(p, 1e-9)
        return out, info
    raise ValueError(kind)


def push_directions(bundle, train_recs, env_norm, train_frac=C.TRAIN_FRAC, window=C.WINDOW):
    """Push direction per training trial (Nm): torque in high-EMG phases minus
    torque in low-EMG phases (same rule as the pipeline's AGC 'auto' mode)."""
    dirs = []
    for r in train_recs:
        cut = _train_part(r, train_frac, window)
        act = env_norm[r['idx']][:cut].sum(axis=1)
        tq = r['tq'][:cut]
        hi, lo = act >= np.percentile(act, 80), act <= np.percentile(act, 20)
        dirs.append(float(tq[hi].mean() - tq[lo].mean()))
    return dirs
