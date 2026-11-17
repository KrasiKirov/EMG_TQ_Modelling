"""Build the windowed train / validation / test data for one subject and protocol (used by EXPERIMENTS/pipeline/run.py).

Steps: load the subject (cached), normalise the EMG on training data, find rest periods from the EMG, build the
target (measured torque minus a passive model), cut windows.
"""

import copy

import numpy as np

from ML.research import config as C
from ML.research import splits
from ML.research import truth
from ML.research.data import load_subject
from ML.research.emg_norm import normalise, push_directions
from ML.research.passive import PassiveModel, rest_segments


def apply_metadata_passive(bundle, subject):
    """Fill empty static passive maps from ML.config.METADATA_PASSIVE (IES only); a measured map is never replaced."""
    from ML.config import METADATA_PASSIVE
    table = METADATA_PASSIVE.get(subject)
    st = bundle['maps']['static']
    if not table or any(len(st.get(s) or []) for s in ('S1', 'S2')):
        return bundle
    out = copy.copy(bundle)
    out['maps'] = dict(bundle['maps'], static={'S1': sorted(table), 'S2': sorted(table)})
    return out


def prepare(subject, protocol, cfg):
    """Return (data, details): the windowed arrays of splits.build and a summary of the choices made."""
    bundle = load_subject(subject, envelope=cfg.get('envelope', 'zerophase'), trim_end_s=cfg.get('trim_end_s', 0.0),
                          lp_order=cfg.get('lp_order', 4))
    if cfg.get('metadata_passive'):
        bundle = apply_metadata_passive(bundle, subject)
    recs = bundle['active']
    spec = splits.PROTOCOLS[protocol]
    train_recs = [r for r in recs if r['session'] in spec['train']]
    tf_ = 0.6 if protocol == 'pooled' else C.TRAIN_FRAC
    val_end = 0.8 if protocol == 'pooled' else 1.0
    window = cfg.get('window', C.WINDOW)

    feats, emg_info = normalise(cfg['emg'], bundle, train_recs, recs, train_frac=tf_, val_end_frac=val_end,
                                window=window)
    base_env, _ = normalise('trainmax', bundle, train_recs, recs, train_frac=tf_, val_end_frac=val_end,
                            window=window)
    tr_segs, va_segs = rest_segments(train_recs, base_env, train_frac=tf_, val_end_frac=val_end, window=window)
    if cfg['passive'] == 'none':              # total torque: nothing subtracted
        targets, pinfo = {r['idx']: r['tq'].copy() for r in recs}, {'kind': 'none'}
    elif cfg['passive'] == 'restanchor':
        targets, pinfo = restanchor_targets(bundle, protocol)
    else:
        pm = PassiveModel(cfg['passive'], bundle).fit(train_recs, tr_segs, va_segs)
        targets, pinfo = {r['idx']: r['tq'] - pm.predict(r) for r in recs}, pm.describe()
    rest = rest_flags(bundle, protocol)
    step = cfg.get('input_step', 1)
    gap = bool(cfg.get('val_gap', False))
    data = splits.build(protocol, recs, feats, targets, window=window, rest=rest, step=step,
                        train_stride=cfg.get('train_stride', 1), val_gap=gap)
    if cfg.get('state', 'reset') == 'carry':
        data['streams'] = splits.streams(protocol, recs, feats, targets, window=window, step=step, val_gap=gap)
    details = {'passive': pinfo, 'emg': {k: v for k, v in emg_info.items() if k in ('kind', 'rmin_frac', 'transductive', 'auto_decision')},
               'push_directions': push_directions(bundle, train_recs, base_env, train_frac=tf_, window=window),
               'n_windows': {p: int(len(data[f'y_{p}'])) for p in ('train', 'val', 'test')},
               'rest_frac_train': float(np.mean(data['meta_train']['rest']))}
    return data, details


def restanchor_targets(bundle, protocol):
    """Target = measured torque minus the in-trial passive baseline through the EMG rest segments (truth.py).
    Training sessions only search the training part of each trial, so no test torque enters a label."""
    spec = splits.PROTOCOLS[protocol]
    targets, n_segs = {}, {}
    for sess in ('S1', 'S2'):
        limit = 0.8 if (protocol == 'pooled') else None
        segs = truth.session_rest(bundle, sess, max_frac=limit if sess in spec['train'] else None)
        for r in bundle['active']:
            if r['session'] == sess:
                targets[r['idx']] = r['tq'] - truth.baseline(r, segs[r['idx']])
                n_segs[r['idx']] = len(segs[r['idx']])
    return targets, {'kind': 'restanchor', 'sigma_s': truth.SIGMA_S, 'n_rest_segments_per_trial': n_segs}


def rest_flags(bundle, protocol):
    """{trial idx: (T,) bool} EMG-detected rest per sample, for the rest_weight option and rest-window metrics."""
    spec = splits.PROTOCOLS[protocol]
    out = {}
    for sess in ('S1', 'S2'):
        limit = 0.8 if (protocol == 'pooled' and sess in spec['train']) else None
        segs = truth.session_rest(bundle, sess, max_frac=limit)
        for r in bundle['active']:
            if r['session'] == sess:
                out[r['idx']] = truth.rest_mask(len(r['tq']), segs[r['idx']])
    return out
