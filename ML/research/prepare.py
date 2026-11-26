"""Build the windowed train / validation / test data for one subject (used by EXPERIMENTS/pipeline/run.py).

Steps: load the subject (cached), normalise the EMG on training data, find rest periods from the EMG, build the
target (measured torque minus the passive torque), cut windows.
"""

import copy

import numpy as np

from ML.config import WINDOW
from ML.research import splits
from ML.research.data import load_subject
from ML.research.emg_norm import normalise, push_directions
from ML.research.passive import passive_torque
from ML.research.rest_flags import rest_flags


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
    spec = splits.SESSION_SPLITS[protocol]
    train_recs = [r for r in recs if r['session'] in spec['train']]
    window = cfg.get('window', WINDOW)

    feats, emg_info = normalise(cfg['emg'], bundle, train_recs, recs, window=window,
                                scale_part=cfg.get('scale_part', 'train_val'))
    base_env, _ = normalise('trainmax', bundle, train_recs, recs, window=window)
    if cfg['passive'] == 'none':              # total torque: nothing subtracted
        targets, pinfo = {r['idx']: r['tq'].copy() for r in recs}, {'kind': 'none'}
    else:
        targets = {r['idx']: r['tq'] - passive_torque(cfg['passive'], bundle, r) for r in recs}
        pinfo = {'kind': cfg['passive']}
    rest = rest_flags(bundle)
    gap = bool(cfg.get('val_gap', False))
    data = splits.build(protocol, recs, feats, targets, window=window, rest=rest, val_gap=gap)
    if cfg.get('state', 'reset') == 'carry':
        data['streams'] = splits.streams(protocol, recs, feats, targets, window=window, val_gap=gap)
    data['trial_len'] = {r['idx']: len(r['t']) for r in recs}
    data['train_recs'] = train_recs
    details = {'passive': pinfo, 'emg': {k: v for k, v in emg_info.items() if k in ('kind', 'rmin_frac', 'transductive', 'auto_decision', 'scale_part')},
               'push_directions': push_directions(bundle, train_recs, base_env, window=window),
               'n_windows': {p: int(len(data[f'y_{p}'])) for p in ('train', 'val', 'test')},
               'rest_frac_train': float(np.mean(data['meta_train']['rest']))}
    return data, details

