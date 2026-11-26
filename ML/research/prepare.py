"""Build the windowed train / validation / test data for one subject (used by EXPERIMENTS/pipeline/run.py).

Steps: load the subject (cached), fill IES's empty passive maps from its metadata, scale the EMG on training data,
build the target (measured torque minus the passive torque), find rest periods from the EMG, cut windows.
"""

import copy

import numpy as np

from ML.research import splits
from ML.research.data import load_subject
from ML.research.emg_norm import normalise
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
    """Return (data, details): the windowed arrays of splits.build and a summary of the choices made.

    cfg: passive ('static' or 'none'), window (steps), envelope ('causal' or 'zerophase'), state ('reset' or 'carry').
    """
    bundle = apply_metadata_passive(load_subject(subject, envelope=cfg['envelope']), subject)
    recs = bundle['active']
    spec = splits.SESSION_SPLITS[protocol]
    train_recs = [r for r in recs if r['session'] in spec['train']]
    window = cfg['window']

    feats, _ = normalise(train_recs, recs, window=window)
    if cfg['passive'] == 'none':              # total torque: nothing subtracted
        targets = {r['idx']: r['tq'].copy() for r in recs}
    else:
        targets = {r['idx']: r['tq'] - passive_torque(bundle, r) for r in recs}
    rest = rest_flags(bundle)
    data = splits.build(protocol, recs, feats, targets, window=window, rest=rest)
    if cfg['state'] == 'carry':
        data['streams'] = splits.streams(protocol, recs, feats, targets, window=window)
    data['trial_len'] = {r['idx']: len(r['t']) for r in recs}
    data['train_recs'] = train_recs
    details = {'passive': {'kind': cfg['passive']}, 'emg': {'kind': 'trainmax', 'scale_part': 'train'},
               'n_windows': {p: int(len(data[f'y_{p}'])) for p in ('train', 'val', 'test')},
               'rest_frac_train': float(np.mean(data['meta_train']['rest']))}
    return data, details
