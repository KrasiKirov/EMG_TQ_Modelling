"""Build the windowed train / validation / test data for one subject (used by EXPERIMENTS/pipeline/run.py).

Steps: load the subject (cached), scale the EMG on training data, build the target (measured torque minus the passive
torque of the trial's own session), cut windows.
"""

from ML.preprocessing.passive_map import lookup_passive_torque
from ML.research import splits
from ML.research.data import load_subject
from ML.research.emg_norm import normalise


def prepare(subject, cfg):
    """Return (data, details): the windowed arrays of splits.build and a summary of the choices made.

    cfg: passive ('static' or 'none'), window (steps), state ('reset' or 'carry').
    """
    bundle = load_subject(subject)
    recs = bundle['active']
    train_recs = [r for r in recs if r['session'] in splits.TRAIN_SESSIONS]
    window = cfg['window']

    feats, _ = normalise(train_recs, recs, window=window)
    if cfg['passive'] == 'none':              # total torque: nothing subtracted
        targets = {r['idx']: r['tq'].copy() for r in recs}
    else:
        targets = {r['idx']: r['tq'] - lookup_passive_torque(bundle['maps'][r['session']], r['pos']) for r in recs}
    data = splits.build(recs, feats, targets, window=window)
    if cfg['state'] == 'carry':
        data['streams'] = splits.streams(recs, feats, targets, window=window)
    data['trial_len'] = {r['idx']: len(r['t']) for r in recs}
    data['train_recs'] = train_recs
    details = {'passive': {'kind': cfg['passive']}, 'emg': {'kind': 'trainmax', 'scale_part': 'train'},
               'n_windows': {p: int(len(data[f'y_{p}'])) for p in ('train', 'val', 'test')}}
    return data, details
