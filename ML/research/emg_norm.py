"""EMG scaling: each channel's envelope is divided by its largest value over the training part of the session-1 trials
(before the validation block), so neither validation nor the test session shapes the scale."""

import numpy as np

from ML.config import TRAIN_FRAC, WINDOW


def _train_part(rec, window=WINDOW):
    """Number of samples at the start of a training trial that its training windows cover."""
    n_win = len(rec['t']) - window + 1
    return int(n_win * TRAIN_FRAC) + window - 1


def normalise(train_recs, all_recs, window=WINDOW):
    """({trial idx: (T, 4) scaled envelopes} for every record in ``all_recs``, (4,) per-channel scale from
    ``train_recs``). A channel whose maximum is 0 is left unscaled."""
    emg_max = np.max(np.stack([r['env_raw'][:_train_part(r, window)].max(axis=0) for r in train_recs]), axis=0)
    emg_max = np.where(emg_max > 0, emg_max, 1.0)
    return {r['idx']: r['env_raw'] / emg_max for r in all_recs}, emg_max
