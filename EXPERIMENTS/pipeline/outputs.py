"""Saving one run: <stem>_val.npz (validation predictions), <stem>.npz (test predictions) and <stem>.json (settings,
training summary, provenance). Each file is written to a temporary name first and then renamed, so a stopped run
never leaves a half-written file under its final name."""
import json
import os

import numpy as np


def _save_npz(path, **arrays):
    tmp = f'{path}.tmp'
    with open(tmp, 'wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def write_run(stem, data, pred, val_pred, res):
    """``stem`` = results folder / 'seed<k>'; ``res`` is the run summary written as JSON (dict order kept)."""
    from ML.research.splits import common_mask
    stem = str(stem)
    mv = data['meta_val']
    _save_npz(stem + '_val.npz', y=data['y_val'].astype(np.float32),
              y_pred=val_pred.astype(np.float32), trial=mv['trial'].astype(np.int16),
              last=mv['last'].astype(np.int32), angle=mv['angle'].astype(np.float32),
              session=(mv['session'] == 'S2').astype(np.int8), t=mv['t'].astype(np.float32),
              common=common_mask(mv, data['trial_len']))
    y, meta = data['y_test'].astype(np.float64), data['meta_test']
    _save_npz(stem + '.npz', y_own_target=y.astype(np.float32), y_pred=pred.astype(np.float32),
              trial=meta['trial'].astype(np.int16), angle=meta['angle'].astype(np.float32),
              session=(meta['session'] == 'S2').astype(np.int8), t=meta['t'].astype(np.float32))
    with open(stem + '.json.tmp', 'w') as f:
        json.dump(res, f, indent=1, default=float)
    os.replace(stem + '.json.tmp', stem + '.json')
