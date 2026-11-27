"""ML.dataset: EMG scaling on the training part only, the split's purge gap, every window kept, common targets."""
import copy

import numpy as np
import pytest

from EXPERIMENTS.pipeline.fitters import drop_last_val_window
from ML import config as RC
from ML.dataset import splits as SP
from ML.dataset.emg_norm import normalise

def _recs(b, sess='S1'):
    return [r for r in b['active'] if r['session'] == sess]


# --- EMG scale (D20) --------------------------------------------------------------------------------------------

def test_scale_from_train_part_only(synthetic_bundle):
    b = synthetic_bundle
    tr = _recs(b)
    _, scale = normalise(tr, b['active'], window=20)
    b2 = copy.deepcopy(b)
    for r in b2['active']:
        cut = int((len(r['t']) - 19) * RC.TRAIN_FRAC) + 19
        if r['session'] == 'S1':
            r['env_raw'][cut:] *= 50.0           # validation block
        else:
            r['env_raw'] *= 50.0                  # test session
    _, scale2 = normalise(_recs(b2), b2['active'], window=20)
    assert np.allclose(scale, scale2)


def test_scale_is_train_part_maximum(synthetic_bundle):
    b = synthetic_bundle
    tr = _recs(b)
    feats, scale = normalise(tr, b['active'], window=20)
    want = np.max(np.stack([r['env_raw'][:int((len(r['t']) - 19) * RC.TRAIN_FRAC) + 19].max(axis=0) for r in tr]),
                  axis=0)
    assert np.allclose(scale, want)
    assert np.allclose(feats[tr[0]['idx']], tr[0]['env_raw'] / want)


# --- splits: purge gap, every window, common targets (D2, D16, D24) ----------------------------------------------------

def _build(b, window):
    recs = b['active']
    feats = {r['idx']: r['env_raw'] for r in recs}
    targets = {r['idx']: r['tq'] for r in recs}
    return SP.build(recs, feats, targets, window=window)


@pytest.mark.parametrize('window', [20, 1000])
def test_purge_gap_one_window_span(synthetic_bundle, window):
    d = _build(synthetic_bundle, window)
    for i in np.unique(d['meta_train']['trial']):
        last_tr = d['meta_train']['last'][d['meta_train']['trial'] == i].max()
        first_va = d['meta_val']['last'][d['meta_val']['trial'] == i].min()
        assert first_va - last_tr == (window - 1) + 1


def test_build_keeps_every_training_window(synthetic_bundle):
    d = _build(synthetic_bundle, 20)
    assert 'n_train_full' not in d
    m = d['meta_train']
    for i in np.unique(m['trial']):
        assert np.all(np.diff(m['last'][m['trial'] == i]) == 1)


def test_common_val_targets(synthetic_bundle):
    keys = []
    for w in (20, 1000):
        d = drop_last_val_window(_build(synthetic_bundle, w))
        m = SP.common_mask(d['meta_val'], {r['idx']: len(r['t']) for r in synthetic_bundle['active']})
        keys.append(set(zip(d['meta_val']['trial'][m].tolist(), d['meta_val']['last'][m].tolist())))
    assert keys[0] == keys[1] and len(keys[0]) > 0


def test_d24_targets_every_trial(synthetic_bundle):
    recs = _recs(synthetic_bundle)
    SP.check_common_targets(recs)                             # 90 s trials pass
    short = copy.deepcopy(recs)
    short[3]['t'] = short[3]['t'][:6000]
    with pytest.raises(ValueError, match=f"trial {short[3]['idx']}"):
        SP.check_common_targets(short)
