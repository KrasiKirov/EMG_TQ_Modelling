import copy
import json

import numpy as np
import pytest

from EXPERIMENTS.pipeline import run as PR
from ML.research import config as RC, splits as SP
from ML.research.emg_norm import normalise

UNIT = dict(batch=256, lr=0.003, lr_factor=0.5, lr_patience=6, early=50, epochs=2)


def _recs(b, sess='S1'):
    return [r for r in b['active'] if r['session'] == sess]


# --- EMG scale (D20) --------------------------------------------------------------------------------------------

def test_scale_from_train_part_only(synthetic_bundle):
    b = synthetic_bundle
    tr = _recs(b)
    _, info = normalise('trainmax', b, tr, b['active'], window=20, scale_part='train')
    b2 = copy.deepcopy(b)
    for r in b2['active']:
        cut = int((len(r['t']) - 19) * RC.TRAIN_FRAC) + 19
        if r['session'] == 'S1':
            r['env_raw'][cut:] *= 50.0           # validation block
        else:
            r['env_raw'] *= 50.0                  # test session
        r['env_max1k'] = r['env_raw'].max(axis=0) * 1.1
    _, info2 = normalise('trainmax', b2, _recs(b2), b2['active'], window=20, scale_part='train')
    assert np.allclose(info['train_max'], info2['train_max'])


def test_scale_default_unchanged(synthetic_bundle):
    b = synthetic_bundle
    tr = _recs(b)
    feats, info = normalise('trainmax', b, tr, b['active'], window=20)
    want = np.max(np.stack([r['env_max1k'] for r in tr]), axis=0)
    assert np.allclose(info['train_max'], want)
    assert np.allclose(feats[tr[0]['idx']], tr[0]['env_raw'] / want)


# --- splits: purge gap, every window, common targets (D2, D16, D24) ----------------------------------------------------

def _build(b, window):
    recs = b['active']
    feats = {r['idx']: r['env_raw'] for r in recs}
    targets = {r['idx']: r['tq'] for r in recs}
    return SP.build('S1toS2', recs, feats, targets, window=window, val_gap=True)


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
        d = PR.drop_last_val_window(_build(synthetic_bundle, w))
        m = SP.common_mask(d['meta_val'], {r['idx']: len(r['t']) for r in synthetic_bundle['active']})
        keys.append(set(zip(d['meta_val']['trial'][m].tolist(), d['meta_val']['last'][m].tolist())))
    assert keys[0] == keys[1] and len(keys[0]) > 0


def test_d24_targets_every_trial(synthetic_bundle):
    recs = _recs(synthetic_bundle)
    SP.check_common_targets(recs, min_s=5.0)                  # 90 s trials pass
    short = copy.deepcopy(recs)
    short[3]['t'] = short[3]['t'][:6000]
    with pytest.raises(ValueError, match=f"trial {short[3]['idx']}"):
        SP.check_common_targets(short, min_s=5.0)


# --- training: clipping, logging, validation predictions -----------------------------------------

def _data(window=20):
    from ML.research.run import prepare
    rcfg = dict(passive='static', emg='trainmax', window=window, envelope='causal', trim_end_s=1.0,
                state='reset', lp_order=2, val_gap=True, metadata_passive=True, scale_part='train')
    return PR.drop_last_val_window(prepare('SYN', 'S1toS2', rcfg)[0])


def test_clipnorm_windowed_applied(patch_subject, monkeypatch):
    from ML.research import train as RT
    monkeypatch.setitem(RC.PROTOCOLS, 'unit', UNIT)
    _, info = RT.fit_predict(_data(), 'lstm8', protocol='unit', shuffle=False, clipnorm=1.0)
    assert info['clipnorm'] == 1.0 and info['optimizer_clipnorm'] == 1.0






def _cfg(**over):
    cfg = dict(PR.BASE, passive='static', envelope='causal', trim_end_s=1.0, lp_order=2, val_gap=True,
               metadata_passive=True, shuffle=False, train_protocol='unit', units=4, layers=1, dropout=0.0,
               scale_part='train', clipnorm=1.0)
    cfg.update(over)
    return cfg


@pytest.fixture
def run_unit(patch_subject, tmp_results, monkeypatch):
    from EXPERIMENTS.pipeline import provenance as PV
    monkeypatch.setitem(RC.PROTOCOLS, 'unit', UNIT)
    monkeypatch.setattr(PR, 'protocol_hash', lambda *a: 'p' * 64)
    monkeypatch.setattr(PV, 'collect', lambda **kw: dict(git_commit='abc', git_dirty=False, data_sha256={},
                                                         versions={}, threads=1, host='h', protocol_hash=None))

    def go(exp, **over):
        monkeypatch.setattr(PR, 'load_config', lambda e: ({'id': e}, _cfg(**over)))
        PR.main(['--exp', exp, '--subject', 'SYN', '--seed', '0'])
        stem = tmp_results / 'results_v2' / exp / 'SYN' / 'S1toS2' / 'seed0'
        return json.loads(stem.with_suffix('.json').read_text()), np.load(str(stem) + '_val.npz')
    return go


def test_val_mae_recorded(run_unit):
    res, v = run_unit('TS2_unit_a')
    assert np.isclose(res['train_info']['val_mae'], np.mean(np.abs(v['y_pred'] - v['y'])))
    assert v['common'].dtype == bool and v['common'].any()


def test_val_file_has_no_test_predictions(run_unit):
    _, v = run_unit('TS2_unit_b')
    assert set(np.unique(v['session']).tolist()) == {0}       # 0 = S1


def test_info_logs_clipnorm_cap_shuffle(run_unit):
    for exp, over in (('TS2_unit_w', {}), ('TS2_unit_c', dict(state='carry', chunk=200))):
        res, v = run_unit(exp, **over)
        ti = res['train_info']
        assert ti['clipnorm'] == 1.0 and ti['max_epochs'] == UNIT['epochs'] and ti['shuffle'] is False
        assert np.isclose(ti['val_mae'], np.mean(np.abs(v['y_pred'] - v['y'])))




def test_ridge_lagged_run(run_unit):
    res, v = run_unit('FE_unit_ridge', model='ridge_lagged', window=51)
    ti = res['train_info']
    assert str(ti['alpha']) in ti['val_mae_by_alpha'] and ti['n_params'] == 50
    assert np.isclose(ti['val_mae'], np.mean(np.abs(v['y_pred'] - v['y'])))
