"""Run settings: the defaults, the retired settings (overlap 2026-11-24; balance, input_step, pf_extra, rest_weight
2026-11-26) and spec loading."""
import pytest

from EXPERIMENTS.pipeline import run as PR
from EXPERIMENTS.pipeline import settings as ST
from EXPERIMENTS.pipeline.spec import CONFIGS, SpecError

BASE_2026_11_26 = dict(
    passive='legacy', emg='trainmax', units=8, layers=2, dropout=0.3, model='lstm',
    window=20, lr=None, train_protocol='orig', extra_input=None, shuffle=True, envelope='zerophase',
    trim_end_s=0.0, state='reset', lp_order=4, val_gap=False, metadata_passive=False,
    chunk=None, clipnorm=None, scale_part='train_val', batch=None)


def test_base_frozen():
    assert PR.BASE == BASE_2026_11_26
    assert list(PR.BASE) == list(BASE_2026_11_26)


def test_every_committed_spec_validates():
    for p in sorted(CONFIGS.glob('*.json')):
        ST.load_config(p.stem)


@pytest.mark.parametrize('key,value', [('passive', 'drift'), ('emg', 'agc_soft'), ('model', 'moe')])
def test_removed_value_is_rejected_by_name(key, value):
    with pytest.raises(SpecError, match=key):
        ST.validate(dict(ST.BASE, **{key: value}))


def test_unknown_key_is_rejected():
    with pytest.raises(SpecError, match='typo_key'):
        ST.validate(dict(ST.BASE, typo_key=1))


@pytest.mark.parametrize('over', [dict(model='ridge'), dict(extra_input='passive_trial_value')])
def test_carry_limits(over):
    with pytest.raises(SpecError, match='carry'):
        ST.validate(dict(ST.BASE, state='carry', **over))


def test_run_reexports_settings():
    assert PR.BASE is ST.BASE and PR.load_config is ST.load_config


def test_base_has_no_retired_keys():
    assert not set(ST.RETIRED) & set(ST.BASE)


def test_strip_retired_accepts_full_overlap_values():
    assert ST.strip_retired(dict(ST.BASE, train_stride=1, epoch_windows='full')) == ST.BASE
    assert ST.strip_retired(dict(ST.BASE, epoch_windows=None)) == ST.BASE


def test_strip_retired_accepts_plain_mse():
    assert ST.strip_retired(dict(ST.BASE, balance='none')) == ST.BASE


def test_strip_retired_refuses_balance_scale():
    with pytest.raises(SpecError, match='balance was retired on 2026-11-26'):
        ST.strip_retired(dict(ST.BASE, balance='scale'))


def test_strip_retired_input_step():
    assert ST.strip_retired(dict(ST.BASE, input_step=1)) == ST.BASE
    with pytest.raises(SpecError, match='input_step was retired on 2026-11-26'):
        ST.strip_retired(dict(ST.BASE, input_step=5))


@pytest.mark.parametrize('over', [dict(train_stride=5), dict(epoch_windows=1000)])
def test_strip_retired_refuses_overlap(over):
    with pytest.raises(SpecError, match='retired on 2026-11-24'):
        ST.strip_retired(dict(ST.BASE, **over))


def test_full_overlap_spec_still_loads():
    cfg = ST.load_config('FE_cell_w20_full')[1]
    assert 'train_stride' not in cfg and 'epoch_windows' not in cfg and cfg['window'] == 20


def test_strip_retired_pf_extra_and_rest_weight():
    assert ST.strip_retired(dict(ST.BASE, pf_extra=None, rest_weight=None)) == ST.BASE
    for over in (dict(pf_extra=[2, 3.0]), dict(rest_weight=0.0)):
        with pytest.raises(SpecError, match='retired on 2026-11-26'):
            ST.strip_retired(dict(ST.BASE, **over))
