"""Run settings: the defaults, the retired settings (overlap 2026-11-24; balance, input_step, pf_extra, rest_weight and
the study-v2-only settings 2026-11-26) and spec loading."""
import pytest

from EXPERIMENTS.pipeline import run as PR
from EXPERIMENTS.pipeline import settings as ST
from EXPERIMENTS.pipeline.spec import CONFIGS, SpecError

BASE_2026_11_26 = dict(passive='static', units=8, layers=2, dropout=0.3, model='lstm', window=20, lr=None,
                       envelope='causal', state='reset', chunk=None, batch=None)


def test_base_frozen():
    assert PR.BASE == BASE_2026_11_26
    assert list(PR.BASE) == list(BASE_2026_11_26)


def test_every_committed_spec_validates():
    for p in sorted(CONFIGS.glob('*.json')):
        ST.load_config(p.stem)


@pytest.mark.parametrize('key,value', [('passive', 'legacy'), ('model', 'ridge'), ('envelope', 'raw')])
def test_removed_value_is_rejected_by_name(key, value):
    with pytest.raises(SpecError, match=key):
        ST.validate(dict(ST.BASE, **{key: value}))


def test_unknown_key_is_rejected():
    with pytest.raises(SpecError, match='typo_key'):
        ST.validate(dict(ST.BASE, typo_key=1))


def test_carry_limits():
    with pytest.raises(SpecError, match='carry'):
        ST.validate(dict(ST.BASE, state='carry', model='ridge_lagged'))


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


V2_VALUES = dict(emg='trainmax', scale_part='train', shuffle=False, extra_input=None, trim_end_s=1.0, lp_order=2,
                 val_gap=True, metadata_passive=True, train_protocol='study400', clipnorm=1.0)


def test_strip_retired_study_v2_values():
    assert ST.strip_retired(dict(ST.BASE, **V2_VALUES)) == ST.BASE
    assert ST.strip_retired(dict(ST.BASE, model='ridge_lagged', train_protocol='orig')) == dict(ST.BASE,
                                                                                               model='ridge_lagged')


@pytest.mark.parametrize('over', [dict(emg='mvc_global'), dict(scale_part='train_val'), dict(shuffle=True),
                                  dict(extra_input='passive_trial_value'), dict(trim_end_s=0.0), dict(lp_order=4),
                                  dict(val_gap=False), dict(metadata_passive=False), dict(train_protocol='fast'),
                                  dict(clipnorm=None), dict(train_protocol='orig')])
def test_strip_retired_refuses_older_values(over):
    with pytest.raises(SpecError, match='older-options-2026-11-26'):
        ST.strip_retired(dict(ST.BASE, **over))


def test_only_study_v2_specs_and_the_template_remain():
    ids = [p.stem for p in CONFIGS.glob('*.json')]
    assert ids and all(i.startswith(('TS2_', 'FE_')) or i == 'TEMPLATE_reference' for i in ids)
