"""Run settings: the defaults are frozen (final runs are checked against config_hash(BASE + changes))."""
import pytest

from EXPERIMENTS.pipeline import run as PR
from EXPERIMENTS.pipeline import settings as ST
from EXPERIMENTS.pipeline.spec import CONFIGS, SpecError

BASE_2026_11_22 = dict(
    passive='legacy', emg='trainmax', units=8, layers=2, dropout=0.3, model='lstm', balance='none', pf_extra=None,
    window=20, lr=None, train_protocol='orig', rest_weight=None, extra_input=None, shuffle=True, envelope='zerophase',
    trim_end_s=0.0, input_step=1, train_stride=1, state='reset', lp_order=4, val_gap=False, metadata_passive=False,
    chunk=None, clipnorm=None, epoch_windows=None, scale_part='train_val', batch=None)


def test_base_frozen():
    assert PR.BASE == BASE_2026_11_22
    assert list(PR.BASE) == list(BASE_2026_11_22)


def test_every_committed_spec_validates():
    for p in sorted(CONFIGS.glob('*.json')):
        ST.load_config(p.stem)


@pytest.mark.parametrize('key,value', [('passive', 'drift'), ('emg', 'agc_soft'), ('balance', 'weights'),
                                       ('model', 'moe'), ('pf_extra', [2, 3.0]), ('rest_weight', 0.0)])
def test_removed_value_is_rejected_by_name(key, value):
    with pytest.raises(SpecError, match=key):
        ST.validate(dict(ST.BASE, **{key: value}))


def test_unknown_key_is_rejected():
    with pytest.raises(SpecError, match='typo_key'):
        ST.validate(dict(ST.BASE, typo_key=1))


@pytest.mark.parametrize('over', [dict(model='ridge'), dict(extra_input='passive_trial_value'), dict(train_stride=5),
                                  dict(epoch_windows='full')])
def test_carry_limits(over):
    with pytest.raises(SpecError, match='carry'):
        ST.validate(dict(ST.BASE, state='carry', **over))


def test_run_reexports_settings():
    assert PR.BASE is ST.BASE and PR.load_config is ST.load_config
