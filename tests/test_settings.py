"""Run settings: the defaults are frozen (final runs are checked against config_hash(BASE + changes))."""
from EXPERIMENTS.pipeline import run as PR

BASE_2026_11_22 = dict(
    passive='legacy', emg='trainmax', units=8, layers=2, dropout=0.3, model='lstm', balance='none', pf_extra=None,
    window=20, lr=None, train_protocol='orig', rest_weight=None, extra_input=None, shuffle=True, envelope='zerophase',
    trim_end_s=0.0, input_step=1, train_stride=1, state='reset', lp_order=4, val_gap=False, metadata_passive=False,
    chunk=None, clipnorm=None, epoch_windows=None, scale_part='train_val', batch=None)


def test_base_frozen():
    assert PR.BASE == BASE_2026_11_22
    assert list(PR.BASE) == list(BASE_2026_11_22)
