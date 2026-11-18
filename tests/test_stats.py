import numpy as np

from EXPERIMENTS.analysis import stats as ST

P6 = ['CH', 'EG', 'HM', 'IES', 'JM', 'YES']


def test_effects_identical_zero():
    arm = {p: {0: 0.2, 1: 0.3} for p in P6}
    assert all(v == 0 for v in ST.person_effects(arm, arm).values())


def test_effects_log_ratio_of_medians():
    arm = {'A': {0: 0.2, 1: 0.4, 2: 0.3}}
    ref = {'A': {0: 0.1, 1: 0.2}}
    assert np.isclose(ST.person_effects(arm, ref)['A'], np.log(0.3 / 0.15))


def test_t_interval_known_values():
    eff = dict(zip(P6, [-0.1, -0.2, -0.3, -0.1, -0.2, -0.3]))
    m, lo, hi = ST.t_interval(eff)
    assert np.isclose(m, -0.2)
    assert np.isclose(hi - m, 2.5706 * 0.08944 / np.sqrt(6), atol=1e-4)
    assert np.isclose(m - lo, hi - m)


def test_decide_requires_6_of_6():
    eff = dict(zip(P6, [-0.2, -0.2, -0.2, -0.2, -0.2, 0.1]))
    assert ST.decide(eff) == 'no clear difference'


def test_decide_requires_5pct():
    assert ST.decide(dict.fromkeys(P6, np.log(1.03))) == 'no clear difference'
    assert ST.decide(dict.fromkeys(P6, np.log(1.06))) == 'worse'
    assert ST.decide(dict.fromkeys(P6, np.log(1 / 1.06))) == 'better'
