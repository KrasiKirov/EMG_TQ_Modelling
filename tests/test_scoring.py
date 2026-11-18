import numpy as np

from EXPERIMENTS.pipeline import scoring as SC


def test_nmae_known_values():
    y, p, t = np.array([0., 2., 4.]), np.array([1., 2., 3.]), np.array([0., 1., 2.])
    row = SC.score_trial(y, p, t, t0=0.0, scale=4.0, score_from_s=0.0)
    assert np.isclose(row['mae'], 2 / 3)
    assert np.isclose(row['nmae'], 1 / 6)
    assert np.isclose(row['rmse'], np.sqrt(2 / 3))
    assert np.isclose(row['nrmse_pos'], np.sqrt(2 / 3) / 4)
    assert row['n'] == 3


def test_score_from_common_start():
    t_meas = np.arange(0, 20, 0.01)
    y = np.sin(t_meas)
    a = SC.score_trial(y[20:], y[20:] + 0.1, t_meas[20:], t0=0.0, scale=1.0, score_from_s=1.0)    # starts at 0.2 s
    b = SC.score_trial(y[100:], y[100:] + 0.1, t_meas[100:], t0=0.0, scale=1.0, score_from_s=1.0)  # starts at 1.0 s
    assert a['n'] == b['n'] == len(t_meas) - 100


def test_nrmse_pos_range():
    y = np.array([1., 3., 5., 3.])
    row = SC.score_trial(y, y + 1.0, np.arange(4.), t0=0.0, scale=5.0)
    assert np.isclose(row['nrmse_pos'], 1.0 / 4.0)


def test_person_scale_ies_metadata_target(synthetic_bundle):
    r = synthetic_bundle['active'][0]
    target = r['tq'] - 10.0 * r['angle']            # a passive model applied, no holds needed
    s = SC.person_scale(target)
    assert s > 0 and np.isclose(s, np.percentile(np.abs(target), 99))


def test_old_signature_unchanged():
    y, p, t = np.array([0., 2., 4.]), np.array([1., 2., 3.]), np.array([5., 6., 7.])
    row = SC.score_trial(y, p, t)
    assert 'nmae' not in row and np.isclose(row['mae'], 2 / 3)
