import numpy as np
from EXPERIMENTS.pipeline import scoring as SC

t = np.arange(0, 90, 0.01)
y = 5 * np.sin(t)


def test_metrics_perfect():
    m = SC.metrics(y, y.copy())
    assert m['r2'] == 1 and m['mae'] == 0


def test_breakdown_separates_drift_from_shape():
    p = 0.5 * y + 2                                  # pure gain+offset drift
    row, shown = SC.score_trial(y, p, t, {'type': 'none'})
    assert row['err_raw_mae'] > 1 and row['err_after_gain_offset_mae'] < 1e-6
    assert np.allclose(shown, p)                     # default path never changes predictions
    assert np.isclose(row['mae'], np.mean(abs(p - y)))   # scored on the whole trial


def test_calibrated_experiment_scores_from_score_start():
    p = 0.5 * y + 2
    row, shown = SC.score_trial(y, p, t, {'type': 'gain_offset', 'seconds': 10.0, 'score_from_s': 30.0})
    assert row['mae'] < 1e-6 and np.allclose(shown, y, atol=1e-6)


def test_gain_offset_matches_calibration_test_rule():
    import sys; sys.path.insert(0, 'EXPERIMENTS/analysis')
    from calibration_test import correct
    rng = np.random.default_rng(0); p = y + rng.normal(0, 1, y.size)
    m = t < 10
    assert np.allclose(SC.gain_offset(p[m], y[m], p), correct(p[m], y[m], p, 'gain+offset'))
