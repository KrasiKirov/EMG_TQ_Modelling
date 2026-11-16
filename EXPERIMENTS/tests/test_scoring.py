import numpy as np
from EXPERIMENTS.pipeline import scoring as SC

t = np.arange(0, 90, 0.01)
y = 5 * np.sin(t)


def test_metrics_perfect():
    m = SC.metrics(y, y.copy())
    assert m['r2'] == 1 and m['mae'] == 0


def test_breakdown_separates_drift_from_shape():
    p = 0.5 * y + 2                                  # pure gain+offset drift
    row, shown = SC.score_trial(y, p, t)
    assert row['err_raw_mae'] > 1 and row['err_after_gain_offset_mae'] < 1e-6
    assert np.allclose(shown, p)                     # scoring never changes predictions
    assert np.isclose(row['mae'], np.mean(abs(p - y)))   # scored on the whole trial


def test_score_trial_takes_no_calibration():
    import pytest
    with pytest.raises(TypeError):
        SC.score_trial(y, y, t, {'type': 'none'})       # old 4-argument call must fail, not be read as the breakdown
