"""Input options used by the window / training-order sweep: causal envelope, end trim, window step, training stride."""
import numpy as np
import pandas as pd

from ML.preprocessing.emg_envelope import extract_envelope, extract_envelope_causal
from ML.research import data as D, splits


def _emg(n=12000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=n) * (1 + np.sin(np.arange(n) / 700.0))


def test_causal_envelope_ignores_the_future():
    x = _emg()
    y = x.copy(); y[6000:] = _emg(seed=1)[6000:]                 # change only the future
    assert np.array_equal(extract_envelope_causal(x)[:6000], extract_envelope_causal(y)[:6000])
    assert not np.allclose(extract_envelope(x)[5000:6000], extract_envelope(y)[5000:6000])   # zero-phase looks ahead


def _trial(n=12000):
    cols = {c: _emg(n, k) for k, c in enumerate(['gm', 'gl', 'sol', 'ta'])}
    df = pd.DataFrame({**cols, D.db.POSITION_COL: np.full(n, -0.2), D.db.TORQUE_COL: np.sin(np.arange(n) / 900.0)})
    df.attrs.update(trial_index=3, comment='p1 test')
    return df


def test_default_processing_is_unchanged():
    r = D._process_active(_trial(), ['gm', 'gl', 'sol', 'ta'], 'S1')
    assert len(r['env_raw']) == 1100 and r['t'][0] == 1.0
    full = np.column_stack([extract_envelope(_trial()[c].values) for c in ['gm', 'gl', 'sol', 'ta']])
    assert np.allclose(r['env_max1k'], full.max(axis=0))       # legacy: max over the whole trial


def test_causal_with_end_trim():
    df = _trial()
    r = D._process_active(df, ['gm', 'gl', 'sol', 'ta'], 'S1', envelope='causal', trim_end_s=1.0)
    assert len(r['env_raw']) == len(r['tq']) == len(r['t']) == 1000      # 12 s - 1 s - 1 s at 100 Hz
    assert r['t'][0] == 1.0 and abs(r['t'][-1] - 10.99) < 1e-9
    kept = np.column_stack([extract_envelope_causal(df[c].values) for c in ['gm', 'gl', 'sol', 'ta']])[1000:-1000]
    assert np.allclose(r['env_max1k'], kept.max(axis=0))       # scale taken after trimming
    assert np.allclose(r['env_raw'], kept[::10])


def test_window_step_spans_more_time():
    T = 100
    feat, pos, y = np.arange(T * 4, dtype=float).reshape(T, 4), np.zeros(T), np.arange(T, dtype=float)
    X, yy, last = splits.windows(feat, pos, y, window=4, step=5)
    assert X.shape == (T - 15, 4, 5)
    assert list(X[0, :, 0]) == [0, 20, 40, 60] and yy[0] == 15 and last[0] == 15


def _recs():
    rng = np.random.default_rng(0)
    out, feats, tg = [], {}, {}
    for i, s in enumerate(['S1', 'S1', 'S2']):
        T = 300
        out.append({'idx': i, 'session': s, 'angle': 0.1 * i, 'pos': np.full(T, 0.1 * i), 't': np.arange(T) / 100})
        feats[i], tg[i] = rng.random((T, 4)), rng.random(T)
    return out, feats, tg


def test_train_stride_thins_training_windows_only():
    recs, feats, tg = _recs()
    full = splits.build('S1toS2', recs, feats, tg, window=20)
    thin = splits.build('S1toS2', recs, feats, tg, window=20, train_stride=4)
    assert len(thin['y_train']) == sum(len(range(0, k, 4)) for k in [int(281 * 0.8)] * 2)
    assert np.array_equal(thin['y_val'], full['y_val']) and np.array_equal(thin['y_test'], full['y_test'])
    assert np.array_equal(thin['y_train'][:3], full['y_train'][[0, 4, 8]])


def test_lowpass_order_changes_only_the_lowpass():
    from scipy.signal import butter, sosfilt
    x = _emg()
    e2 = extract_envelope_causal(x, lp_order=2)
    assert not np.allclose(e2[2000:], extract_envelope_causal(x)[2000:])          # differs from the 4th-order default
    # same band-pass (4th order) as the default, then a 2nd-order 2 Hz low-pass
    xb = x - x[:1000].mean()
    rect = np.abs(sosfilt(butter(4, [30 / 500, 300 / 500], 'band', output='sos'), xb))
    ref = sosfilt(butter(2, 2 / 500, 'low', output='sos'), rect)
    assert np.allclose(e2[3000:], ref[3000:], rtol=1e-3, atol=1e-6)               # equal once the start-up has settled


def test_lowpass_order_reaches_the_records():
    df = _trial()
    r4 = D._process_active(df, ['gm', 'gl', 'sol', 'ta'], 'S1', envelope='causal', trim_end_s=1.0)
    r2 = D._process_active(df, ['gm', 'gl', 'sol', 'ta'], 'S1', envelope='causal', trim_end_s=1.0, lp_order=2)
    kept = np.column_stack([extract_envelope_causal(df[c].values, lp_order=2) for c in ['gm', 'gl', 'sol', 'ta']])[1000:-1000]
    assert np.allclose(r2['env_raw'], kept[::10]) and not np.allclose(r2['env_raw'], r4['env_raw'])
    assert r2['lp_order'] == 2 and r4['lp_order'] == 4


def test_val_gap_keeps_validation_inputs_out_of_the_training_part():
    recs, feats, tg = _recs()
    for window, step in ((20, 1), (10, 5)):
        span = (window - 1) * step
        d = splits.build('S1toS2', recs, feats, tg, window=window, step=step, val_gap=True)
        plain = splits.build('S1toS2', recs, feats, tg, window=window, step=step)
        for tr in (0, 1):
            last_train = d['meta_train']['last'][d['meta_train']['trial'] == tr].max()
            first_val_input = d['meta_val']['last'][d['meta_val']['trial'] == tr].min() - span
            assert first_val_input > last_train                        # no validation input inside the training part
            assert first_val_input == last_train + 1                   # and nothing more is dropped
        assert np.array_equal(d['y_train'], plain['y_train']) and np.array_equal(d['y_test'], plain['y_test'])
        assert len(d['y_val']) == len(plain['y_val']) - 2 * span


def test_val_gap_in_carried_state_streams():
    recs, feats, tg = _recs()
    plain = splits.streams('S1toS2', recs, feats, tg, window=20)
    gap = splits.streams('S1toS2', recs, feats, tg, window=20, val_gap=True)
    for a, b in zip(plain['train'], gap['train']):
        assert np.array_equal(a['train_mask'], b['train_mask'])
        first_plain, first_gap = np.argmax(a['val_mask']), np.argmax(b['val_mask'])
        assert first_gap == first_plain + 19 and b['val_mask'].sum() == a['val_mask'].sum() - 19


def test_metadata_passive_fills_only_missing_maps():
    from ML.config import METADATA_PASSIVE
    from ML.research.run import apply_metadata_passive
    empty = {'maps': {'static': {'S1': [], 'S2': []}, 'legacy': {'S1': [], 'S2': []}, 'worksheet': None}}
    out = apply_metadata_passive(empty, 'IES')
    assert out['maps']['static']['S1'] == out['maps']['static']['S2'] == sorted(METADATA_PASSIVE['IES'])
    assert empty['maps']['static']['S1'] == []                                       # input bundle untouched
    has = {'maps': {'static': {'S1': [(0.0, 1.0), (0.1, 2.0)], 'S2': [(0.0, 3.0), (0.1, 4.0)]}}}
    assert apply_metadata_passive(has, 'IES')['maps']['static'] == has['maps']['static']   # never replaces a real map
    assert apply_metadata_passive(empty, 'EG')['maps']['static']['S1'] == []             # only subjects with a table
