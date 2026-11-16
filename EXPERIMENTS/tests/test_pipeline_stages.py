import numpy as np

from EXPERIMENTS.analysis.pipeline_stages import stages_1k, strongest_rise
from ML.preprocessing.emg_envelope import extract_envelope_causal


def _emg(seed=0, n=6000):
    rng = np.random.default_rng(seed)
    burst = 1 + 4 * (np.arange(n) > 3000)
    return 0.3 + rng.normal(0, 1e-4, n) * burst


def test_stage_chain_matches_library_causal_envelope():
    x = _emg()
    for order in (2, 4):
        assert np.allclose(stages_1k(x, lp_order=order)['env'], extract_envelope_causal(x, lp_order=order))


def test_rectified_is_absolute_band_passed_and_offset_removed():
    st = stages_1k(_emg())
    assert np.array_equal(st['rect'], np.abs(st['band']))
    assert abs(np.mean(st['demeaned'][:1000])) < 1e-12


def test_split_structure_reports_purged_boundary_and_first_test_label():
    from ML.research import splits
    from EXPERIMENTS.pipeline.run import drop_last_val_window
    from EXPERIMENTS.analysis.pipeline_stages import split_structure
    T = 200
    t = 1.0 + np.arange(T) / 100
    recs = [{'idx': 1, 'session': 'S1', 'angle': -0.3, 't': t, 'pos': np.full(T, -0.3)},
            {'idx': 2, 'session': 'S2', 'angle': -0.3, 't': t, 'pos': np.full(T, -0.3)}]
    feats = {i: np.zeros((T, 4)) for i in (1, 2)}
    targets = {i: np.zeros(T) for i in (1, 2)}
    data = drop_last_val_window(splits.build('S1toS2', recs, feats, targets, window=20, val_gap=True))
    s = split_structure(recs, data, window=20, batch=64)
    s1, s2 = s['trials']
    # 181 windows, the first 144 train: labels at samples 19..162
    assert (s1['train']['n'], s1['train']['first_label'], s1['train']['last_label']) == (144, 1.19, 2.62)
    # purged: the first validation window starts right after the last training input and is labelled 19 samples later
    assert (s1['val']['first_input'], s1['val']['first_label'], s1['val']['last_label']) == (2.63, 2.82, 2.98)
    assert (s2['test']['first_label'], s2['test']['last_label'], s2['test']['n']) == (1.19, 2.99, 181)
    assert s['updates_per_epoch'] == 3


def test_window_settings_report_time_overlap_and_updates_per_epoch():
    from EXPERIMENTS.analysis.pipeline_stages import window_settings
    s = {r['id']: r for r in window_settings(8810)}
    assert set(s) >= {'W00_reference', 'WL1_0500ms', 'WO4_00', 'WC3_carry_1s', 'WC4_carry_10s'}
    assert 'WL1_0500ms_lp4' not in s                       # only the settings the study compares
    assert (s['W00_reference']['train_windows'], s['W00_reference']['updates_per_epoch']) == (56256, 879)
    assert s['WL1_0500ms']['overlap_pct'] == 98.0           # 500 ms window moved 10 ms, not 10 steps moved 1
    assert s['WO4_00']['overlap_pct'] == 0
    assert (s['WC3_carry_1s']['updates_per_epoch'], s['WC4_carry_10s']['updates_per_epoch']) == (71, 8)


def test_equal_update_settings_repeat_their_windows_to_match_the_reference_epoch():
    from EXPERIMENTS.analysis.pipeline_stages import window_settings
    s = {r['id']: r for r in window_settings(8810)}
    eq = s['WO4_00_eq']
    # 2,816 distinct windows, fed 20 times per epoch: 880 updates, like the reference's 879
    assert (eq['train_windows'], eq['passes_per_epoch'], eq['updates_per_epoch']) == (2816, 20, 880)
    assert (s['WO2_75_eq']['passes_per_epoch'], s['WO3_50_eq']['passes_per_epoch']) == (5, 10)
    assert s['W00_reference']['passes_per_epoch'] == 1


def test_strongest_rise_finds_the_burst_onset():
    f = np.zeros((3000, 4))
    f[1500:] = 1.0
    j = strongest_rise(f)
    assert 1400 <= j <= 1500
