import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import pandas as pd

from ML.evaluation.metrics import compute_metrics
from ML.evaluation.report import evaluate, write_json
from ML.models.baselines import PositionModel, RidgeModel
from ML.preprocessing.calibration import PassiveCalibration, fit_passive
from ML.preprocessing.emg_envelope import detect_emg_columns, process_trials
from ML.preprocessing.flb_reader import read_flb
from ML.preprocessing.pipeline import PipelineConfig, build_bundle, process_block, source_normalize
from ML.preprocessing.trial_manifest import CHANNELS, inventory


def trial(index=0, comment='active test p1', position=.1, n=12000):
    rng = np.random.default_rng(index)
    time = np.arange(n) / 1000
    frame = pd.DataFrame(dict(time=time, position=np.full(n, position),
                              torque=np.sin(time) + position * 3))
    for c in CHANNELS:
        frame[c] = rng.normal(0, .02, n) * (1 + .3 * np.sin(time))
    frame.attrs = dict(trial_index=index, comment=comment, domainIncr=.001)
    return frame


def bundle(trials=None, config=None):
    trials = trials or [trial(), trial(1, 'active retest p1')]
    config = config or PipelineConfig(target_mode='measured_torque', edge_trim_s=.2, window=10)
    return build_bundle(trials, inventory('TEST', trials, 'fixture-hash'), config)


class AuditedPipelineTests(unittest.TestCase):
    def test_raw_support_is_disjoint_and_windows_do_not_cross_boundary(self):
        b = bundle()
        tr, val = b['train'], b['val']
        self.assertLessEqual(tr.metadata['raw_stop'].max(), val.metadata['raw_start'].min())
        self.assertTrue(np.all(tr.metadata['window_start'] >= tr.metadata['raw_start']))
        self.assertTrue(np.all(tr.metadata['sample_index'] < tr.metadata['raw_stop']))

    def test_validation_and_retest_cannot_change_training(self):
        raw = [trial(), trial(1, 'active retest p1')]
        b = bundle(raw)
        changed = [f.copy(deep=True) for f in raw]
        val_start = int(b['val'].metadata['raw_start'].min())
        changed[0].loc[val_start:, list(CHANNELS)] *= 1000
        changed[0].loc[val_start:, 'torque'] += 1000
        changed[1].loc[:, list(CHANNELS)] *= 1000
        other = bundle(changed)
        np.testing.assert_array_equal(b['train'].X, other['train'].X)
        np.testing.assert_array_equal(b['train'].y, other['train'].y)
        self.assertEqual(b.preprocessing['emg_scales'], other.preprocessing['emg_scales'])

    def test_repeated_preprocessing_is_nonmutating(self):
        raw = [trial()]
        before = raw[0].copy(deep=True)
        a, _ = process_trials(raw)
        b, _ = process_trials(raw)
        pd.testing.assert_frame_equal(raw[0], before)
        self.assertEqual(detect_emg_columns(a[0]), list(CHANNELS))
        pd.testing.assert_frame_equal(a[0], b[0])

    def test_missing_passive_does_not_become_measured_torque(self):
        b = bundle(config=PipelineConfig(target_mode='active_torque'))
        self.assertEqual(len(b['train'].y), 0)
        self.assertTrue(any(c.get('reason') == 'No supported passive calibration' for c in b.coverage))

    def test_retest_passive_cannot_change_calibration(self):
        raw = [trial(0, 'passive test p1', -.1), trial(1, 'passive test p2', .2),
               trial(2, 'passive retest p1', -.1)]
        for i, f in enumerate(raw):
            f['torque'] = float(i)
        m = inventory('TEST', raw, 'fixture')
        before = fit_passive(raw, m)
        raw[-1]['torque'] = 999.
        after = fit_passive(raw, m)
        self.assertEqual(before.to_dict(), after.to_dict())
        self.assertEqual(len(before.positions), 2)

    def test_noisy_passive_and_later_tail_plateau_rejected(self):
        raw = [trial(0, 'passive test p1', -.1)]
        raw[0]['torque'] = np.random.default_rng(2).normal(0, 15., len(raw[0]))
        raw[0].loc[6000:, 'torque'] = 7.8
        cal = fit_passive(raw, inventory('TEST', raw, 'fixture'))
        self.assertEqual(len(cal.positions), 0)

    def test_boundary_support_is_flagged_not_extrapolated(self):
        cal = PassiveCalibration([-.1, .1], [2., -2.], [])
        values = cal.predict(np.array([-.3, 0., .3]))
        self.assertTrue(np.isnan(values[[0, 2]]).all())
        self.assertAlmostEqual(values[1], 0.)

    def test_manifest_override_requires_evidence(self):
        with self.assertRaises(ValueError):
            inventory('TEST', [trial()], 'fixture', {'1': {'kind': 'passive'}})

    def test_qc_respects_explicit_valid_interval(self):
        raw = trial()
        raw.loc[6000:, 'position'] = np.linspace(0, 1, len(raw)-6000)
        m = inventory('TEST', [raw], 'fixture', {'1': {
            'valid_intervals': [[0, 6000]], 'evidence': 'fixture: stationary interval before movement'}})
        self.assertEqual(m['trials'][0]['status'], 'included')

    def test_calibration_budget_never_filters_outside_prefix(self):
        from ML.training.benchmark import calibration_subset
        raw = [trial(n=40000), trial(1, 'active retest p1', n=40000)]
        original = bundle(raw)
        selected = calibration_subset(original, 15, 1000)
        self.assertEqual(selected, {'1': [[0, 15000]]})
        manifest = copy.deepcopy(original.manifest)
        manifest['trials'][0]['valid_intervals'] = selected['1']
        a = build_bundle(raw, manifest, original.config)
        altered = [f.copy(deep=True) for f in raw]
        altered[0].loc[15000:, list(CHANNELS)] *= 1000
        b = build_bundle(altered, manifest, original.config)
        np.testing.assert_array_equal(a['train'].X, b['train'].X)
        np.testing.assert_array_equal(a['val'].X, b['val'].X)

    def test_ambiguous_yes_trial_is_excluded(self):
        m = inventory('YES', [trial(26, 'passive retest p2')], 'fixture')
        self.assertEqual(m['trials'][0]['status'], 'ambiguous')

    def test_evidenced_override_is_applied_before_classification(self):
        m = inventory('YES', [trial(26, 'passive retest p2')], 'fixture', {'27': {
            'kind': 'active', 'session': 'test', 'position_id': 'p2', 'status': 'included',
            'evidence': 'Synthetic fixture evidence, not a correction to the real recording'}})
        self.assertEqual(m['trials'][0]['kind'], 'active')
        self.assertEqual(m['trials'][0]['status'], 'included')
        self.assertEqual(m['trials'][0]['reason'], '')

    def test_short_nonfinite_and_wrong_sample_rate_rejected(self):
        for frame in [trial(n=20), trial(), trial()]:
            if len(frame) > 20 and not hasattr(self, '_nan_checked'):
                frame.loc[0, 'gm'] = np.nan
                self._nan_checked = True
            elif len(frame) > 20:
                frame.attrs['domainIncr'] = .002
            with self.assertRaises(ValueError):
                process_block(frame, 0, len(frame), PipelineConfig())

    def test_window_parameter_and_normalization_are_consistent(self):
        b = bundle(config=PipelineConfig(target_mode='measured_torque', window=50))
        self.assertEqual(b['train'].X.shape[1:], (50, 5))
        self.assertLessEqual(float(b['train'].X[:, :, :4].max()), 1.00001)

    def test_source_only_normalization_ignores_target_amplitude(self):
        source, target = bundle(), bundle()
        _, _, scales = source_normalize([source], target)
        changed = copy.deepcopy(target)
        changed.preprocessing['emg_scales'] = [v * 100 for v in changed.preprocessing['emg_scales']]
        _, _, scales2 = source_normalize([source], changed)
        np.testing.assert_array_equal(scales, scales2)

    def test_metrics_reject_broadcasting_and_report_empty(self):
        with self.assertRaises(ValueError):
            compute_metrics(np.ones(3), np.ones((3, 1)))
        self.assertTrue(np.isnan(compute_metrics(np.array([]), np.array([]))['r2']))
        self.assertTrue(np.isnan(compute_metrics(np.ones(4), np.zeros(4))['r2']))

    def test_invalid_raw_support_never_wraps_negative_indices(self):
        frame = trial()
        for start, stop in [(0, -10), (-1, 8000), (500, 100), (0, len(frame)+1)]:
            with self.assertRaises(ValueError):
                process_block(frame, start, stop, PipelineConfig())

    def test_nonfinite_manifest_can_be_serialized_strictly(self):
        frame = trial()
        frame.loc[2, 'torque'] = np.inf
        result = inventory('TEST', [frame], 'fixture')
        self.assertEqual(result['trials'][0]['status'], 'excluded')
        json.dumps(result, allow_nan=False)

    def test_qc_discontinuities_do_not_bridge_valid_intervals(self):
        frame = trial()
        frame.loc[6000:, 'torque'] += 1000
        result = inventory('TEST', [frame], 'fixture', {'1': {
            'valid_intervals': [[0, 4000], [8000, 12000]], 'evidence': 'fixture'}})
        self.assertLess(result['trials'][0]['signal_qc']['torque']['max_adjacent_step'], .01)

    def test_report_retains_trial_identity(self):
        b = bundle()
        result = evaluate(b['test'], b['test'].y)
        self.assertIn('TEST/2', result['trials'])
        self.assertEqual(result['mean_subject_position_rmse'], 0.)

    def test_fresh_package_load_matches_baseline(self):
        from ML.inference import Predictor
        b = bundle()
        model = RidgeModel().fit(b['train'].X, b['train'].y)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model.save(root / 'model.npz')
            write_json(root / 'package.json', dict(schema_version=1, kind='ridge', target_offset=0.,
                        target_scale=1., preprocessing=b.preprocessing, input_shape=[b.config.window, 5]))
            restored = Predictor(root)
            np.testing.assert_allclose(restored.predict_windows(b['test'].X), model.predict(b['test'].X))
            raw = trial(1, 'active retest p1').drop(columns=['torque'])
            result = restored.predict_recording(raw)
            self.assertTrue(np.isfinite(result['prediction_nm']).all())

    def test_coincident_position_groups_are_averaged(self):
        X = np.zeros((4, 3, 5))
        model = PositionModel().fit(X, np.array([1., 1., 3., 3.]), np.array(['a', 'a', 'b', 'b']))
        np.testing.assert_array_equal(model.predict(X), np.full(4, 2.))

    def test_neural_package_reload_and_invalid_scaling(self):
        from ML.inference import Predictor
        from ML.models.lstm import build_model
        b = bundle()
        model = build_model(b.config.window, 5)
        X = b['test'].X[:20]
        expected = np.asarray(model(X, training=False)).ravel() * 2.5 + 1.2
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model.save(root / 'model.keras')
            package = dict(schema_version=1, kind='lstm', target_offset=1.2, target_scale=2.5,
                           preprocessing=b.preprocessing, input_shape=[b.config.window, 5])
            write_json(root / 'package.json', package)
            np.testing.assert_allclose(Predictor(root).predict_windows(X), expected, atol=1e-6)
            package['target_scale'] = 0.
            write_json(root / 'package.json', package)
            with self.assertRaises(ValueError):
                Predictor(root)

    def test_sweep_rules_reject_worse_positions_and_require_speed_gain(self):
        from ML.training.sweep import qualifies
        reference = dict(mean=1., worst=2., seconds=100., by_seed={'11': 1., '23': 1., '37': 1.})
        candidate = dict(mean=.9, worst=2., seconds=80., by_seed={'11': .9, '23': .9, '37': .9})
        self.assertTrue(qualifies(candidate, reference))
        self.assertFalse(qualifies(candidate, reference, efficiency=True))
        candidate['seconds'] = 70.
        self.assertTrue(qualifies(candidate, reference, efficiency=True))
        candidate['worst'] = 2.2
        self.assertFalse(qualifies(candidate, reference))


class BinaryReaderTests(unittest.TestCase):
    def test_float32_fixture_channel_order_and_interval(self):
        names = ['Position', 'Torque', 'TA EMG', 'MG EMG', 'Sol EMG', 'LG EMG']
        def integer(value):
            return struct.pack('<i', value)
        def string(value):
            return integer(len(value)) + value.encode('latin-1')
        header = b''.join(map(integer, [4, 2, 1, 6, 100, 4]))
        header += string('Time') + struct.pack('<ff', .002, 0) + string('active test p1')
        header += b''.join(string(n) for n in names) + struct.pack('<12d', *([0.] * 12))
        signals = np.tile(np.arange(6, dtype='<f4'), (100, 1))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'fixture.flb'
            path.write_bytes(header + signals.tobytes(order='F'))
            frame = read_flb(path)[0]
            self.assertAlmostEqual(frame.attrs['domainIncr'], .002)
            self.assertTrue(np.all(frame.gm == 3))
            self.assertTrue(np.all(frame.gl == 5))
            path.write_bytes(path.read_bytes() + b'\x01')
            with self.assertRaises(IOError):
                read_flb(path)


if __name__ == '__main__':
    unittest.main()
