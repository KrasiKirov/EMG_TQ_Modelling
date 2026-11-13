"""Tests for EMG pipeline intermediates and isometric trial selection."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ML.preprocessing.emg_envelope import extract_envelope
from ML.plot_emg_pipeline import (
    isometric_active_trials,
    split_trainval_retest,
    pick_pipeline_trial,
    resolve_muscle,
)


def _make_trial(comment, trial_index, n=2000, torque_std=1.0, emg_offset=0.03):
    rng = np.random.default_rng(trial_index + 1)
    df = pd.DataFrame({
        'time': np.arange(n) / 1000.0,
        'position': np.full(n, 0.2),
        'torque': rng.normal(5.0, torque_std, n),
        'gm': rng.normal(emg_offset, 0.01, n),
        'gl': rng.normal(0.0, 0.01, n),
        'sol': rng.normal(0.0, 0.01, n),
        'ta': rng.normal(0.0, 0.01, n),
    })
    df.attrs['comment'] = comment
    df.attrs['trial_index'] = trial_index
    return df


class TestExtractEnvelopeIntermediates(unittest.TestCase):
    def test_return_intermediates_includes_filter_stages(self):
        rng = np.random.default_rng(0)
        raw = rng.normal(0.04, 0.02, 4000)
        envelope, stages = extract_envelope(raw, return_intermediates=True)

        self.assertIsInstance(stages, dict)
        self.assertIn('demeaned', stages)
        self.assertIn('filtered', stages)
        self.assertIn('rectified', stages)
        np.testing.assert_allclose(np.mean(stages['demeaned']), 0.0, atol=1e-12)
        np.testing.assert_allclose(stages['rectified'], np.abs(stages['filtered']))
        self.assertTrue(np.all(envelope >= 0))
        self.assertEqual(len(envelope), len(raw))


class TestTrialSelection(unittest.TestCase):
    def setUp(self):
        self.trials = [
            _make_trial('MVC p1', 0),
            _make_trial('passive p1', 1),
            _make_trial('test p1', 2, torque_std=1.0),
            _make_trial('test p2 ramp', 3, torque_std=40.0),
            _make_trial('retest p1', 4, torque_std=1.0),
            _make_trial('test p3', 5, torque_std=2.0),
        ]

    def test_isometric_active_drops_mvc_passive_and_high_std(self):
        active = isometric_active_trials(self.trials)
        comments = [df.attrs['comment'] for df in active]
        self.assertEqual(comments, ['test p1', 'retest p1', 'test p3'])

    def test_split_puts_test_in_trainval_and_retest_held_out(self):
        active = isometric_active_trials(self.trials)
        trainval, retest = split_trainval_retest(active)
        self.assertEqual(
            [df.attrs['comment'] for df in trainval],
            ['test p1', 'test p3'],
        )
        self.assertEqual(
            [df.attrs['comment'] for df in retest],
            ['retest p1'],
        )

    def test_pick_explicit_trial_and_default_first_test_session(self):
        active = isometric_active_trials(self.trials)
        chosen = pick_pipeline_trial(active, trial_number=6)
        self.assertEqual(chosen.attrs['comment'], 'test p3')
        default = pick_pipeline_trial(active, trial_number=None)
        self.assertEqual(default.attrs['comment'], 'test p1')

    def test_resolve_muscle_aliases(self):
        self.assertEqual(resolve_muscle('1'), 'gm')
        self.assertEqual(resolve_muscle('Muscle 1'), 'gm')
        self.assertEqual(resolve_muscle('MG'), 'gm')


if __name__ == '__main__':
    unittest.main()
