"""Regression checks for the explicitly reconstructed YES coverage."""
import unittest
from pathlib import Path
from unittest.mock import patch

from ML.preprocessing.flb_reader import read_flb
from ML.preprocessing.dataset_builder import classify_trials
from ML.preprocessing.yes_reconstruction import reconstruct_yes, RETEST_RECORDS


class YesReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(__file__).resolve().parents[1] / 'data' / 'YES_281124.flb'
        cls.worksheet = cls.path.with_name('reklab_trial_worksheet_YES_20241128.pdf')
        if not cls.path.exists() or not cls.worksheet.exists():
            raise unittest.SkipTest(
                'YES reconstruction integration fixtures are not available')
        cls.trials = read_flb(cls.path)

    def test_eight_positions_without_changing_raw_comments(self):
        comments = [d.attrs['comment'] for d in self.trials]
        frames, passive, audit = reconstruct_yes(self.trials, self.path)
        _, _, active = classify_trials(frames)
        retained = [d for d in active if d.torque.std() <= 30]
        retest = [d for d in retained if d.attrs['session'] == 'retest']
        train = [d for d in retained if d.attrs['session'] == 'test']
        self.assertEqual([d.attrs['trial_index'] + 1 for d in retest],
                         [27, 29, 31, 33, 35, 37, 39, 41])
        self.assertEqual([d.attrs['trial_index'] + 1 for d in train],
                         [10, 11, 13, 15, 17, 19, 21, 23, 25])
        self.assertEqual([d.attrs['comment'] for d in self.trials], comments)
        self.assertEqual(len(passive), 8)
        self.assertEqual([t for _, t in passive], [10.6, 6.7, 3.5, 2.4, .73, -.38, -2.5, -5.])
        self.assertEqual(len(audit['retest_records']), 8)
        self.assertIn('inferred', audit['confidence'])

    def test_different_source_rejected(self):
        with patch('ML.preprocessing.yes_reconstruction._sha256', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'hashes'):
                reconstruct_yes(self.trials, self.path)

    def test_unexpected_metadata_rejected(self):
        frames = [d.copy(deep=True) for d in self.trials]
        frames[26].attrs['comment'] = 'different recording'
        with self.assertRaisesRegex(ValueError, 'trial 27'):
            reconstruct_yes(frames, self.path)


if __name__ == '__main__':
    unittest.main()
