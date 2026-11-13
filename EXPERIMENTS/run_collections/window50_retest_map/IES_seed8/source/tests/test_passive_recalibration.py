import unittest

from ML.evaluation.passive_recalibration import audit_passive_recalibration
from ML.preprocessing.trial_manifest import inventory
from ML.tests.test_audited_pipeline import trial


class PassiveRecalibrationTests(unittest.TestCase):
    def test_audit_compares_retest_passive_only(self):
        source_p1 = trial(0, 'passive test p1', -.1)
        source_p2 = trial(1, 'passive test p2', .2)
        retest_p1 = trial(2, 'passive retest p1', -.1)
        active_retest = trial(3, 'active retest p2', .2)
        source_p1['torque'] = 1.
        source_p2['torque'] = 3.
        retest_p1['torque'] = 2.1
        active_retest['torque'] = 999.
        raw = [source_p1, source_p2, retest_p1, active_retest]
        manifest = inventory('TEST', raw, 'fixture-hash')

        report = audit_passive_recalibration(raw, manifest, session='retest')

        self.assertTrue(report['audit_only'])
        self.assertFalse(report['active_torque_used'])
        self.assertEqual(report['active_trials_used'], 0)
        self.assertEqual(report['summary']['session_point_count'], 1)
        self.assertAlmostEqual(report['records'][0]['delta_session_minus_source_nm'], 1.1,
                               places=3)
        self.assertTrue(report['records'][0]['drift_flag'])

    def test_audit_rejects_invalid_threshold(self):
        raw = [trial(0, 'passive test p1', -.1), trial(1, 'passive retest p1', -.1)]
        manifest = inventory('TEST', raw, 'fixture-hash')
        with self.assertRaises(ValueError):
            audit_passive_recalibration(raw, manifest, drift_flag_nm=0.)


if __name__ == '__main__':
    unittest.main()
