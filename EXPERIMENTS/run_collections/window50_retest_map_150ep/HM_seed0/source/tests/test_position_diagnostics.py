import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ML.evaluation.position_diagnostics import (audit_project_convention,
                                                diagnose_run, signed_group)


class PositionDiagnosticTests(unittest.TestCase):
    def test_signed_groups_are_numeric_and_direction_neutral(self):
        self.assertEqual(signed_group(-.1), "negative")
        self.assertEqual(signed_group(.1), "positive")
        self.assertEqual(signed_group(0.), "near_zero")
        self.assertEqual(signed_group(.001, zero_tolerance=.01), "near_zero")

    def test_audit_reports_intended_convention_without_claiming_sensor_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "exp_model").mkdir()
            (root / "ES_sandbox").mkdir()
            (root / "exp_model/initialize_input_waveform.m").write_text(
                "maxPF = -0.5; maxDF = +0.2;\n")
            (root / "ES_sandbox/source.m").write_text(
                "maxPF=-0.50; maxDF=+0.25;\n")
            audit = audit_project_convention(root, (
                "exp_model/initialize_input_waveform.m",
                "ES_sandbox/source.m"))
            self.assertEqual(audit["status"], "intended_convention_found")
            self.assertFalse(audit["physical_sensor_polarity_verified"])

    def test_diagnose_run_uses_manifest_angles_and_ranks_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "HM_position_seed1").mkdir()
            (root / "HM_manifest.json").write_text(json.dumps({
                "subject": "HM", "trials": [
                    {"kind": "active", "position_id": "p1", "mean_position_rad": -.2},
                    {"kind": "active", "position_id": "p2", "mean_position_rad": .2},
                ]}))
            (root / "HM_position_seed1/package.json").write_text(json.dumps(
                {"kind": "position", "seed": 1}))
            np.savez(root / "HM_position_seed1/test_predictions.npz",
                     y_true=np.array([0., 0., 1., 1.]),
                     y_pred=np.array([0., 2., 1., 1.]),
                     position_id=np.array(["p1", "p1", "p2", "p2"]),
                     subject=np.array(["HM"] * 4),
                     trial=np.array([1, 1, 2, 2]))
            report = diagnose_run(root, root)
            self.assertEqual(len(report["records"]), 2)
            self.assertEqual(report["records"][0]["signed_group"], "negative")
            self.assertEqual(report["worst_positions"][0]["position_id"], "p1")
            self.assertFalse(report["position_convention"]["physical_sensor_polarity_verified"])


if __name__ == "__main__":
    unittest.main()
