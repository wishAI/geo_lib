from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT.parent.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT.parent.parent))

from algorithms.urdf_learn_wasd_walk.reward_probe import (
    RewardValidationReport,
    build_validation_report,
    validate_reward_report,
)


class RewardProbeTests(unittest.TestCase):
    def test_validation_report_passes_sign_checks(self) -> None:
        report = build_validation_report()
        validate_reward_report(report)

    def test_report_values_show_expected_ordering(self) -> None:
        report: RewardValidationReport = build_validation_report()

        self.assertGreater(report.lin_tracking_perfect, report.lin_tracking_bad)
        self.assertGreater(report.yaw_tracking_perfect, report.yaw_tracking_bad)
        self.assertGreater(report.orientation_tilted, report.orientation_flat)
        self.assertGreater(report.action_rate_large, report.action_rate_zero)


if __name__ == "__main__":
    unittest.main()
