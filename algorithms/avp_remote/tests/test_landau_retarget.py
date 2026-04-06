import math
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from asset_setup import prepare_landau_inputs
from landau_retarget import (
    LEFT_ARM_CHAIN,
    RIGHT_ARM_CHAIN,
    LandauUpperBodyRetargeter,
    find_joint_chain,
)


class TestLandauRetarget(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        prepared = prepare_landau_inputs(refresh=False)
        cls.retargeter = LandauUpperBodyRetargeter(
            urdf_path=prepared.urdf_path,
            skeleton_json_path=prepared.skeleton_json_path,
            snapshot_path=MODULE_ROOT / "avp_snapshot.json",
        )

    def test_find_joint_chain_matches_expected_arm_order(self):
        prepared = prepare_landau_inputs(refresh=False)
        self.assertEqual(
            find_joint_chain(prepared.urdf_path, "spine_03_x", "hand_l"),
            LEFT_ARM_CHAIN,
        )
        self.assertEqual(
            find_joint_chain(prepared.urdf_path, "spine_03_x", "hand_r"),
            RIGHT_ARM_CHAIN,
        )

    def test_snapshot_retarget_produces_finite_joint_values_within_limits(self):
        pose = self.retargeter.retarget_frame(self.retargeter.snapshot_frame)

        self.assertIn("neck_x", pose)
        self.assertIn("head_x", pose)
        for joint_name in (*LEFT_ARM_CHAIN, *RIGHT_ARM_CHAIN, "index1_l", "index1_r", "thumb2_l", "thumb2_r"):
            self.assertIn(joint_name, pose)

        arm_magnitude = sum(abs(float(pose[joint_name])) for joint_name in (*LEFT_ARM_CHAIN, *RIGHT_ARM_CHAIN))
        self.assertGreater(arm_magnitude, 0.2)

        for joint_name, joint_value in pose.items():
            self.assertTrue(math.isfinite(float(joint_value)), joint_name)
            spec = self.retargeter.joint_specs.get(joint_name)
            if spec is None:
                continue
            self.assertGreaterEqual(float(joint_value), spec.lower - 1.0e-6, joint_name)
            self.assertLessEqual(float(joint_value), spec.upper + 1.0e-6, joint_name)

    def test_hand_calibration_scales_are_positive(self):
        for side in ("left", "right"):
            calibration = self.retargeter.calibration[side]
            self.assertGreater(calibration.scale, 0.0)
            self.assertEqual(calibration.rotation_offset.shape, (3, 3))
            should_be_identity = calibration.rotation_offset @ calibration.rotation_offset.T
            np.testing.assert_allclose(should_be_identity, np.eye(3), atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
