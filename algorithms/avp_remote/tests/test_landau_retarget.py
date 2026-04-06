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
    _side_suffix,
    _tracking_matrix_world,
    find_joint_chain,
)
from landau_pose import apply_joint_positions_to_local_matrices, world_matrices_from_local


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

    def test_snapshot_arm_solve_tracks_clipped_wrist_targets(self):
        pose = self.retargeter.retarget_frame(self.retargeter.snapshot_frame)
        local_matrices = apply_joint_positions_to_local_matrices(self.retargeter.records, pose)
        world_matrices = world_matrices_from_local(self.retargeter.records, local_matrices)
        world_by_name = {
            record.name: world_matrices[record.index]
            for record in self.retargeter.records
        }

        for side in ("left", "right"):
            wrist_world = _tracking_matrix_world(self.retargeter.snapshot_frame[f"{side}_wrist"])
            self.assertIsNotNone(wrist_world)

            target_base = self.retargeter.base_world_inv @ wrist_world
            clipped_target = self.retargeter._clamp_hand_base_position(side, target_base[:3, 3])
            solved_base = self.retargeter.base_world_inv @ world_by_name[f"hand_{_side_suffix(side)}"]

            np.testing.assert_allclose(solved_base[:3, 3], clipped_target, atol=3.0e-2)
            self.assertLess(abs(float(solved_base[1, 3])), 3.0e-2)
            self.assertGreater(float(solved_base[2, 3]), 0.18)

    def test_snapshot_finger_bases_do_not_saturate(self):
        pose = self.retargeter.retarget_frame(self.retargeter.snapshot_frame)

        for joint_name in (
            "index1_base_l",
            "middle1_base_l",
            "ring1_base_l",
            "pinky1_base_l",
            "index1_base_r",
            "middle1_base_r",
            "ring1_base_r",
            "pinky1_base_r",
        ):
            self.assertLess(abs(float(pose[joint_name])), 0.35, joint_name)

        self.assertLess(abs(float(pose["thumb1_l"])), 0.8)
        self.assertLess(abs(float(pose["thumb1_r"])), 0.8)


if __name__ == "__main__":
    unittest.main()
