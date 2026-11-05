import math
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from asset_setup import prepare_landau_inputs
from landau_mapping_config import JOINT_OUTPUT_RULES, UNTRACKED_POSE_DEFAULTS
from landau_retarget import (
    LEFT_ARM_CHAIN,
    RIGHT_ARM_CHAIN,
    LandauUpperBodyRetargeter,
    _side_suffix,
    _tracking_matrix_world,
    find_joint_chain,
)


class TestScalarPalmBasis(unittest.TestCase):
    def setUp(self):
        self.retargeter = LandauUpperBodyRetargeter.__new__(LandauUpperBodyRetargeter)
        self.retargeter.robot_hand_basis = {'left': np.eye(3)}
        self.points = {name: np.array(point) for name, point in {
            'indexMetacarpal': [.02,.02,0], 'littleMetacarpal': [-.02,.02,0],
            'indexKnuckle': [.02,.04,0], 'middleKnuckle': [0,.04,0],
            'ringKnuckle': [-.02,.04,0], 'thumbKnuckle': [.015,.015,.005],
            'thumbIntermediateBase': [.02,.025,.005],
            'thumbIntermediateTip': [.025,.035,.01], 'thumbTip': [.025,.045,.02],
        }.items()}
        self.retargeter._wrist_local_positions = lambda side, frame: self.points
        self.retargeter._mapped_joint_value = lambda name, value: value

    def test_anatomical_basis_preserves_asymmetric_points(self):
        actual = self.retargeter._aligned_hand_local_positions('left', {})
        for name, point in self.points.items():
            np.testing.assert_allclose(actual[name], point, atol=1e-12)

    def test_mirrored_thumb_axes(self):
        left, right = {}, {}
        self.retargeter._apply_thumb('l', self.points, left)
        self.retargeter._apply_thumb('r', self.points, right)
        for segment in ('thumb2', 'thumb3'):
            self.assertGreater(left[segment+'_l'], 0)
            self.assertAlmostEqual(left[segment+'_l'], -right[segment+'_r'])

    def test_degenerate_palm_and_thumb_are_omitted(self):
        self.points['littleMetacarpal'] = self.points['indexMetacarpal'].copy()
        self.assertIsNone(self.retargeter._aligned_hand_local_positions('left', {}))
        self.points['thumbIntermediateTip'] = self.points['thumbIntermediateBase'].copy()
        pose = {}
        self.retargeter._apply_thumb('l', self.points, pose)
        self.assertEqual(pose, {})
        del self.points['indexKnuckle']
        self.assertIsNone(self.retargeter._aligned_hand_local_positions('left', {}))


class TestLandauRetarget(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        prepared = prepare_landau_inputs(refresh=False)
        cls.retargeter = LandauUpperBodyRetargeter(
            urdf_path=prepared.urdf_path,
            skeleton_json_path=prepared.skeleton_json_path,
            snapshot_path=MODULE_ROOT / "avp_snapshot.json",
            use_trac_ik=False,
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
        self.assertEqual(self.retargeter.joint_specs["hand_l"].name, "left_wrist_pitch_joint")
        self.assertEqual(self.retargeter.joint_specs["forearm_twist_r"].name, "right_forearm_roll_joint")

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
        world_by_name = self.retargeter._base_relative_world_map(pose)

        for side in ("left", "right"):
            wrist_world = _tracking_matrix_world(self.retargeter.snapshot_frame[f"{side}_wrist"])
            self.assertIsNotNone(wrist_world)

            target_base = self.retargeter.base_world_inv @ wrist_world
            clipped_target = self.retargeter._clamp_hand_base_position(side, target_base[:3, 3])
            solved_base = world_by_name[f"hand_{_side_suffix(side)}"]

            np.testing.assert_allclose(solved_base[:3, 3], clipped_target, atol=7.0e-2)
            self.assertLess(abs(float(solved_base[1, 3])), 3.0e-2)
            self.assertGreater(float(solved_base[2, 3]), 0.15)

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

    def test_snapshot_matches_measured_palm_orientation_within_joint_limits(self):
        from urdf_kinematics import world_map_from_joint_specs
        self.retargeter.left_seed[:] = 0
        self.retargeter.right_seed[:] = 0
        frame = self.retargeter.snapshot_frame
        pose = self.retargeter.retarget_frame(frame)
        worlds = world_map_from_joint_specs(self.retargeter.urdf_joint_specs, pose, pose_key="child_link")
        for side in ("left", "right"):
            actual = (self.retargeter.base_world_inv @ worlds["hand_" + _side_suffix(side)])[:3, :3]
            target = self.retargeter._target_hand_base_rotation(side, frame)
            degrees = math.degrees(math.acos(float(np.clip((np.trace(actual.T @ target) - 1) / 2, -1, 1))))
            self.assertLess(degrees, 12, side)
            self.assertGreater(abs(pose["forearm_twist_" + _side_suffix(side)]), .5)

    def test_snapshot_retarget_explicitly_zeros_untracked_leg_pose_keys(self):
        pose = self.retargeter.retarget_frame(self.retargeter.snapshot_frame)

        for joint_name, joint_value in UNTRACKED_POSE_DEFAULTS.items():
            self.assertIn(joint_name, pose)
            self.assertAlmostEqual(float(pose[joint_name]), float(joint_value), places=7)

    def test_mapping_config_covers_all_non_default_snapshot_outputs(self):
        pose = self.retargeter.retarget_frame(self.retargeter.snapshot_frame)

        for joint_name in pose:
            if joint_name in UNTRACKED_POSE_DEFAULTS:
                continue
            self.assertIn(joint_name, JOINT_OUTPUT_RULES)


if __name__ == "__main__":
    unittest.main()
