from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from asset_paths import default_g1_urdf_path
from g1_pose import estimate_urdf_root_height, g1_hand_pose_summary, map_landau_pose_to_g1_pose


class G1PoseTests(unittest.TestCase):
    def test_map_landau_pose_to_g1_pose_covers_upper_body_and_hands(self) -> None:
        landau_pose = {
            'spine_02_x': 0.1,
            'shoulder_l': 0.2,
            'arm_stretch_l': 1.1,
            'forearm_stretch_l': -0.8,
            'hand_l': -0.2,
            'arm_twist_l': 0.3,
            'forearm_twist_l': 0.4,
            'thumb1_l': 0.5,
            'thumb2_l': 0.6,
            'thumb3_l': 0.7,
            'index1_base_l': 0.1,
            'index1_l': 0.8,
            'index2_l': 0.7,
            'index3_l': 0.6,
            'middle1_base_l': 0.05,
            'middle1_l': 0.9,
            'middle2_l': 0.8,
            'middle3_l': 0.7,
            'ring1_base_l': -0.02,
            'ring1_l': 0.6,
            'ring2_l': 0.5,
            'ring3_l': 0.4,
            'pinky1_base_l': -0.08,
            'pinky1_l': 0.5,
            'pinky2_l': 0.4,
            'pinky3_l': 0.3,
        }

        g1_pose = map_landau_pose_to_g1_pose(landau_pose)

        self.assertAlmostEqual(g1_pose['torso_joint'], 0.03)
        self.assertLess(g1_pose['left_shoulder_pitch_joint'], -0.9)
        self.assertGreater(g1_pose['left_elbow_pitch_joint'], 1.0)
        self.assertAlmostEqual(g1_pose['L_thumb_proximal_yaw_joint'], 0.36)
        self.assertAlmostEqual(g1_pose['L_thumb_proximal_pitch_joint'], 0.1965)
        self.assertGreater(g1_pose['L_index_proximal_joint'], 0.35)
        self.assertGreater(g1_pose['L_middle_proximal_joint'], 0.4)
        self.assertGreater(g1_pose['L_ring_proximal_joint'], 0.25)
        self.assertGreater(g1_pose['L_pinky_proximal_joint'], 0.2)

        left_hand = g1_hand_pose_summary(g1_pose, 'left')
        self.assertEqual(left_hand['thumb'], (0.36, 0.1965))

    def test_map_landau_pose_to_g1_pose_honors_joint_limits(self) -> None:
        g1_pose = map_landau_pose_to_g1_pose(
            {'arm_stretch_r': 10.0},
            joint_limits={'right_shoulder_pitch_joint': (-0.5, 0.5)},
        )

        self.assertEqual(g1_pose['right_shoulder_pitch_joint'], -0.5)

    @unittest.skipUnless(default_g1_urdf_path().exists(), 'requires cloned H1_2 helper repo')
    def test_estimate_urdf_root_height_matches_g1_rest_scale(self) -> None:
        height = estimate_urdf_root_height(default_g1_urdf_path())

        self.assertGreater(height, 0.7)
        self.assertLess(height, 1.1)


if __name__ == '__main__':
    unittest.main()
