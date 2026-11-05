import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from asset_paths import default_g1_urdf_path
from avp_g1_pose import estimate_urdf_root_height, h1_2_hand_pose_summary, map_landau_pose_to_h1_2_pose, world_map_from_urdf_pose


class TestG1Pose(unittest.TestCase):
    @unittest.skipUnless(default_g1_urdf_path().exists(), "requires cloned H1_2 helper repo")
    def test_mirrored_landau_curl_closes_both_hands_with_real_h1_limits(self):
        from urdf_kinematics import load_joint_limits
        limits = load_joint_limits(default_g1_urdf_path())
        source = {f"{finger}{i}_{side}": -1.4
                  for side in ("l", "r")
                  for finger in ("thumb", "index", "middle", "ring", "pinky")
                  for i in (1, 2, 3)}
        source.update(thumb1_l=-.4, thumb1_r=.4, thumb2_l=.9, thumb3_l=.9, thumb2_r=-.9, thumb3_r=-.9)
        closed = map_landau_pose_to_h1_2_pose(source, joint_limits=limits)
        self.assertEqual(h1_2_hand_pose_summary(closed, 'left')['thumb'],
                         h1_2_hand_pose_summary(closed, 'right')['thumb'])
        for prefix in ("L", "R"):
            self.assertGreater(closed[f"{prefix}_thumb_proximal_pitch_joint"], .25)
            for finger in ("index", "middle", "ring", "pinky"):
                self.assertGreater(closed[f"{prefix}_{finger}_proximal_joint"], .7)

    def test_map_landau_pose_to_g1_pose_covers_upper_body_and_hands(self):
        landau_pose = {
            "spine_02_x": 0.1,
            "shoulder_l": 0.2,
            "arm_stretch_l": 1.1,
            "forearm_stretch_l": -0.8,
            "hand_l": -0.2,
            "arm_twist_l": 0.3,
            "forearm_twist_l": 0.4,
            "thumb1_l": 0.5,
            "thumb2_l": 0.6,
            "thumb3_l": 0.7,
            "index1_base_l": 0.1,
            "index1_l": -0.8,
            "index2_l": -0.7,
            "index3_l": -0.6,
            "middle1_base_l": 0.05,
            "middle1_l": -0.9,
            "middle2_l": -0.8,
            "middle3_l": -0.7,
            "ring1_base_l": -0.02,
            "ring1_l": -0.6,
            "ring2_l": -0.5,
            "ring3_l": -0.4,
            "pinky1_base_l": -0.08,
            "pinky1_l": -0.5,
            "pinky2_l": -0.4,
            "pinky3_l": -0.3,
        }

        g1_pose = map_landau_pose_to_h1_2_pose(landau_pose)

        self.assertAlmostEqual(g1_pose["torso_joint"], 0.03)
        self.assertLess(g1_pose["left_shoulder_pitch_joint"], -0.9)
        self.assertGreater(g1_pose["left_shoulder_roll_joint"], 0.35)
        self.assertGreater(g1_pose["left_elbow_pitch_joint"], 1.0)
        self.assertAlmostEqual(g1_pose["L_thumb_proximal_yaw_joint"], 0.08)
        self.assertAlmostEqual(g1_pose["L_thumb_proximal_pitch_joint"], 0.1965)
        self.assertGreater(g1_pose["L_index_proximal_joint"], 0.35)
        self.assertGreater(g1_pose["L_middle_proximal_joint"], 0.4)
        self.assertGreater(g1_pose["L_ring_proximal_joint"], 0.25)
        self.assertGreater(g1_pose["L_pinky_proximal_joint"], 0.2)

        left_hand = h1_2_hand_pose_summary(g1_pose, "left")
        self.assertAlmostEqual(left_hand["thumb"][0], 0.08)
        self.assertAlmostEqual(left_hand["thumb"][1], 0.1965)

    def test_map_landau_pose_to_g1_pose_honors_joint_limits(self):
        g1_pose = map_landau_pose_to_h1_2_pose(
            {"arm_stretch_r": 10.0},
            joint_limits={"right_shoulder_pitch_joint": (-0.5, 0.5)},
        )

        self.assertEqual(g1_pose["right_shoulder_pitch_joint"], -0.5)

    def test_map_landau_pose_to_g1_pose_accepts_hand_override(self):
        g1_pose = map_landau_pose_to_h1_2_pose(
            {"arm_stretch_l": 0.3},
            hand_pose_override={"L_index_proximal_joint": 0.4},
        )

        self.assertAlmostEqual(g1_pose["L_index_proximal_joint"], 0.4)

    def test_map_landau_pose_to_g1_pose_keeps_a_visible_baseline_arm_pose(self):
        g1_pose = map_landau_pose_to_h1_2_pose({})

        self.assertLess(g1_pose["left_shoulder_pitch_joint"], -0.5)
        self.assertGreater(g1_pose["left_shoulder_roll_joint"], 0.3)
        self.assertGreater(g1_pose["left_elbow_pitch_joint"], 0.3)
        self.assertLess(g1_pose["right_shoulder_pitch_joint"], -0.5)
        self.assertLess(g1_pose["right_shoulder_roll_joint"], -0.3)
        self.assertGreater(g1_pose["right_elbow_pitch_joint"], 0.3)

    @unittest.skipUnless(default_g1_urdf_path().exists(), "requires cloned H1_2 helper repo")
    def test_visible_baseline_pose_places_hands_forward_of_torso(self):
        g1_pose = map_landau_pose_to_h1_2_pose({})
        world = world_map_from_urdf_pose(default_g1_urdf_path(), g1_pose)
        left_hand = world["L_hand_base_link"][:3, 3]
        right_hand = world["R_hand_base_link"][:3, 3]

        self.assertGreater(float(left_hand[0]), 0.3)
        self.assertGreater(float(left_hand[1]), 0.18)
        self.assertGreater(float(right_hand[0]), 0.3)
        self.assertLess(float(right_hand[1]), -0.18)

    @unittest.skipUnless(default_g1_urdf_path().exists(), "requires cloned H1_2 helper repo")
    def test_estimate_urdf_root_height_matches_g1_rest_scale(self):
        height = estimate_urdf_root_height(default_g1_urdf_path())

        self.assertGreater(height, 0.7)
        self.assertLess(height, 1.1)


if __name__ == "__main__":
    unittest.main()
