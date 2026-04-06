import unittest
import sys
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from avp_tracking_schema import HAND_JOINT_NAMES
from usd_skeleton_schema import (
    LEFT_FOOT_JOINT,
    LEFT_HAND_JOINTS,
    LEFT_HAND_ROOT_JOINT,
    RIGHT_HAND_JOINTS,
    RIGHT_HAND_ROOT_JOINT,
    SKELETON_JOINT_NAMES,
    build_empty_future_hand_mapping,
    get_missing_expected_joints,
    get_unexpected_joints,
)


class TestUsdSkeletonSchema(unittest.TestCase):
    def test_skeleton_joint_count(self):
        self.assertEqual(len(SKELETON_JOINT_NAMES), 68)

    def test_key_joint_paths_exist(self):
        self.assertIn(LEFT_HAND_ROOT_JOINT, SKELETON_JOINT_NAMES)
        self.assertIn(RIGHT_HAND_ROOT_JOINT, SKELETON_JOINT_NAMES)
        self.assertIn(LEFT_FOOT_JOINT, SKELETON_JOINT_NAMES)

    def test_hand_joint_groups_are_scoped(self):
        self.assertTrue(LEFT_HAND_JOINTS)
        self.assertTrue(RIGHT_HAND_JOINTS)
        self.assertTrue(all(name.startswith(LEFT_HAND_ROOT_JOINT) for name in LEFT_HAND_JOINTS))
        self.assertTrue(all(name.startswith(RIGHT_HAND_ROOT_JOINT) for name in RIGHT_HAND_JOINTS))

    def test_joint_diff_helpers(self):
        subset = SKELETON_JOINT_NAMES[:-1]
        missing = get_missing_expected_joints(subset)
        unexpected = get_unexpected_joints(subset)
        self.assertEqual(len(missing), 1)
        self.assertEqual(unexpected, ())

    def test_future_hand_mapping_template(self):
        mapping = build_empty_future_hand_mapping()
        self.assertEqual(set(mapping.keys()), {"left", "right"})
        self.assertEqual(tuple(mapping["left"].keys()), HAND_JOINT_NAMES)
        self.assertEqual(tuple(mapping["right"].keys()), HAND_JOINT_NAMES)
        self.assertTrue(all(value is None for value in mapping["left"].values()))
        self.assertTrue(all(value is None for value in mapping["right"].values()))


if __name__ == "__main__":
    unittest.main()
