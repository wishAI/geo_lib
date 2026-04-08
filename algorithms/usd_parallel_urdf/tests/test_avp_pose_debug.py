from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from avp_pose_debug import HAND_JOINT_NAMES, extract_tracking_frame, landau_hand_pose_summary


def _stack(scale: float) -> list[list[list[float]]]:
    mats = []
    for index, _ in enumerate(HAND_JOINT_NAMES):
        mat = np.eye(4, dtype=float)
        mat[0, 3] = scale * index
        mats.append(mat.tolist())
    return mats


class AvpPoseDebugTests(unittest.TestCase):
    def test_extract_tracking_frame_uses_arm_stack_when_wrist_missing(self) -> None:
        payload = {
            'head': np.eye(4, dtype=float).tolist(),
            'left_arm': _stack(0.01),
            'right_arm': _stack(0.02),
        }

        frame = extract_tracking_frame(payload)

        self.assertIsNotNone(frame['left_wrist'])
        self.assertIsNotNone(frame['right_wrist'])
        np.testing.assert_allclose(frame['left_wrist'], np.asarray(payload['left_arm'][0], dtype=float))
        np.testing.assert_allclose(frame['right_wrist'], np.asarray(payload['right_arm'][0], dtype=float))

    def test_landau_hand_pose_summary_groups_finger_chains(self) -> None:
        pose = {
            'thumb1_l': 0.1,
            'thumb2_l': 0.2,
            'thumb3_l': 0.3,
            'index1_base_l': 0.4,
            'index1_l': 0.5,
            'index2_l': 0.6,
            'index3_l': 0.7,
        }

        summary = landau_hand_pose_summary(pose, 'left')

        self.assertEqual(summary['thumb'], (0.1, 0.2, 0.3))
        self.assertEqual(summary['index'], (0.4, 0.5, 0.6, 0.7))


if __name__ == '__main__':
    unittest.main()
