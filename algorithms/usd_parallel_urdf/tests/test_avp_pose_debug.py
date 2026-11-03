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


class HeadMappingTests(unittest.TestCase):
    def test_native_debug_uses_real_head_axes_and_joint_name_limits(self):
        from avp_pose_debug import AvpLandauRetargeter, AVP_TO_SCENE_OPTIONS
        from g1_pose import UrdfJointSpec, _joint_motion
        from skeleton_common import rpy_to_matrix
        root = np.eye(4)
        head_origin = np.eye(4)
        head_origin[:3, :3] = rpy_to_matrix([np.pi / 2, 0, 0])
        chain = [UrdfJointSpec('neck_pitch_joint', 'revolute', 'spine_03_x', 'neck_x',
                              root, np.array([1., 0, 0]), -.7, .7),
                 UrdfJointSpec('head_yaw_joint', 'revolute', 'neck_x', 'head_x',
                              head_origin, np.array([0., 1, 0]), -.7, .7)]
        retarget = AvpLandauRetargeter.__new__(AvpLandauRetargeter)
        retarget.head_chain, retarget.head_base_world = chain, root
        for pitch in (-.5, 0, .5):
            for yaw in (-.5, 0, .5):
                world = root.copy()
                for spec, angle in zip(chain, (pitch, yaw)):
                    world = world @ spec.origin @ _joint_motion(spec, angle)
                forward = world[:3, 2]
                right = np.cross(forward, [0, 0, 1]);right /= np.linalg.norm(right)
                target = np.eye(4)
                target[:3, :3] = np.column_stack((right, forward, np.cross(right, forward)))
                raw = np.linalg.inv(AVP_TO_SCENE_OPTIONS.pretransform) @ target
                pose = {};retarget._retarget_head({'head': raw}, pose)
                np.testing.assert_allclose([pose['neck_x'], pose['head_x']], [pitch, yaw], atol=1e-6)
        # Names are deliberately semantic, while emitted pose keys are links.
        target = np.eye(4);target[:3, 1] = [1, 0, 0]
        pose = {};retarget._retarget_head({'head': np.linalg.inv(AVP_TO_SCENE_OPTIONS.pretransform) @ target}, pose)
        self.assertTrue(all(abs(v) <= .7 for v in pose.values()))
        missing = {};retarget._retarget_head({}, missing);self.assertEqual(missing, {})


if __name__ == '__main__':
    unittest.main()
