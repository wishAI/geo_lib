import math
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import landau_pose
import urdf_kinematics


def _rot_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )


def _rot_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


def _rot_z(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


class TestUrdfKinematics(unittest.TestCase):
    def test_rpy_matrix_matches_urdf_convention(self):
        roll = 2.748947
        pitch = -1.256079
        yaw = 0.394193
        expected = _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)

        np.testing.assert_allclose(urdf_kinematics.rpy_matrix(roll, pitch, yaw), expected, atol=1.0e-9)
        np.testing.assert_allclose(landau_pose.rpy_matrix(roll, pitch, yaw), expected, atol=1.0e-9)

    def test_left_hip_pitch_origin_uses_urdf_rpy_order(self):
        urdf_path = MODULE_ROOT / "inputs" / "landau_v10" / "landau_v10_parallel_mesh.urdf"
        specs = urdf_kinematics.load_urdf_joint_specs(urdf_path)
        left_hip_pitch = specs["left_hip_pitch_joint"]

        expected_row0 = np.array([0.28580513040669403, 0.01889961027516769, 0.9581013684181123, 0.059407])
        np.testing.assert_allclose(left_hip_pitch.origin[0], expected_row0, atol=5.0e-6)


class TestSkinMapping(unittest.TestCase):
    def test_serial_arms_keep_skin_connected_and_preserve_hand_targets(self):
        folder = MODULE_ROOT / 'inputs/landau_v10'
        records = landau_pose.load_skeleton_records(folder / 'landau_v10_skeleton.json')
        specs = urdf_kinematics.load_urdf_joint_specs(folder / 'landau_v10_parallel_mesh.urdf')
        mapper = landau_pose.UrdfSkinMapper(records, specs)
        rest = urdf_kinematics.world_map_from_joint_specs(specs)
        skin_rest = landau_pose.world_matrices_from_local(records, [r.local_matrix for r in records])
        for actual, record in zip(mapper.apply({}), records):
            np.testing.assert_allclose(actual, record.local_matrix, atol=1e-12)
        for angle in (-1.2, -.5, .5, 1.2):
            pose = {f'{part}_{side}': value for side in ('l', 'r') for part, value in
                    [('arm_stretch', -.7), ('arm_twist', angle), ('forearm_stretch', .8),
                     ('forearm_twist', -angle), ('hand', .3)]}
            worlds = urdf_kinematics.world_map_from_joint_specs(specs, pose, pose_key='child_link')
            skin = landau_pose.world_matrices_from_local(records, mapper.apply(pose))
            for record in records:
                name = record.name
                if name not in ('arm_stretch_l', 'arm_stretch_r'):
                    expected = worlds[name] @ np.linalg.inv(rest[name]) @ skin_rest[record.index]
                    np.testing.assert_allclose(skin[record.index], expected, atol=1e-12)
                else:
                    elbow = name.replace('arm_stretch_', 'forearm_stretch_')
                    delta = skin[record.index] @ np.linalg.inv(skin_rest[record.index])
                    endpoint = delta @ rest[elbow][:, 3]
                    np.testing.assert_allclose(endpoint, worlds[elbow][:, 3], atol=1e-7)

    def test_leg_owners_preserve_off_axis_skin_and_distal_targets(self):
        folder = MODULE_ROOT / 'inputs/landau_v10'
        records = landau_pose.load_skeleton_records(folder / 'landau_v10_skeleton.json')
        specs = urdf_kinematics.load_urdf_joint_specs(folder / 'landau_v10_parallel_mesh.urdf')
        mapper=landau_pose.UrdfSkinMapper(records,specs)
        rest=urdf_kinematics.world_map_from_joint_specs(specs)
        binds=landau_pose.world_matrices_from_local(records,[r.local_matrix for r in records])
        for angle in (-.4,.4):
            pose={f'{part}_{side}':value for side in ('l','r') for part,value in
                  [('thigh_stretch',-.2),('thigh_twist',angle),('leg_stretch',.7),('leg_twist',angle),('foot',-.1),('toes_01',.2)]}
            worlds=urdf_kinematics.world_map_from_joint_specs(specs,pose,pose_key='child_link')
            actual=landau_pose.world_matrices_from_local(records,mapper.apply(pose))
            for r in records:
                if not r.name.startswith(('thigh_','leg_','foot_','toes_')):continue
                owner=r.name.replace('thigh_stretch_','thigh_twist_').replace('leg_stretch_','leg_twist_')
                expected=worlds[owner]@np.linalg.inv(rest[owner])@binds[r.index]
                np.testing.assert_allclose(actual[r.index],expected,atol=1e-12)


if __name__ == "__main__":
    unittest.main()
