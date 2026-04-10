from __future__ import annotations

import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from skeleton_common import (
    BodyBasis,
    build_urdf_model,
    build_link_geometries,
    generate_urdf_text,
    infer_body_basis_world,
    infer_hand_width_axes_world,
    infer_joint_axis,
    infer_lateral_axis_world,
    matrix_to_xyz_rpy,
    rpy_to_matrix,
)


class SkeletonCommonTests(unittest.TestCase):
    @staticmethod
    def _body_basis() -> BodyBasis:
        return BodyBasis(
            lateral_world=np.array([1.0, 0.0, 0.0], dtype=float),
            up_world=np.array([0.0, 0.0, 1.0], dtype=float),
            forward_world=np.array([0.0, 1.0, 0.0], dtype=float),
        )

    def test_rpy_round_trip(self) -> None:
        rpy = np.array([0.31, -0.44, 1.27], dtype=float)
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = rpy_to_matrix(rpy)
        matrix[:3, 3] = np.array([0.1, -0.2, 0.3], dtype=float)

        xyz, extracted_rpy = matrix_to_xyz_rpy(matrix)

        np.testing.assert_allclose(xyz, [0.1, -0.2, 0.3], atol=1e-9)
        np.testing.assert_allclose(extracted_rpy, rpy, atol=1e-9)

    def test_leaf_link_gets_box_fallback_geometry(self) -> None:
        records = [
            {
                'index': 0,
                'name': 'root_x',
                'incoming_length': 0.0,
                'local_xyz': np.zeros(3, dtype=float),
                'parent_index': -1,
            },
            {
                'index': 1,
                'name': 'hand_l',
                'incoming_length': 0.08,
                'local_xyz': np.array([0.08, 0.0, 0.0], dtype=float),
                'parent_index': 0,
            },
        ]

        geoms = build_link_geometries(records)

        self.assertEqual(geoms['hand_l'][0]['kind'], 'sphere')
        self.assertEqual(geoms['hand_l'][1]['kind'], 'box')

    def test_generate_urdf_includes_base_link_root_fixup(self) -> None:
        root_matrix = np.eye(4, dtype=float)
        root_matrix[:3, 3] = np.array([0.0, 0.0, 0.25], dtype=float)
        child_matrix = np.eye(4, dtype=float)
        child_matrix[:3, 3] = np.array([0.1, 0.0, 0.0], dtype=float)

        records = [
            {
                'index': 0,
                'path': 'root_x',
                'name': 'root_x',
                'parent_index': -1,
                'parent_path': None,
                'parent_name': None,
                'children': ['root_x/child'],
                'child_names': ['child'],
                'local_matrix': root_matrix,
                'world_matrix': root_matrix,
                'local_xyz': np.array([0.0, 0.0, 0.25], dtype=float),
                'local_rpy': np.zeros(3, dtype=float),
                'world_xyz': np.array([0.0, 0.0, 0.25], dtype=float),
                'world_rpy': np.zeros(3, dtype=float),
                'incoming_length': 0.0,
                'axis': np.array([0.0, 1.0, 0.0], dtype=float),
                'limits': (-1.0, 1.0),
            },
            {
                'index': 1,
                'path': 'root_x/child',
                'name': 'child',
                'parent_index': 0,
                'parent_path': 'root_x',
                'parent_name': 'root_x',
                'children': [],
                'child_names': [],
                'local_matrix': child_matrix,
                'world_matrix': root_matrix @ child_matrix,
                'local_xyz': np.array([0.1, 0.0, 0.0], dtype=float),
                'local_rpy': np.zeros(3, dtype=float),
                'world_xyz': np.array([0.1, 0.0, 0.25], dtype=float),
                'world_rpy': np.zeros(3, dtype=float),
                'incoming_length': 0.1,
                'axis': np.array([0.0, 1.0, 0.0], dtype=float),
                'limits': (-1.0, 1.0),
            },
        ]

        robot_xml = ET.fromstring(generate_urdf_text('demo', records))
        link_names = {link.attrib['name'] for link in robot_xml.findall('link')}
        fixed_joint = robot_xml.find("./joint[@name='root_x_base_fixed']")

        self.assertIn('base_link', link_names)
        self.assertIsNotNone(fixed_joint)
        self.assertEqual(fixed_joint.attrib['type'], 'fixed')

    def test_generate_urdf_is_pretty_printed(self) -> None:
        records = [
            {
                'index': 0,
                'path': 'root_x',
                'name': 'root_x',
                'parent_index': -1,
                'parent_path': None,
                'parent_name': None,
                'children': [],
                'child_names': [],
                'local_matrix': np.eye(4, dtype=float),
                'world_matrix': np.eye(4, dtype=float),
                'local_xyz': np.zeros(3, dtype=float),
                'local_rpy': np.zeros(3, dtype=float),
                'world_xyz': np.zeros(3, dtype=float),
                'world_rpy': np.zeros(3, dtype=float),
                'incoming_length': 0.0,
                'axis': np.array([0.0, 1.0, 0.0], dtype=float),
                'limits': (-1.0, 1.0),
            },
        ]

        urdf_text = generate_urdf_text('demo', records)

        self.assertTrue(urdf_text.startswith('<?xml'))
        self.assertIn('\n  <link name="base_link">', urdf_text)
        self.assertGreater(len(urdf_text.splitlines()), 5)

    def test_generate_urdf_preserves_high_precision_joint_axis(self) -> None:
        records = [
            {
                'index': 0,
                'path': 'root_x',
                'name': 'root_x',
                'parent_index': -1,
                'parent_path': None,
                'parent_name': None,
                'children': ['root_x/child'],
                'child_names': ['child'],
                'local_matrix': np.eye(4, dtype=float),
                'world_matrix': np.eye(4, dtype=float),
                'local_xyz': np.zeros(3, dtype=float),
                'local_rpy': np.zeros(3, dtype=float),
                'world_xyz': np.zeros(3, dtype=float),
                'world_rpy': np.zeros(3, dtype=float),
                'incoming_length': 0.0,
                'axis': np.array([0.0, 1.0, 0.0], dtype=float),
                'limits': (-1.0, 1.0),
            },
            {
                'index': 1,
                'path': 'root_x/child',
                'name': 'child',
                'parent_index': 0,
                'parent_path': 'root_x',
                'parent_name': 'root_x',
                'children': [],
                'child_names': [],
                'local_matrix': np.eye(4, dtype=float),
                'world_matrix': np.eye(4, dtype=float),
                'local_xyz': np.zeros(3, dtype=float),
                'local_rpy': np.zeros(3, dtype=float),
                'world_xyz': np.zeros(3, dtype=float),
                'world_rpy': np.zeros(3, dtype=float),
                'incoming_length': 0.1,
                'axis': np.array([0.123456, 0.654321, -0.746879], dtype=float),
                'limits': (-1.0, 1.0),
            },
        ]

        robot_xml = ET.fromstring(generate_urdf_text('demo', records))
        axis = robot_xml.find("./joint[@name='child']/axis")

        self.assertIsNotNone(axis)
        self.assertEqual(axis.attrib['xyz'], '0.123456 0.654321 -0.746879')

    def test_infer_lateral_axis_world_prefers_left_right_joint_pair(self) -> None:
        records = [
            {'name': 'thigh_stretch_l', 'world_xyz': np.array([0.12, 0.0, 0.4], dtype=float)},
            {'name': 'thigh_stretch_r', 'world_xyz': np.array([-0.12, 0.0, 0.4], dtype=float)},
        ]

        lateral = infer_lateral_axis_world(records)

        np.testing.assert_allclose(lateral, [1.0, 0.0, 0.0], atol=1e-9)

    def test_infer_joint_axis_uses_non_bone_axis_for_bend_joint(self) -> None:
        local_matrix = np.eye(4, dtype=float)

        axis = infer_joint_axis(
            'leg_stretch_l',
            local_matrix=local_matrix,
            primary_local_axis=np.array([0.0, 1.0, 0.0], dtype=float),
            body_basis=self._body_basis(),
        )

        np.testing.assert_allclose(axis, [1.0, 0.0, 0.0], atol=1e-9)

    def test_infer_joint_axis_aligns_bend_axis_to_world_lateral_direction(self) -> None:
        local_matrix = np.eye(4, dtype=float)
        local_matrix[:3, :3] = np.array(
            [
                [0.28580889, 0.0188995, 0.9580977],
                [0.11888605, -0.99278063, -0.0158798],
                [0.95088321, 0.11844356, -0.28598943],
            ],
            dtype=float,
        )

        axis = infer_joint_axis(
            'thigh_stretch_l',
            local_matrix=local_matrix,
            primary_local_axis=np.array([0.0, 1.0, 0.0], dtype=float),
            body_basis=self._body_basis(),
        )

        np.testing.assert_allclose(local_matrix[:3, :3] @ axis, [1.0, 0.0, 0.0], atol=1e-5)

    def test_infer_joint_axis_keeps_twist_axis_aligned_to_primary_direction(self) -> None:
        axis = infer_joint_axis(
            'leg_twist_l',
            local_matrix=np.eye(4, dtype=float),
            primary_local_axis=np.array([0.0, 1.0, 0.0], dtype=float),
            body_basis=self._body_basis(),
        )

        np.testing.assert_allclose(axis, [0.0, 1.0, 0.0], atol=1e-9)

    def test_infer_joint_axis_uses_world_orientation_for_knee_axis_selection(self) -> None:
        world_matrix = np.eye(4, dtype=float)
        world_matrix[:3, :3] = np.array(
            [
                [0.28580889, 0.0188995, 0.9580977],
                [-0.95088321, -0.11844356, 0.28598943],
                [0.11888605, -0.99278063, -0.0158798],
            ],
            dtype=float,
        )

        axis = infer_joint_axis(
            'leg_stretch_l',
            local_matrix=np.eye(4, dtype=float),
            primary_local_axis=np.array([0.0, 1.0, 0.0], dtype=float),
            body_basis=self._body_basis(),
            world_matrix=world_matrix,
        )

        np.testing.assert_allclose(world_matrix[:3, :3] @ axis, [1.0, 0.0, 0.0], atol=1e-5)

    def test_infer_joint_axis_uses_local_x_for_finger_base_joint(self) -> None:
        axis = infer_joint_axis(
            'index1_base_l',
            local_matrix=np.eye(4, dtype=float),
            primary_local_axis=np.array([0.0, 1.0, 0.0], dtype=float),
            body_basis=self._body_basis(),
        )

        np.testing.assert_allclose(axis, [1.0, 0.0, 0.0], atol=1e-9)

    def test_infer_joint_axis_aligns_finger_curl_to_hand_width_world(self) -> None:
        world_matrix = np.eye(4, dtype=float)
        world_matrix[:3, :3] = np.array(
            [
                [0.70710678, 0.0, 0.70710678],
                [0.0, 1.0, 0.0],
                [-0.70710678, 0.0, 0.70710678],
            ],
            dtype=float,
        )

        axis = infer_joint_axis(
            'index1_l',
            local_matrix=np.eye(4, dtype=float),
            primary_local_axis=np.array([0.0, 1.0, 0.0], dtype=float),
            body_basis=self._body_basis(),
            world_matrix=world_matrix,
            hand_width_world=np.array([1.0, 0.0, 0.0], dtype=float),
        )

        np.testing.assert_allclose(world_matrix[:3, :3] @ axis, [1.0, 0.0, 0.0], atol=1e-9)

    def test_infer_joint_axis_aligns_thumb_curl_to_hand_width_world(self) -> None:
        world_matrix = np.eye(4, dtype=float)
        world_matrix[:3, :3] = np.array(
            [
                [0.5, 0.2, 0.84261498],
                [0.0, 0.97280621, -0.23162053],
                [-0.8660254, 0.11581027, 0.4864031],
            ],
            dtype=float,
        )
        hand_width_world = np.array([0.2, -0.9, 0.4], dtype=float)
        hand_width_world /= np.linalg.norm(hand_width_world)

        axis = infer_joint_axis(
            'thumb2_l',
            local_matrix=np.eye(4, dtype=float),
            primary_local_axis=np.array([0.0, 1.0, 0.0], dtype=float),
            body_basis=self._body_basis(),
            world_matrix=world_matrix,
            hand_width_world=hand_width_world,
        )

        np.testing.assert_allclose(world_matrix[:3, :3] @ axis, hand_width_world, atol=5e-4)

    def test_infer_joint_axis_uses_hinge_plane_normal_for_elbow(self) -> None:
        world_matrix = np.eye(4, dtype=float)
        world_matrix[:3, :3] = np.array(
            [
                [-0.123014, 0.605783, 0.786063],
                [-0.979999, -0.199003, -0.000002],
                [0.156427, -0.770341, 0.618147],
            ],
            dtype=float,
        )
        hinge_normal_world = np.array([-0.786063, 0.0, -0.618147], dtype=float)

        axis = infer_joint_axis(
            'forearm_stretch_l',
            local_matrix=np.eye(4, dtype=float),
            primary_local_axis=np.array([0.0, 1.0, 0.0], dtype=float),
            body_basis=self._body_basis(),
            world_matrix=world_matrix,
            hinge_normal_world=hinge_normal_world,
        )

        expected = hinge_normal_world / np.linalg.norm(hinge_normal_world)
        np.testing.assert_allclose(world_matrix[:3, :3] @ axis, expected, atol=1e-6)

    def test_infer_joint_axis_uses_thumb_base_axis_orthogonal_to_thumb_and_width(self) -> None:
        world_matrix = np.eye(4, dtype=float)
        world_matrix[:3, :3] = np.array(
            [
                [-0.954667, 0.230988, -0.187765],
                [0.0, -0.630769, -0.77597],
                [-0.297676, -0.740793, 0.602175],
            ],
            dtype=float,
        )
        hand_width_world = np.array([0.141308, -0.98773, 0.066496], dtype=float)

        axis = infer_joint_axis(
            'thumb1_l',
            local_matrix=np.eye(4, dtype=float),
            primary_local_axis=np.array([0.0, 1.0, 0.0], dtype=float),
            body_basis=self._body_basis(),
            world_matrix=world_matrix,
            hand_width_world=hand_width_world,
        )

        thumb_direction_world = world_matrix[:3, :3] @ np.array([0.0, 1.0, 0.0], dtype=float)
        expected = np.cross(thumb_direction_world, hand_width_world)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(world_matrix[:3, :3] @ axis, expected, atol=1e-6)

    def test_infer_joint_axis_uses_body_up_for_waist_yaw(self) -> None:
        axis = infer_joint_axis(
            'spine_01_x',
            local_matrix=np.eye(4, dtype=float),
            primary_local_axis=np.array([0.0, 1.0, 0.0], dtype=float),
            body_basis=self._body_basis(),
        )

        np.testing.assert_allclose(axis, [0.0, 0.0, 1.0], atol=1e-9)

    def test_infer_body_basis_world_uses_toe_direction_for_forward_sign(self) -> None:
        records = [
            {'name': 'root_x', 'world_xyz': np.array([0.0, 0.0, 0.0], dtype=float)},
            {'name': 'shoulder_l', 'world_xyz': np.array([0.2, 0.0, 1.0], dtype=float)},
            {'name': 'shoulder_r', 'world_xyz': np.array([-0.2, 0.0, 1.0], dtype=float)},
            {'name': 'spine_03_x', 'world_xyz': np.array([0.0, 0.0, 0.9], dtype=float)},
            {'name': 'foot_l', 'world_xyz': np.array([0.1, 0.1, 0.0], dtype=float)},
            {'name': 'toes_01_l', 'world_xyz': np.array([0.1, 0.25, 0.0], dtype=float)},
            {'name': 'foot_r', 'world_xyz': np.array([-0.1, 0.1, 0.0], dtype=float)},
            {'name': 'toes_01_r', 'world_xyz': np.array([-0.1, 0.25, 0.0], dtype=float)},
        ]

        basis = infer_body_basis_world(records)

        np.testing.assert_allclose(basis.lateral_world, [1.0, 0.0, 0.0], atol=1e-9)
        self.assertGreater(basis.forward_world[1], 0.9)
        self.assertGreater(basis.up_world[2], 0.9)

    def test_infer_hand_width_axes_world_prefers_knuckle_span_toward_thumb(self) -> None:
        def record(name: str, xyz: tuple[float, float, float]) -> dict:
            matrix = np.eye(4, dtype=float)
            matrix[:3, 3] = np.array(xyz, dtype=float)
            return {'name': name, 'world_xyz': np.array(xyz, dtype=float), 'world_matrix': matrix, 'child_names': []}

        records = [
            record('hand_l', (0.0, 0.0, 0.0)),
            record('thumb1_l', (0.08, 0.02, 0.0)),
            record('index1_base_l', (0.06, 0.01, 0.0)),
            record('middle1_base_l', (0.01, 0.0, 0.0)),
            record('ring1_base_l', (-0.03, -0.01, 0.0)),
            record('pinky1_base_l', (-0.06, -0.02, 0.0)),
            record('index1_l', (0.06, 0.08, 0.0)),
            record('middle1_l', (0.01, 0.08, 0.0)),
            record('ring1_l', (-0.03, 0.08, 0.0)),
            record('pinky1_l', (-0.06, 0.08, 0.0)),
            record('index2_l', (0.06, 0.14, 0.0)),
            record('middle2_l', (0.01, 0.14, 0.0)),
            record('ring2_l', (-0.03, 0.14, 0.0)),
            record('pinky2_l', (-0.06, 0.14, 0.0)),
        ]

        hand_axes = infer_hand_width_axes_world(records, self._body_basis())

        np.testing.assert_allclose(hand_axes['l'], [1.0, 0.0, 0.0], atol=1e-9)

    def test_build_urdf_model_adds_virtual_hip_roll_and_serializes_leg_chain(self) -> None:
        def mat(x: float, y: float, z: float) -> np.ndarray:
            matrix = np.eye(4, dtype=float)
            matrix[:3, 3] = np.array([x, y, z], dtype=float)
            return matrix

        records = [
            {
                'index': 0,
                'path': 'root_x',
                'name': 'root_x',
                'parent_index': -1,
                'parent_path': None,
                'parent_name': None,
                'children': ['root_x/thigh_stretch_l'],
                'child_names': ['thigh_stretch_l'],
                'local_matrix': np.eye(4, dtype=float),
                'world_matrix': np.eye(4, dtype=float),
                'local_xyz': np.zeros(3, dtype=float),
                'local_rpy': np.zeros(3, dtype=float),
                'world_xyz': np.zeros(3, dtype=float),
                'world_rpy': np.zeros(3, dtype=float),
                'incoming_length': 0.0,
                'axis': np.array([0.0, 1.0, 0.0], dtype=float),
                'limits': (-1.0, 1.0),
            },
            {
                'index': 1,
                'path': 'root_x/thigh_stretch_l',
                'name': 'thigh_stretch_l',
                'parent_index': 0,
                'parent_path': 'root_x',
                'parent_name': 'root_x',
                'children': ['root_x/thigh_stretch_l/thigh_twist_l', 'root_x/thigh_stretch_l/leg_stretch_l'],
                'child_names': ['thigh_twist_l', 'leg_stretch_l'],
                'local_matrix': mat(0.1, 0.0, 0.0),
                'world_matrix': mat(0.1, 0.0, 0.0),
                'local_xyz': np.array([0.1, 0.0, 0.0], dtype=float),
                'local_rpy': np.zeros(3, dtype=float),
                'world_xyz': np.array([0.1, 0.0, 0.0], dtype=float),
                'world_rpy': np.zeros(3, dtype=float),
                'incoming_length': 0.1,
                'axis': np.array([0.0, 0.0, 1.0], dtype=float),
                'limits': (-1.0, 1.0),
            },
            {
                'index': 2,
                'path': 'root_x/thigh_stretch_l/thigh_twist_l',
                'name': 'thigh_twist_l',
                'parent_index': 1,
                'parent_path': 'root_x/thigh_stretch_l',
                'parent_name': 'thigh_stretch_l',
                'children': [],
                'child_names': [],
                'local_matrix': np.eye(4, dtype=float),
                'world_matrix': mat(0.1, 0.0, 0.0),
                'local_xyz': np.zeros(3, dtype=float),
                'local_rpy': np.zeros(3, dtype=float),
                'world_xyz': np.array([0.1, 0.0, 0.0], dtype=float),
                'world_rpy': np.zeros(3, dtype=float),
                'incoming_length': 0.0,
                'axis': np.array([0.0, 1.0, 0.0], dtype=float),
                'limits': (-1.0, 1.0),
            },
            {
                'index': 3,
                'path': 'root_x/thigh_stretch_l/leg_stretch_l',
                'name': 'leg_stretch_l',
                'parent_index': 1,
                'parent_path': 'root_x/thigh_stretch_l',
                'parent_name': 'thigh_stretch_l',
                'children': ['root_x/thigh_stretch_l/leg_stretch_l/leg_twist_l', 'root_x/thigh_stretch_l/leg_stretch_l/foot_l'],
                'child_names': ['leg_twist_l', 'foot_l'],
                'local_matrix': mat(0.0, 0.2, 0.0),
                'world_matrix': mat(0.1, 0.2, 0.0),
                'local_xyz': np.array([0.0, 0.2, 0.0], dtype=float),
                'local_rpy': np.zeros(3, dtype=float),
                'world_xyz': np.array([0.1, 0.2, 0.0], dtype=float),
                'world_rpy': np.zeros(3, dtype=float),
                'incoming_length': 0.2,
                'axis': np.array([1.0, 0.0, 0.0], dtype=float),
                'limits': (-1.0, 1.0),
            },
            {
                'index': 4,
                'path': 'root_x/thigh_stretch_l/leg_stretch_l/leg_twist_l',
                'name': 'leg_twist_l',
                'parent_index': 3,
                'parent_path': 'root_x/thigh_stretch_l/leg_stretch_l',
                'parent_name': 'leg_stretch_l',
                'children': [],
                'child_names': [],
                'local_matrix': mat(0.0, 0.1, 0.0),
                'world_matrix': mat(0.1, 0.3, 0.0),
                'local_xyz': np.array([0.0, 0.1, 0.0], dtype=float),
                'local_rpy': np.zeros(3, dtype=float),
                'world_xyz': np.array([0.1, 0.3, 0.0], dtype=float),
                'world_rpy': np.zeros(3, dtype=float),
                'incoming_length': 0.1,
                'axis': np.array([0.0, 1.0, 0.0], dtype=float),
                'limits': (-1.0, 1.0),
            },
            {
                'index': 5,
                'path': 'root_x/thigh_stretch_l/leg_stretch_l/foot_l',
                'name': 'foot_l',
                'parent_index': 3,
                'parent_path': 'root_x/thigh_stretch_l/leg_stretch_l',
                'parent_name': 'leg_stretch_l',
                'children': [],
                'child_names': [],
                'local_matrix': mat(0.0, 0.2, 0.0),
                'world_matrix': mat(0.1, 0.4, 0.0),
                'local_xyz': np.array([0.0, 0.2, 0.0], dtype=float),
                'local_rpy': np.zeros(3, dtype=float),
                'world_xyz': np.array([0.1, 0.4, 0.0], dtype=float),
                'world_rpy': np.zeros(3, dtype=float),
                'incoming_length': 0.2,
                'axis': np.array([0.0, 0.0, 1.0], dtype=float),
                'limits': (-1.0, 1.0),
            },
        ]

        links, joints = build_urdf_model(records)
        joints_by_child = {joint.child_link: joint for joint in joints}

        self.assertIn('left_hip_roll_link', links)
        self.assertEqual(joints_by_child['left_hip_roll_link'].name, 'left_hip_roll_joint')
        self.assertEqual(joints_by_child['left_hip_roll_link'].parent_link, 'thigh_stretch_l')
        self.assertEqual(joints_by_child['thigh_twist_l'].parent_link, 'left_hip_roll_link')
        self.assertEqual(joints_by_child['leg_stretch_l'].parent_link, 'thigh_twist_l')
        self.assertEqual(joints_by_child['foot_l'].parent_link, 'leg_twist_l')
        self.assertEqual(links['thigh_twist_l'].render_source_names, ('thigh_stretch_l', 'thigh_twist_l'))
        self.assertEqual(links['leg_twist_l'].render_source_names, ('leg_stretch_l', 'leg_twist_l'))

    def test_build_urdf_model_serializes_forearm_and_hand_under_twist_links(self) -> None:
        def mat(x: float, y: float, z: float) -> np.ndarray:
            matrix = np.eye(4, dtype=float)
            matrix[:3, 3] = np.array([x, y, z], dtype=float)
            return matrix

        records = [
            {
                'index': 0,
                'path': 'root_x',
                'name': 'root_x',
                'parent_index': -1,
                'parent_path': None,
                'parent_name': None,
                'children': ['root_x/arm_twist_l'],
                'child_names': ['arm_twist_l'],
                'local_matrix': np.eye(4, dtype=float),
                'world_matrix': np.eye(4, dtype=float),
                'local_xyz': np.zeros(3, dtype=float),
                'local_rpy': np.zeros(3, dtype=float),
                'world_xyz': np.zeros(3, dtype=float),
                'world_rpy': np.zeros(3, dtype=float),
                'incoming_length': 0.0,
                'axis': np.array([0.0, 1.0, 0.0], dtype=float),
                'limits': (-1.0, 1.0),
            },
            {
                'index': 1,
                'path': 'root_x/arm_stretch_l',
                'name': 'arm_stretch_l',
                'parent_index': 0,
                'parent_path': 'root_x',
                'parent_name': 'root_x',
                'children': ['root_x/arm_twist_l', 'root_x/forearm_stretch_l'],
                'child_names': ['arm_twist_l', 'forearm_stretch_l'],
                'local_matrix': mat(0.0, 0.1, 0.0),
                'world_matrix': mat(0.0, 0.1, 0.0),
                'local_xyz': np.array([0.0, 0.1, 0.0], dtype=float),
                'local_rpy': np.zeros(3, dtype=float),
                'world_xyz': np.array([0.0, 0.1, 0.0], dtype=float),
                'world_rpy': np.zeros(3, dtype=float),
                'incoming_length': 0.1,
                'axis': np.array([1.0, 0.0, 0.0], dtype=float),
                'limits': (-1.0, 1.0),
            },
            {
                'index': 2,
                'path': 'root_x/arm_twist_l',
                'name': 'arm_twist_l',
                'parent_index': 1,
                'parent_path': 'root_x/arm_stretch_l',
                'parent_name': 'arm_stretch_l',
                'children': [],
                'child_names': [],
                'local_matrix': np.eye(4, dtype=float),
                'world_matrix': mat(0.0, 0.1, 0.0),
                'local_xyz': np.zeros(3, dtype=float),
                'local_rpy': np.zeros(3, dtype=float),
                'world_xyz': np.array([0.0, 0.1, 0.0], dtype=float),
                'world_rpy': np.zeros(3, dtype=float),
                'incoming_length': 0.0,
                'axis': np.array([0.0, 1.0, 0.0], dtype=float),
                'limits': (-1.0, 1.0),
            },
            {
                'index': 3,
                'path': 'root_x/forearm_stretch_l',
                'name': 'forearm_stretch_l',
                'parent_index': 1,
                'parent_path': 'root_x/arm_stretch_l',
                'parent_name': 'arm_stretch_l',
                'children': ['root_x/forearm_twist_l', 'root_x/hand_l'],
                'child_names': ['forearm_twist_l', 'hand_l'],
                'local_matrix': mat(0.0, 0.2, 0.0),
                'world_matrix': mat(0.0, 0.3, 0.0),
                'local_xyz': np.array([0.0, 0.2, 0.0], dtype=float),
                'local_rpy': np.zeros(3, dtype=float),
                'world_xyz': np.array([0.0, 0.3, 0.0], dtype=float),
                'world_rpy': np.zeros(3, dtype=float),
                'incoming_length': 0.2,
                'axis': np.array([1.0, 0.0, 0.0], dtype=float),
                'limits': (-1.0, 1.0),
            },
            {
                'index': 4,
                'path': 'root_x/forearm_twist_l',
                'name': 'forearm_twist_l',
                'parent_index': 3,
                'parent_path': 'root_x/forearm_stretch_l',
                'parent_name': 'forearm_stretch_l',
                'children': [],
                'child_names': [],
                'local_matrix': mat(0.0, 0.1, 0.0),
                'world_matrix': mat(0.0, 0.4, 0.0),
                'local_xyz': np.array([0.0, 0.1, 0.0], dtype=float),
                'local_rpy': np.zeros(3, dtype=float),
                'world_xyz': np.array([0.0, 0.4, 0.0], dtype=float),
                'world_rpy': np.zeros(3, dtype=float),
                'incoming_length': 0.1,
                'axis': np.array([0.0, 1.0, 0.0], dtype=float),
                'limits': (-1.0, 1.0),
            },
            {
                'index': 5,
                'path': 'root_x/hand_l',
                'name': 'hand_l',
                'parent_index': 3,
                'parent_path': 'root_x/forearm_stretch_l',
                'parent_name': 'forearm_stretch_l',
                'children': [],
                'child_names': [],
                'local_matrix': mat(0.0, 0.2, 0.0),
                'world_matrix': mat(0.0, 0.5, 0.0),
                'local_xyz': np.array([0.0, 0.2, 0.0], dtype=float),
                'local_rpy': np.zeros(3, dtype=float),
                'world_xyz': np.array([0.0, 0.5, 0.0], dtype=float),
                'world_rpy': np.zeros(3, dtype=float),
                'incoming_length': 0.2,
                'axis': np.array([0.0, 0.0, 1.0], dtype=float),
                'limits': (-1.0, 1.0),
            },
        ]

        links, joints = build_urdf_model(records)
        joints_by_child = {joint.child_link: joint for joint in joints}

        self.assertEqual(joints_by_child['forearm_stretch_l'].parent_link, 'arm_twist_l')
        self.assertEqual(joints_by_child['hand_l'].parent_link, 'forearm_twist_l')
        self.assertEqual(links['arm_stretch_l'].render_source_names, ())
        self.assertEqual(links['arm_twist_l'].render_source_names, ('arm_stretch_l', 'arm_twist_l'))
        self.assertEqual(links['forearm_stretch_l'].render_source_names, ())
        self.assertEqual(links['forearm_twist_l'].render_source_names, ('forearm_stretch_l', 'forearm_twist_l'))

    def test_generate_urdf_transforms_aggregated_spine_geometry_into_distal_link_frame(self) -> None:
        def mat(x: float, y: float, z: float) -> np.ndarray:
            matrix = np.eye(4, dtype=float)
            matrix[:3, 3] = np.array([x, y, z], dtype=float)
            return matrix

        records = [
            {
                'index': 0,
                'path': 'root_x',
                'name': 'root_x',
                'parent_index': -1,
                'parent_path': None,
                'parent_name': None,
                'children': ['root_x/spine_01_x'],
                'child_names': ['spine_01_x'],
                'local_matrix': np.eye(4, dtype=float),
                'world_matrix': np.eye(4, dtype=float),
                'local_xyz': np.zeros(3, dtype=float),
                'local_rpy': np.zeros(3, dtype=float),
                'world_xyz': np.zeros(3, dtype=float),
                'world_rpy': np.zeros(3, dtype=float),
                'incoming_length': 0.0,
                'axis': np.array([0.0, 1.0, 0.0], dtype=float),
                'limits': (-1.0, 1.0),
            },
            {
                'index': 1,
                'path': 'root_x/spine_01_x',
                'name': 'spine_01_x',
                'parent_index': 0,
                'parent_path': 'root_x',
                'parent_name': 'root_x',
                'children': ['root_x/spine_01_x/spine_02_x'],
                'child_names': ['spine_02_x'],
                'local_matrix': mat(0.0, 0.1, 0.0),
                'world_matrix': mat(0.0, 0.1, 0.0),
                'local_xyz': np.array([0.0, 0.1, 0.0], dtype=float),
                'local_rpy': np.zeros(3, dtype=float),
                'world_xyz': np.array([0.0, 0.1, 0.0], dtype=float),
                'world_rpy': np.zeros(3, dtype=float),
                'incoming_length': 0.1,
                'axis': np.array([0.0, 0.0, 1.0], dtype=float),
                'limits': (-1.0, 1.0),
            },
            {
                'index': 2,
                'path': 'root_x/spine_01_x/spine_02_x',
                'name': 'spine_02_x',
                'parent_index': 1,
                'parent_path': 'root_x/spine_01_x',
                'parent_name': 'spine_01_x',
                'children': ['root_x/spine_01_x/spine_02_x/spine_03_x'],
                'child_names': ['spine_03_x'],
                'local_matrix': mat(0.0, 0.2, 0.0),
                'world_matrix': mat(0.0, 0.3, 0.0),
                'local_xyz': np.array([0.0, 0.2, 0.0], dtype=float),
                'local_rpy': np.zeros(3, dtype=float),
                'world_xyz': np.array([0.0, 0.3, 0.0], dtype=float),
                'world_rpy': np.zeros(3, dtype=float),
                'incoming_length': 0.2,
                'axis': np.array([1.0, 0.0, 0.0], dtype=float),
                'limits': (-1.0, 1.0),
            },
            {
                'index': 3,
                'path': 'root_x/spine_01_x/spine_02_x/spine_03_x',
                'name': 'spine_03_x',
                'parent_index': 2,
                'parent_path': 'root_x/spine_01_x/spine_02_x',
                'parent_name': 'spine_02_x',
                'children': [],
                'child_names': [],
                'local_matrix': mat(0.0, 0.3, 0.0),
                'world_matrix': mat(0.0, 0.6, 0.0),
                'local_xyz': np.array([0.0, 0.3, 0.0], dtype=float),
                'local_rpy': np.zeros(3, dtype=float),
                'world_xyz': np.array([0.0, 0.6, 0.0], dtype=float),
                'world_rpy': np.zeros(3, dtype=float),
                'incoming_length': 0.3,
                'axis': np.array([1.0, 0.0, 0.0], dtype=float),
                'limits': (-1.0, 1.0),
            },
        ]

        geoms_by_name = {
            'spine_01_x': [{'kind': 'mesh', 'origin_xyz': [0.0, 0.0, 0.0], 'origin_rpy': [0.0, 0.0, 0.0], 'filename': 'spine_01.stl'}],
            'spine_02_x': [{'kind': 'mesh', 'origin_xyz': [0.0, 0.0, 0.0], 'origin_rpy': [0.0, 0.0, 0.0], 'filename': 'spine_02.stl'}],
            'spine_03_x': [{'kind': 'mesh', 'origin_xyz': [0.0, 0.0, 0.0], 'origin_rpy': [0.0, 0.0, 0.0], 'filename': 'spine_03.stl'}],
        }

        robot_xml = ET.fromstring(generate_urdf_text('demo', records, geoms_by_name=geoms_by_name))

        spine_link = robot_xml.find("./link[@name='spine_03_x']")
        self.assertIsNotNone(spine_link)
        visuals = spine_link.findall('visual')
        self.assertEqual(len(visuals), 3)
        origins = {visual.find('geometry/mesh').attrib['filename']: visual.find('origin').attrib['xyz'] for visual in visuals}
        self.assertEqual(origins['spine_03.stl'], '0.000000 0.000000 0.000000')
        self.assertEqual(origins['spine_02.stl'], '0.000000 -0.300000 0.000000')
        self.assertEqual(origins['spine_01.stl'], '0.000000 -0.500000 0.000000')


if __name__ == '__main__':
    unittest.main()
