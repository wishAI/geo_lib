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
    build_link_geometries,
    generate_urdf_text,
    infer_joint_axis,
    infer_lateral_axis_world,
    matrix_to_xyz_rpy,
    rpy_to_matrix,
)


class SkeletonCommonTests(unittest.TestCase):
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
            lateral_axis_world=np.array([1.0, 0.0, 0.0], dtype=float),
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
            lateral_axis_world=np.array([1.0, 0.0, 0.0], dtype=float),
        )

        np.testing.assert_allclose(axis, [0.0, 0.0, 1.0], atol=1e-9)

    def test_infer_joint_axis_keeps_twist_axis_aligned_to_primary_direction(self) -> None:
        axis = infer_joint_axis(
            'leg_twist_l',
            local_matrix=np.eye(4, dtype=float),
            primary_local_axis=np.array([0.0, 1.0, 0.0], dtype=float),
            lateral_axis_world=np.array([1.0, 0.0, 0.0], dtype=float),
        )

        np.testing.assert_allclose(axis, [0.0, 1.0, 0.0], atol=1e-9)


if __name__ == '__main__':
    unittest.main()
