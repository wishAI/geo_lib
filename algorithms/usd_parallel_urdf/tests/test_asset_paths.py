from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from asset_paths import (
    asset_tag,
    default_avp_snapshot_path,
    default_g1_urdf_path,
    resolve_asset_paths,
)
from skeleton_common import build_animation_clip, interpolate_pose_dict


class AssetPathTests(unittest.TestCase):
    def test_asset_paths_use_input_stem_for_outputs(self) -> None:
        usd_path = Path('/tmp/models/example-bunny.usdc')
        output_dir = Path('/tmp/out')

        paths = resolve_asset_paths(usd_path=usd_path, output_dir=output_dir)

        self.assertEqual(paths.asset_tag, 'example-bunny')
        self.assertEqual(paths.primitive_robot_name, 'example-bunny_parallel')
        self.assertEqual(paths.mesh_robot_name, 'example-bunny_parallel_mesh')
        self.assertEqual(paths.skeleton_json, output_dir / 'example-bunny_skeleton.json')
        self.assertEqual(paths.primitive_urdf, output_dir / 'example-bunny_parallel.urdf')
        self.assertEqual(paths.mesh_urdf, output_dir / 'example-bunny_parallel_mesh.urdf')
        self.assertEqual(paths.mesh_output_dir, output_dir / 'mesh_collision_stl' / 'example-bunny')

    def test_asset_tag_sanitizes_spaces(self) -> None:
        self.assertEqual(asset_tag(Path('/tmp/My Model v2.usdc')), 'My_Model_v2')

    def test_default_auxiliary_paths_point_inside_repo(self) -> None:
        self.assertIn('geo_lib', str(default_avp_snapshot_path()))
        self.assertEqual(default_g1_urdf_path().name, 'h1_2.urdf')


class AnimationHelperTests(unittest.TestCase):
    def test_interpolate_pose_dict_blends_missing_joints_from_zero(self) -> None:
        pose = interpolate_pose_dict({'hip_l': 1.0}, {'hip_r': -0.5}, 0.25)

        self.assertAlmostEqual(pose['hip_l'], 0.75)
        self.assertAlmostEqual(pose['hip_r'], -0.125)

    def test_build_animation_clip_filters_unknown_joint_names(self) -> None:
        records = [
            {'name': 'arm_stretch_r'},
            {'name': 'arm_stretch_l'},
            {'name': 'thigh_stretch_l'},
            {'name': 'thigh_stretch_r'},
            {'name': 'leg_stretch_l'},
            {'name': 'leg_stretch_r'},
            {'name': 'foot_l'},
            {'name': 'foot_r'},
            {'name': 'toes_01_l'},
            {'name': 'toes_01_r'},
            {'name': 'spine_01_x'},
            {'name': 'spine_02_x'},
            {'name': 'neck_x'},
            {'name': 'forearm_stretch_r'},
            {'name': 'forearm_stretch_l'},
            {'name': 'shoulder_r'},
            {'name': 'shoulder_l'},
        ]

        clip = build_animation_clip(records, 'walk_cycle')

        self.assertEqual([name for name, _, _ in clip], ['walk', 'walk_right'])
        self.assertTrue(all(duration > 0.0 for _, _, duration in clip))
        self.assertTrue(all(isinstance(pose, dict) for _, pose, _ in clip))


if __name__ == '__main__':
    unittest.main()
