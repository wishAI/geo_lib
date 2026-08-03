from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from config import DEFAULT_MESH_BUILD_CONFIG, resolve_link_mesh_policy, resolve_lowpoly_link_config


class MeshConfigTests(unittest.TestCase):
    def test_head_override_is_higher_detail_and_tighter_fit(self) -> None:
        default_cfg = DEFAULT_MESH_BUILD_CONFIG.lowpoly_default
        head_cfg = resolve_lowpoly_link_config(DEFAULT_MESH_BUILD_CONFIG, 'head_x')

        self.assertGreater(head_cfg.target_cells[0], default_cfg.target_cells[0])
        self.assertGreaterEqual(head_cfg.target_face_ratio, default_cfg.target_face_ratio)
        self.assertLess(head_cfg.max_extent_ratio_xyz[0], default_cfg.max_extent_ratio_xyz[0])

    def test_unknown_link_uses_default_config(self) -> None:
        resolved = resolve_lowpoly_link_config(DEFAULT_MESH_BUILD_CONFIG, 'not_a_real_link')
        self.assertEqual(resolved, DEFAULT_MESH_BUILD_CONFIG.lowpoly_default)
        self.assertGreater(len(resolved.alpha_radius_ratios), 0)

    def test_default_mesh_simplifier_uses_alpha_shape(self) -> None:
        self.assertEqual(DEFAULT_MESH_BUILD_CONFIG.mesh_simplify_mode, 'alpha_shape')

    def test_finger_links_use_higher_precision_alpha_config(self) -> None:
        default_cfg = DEFAULT_MESH_BUILD_CONFIG.lowpoly_default
        finger_cfg = resolve_lowpoly_link_config(DEFAULT_MESH_BUILD_CONFIG, 'index2_l')

        self.assertLess(finger_cfg.min_pitch, default_cfg.min_pitch)
        self.assertGreater(finger_cfg.target_cells[0], default_cfg.target_cells[0])
        self.assertGreater(finger_cfg.alpha_max_points, default_cfg.alpha_max_points)
        self.assertLess(finger_cfg.alpha_radius_ratios[0], default_cfg.alpha_radius_ratios[0])

    def test_body_links_use_body_precision_group(self) -> None:
        default_cfg = DEFAULT_MESH_BUILD_CONFIG.lowpoly_default
        body_cfg = resolve_lowpoly_link_config(DEFAULT_MESH_BUILD_CONFIG, 'spine_02_x')

        self.assertLess(body_cfg.max_faces, default_cfg.max_faces)
        self.assertGreater(body_cfg.target_cells[0], default_cfg.target_cells[0])

    def test_finger_links_default_to_alpha_shape_policy(self) -> None:
        policy = resolve_link_mesh_policy(DEFAULT_MESH_BUILD_CONFIG, 'middle2_r')

        self.assertEqual(policy.mesh_method, 'alpha_shape')
        self.assertEqual(policy.marching_axis_mode, 'local')

    def test_foot_toe_and_hand_alignment_are_configured(self) -> None:
        self.assertEqual(resolve_link_mesh_policy(DEFAULT_MESH_BUILD_CONFIG, 'foot_l').marching_axis_mode, 'local')
        self.assertEqual(resolve_link_mesh_policy(DEFAULT_MESH_BUILD_CONFIG, 'toes_01_r').marching_axis_mode, 'local')
        self.assertEqual(resolve_link_mesh_policy(DEFAULT_MESH_BUILD_CONFIG, 'hand_l').marching_axis_mode, 'local')


if __name__ == '__main__':
    unittest.main()
