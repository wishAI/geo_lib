from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from config import DEFAULT_MESH_BUILD_CONFIG, resolve_lowpoly_link_config


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


if __name__ == '__main__':
    unittest.main()
