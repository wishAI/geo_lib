import unittest
import sys
from pathlib import Path

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from avp_transform_utils import TransformOptions, build_xyz_transform, to_usd_world


def _rot_x_90():
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )


class TestAvpTransformUtils(unittest.TestCase):
    def test_build_xyz_transform_preserves_rotation_with_translation(self):
        mat = build_xyz_transform((90.0, 0.0, 0.0), (1.0, 2.0, 3.0))
        self.assertTrue(np.allclose(mat[:3, :3], _rot_x_90(), atol=1e-6))
        self.assertTrue(np.allclose(mat[3, :3], [1.0, 2.0, 3.0], atol=1e-6))

    def test_build_xyz_transform_identity_rotation(self):
        mat = build_xyz_transform((0.0, 0.0, 0.0), (0.5, -0.25, 1.25))
        self.assertTrue(np.allclose(mat[:3, :3], np.eye(3), atol=1e-6))
        self.assertTrue(np.allclose(mat[3, :3], [0.5, -0.25, 1.25], atol=1e-6))

    def test_build_xyz_transform_applies_scale(self):
        mat = build_xyz_transform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), scale_xyz=(0.5, 2.0, 3.0))
        self.assertTrue(
            np.allclose(
                mat[:3, :3],
                np.diag([0.5, 2.0, 3.0]),
                atol=1e-6,
            )
        )

    def test_to_usd_world_transposes_column_major_input(self):
        mat = np.arange(16, dtype=float).reshape(4, 4)
        world = to_usd_world(mat, options=TransformOptions(column_major=True))
        self.assertTrue(np.allclose(world, mat.T, atol=1e-6))

    def test_to_usd_world_applies_pre_then_post(self):
        mat = np.eye(4, dtype=float)
        pre = build_xyz_transform((0.0, 0.0, 90.0), (1.0, 0.0, 0.0))
        post = build_xyz_transform((0.0, 180.0, 0.0), (0.0, 2.0, 0.0))
        world = to_usd_world(
            mat,
            options=TransformOptions(
                column_major=False,
                pretransform=pre,
                posttransform=post,
            ),
        )
        expected = pre @ mat @ post
        self.assertTrue(np.allclose(world, expected, atol=1e-6))

    def test_to_usd_world_none_transforms_match_disabled_behavior(self):
        mat = np.arange(16, dtype=float).reshape(4, 4)
        world = to_usd_world(
            mat,
            options=TransformOptions(
                column_major=False,
                pretransform=None,
                posttransform=None,
            ),
        )
        self.assertTrue(np.allclose(world, mat, atol=1e-6))

    def test_to_usd_world_none_input(self):
        self.assertIsNone(to_usd_world(None, options=TransformOptions()))

    def test_to_usd_world_rejects_non_4x4_matrix(self):
        with self.assertRaises(ValueError):
            to_usd_world(np.eye(3, dtype=float), options=TransformOptions())


if __name__ == "__main__":
    unittest.main()
