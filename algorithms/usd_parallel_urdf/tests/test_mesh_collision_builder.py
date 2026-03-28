from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from config import LowpolyMeshConfig
from mesh_collision_builder import _cluster_mesh_vertices, _fit_vertices_to_reference_bounds


class MeshCollisionBuilderTests(unittest.TestCase):
    def test_fit_vertices_to_reference_bounds_shrinks_inflated_mesh(self) -> None:
        reference_points = np.array(
            [
                [-0.10, -0.25, -0.15],
                [0.10, -0.25, -0.15],
                [-0.10, 0.25, 0.15],
                [0.10, 0.25, 0.15],
            ],
            dtype=float,
        )
        inflated_vertices = np.array(
            [
                [-0.16, -0.32, -0.22],
                [0.19, -0.32, -0.22],
                [0.19, 0.34, -0.22],
                [-0.16, 0.34, -0.22],
                [-0.16, -0.32, 0.24],
                [0.19, -0.32, 0.24],
                [0.19, 0.34, 0.24],
                [-0.16, 0.34, 0.24],
            ],
            dtype=float,
        )
        config = LowpolyMeshConfig(
            fit_margin_ratio=0.01,
            fit_margin_min=0.0,
            max_extent_ratio_xyz=(1.04, 1.04, 1.04),
        )

        fitted_vertices, details = _fit_vertices_to_reference_bounds(inflated_vertices, reference_points, config)
        fitted_extent = fitted_vertices.max(axis=0) - fitted_vertices.min(axis=0)
        reference_extent = reference_points.max(axis=0) - reference_points.min(axis=0)

        np.testing.assert_array_less(fitted_extent, reference_extent * 1.040001)
        self.assertTrue(details['fit_applied'])

    def test_cluster_mesh_vertices_merges_duplicate_cells(self) -> None:
        vertices = np.array(
            [
                [0.00, 0.00, 0.00],
                [0.01, 0.00, 0.00],
                [0.00, 0.01, 0.00],
                [1.00, 0.00, 0.00],
                [1.01, 0.00, 0.00],
                [1.00, 0.01, 0.00],
            ],
            dtype=float,
        )
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)

        clustered_vertices, clustered_faces = _cluster_mesh_vertices(vertices, faces, cell_size=0.05)

        self.assertEqual(len(clustered_vertices), 0)
        self.assertEqual(len(clustered_faces), 0)


if __name__ == '__main__':
    unittest.main()
