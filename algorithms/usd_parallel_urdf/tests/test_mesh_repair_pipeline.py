from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from mesh_repair_pipeline import solid_angle_winding_number, triangles_to_mesh_arrays


class MeshRepairPipelineTests(unittest.TestCase):
    def test_solid_angle_winding_number_distinguishes_inside_outside(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        faces = np.array(
            [
                [0, 2, 1],
                [0, 1, 3],
                [1, 2, 3],
                [2, 0, 3],
            ],
            dtype=np.int64,
        )
        triangles = vertices[faces]
        points = np.array([[0.1, 0.1, 0.1], [2.0, 2.0, 2.0]], dtype=float)

        winding = solid_angle_winding_number(points, triangles)

        self.assertGreater(winding[0], 0.5)
        self.assertLess(winding[1], 0.5)

    def test_triangles_to_mesh_arrays_merges_shared_vertices(self) -> None:
        triangles = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            ],
            dtype=float,
        )

        vertices, faces = triangles_to_mesh_arrays(triangles, tolerance=1e-9)

        self.assertEqual(len(vertices), 4)
        self.assertEqual(len(faces), 2)


if __name__ == '__main__':
    unittest.main()
