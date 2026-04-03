from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

try:
    from .problem import ProblemSpec
    from .solver import SolutionResult
except ImportError:  # pragma: no cover - script mode
    from problem import ProblemSpec
    from solver import SolutionResult


COLORS = [
    "0.11 0.44 0.32 0.55",
    "0.25 0.41 0.59 0.55",
    "0.79 0.49 0.36 0.55",
    "0.82 0.48 0.13 0.55",
    "0.48 0.19 0.42 0.55",
]


def render_mujoco_debug(problem: ProblemSpec, solution: SolutionResult, output_path: str | Path) -> Path:
    """
    Render a quick MuJoCo debug image.

    MuJoCo is used here as a headless scene/debug renderer for board layout sanity checks.
    The debug view uses each placement's axis-aligned bounding box rather than the exact polygon.
    Exact geometric validation remains in the shapely-based solver.
    """

    boards = [board.polygon.to_polygon(name=f"board:{board.board_id}") for board in problem.boards]
    placements_by_board: dict[str, list[Any]] = {board.board_id: [] for board in problem.boards}
    for placement in solution.placements:
        placements_by_board[placement.board_id].append(placement)

    geoms: list[str] = []
    camera_targets: list[tuple[float, float]] = []
    max_extent = 1.0

    for board_index, (board_spec, board_polygon) in enumerate(zip(problem.boards, boards)):
        min_x, min_y, max_x, max_y = board_polygon.bounds
        width = max_x - min_x
        height = max_y - min_y
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        offset_x = board_index * (width + 80.0)
        camera_targets.append((offset_x + center_x, center_y))
        max_extent = max(max_extent, width, height)

        geoms.append(
            f'<geom name="{escape(board_spec.board_id)}" type="box" pos="{offset_x + center_x:.4f} {center_y:.4f} -1" '
            f'size="{width / 2.0:.4f} {height / 2.0:.4f} 1" rgba="0.95 0.93 0.88 1"/>'
        )

        for placement_index, placement in enumerate(placements_by_board[board_spec.board_id]):
            p_min_x, p_min_y, p_max_x, p_max_y = placement.polygon.bounds
            p_width = max(p_max_x - p_min_x, 1.0)
            p_height = max(p_max_y - p_min_y, 1.0)
            center_x = offset_x + (p_min_x + p_max_x) / 2.0
            center_y = (p_min_y + p_max_y) / 2.0
            z = 0.6 + placement_index * 0.03
            color = COLORS[placement_index % len(COLORS)]
            geoms.append(
                f'<geom type="box" pos="{center_x:.4f} {center_y:.4f} {z:.4f}" size="{p_width / 2.0:.4f} {p_height / 2.0:.4f} 0.15" rgba="{color}"/>'
            )
            geoms.append(
                f'<geom type="sphere" pos="{offset_x + placement.centroid_x:.4f} {placement.centroid_y:.4f} {z + 0.25:.4f}" size="1.2" rgba="0.12 0.12 0.12 1"/>'
            )

    if camera_targets:
        cam_x = sum(x for x, _ in camera_targets) / len(camera_targets)
        cam_y = sum(y for _, y in camera_targets) / len(camera_targets)
    else:
        cam_x = cam_y = 0.0

    distance = max_extent * 2.2 + max(0, len(boards) - 1) * 40.0
    mjcf = f"""
    <mujoco model="widget_nesting_debug">
      <visual>
        <global offwidth="1400" offheight="900"/>
        <headlight ambient="0.85 0.85 0.85" diffuse="0.35 0.35 0.35" specular="0 0 0"/>
      </visual>
      <worldbody>
        {''.join(geoms)}
      </worldbody>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[0] = cam_x
    camera.lookat[1] = cam_y
    camera.lookat[2] = 0.0
    camera.distance = distance
    camera.azimuth = 90.0
    camera.elevation = -90.0
    renderer = mujoco.Renderer(model, width=1200, height=800)
    renderer.update_scene(data, camera=camera)
    pixels = renderer.render()
    renderer.close()

    image = Image.fromarray(np.asarray(pixels))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output
