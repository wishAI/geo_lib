from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from .camera_sampler import CameraPose
from .config import SceneConfig


PALETTE = [
    (0.78, 0.78, 0.78, 1.0),
    (0.91, 0.48, 0.28, 1.0),
    (0.29, 0.62, 0.82, 1.0),
    (0.38, 0.71, 0.34, 1.0),
]


@dataclass
class Board:
    name: str
    size_m: np.ndarray
    position_world: np.ndarray
    yaw_deg: float
    is_bottom: bool

    @property
    def rotation_world_from_board(self) -> np.ndarray:
        yaw_rad = math.radians(self.yaw_deg)
        c = math.cos(yaw_rad)
        s = math.sin(yaw_rad)
        return np.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    @property
    def transform_world_from_board(self) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = self.rotation_world_from_board
        transform[:3, 3] = self.position_world
        return transform


@dataclass
class SceneSpec:
    bottom_board: Board
    vertical_boards: list[Board]

    @property
    def bottom_plane_center_world(self) -> np.ndarray:
        top_z = self.bottom_board.position_world[2] + 0.5 * self.bottom_board.size_m[2]
        return np.array([0.0, 0.0, top_z], dtype=float)


def _yaw_to_quat_wxyz(yaw_deg: float) -> tuple[float, float, float, float]:
    half = math.radians(yaw_deg) / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _rotation_to_quat_wxyz(rot: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qw, qx, qy, qz])
    quat /= np.linalg.norm(quat)
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def generate_scene(scene_cfg: SceneConfig, rng: np.random.Generator) -> SceneSpec:
    bottom_size = np.array(scene_cfg.bottom_size_m, dtype=float)
    bottom_board = Board(
        name="bottom_board",
        size_m=bottom_size,
        position_world=np.array([0.0, 0.0, bottom_size[2] / 2.0], dtype=float),
        yaw_deg=0.0,
        is_bottom=True,
    )

    count = int(
        rng.integers(
            scene_cfg.vertical_board_count_min,
            scene_cfg.vertical_board_count_max + 1,
        )
    )
    count = max(2, min(3, count))

    yaw_options = [0.0, 90.0, 0.0]
    offset_options = [
        np.array([0.00, -0.14]),
        np.array([0.00, 0.00]),
        np.array([0.00, 0.14]),
    ]

    vertical_boards: list[Board] = []
    for i in range(count):
        length = float(
            rng.uniform(
                scene_cfg.vertical_length_range_m[0],
                scene_cfg.vertical_length_range_m[1],
            )
        )
        thickness = float(
            rng.uniform(
                scene_cfg.vertical_thickness_range_m[0],
                scene_cfg.vertical_thickness_range_m[1],
            )
        )
        height = float(
            rng.uniform(
                scene_cfg.vertical_height_range_m[0],
                scene_cfg.vertical_height_range_m[1],
            )
        )

        jitter_xy = rng.normal(loc=0.0, scale=0.01, size=2)
        xy = offset_options[i] + jitter_xy
        yaw = yaw_options[i]

        position = np.array([xy[0], xy[1], bottom_size[2] + 0.5 * height], dtype=float)
        vertical_boards.append(
            Board(
                name=f"vertical_board_{i:03d}",
                size_m=np.array([length, thickness, height], dtype=float),
                position_world=position,
                yaw_deg=yaw,
                is_bottom=False,
            )
        )

    return SceneSpec(bottom_board=bottom_board, vertical_boards=vertical_boards)


def build_mjcf(scene: SceneSpec, cameras: list[CameraPose], fov_y_deg: float) -> str:
    geoms: list[str] = []
    boards = [scene.bottom_board, *scene.vertical_boards]
    for idx, board in enumerate(boards):
        half = board.size_m / 2.0
        quat = _yaw_to_quat_wxyz(board.yaw_deg)
        rgba = PALETTE[min(idx, len(PALETTE) - 1)]
        geoms.append(
            "    <geom "
            f'name="{board.name}" type="box" '
            f'pos="{board.position_world[0]:.6f} {board.position_world[1]:.6f} {board.position_world[2]:.6f}" '
            f'size="{half[0]:.6f} {half[1]:.6f} {half[2]:.6f}" '
            f'quat="{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}" '
            f'rgba="{rgba[0]:.3f} {rgba[1]:.3f} {rgba[2]:.3f} {rgba[3]:.3f}"/>'
        )

    cam_entries: list[str] = []
    for cam in cameras:
        quat = _rotation_to_quat_wxyz(cam.rotation_world_from_camera_mj)
        cam_entries.append(
            "    <camera "
            f'name="{cam.name}" '
            f'pos="{cam.position_world[0]:.6f} {cam.position_world[1]:.6f} {cam.position_world[2]:.6f}" '
            f'quat="{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}" '
            f'fovy="{fov_y_deg:.3f}"/>'
        )

    xml = (
        "<mujoco model=\"fake_ship_part\">\n"
        "  <compiler angle=\"degree\"/>\n"
        "  <option gravity=\"0 0 0\"/>\n"
        "  <visual>\n"
        "    <map znear=\"0.01\" zfar=\"20\"/>\n"
        "  </visual>\n"
        "  <worldbody>\n"
        "    <light name=\"key\" pos=\"0 0 2\" dir=\"0 0 -1\"/>\n"
        + "\n".join(geoms)
        + "\n"
        + "\n".join(cam_entries)
        + "\n"
        "  </worldbody>\n"
        "</mujoco>\n"
    )
    return xml


def scene_metadata(scene: SceneSpec) -> dict:
    boards = [scene.bottom_board, *scene.vertical_boards]
    payload = {
        "bottom_board": scene.bottom_board.name,
        "bottom_plane_center_world": scene.bottom_plane_center_world.tolist(),
        "vertical_board_count": len(scene.vertical_boards),
        "boards": [],
    }

    for board in boards:
        payload["boards"].append(
            {
                "name": board.name,
                "is_bottom": board.is_bottom,
                "size_m": board.size_m.tolist(),
                "position_world": board.position_world.tolist(),
                "yaw_deg": board.yaw_deg,
                "transform_world_from_board": board.transform_world_from_board.tolist(),
            }
        )

    return payload


def save_scene_metadata(scene: SceneSpec, target_path: str | Path) -> None:
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(scene_metadata(scene), f, indent=2)
