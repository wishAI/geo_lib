from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil

os.environ.setdefault('MUJOCO_GL', 'egl')

import mujoco
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SemanticLayout:
    width_m: float
    height_m: float
    resolution_m: float
    wall_grid: np.ndarray
    furniture_grid: np.ndarray

    @property
    def occupied_grid(self) -> np.ndarray:
        return np.logical_or(self.wall_grid, self.furniture_grid)

    @property
    def shape(self) -> tuple[int, int]:
        return self.wall_grid.shape


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


def _require_stage_dir(path: str | Path, folder_name: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if folder_name not in resolved.parts:
        raise ValueError(f'{label} path must be inside a `{folder_name}` directory: {resolved}')
    return resolved


def _world_to_grid(layout: SemanticLayout, x_m: float, y_m: float) -> tuple[int, int]:
    col = int(np.clip((x_m + layout.width_m / 2.0) / layout.resolution_m, 0, layout.shape[1] - 1))
    row = int(np.clip((layout.height_m / 2.0 - y_m) / layout.resolution_m, 0, layout.shape[0] - 1))
    return row, col


def _grid_to_world(layout: SemanticLayout, row: int, col: int) -> tuple[float, float]:
    x_m = -layout.width_m / 2.0 + (col + 0.5) * layout.resolution_m
    y_m = layout.height_m / 2.0 - (row + 0.5) * layout.resolution_m
    return x_m, y_m


def _inflate(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    inflated = mask.copy()
    if radius_cells <= 0:
        return inflated
    height, width = mask.shape
    for row, col in np.argwhere(mask):
        for rr in range(max(0, row - radius_cells), min(height, row + radius_cells + 1)):
            for cc in range(max(0, col - radius_cells), min(width, col + radius_cells + 1)):
                if (rr - row) ** 2 + (cc - col) ** 2 <= radius_cells ** 2:
                    inflated[rr, cc] = True
    return inflated


def _nearest_free(mask: np.ndarray, row: int, col: int) -> tuple[int, int]:
    if not mask[row, col]:
        return row, col
    for radius in range(1, max(mask.shape)):
        for rr in range(max(0, row - radius), min(mask.shape[0], row + radius + 1)):
            for cc in range(max(0, col - radius), min(mask.shape[1], col + radius + 1)):
                if not mask[rr, cc]:
                    return rr, cc
    return row, col


def _corner_targets(inflated_occ: np.ndarray, max_targets: int = 18) -> list[tuple[int, int]]:
    free = np.logical_not(inflated_occ)
    raw = []
    for row in range(1, inflated_occ.shape[0] - 1):
        for col in range(1, inflated_occ.shape[1] - 1):
            if not free[row, col]:
                continue
            n = inflated_occ[row - 1, col]
            s = inflated_occ[row + 1, col]
            e = inflated_occ[row, col + 1]
            w = inflated_occ[row, col - 1]
            if (n and w) or (n and e) or (s and w) or (s and e):
                raw.append((row, col))
    selected = []
    for row, col in raw:
        if any((row - sr) ** 2 + (col - sc) ** 2 < 36 for sr, sc in selected):
            continue
        selected.append((row, col))
        if len(selected) >= max_targets:
            break
    return selected


def _astar(mask: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    import heapq

    if start == goal:
        return [start]
    neighbors = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    ]
    queue = [(0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    while queue:
        _, current = heapq.heappop(queue)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))
        for dr, dc, move_cost in neighbors:
            nr = current[0] + dr
            nc = current[1] + dc
            if not (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]):
                continue
            if mask[nr, nc]:
                continue
            tentative = g_score[current] + move_cost
            node = (nr, nc)
            if tentative >= g_score.get(node, float('inf')):
                continue
            came_from[node] = current
            g_score[node] = tentative
            heuristic = math.hypot(goal[0] - nr, goal[1] - nc)
            heapq.heappush(queue, (tentative + heuristic, node))
    return [start]


def _build_route(layout: SemanticLayout, start_pose: Pose2D, robot_radius_m: float = 0.18) -> list[tuple[float, float]]:
    inflated = _inflate(layout.occupied_grid, max(1, int(math.ceil(robot_radius_m / layout.resolution_m))))
    start = _nearest_free(inflated, *_world_to_grid(layout, start_pose.x, start_pose.y))
    remaining = _corner_targets(inflated)
    ordered = [start]
    current = start
    while remaining:
        target = min(remaining, key=lambda cell: (cell[0] - current[0]) ** 2 + (cell[1] - current[1]) ** 2)
        ordered.extend(_astar(inflated, current, target)[1:])
        remaining.remove(target)
        current = target
    sparse = [ordered[0]]
    for cell in ordered[4::4]:
        if cell != sparse[-1]:
            sparse.append(cell)
    if ordered[-1] != sparse[-1]:
        sparse.append(ordered[-1])
    return [_grid_to_world(layout, row, col) for row, col in sparse]


def _cast_lidar(layout: SemanticLayout, pose: Pose2D, beam_angles: np.ndarray, max_range_m: float) -> np.ndarray:
    ranges = np.full(beam_angles.shape, max_range_m, dtype=float)
    step_m = layout.resolution_m * 0.5
    ray_start_m = max(0.08, layout.resolution_m)
    occupied = layout.occupied_grid
    for index, rel_angle in enumerate(beam_angles):
        heading = pose.yaw + float(rel_angle)
        distance = ray_start_m
        while distance < max_range_m:
            px = pose.x + distance * math.cos(heading)
            py = pose.y + distance * math.sin(heading)
            if abs(px) > layout.width_m / 2.0 or abs(py) > layout.height_m / 2.0:
                ranges[index] = distance
                break
            row, col = _world_to_grid(layout, px, py)
            if occupied[row, col]:
                ranges[index] = distance
                break
            distance += step_m
    return ranges


class OccupancyMapper:
    def __init__(self, layout: SemanticLayout):
        self.layout = layout
        self.log_odds = np.zeros(layout.shape, dtype=float)

    def update(self, pose: Pose2D, beam_angles: np.ndarray, ranges: np.ndarray, max_range_m: float) -> None:
        for rel_angle, lidar_range in zip(beam_angles, ranges):
            heading = pose.yaw + float(rel_angle)
            distance = 0.0
            hit = float(lidar_range) < max_range_m - self.layout.resolution_m
            while distance < min(float(lidar_range), max_range_m):
                px = pose.x + distance * math.cos(heading)
                py = pose.y + distance * math.sin(heading)
                if abs(px) > self.layout.width_m / 2.0 or abs(py) > self.layout.height_m / 2.0:
                    break
                row, col = _world_to_grid(self.layout, px, py)
                self.log_odds[row, col] -= 0.18
                distance += self.layout.resolution_m * 0.45
            if hit:
                px = pose.x + float(lidar_range) * math.cos(heading)
                py = pose.y + float(lidar_range) * math.sin(heading)
                if abs(px) <= self.layout.width_m / 2.0 and abs(py) <= self.layout.height_m / 2.0:
                    row, col = _world_to_grid(self.layout, px, py)
                    self.log_odds[row, col] += 0.95

    def ros_map(self) -> np.ndarray:
        img = np.full(self.layout.shape, 205, dtype=np.uint8)
        occupied = _inflate(self.log_odds > 0.45, 1)
        free_seed = np.logical_and(self.log_odds < -0.10, np.logical_not(occupied))
        free = np.logical_and(_inflate(free_seed, 1), np.logical_not(occupied))
        img[free] = 254
        img[occupied] = 0
        return img


class PlanarRobotSimulator:
    def __init__(self, layout: SemanticLayout, scene_xml: str, start_pose: Pose2D, route_xy: list[tuple[float, float]], robot_radius_m: float = 0.18):
        self.layout = layout
        self.scene_xml = scene_xml
        self.robot_radius_m = robot_radius_m
        safe_mask = _inflate(layout.occupied_grid, max(1, int(math.ceil(robot_radius_m / layout.resolution_m))))
        start_row, start_col = _world_to_grid(layout, start_pose.x, start_pose.y)
        start_row, start_col = _nearest_free(safe_mask, start_row, start_col)
        start_xy = _grid_to_world(layout, start_row, start_col)
        self.route_xy = [start_xy, *route_xy[1:]] if route_xy else [start_xy]
        self.model = mujoco.MjModel.from_xml_string(scene_xml)
        self.data = mujoco.MjData(self.model)
        self.pose = Pose2D(start_xy[0], start_xy[1], start_pose.yaw)
        self.set_pose(self.pose)

    def set_pose(self, pose: Pose2D) -> None:
        self.pose = pose
        self.data.qpos[0] = pose.x
        self.data.qpos[1] = pose.y
        self.data.qpos[2] = pose.yaw
        mujoco.mj_forward(self.model, self.data)

    def run(
        self,
        timeout_s: float,
        dt: float = 0.05,
        speed_mps: float = 2.4,
        angular_speed_radps: float = 8.0,
        lidar_beams: int = 181,
        lidar_range_m: float = 5.5,
        snapshot_period_s: float = 10.0,
    ) -> dict[str, object]:
        mapper = OccupancyMapper(self.layout)
        beam_angles = np.linspace(-math.pi, math.pi, lidar_beams, endpoint=False)
        waypoint_index = 1
        elapsed = 0.0
        trajectory = []
        min_ranges = []
        snapshots = []
        next_snapshot_s = snapshot_period_s if snapshot_period_s > 0.0 else None
        safe_mask = _inflate(self.layout.occupied_grid, max(1, int(math.ceil(self.robot_radius_m / self.layout.resolution_m))))
        while elapsed < timeout_s:
            if waypoint_index < len(self.route_xy):
                target_x, target_y = self.route_xy[waypoint_index]
                dx = target_x - self.pose.x
                dy = target_y - self.pose.y
                distance = math.hypot(dx, dy)
                if distance < 0.08:
                    waypoint_index += 1
                else:
                    target_yaw = math.atan2(dy, dx)
                    yaw_error = (target_yaw - self.pose.yaw + math.pi) % (2.0 * math.pi) - math.pi
                    yaw_step = float(np.clip(yaw_error, -angular_speed_radps * dt, angular_speed_radps * dt))
                    yaw = (self.pose.yaw + yaw_step + math.pi) % (2.0 * math.pi) - math.pi
                    move = min(distance, speed_mps * dt)
                    ux = dx / distance
                    uy = dy / distance
                    candidate = Pose2D(x=self.pose.x + move * ux, y=self.pose.y + move * uy, yaw=yaw)
                    row, col = _world_to_grid(self.layout, candidate.x, candidate.y)
                    if not safe_mask[row, col]:
                        self.set_pose(candidate)
                    else:
                        for scale in (0.5, 0.25):
                            candidate = Pose2D(x=self.pose.x + move * scale * ux, y=self.pose.y + move * scale * uy, yaw=yaw)
                            row, col = _world_to_grid(self.layout, candidate.x, candidate.y)
                            if not safe_mask[row, col]:
                                self.set_pose(candidate)
                                break
            ranges = _cast_lidar(self.layout, self.pose, beam_angles, lidar_range_m)
            mapper.update(self.pose, beam_angles, ranges, lidar_range_m)
            trajectory.append({'t': elapsed, 'x': self.pose.x, 'y': self.pose.y, 'yaw': self.pose.yaw})
            min_ranges.append(float(np.min(ranges)))
            elapsed += dt
            while next_snapshot_s is not None and elapsed + 1e-9 >= next_snapshot_s:
                snapshots.append({'time_s': float(next_snapshot_s), 'map_img': mapper.ros_map().copy()})
                next_snapshot_s += snapshot_period_s
        return {
            'elapsed_s': elapsed,
            'trajectory': trajectory,
            'mapper': mapper,
            'snapshots': snapshots,
            'scan_min_range_m': float(min(min_ranges)) if min_ranges else lidar_range_m,
            'scan_max_range_m': float(max(min_ranges)) if min_ranges else lidar_range_m,
            'final_pose': self.pose,
            'waypoints_completed': waypoint_index,
        }


def detect_ros2_environment() -> dict[str, object]:
    ros_root = Path('/opt/ros')
    distros = sorted(path.name for path in ros_root.glob('*') if path.is_dir()) if ros_root.exists() else []
    return {
        'ros_root_exists': ros_root.exists(),
        'installed_distros': distros,
        'ros2_in_path': shutil.which('ros2') is not None,
        'map_format': 'ROS2 map_server compatible pgm+yaml',
    }


def sync_scene_input(source_dir: str | Path, input_dir: str | Path) -> list[str]:
    source_root = Path(source_dir)
    input_root = _require_stage_dir(input_dir, 'inputs', 'Mapping input')
    package = json.loads((source_root / 'scene_package.json').read_text(encoding='utf-8'))
    required = [
        package['scene_xml'],
        package['semantic_layout'],
        package['layout_preview'],
        package['start_pose'],
        package['scene_summary'],
        'scene_package.json',
    ]
    input_root.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in required:
        shutil.copyfile(source_root / name, input_root / name)
        copied.append(name)
    return copied


def load_scene_input(input_dir: str | Path) -> tuple[SemanticLayout, str, Pose2D, dict[str, object]]:
    input_root = _require_stage_dir(input_dir, 'inputs', 'Mapping input')
    package = json.loads((input_root / 'scene_package.json').read_text(encoding='utf-8'))
    if package.get('stage') != 'svg_scene_builder':
        raise ValueError('Input directory is not a valid svg_scene_builder package')
    bundle = np.load(input_root / package['semantic_layout'])
    layout = SemanticLayout(
        width_m=float(bundle['width_m']),
        height_m=float(bundle['height_m']),
        resolution_m=float(bundle['resolution_m']),
        wall_grid=bundle['wall_grid'].astype(bool),
        furniture_grid=bundle['furniture_grid'].astype(bool),
    )
    start_payload = json.loads((input_root / package['start_pose']).read_text(encoding='utf-8'))
    start_pose = Pose2D(start_payload['x'], start_payload['y'], start_payload['yaw'])
    scene_xml = (input_root / package['scene_xml']).read_text(encoding='utf-8')
    return layout, scene_xml, start_pose, package


def _write_map_files(base_path: Path, layout: SemanticLayout, map_img: np.ndarray) -> None:
    Image.fromarray(map_img, mode='L').save(base_path.with_suffix('.pgm'))
    base_path.with_suffix('.yaml').write_text(
        '\n'.join(
            [
                f'image: {base_path.with_suffix(".pgm").name}',
                f'resolution: {layout.resolution_m:.4f}',
                f'origin: [{-layout.width_m / 2.0:.4f}, {-layout.height_m / 2.0:.4f}, 0.0]',
                'negate: 0',
                'occupied_thresh: 0.65',
                'free_thresh: 0.196',
            ]
        )
        + '\n',
        encoding='utf-8',
    )


def _save_map(output_dir: Path, layout: SemanticLayout, map_img: np.ndarray) -> dict[str, float]:
    _write_map_files(output_dir / 'map', layout, map_img)
    total = float(map_img.size)
    return {
        'occupied_ratio': float(np.count_nonzero(map_img == 0) / total),
        'free_ratio': float(np.count_nonzero(map_img == 254) / total),
        'unknown_ratio': float(np.count_nonzero(map_img == 205) / total),
    }


def _snapshot_name(time_s: float) -> str:
    rounded = round(time_s)
    if abs(time_s - rounded) < 1e-9:
        return f'map_{int(rounded):03d}s'
    return f'map_{time_s:06.1f}s'.replace('.', 'p')


def _save_snapshots(output_dir: Path, layout: SemanticLayout, snapshots: list[dict[str, object]]) -> list[str]:
    if not snapshots:
        return []
    snapshot_dir = output_dir / 'snapshots'
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for snapshot in snapshots:
        stem = _snapshot_name(float(snapshot['time_s']))
        _write_map_files(snapshot_dir / stem, layout, snapshot['map_img'])
        saved.append(f'snapshots/{stem}.pgm')
    return saved


def run_mapping_pipeline(
    input_dir: str | Path,
    output_dir: str | Path,
    timeout_s: float = 60.0,
    snapshot_period_s: float = 10.0,
) -> dict[str, object]:
    input_root = _require_stage_dir(input_dir, 'inputs', 'Mapping input')
    output_root = _require_stage_dir(output_dir, 'outputs', 'Mapping output')
    output_root.mkdir(parents=True, exist_ok=True)
    layout, scene_xml, start_pose, package = load_scene_input(input_root)
    route_xy = _build_route(layout, start_pose, robot_radius_m=float(package.get('robot_radius_m', 0.18)))
    sim = PlanarRobotSimulator(layout, scene_xml, start_pose, route_xy, robot_radius_m=float(package.get('robot_radius_m', 0.18)))
    result = sim.run(timeout_s=timeout_s, snapshot_period_s=snapshot_period_s)
    map_img = result['mapper'].ros_map()
    ratios = _save_map(output_root, layout, map_img)
    saved_snapshots = _save_snapshots(output_root, layout, result['snapshots'])
    summary = {
        'ros2_environment': detect_ros2_environment(),
        'scene_bbox': package['bbox'],
        'elapsed_s': result['elapsed_s'],
        'scan_min_range_m': result['scan_min_range_m'],
        'scan_max_range_m': result['scan_max_range_m'],
        'route_waypoints': len(route_xy),
        'waypoints_completed': result['waypoints_completed'],
        'final_pose': {'x': result['final_pose'].x, 'y': result['final_pose'].y, 'yaw': result['final_pose'].yaw},
        'map_ratios': ratios,
        'input_dir': str(input_root),
        'output_dir': str(output_root),
        'input_stage': package['stage'],
        'snapshot_period_s': snapshot_period_s,
        'saved_map_snapshots': saved_snapshots,
    }
    (output_root / 'trajectory.json').write_text(json.dumps(result['trajectory'], indent=2), encoding='utf-8')
    (output_root / 'route.json').write_text(json.dumps(route_xy, indent=2), encoding='utf-8')
    (output_root / 'mapping_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output_root / 'input_package_snapshot.json').write_text(json.dumps(package, indent=2), encoding='utf-8')
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run simple auto SLAM mapping from a copied scene package')
    parser.add_argument('--input', type=Path, default=Path(__file__).resolve().parent / 'inputs' / 'sample_scene')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'outputs' / 'sample_run')
    parser.add_argument('--timeout', type=float, default=60.0)
    parser.add_argument('--snapshot-period', type=float, default=10.0)
    parser.add_argument('--copy-from', type=Path, default=None, help='Optional scene-builder output directory to copy into --input before running')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.copy_from is not None:
        sync_scene_input(args.copy_from, args.input)
    summary = run_mapping_pipeline(
        input_dir=args.input,
        output_dir=args.output,
        timeout_s=args.timeout,
        snapshot_period_s=args.snapshot_period,
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
