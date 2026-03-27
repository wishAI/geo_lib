from __future__ import annotations

import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.svg_auto_slam_mapping.pipeline import (
    PlanarRobotSimulator,
    build_scene_spec,
    load_semantic_layout,
    run_mapping_pipeline,
)


@pytest.fixture(scope='session')
def sample_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_dir = tmp_path_factory.mktemp('svg_auto_slam_mapping') / 'sample_run'
    run_mapping_pipeline(output_dir=output_dir, timeout_s=5.0)
    return output_dir


def test_layout_and_scene_ranges() -> None:
    layout = load_semantic_layout(REPO_ROOT / 'algorithms/svg_auto_slam_mapping/svg_room_map.svg')
    scene = build_scene_spec(layout)
    assert 8.5 <= scene.width_m <= 10.0
    assert 4.5 <= scene.height_m <= 5.5
    assert 8 <= len(scene.wall_geoms) <= 120
    assert 4 <= len(scene.furniture_geoms) <= 80
    occupied_ratio = float(np.count_nonzero(layout.occupied_grid) / layout.occupied_grid.size)
    assert 0.10 <= occupied_ratio <= 0.55


def test_mujoco_scene_loads_and_bbox(sample_output: Path) -> None:
    xml = (sample_output / 'scene.xml').read_text(encoding='utf-8')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.ngeom >= 10
    assert model.nq == 3
    summary = json.loads((sample_output / 'mapping_summary.json').read_text(encoding='utf-8'))
    extent = np.array(summary['scene_bbox']['extent'], dtype=float)
    assert 8.5 <= extent[0] <= 10.0
    assert 4.5 <= extent[1] <= 5.5
    assert 2.0 <= extent[2] <= 3.0


def test_robot_moves_and_lidar_sees_obstacles() -> None:
    layout = load_semantic_layout(REPO_ROOT / 'algorithms/svg_auto_slam_mapping/svg_room_map.svg')
    scene = build_scene_spec(layout)
    route = [(0.0, 0.0), (1.0, 0.0), (1.0, -1.0), (-1.0, -1.0)]
    sim = PlanarRobotSimulator(layout, scene, route)
    result = sim.run(timeout_s=1.0, dt=0.05)
    final_pose = result['final_pose']
    assert abs(final_pose.x) > 0.2 or abs(final_pose.y) > 0.2
    assert 0.05 <= result['scan_min_range_m'] <= 5.5


def test_pipeline_outputs_ros_map_and_ratios(sample_output: Path) -> None:
    map_path = sample_output / 'map.pgm'
    yaml_path = sample_output / 'map.yaml'
    summary = json.loads((sample_output / 'mapping_summary.json').read_text(encoding='utf-8'))
    assert map_path.exists()
    assert yaml_path.exists()
    assert summary['elapsed_s'] <= 5.1
    ratios = summary['map_ratios']
    assert 0.02 <= ratios['occupied_ratio'] <= 0.35
    assert 0.10 <= ratios['free_ratio'] <= 0.80
    assert 0.10 <= ratios['unknown_ratio'] <= 0.90
    assert abs(ratios['occupied_ratio'] + ratios['free_ratio'] + ratios['unknown_ratio'] - 1.0) < 1e-6
    image = np.array(Image.open(map_path))
    unique = set(np.unique(image).tolist())
    assert {0, 205, 254}.intersection(unique)


def test_ros2_environment_info_recorded(sample_output: Path) -> None:
    summary = json.loads((sample_output / 'mapping_summary.json').read_text(encoding='utf-8'))
    env = summary['ros2_environment']
    assert 'installed_distros' in env
    assert env['map_format'] == 'ROS2 map_server compatible pgm+yaml'
