from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import mujoco
import numpy as np
from PIL import Image
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.simple_auto_slam_mapping.mapping import (
    PlanarRobotSimulator,
    load_scene_input,
    run_mapping_pipeline,
    sync_scene_input,
)
from algorithms.svg_scene_builder.builder import run_scene_builder


@pytest.fixture(scope='session')
def staged_paths(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    root = tmp_path_factory.mktemp('svg_stage_split')
    scene_output = root / 'scene_output'
    mapping_input = root / 'inputs' / 'sample_scene'
    mapping_output = root / 'outputs' / 'sample_run'
    run_scene_builder(output_dir=scene_output, svg_path=REPO_ROOT / 'algorithms/svg_scene_builder/svg_room_map.svg')
    sync_scene_input(scene_output, mapping_input)
    shutil.rmtree(scene_output)
    return mapping_input, mapping_output, root


def test_mapping_uses_copied_input_only(staged_paths: tuple[Path, Path, Path]) -> None:
    mapping_input, mapping_output, _ = staged_paths
    summary = run_mapping_pipeline(input_dir=mapping_input, output_dir=mapping_output, timeout_s=5.0)
    assert summary['input_stage'] == 'svg_scene_builder'
    assert Path(summary['input_dir']) == mapping_input.resolve()
    assert Path(summary['output_dir']) == mapping_output.resolve()
    assert summary['elapsed_s'] <= 5.1


def test_robot_moves_and_lidar_with_staged_input(staged_paths: tuple[Path, Path, Path]) -> None:
    mapping_input, _, _ = staged_paths
    layout, scene_xml, start_pose, _ = load_scene_input(mapping_input)
    model = mujoco.MjModel.from_xml_string(scene_xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.ngeom >= 10
    assert model.nq == 3
    row = int((layout.height_m / 2.0 - start_pose.y) / layout.resolution_m)
    col = int((start_pose.x + layout.width_m / 2.0) / layout.resolution_m)
    row = max(0, min(layout.shape[0] - 1, row))
    col = max(0, min(layout.shape[1] - 1, col))
    assert not layout.occupied_grid[row, col]


def test_mapping_outputs_ros_map_ratios_and_snapshots(staged_paths: tuple[Path, Path, Path]) -> None:
    mapping_input, mapping_output, _ = staged_paths
    summary = run_mapping_pipeline(input_dir=mapping_input, output_dir=mapping_output, timeout_s=2.5, snapshot_period_s=1.0)
    image = np.array(Image.open(mapping_output / 'map.pgm'))
    ratios = summary['map_ratios']
    assert image.shape[0] >= 200
    assert image.shape[1] >= 400
    assert 0.01 <= ratios['occupied_ratio'] <= 0.35
    assert 0.10 <= ratios['free_ratio'] <= 0.80
    assert 0.10 <= ratios['unknown_ratio'] <= 0.90
    assert abs(ratios['occupied_ratio'] + ratios['free_ratio'] + ratios['unknown_ratio'] - 1.0) < 1e-6
    assert {0, 205, 254}.intersection(set(np.unique(image).tolist()))
    assert summary['saved_map_snapshots'] == ['snapshots/map_001s.pgm', 'snapshots/map_002s.pgm']
    assert (mapping_output / 'snapshots' / 'map_001s.pgm').exists()
    assert (mapping_output / 'snapshots' / 'map_001s.yaml').exists()
    assert (mapping_output / 'snapshots' / 'map_002s.pgm').exists()


def test_mapping_stops_when_route_finishes(staged_paths: tuple[Path, Path, Path]) -> None:
    mapping_input, mapping_output, _ = staged_paths
    summary = run_mapping_pipeline(input_dir=mapping_input, output_dir=mapping_output, timeout_s=60.0, snapshot_period_s=10.0)
    assert summary['stop_reason'] == 'route_complete'
    assert summary['elapsed_s'] < 60.0
    assert summary['route_completed_at_s'] == pytest.approx(summary['elapsed_s'])
    assert summary['waypoints_completed'] == summary['route_waypoints']
    assert summary['waypoints_skipped'] >= 0


def test_mapping_can_continue_after_route_completion(staged_paths: tuple[Path, Path, Path]) -> None:
    mapping_input, _, _ = staged_paths
    layout, scene_xml, start_pose, _ = load_scene_input(mapping_input)
    sim = PlanarRobotSimulator(layout, scene_xml, start_pose, route_xy=[(start_pose.x, start_pose.y)])
    result = sim.run(timeout_s=0.2, snapshot_period_s=0.1, stop_on_route_complete=False)
    assert result['stop_reason'] == 'timeout_after_route_complete'
    assert result['route_completed_at_s'] is not None
    assert result['elapsed_s'] >= 0.2
    assert [snapshot['time_s'] for snapshot in result['snapshots']] == [0.1, 0.2]


def test_mapping_rejects_paths_outside_inputs_outputs(staged_paths: tuple[Path, Path, Path]) -> None:
    mapping_input, _, root = staged_paths
    bad_input = root / 'bad_input'
    bad_output = root / 'bad_output'
    with pytest.raises(ValueError, match='inside a `inputs` directory'):
        sync_scene_input(REPO_ROOT / 'algorithms/svg_scene_builder/outputs/sample_scene', bad_input)
    with pytest.raises(ValueError, match='inside a `outputs` directory'):
        run_mapping_pipeline(input_dir=mapping_input, output_dir=bad_output, timeout_s=0.1)


def test_no_builder_import_in_mapping() -> None:
    source = (REPO_ROOT / 'algorithms/simple_auto_slam_mapping/mapping.py').read_text(encoding='utf-8')
    assert 'algorithms.svg_scene_builder' not in source
