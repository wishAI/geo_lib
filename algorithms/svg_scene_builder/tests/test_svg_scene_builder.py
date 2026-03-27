from __future__ import annotations

import json
from pathlib import Path
import sys

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.svg_scene_builder.builder import load_semantic_layout, run_scene_builder


def test_layout_and_scene_package(tmp_path: Path) -> None:
    output_dir = tmp_path / 'sample_scene'
    summary = run_scene_builder(output_dir=output_dir, svg_path=REPO_ROOT / 'algorithms/svg_scene_builder/svg_room_map.svg')
    package = json.loads((output_dir / 'scene_package.json').read_text(encoding='utf-8'))
    scene_summary = json.loads((output_dir / 'scene_summary.json').read_text(encoding='utf-8'))
    assert summary['scene_package']['stage'] == 'svg_scene_builder'
    assert package['scene_xml'] == 'scene.xml'
    assert scene_summary['total_geom_count'] >= 10
    assert 8.5 <= package['width_m'] <= 10.0
    assert 4.5 <= package['height_m'] <= 5.5
    assert scene_summary['layout_shape'][0] >= 200
    assert scene_summary['layout_shape'][1] >= 400


def test_start_pose_and_scene_load(tmp_path: Path) -> None:
    output_dir = tmp_path / 'sample_scene'
    run_scene_builder(output_dir=output_dir, svg_path=REPO_ROOT / 'algorithms/svg_scene_builder/svg_room_map.svg')
    xml = (output_dir / 'scene.xml').read_text(encoding='utf-8')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pose = json.loads((output_dir / 'start_pose.json').read_text(encoding='utf-8'))
    layout = load_semantic_layout(REPO_ROOT / 'algorithms/svg_scene_builder/svg_room_map.svg')
    row, col = int((layout.height_m / 2.0 - pose['y']) / layout.resolution_m), int((pose['x'] + layout.width_m / 2.0) / layout.resolution_m)
    row = max(0, min(layout.shape[0] - 1, row))
    col = max(0, min(layout.shape[1] - 1, col))
    assert model.ngeom >= 10
    assert model.nq == 3
    assert not layout.occupied_grid[row, col]
    assert abs(pose['yaw']) < 1e-6


def test_no_mapping_import_in_builder() -> None:
    source = (REPO_ROOT / 'algorithms/svg_scene_builder/builder.py').read_text(encoding='utf-8')
    assert 'algorithms.simple_auto_slam_mapping' not in source
