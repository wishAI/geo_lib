from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.fake_cloud.camera_sampler import sample_cameras
from algorithms.fake_cloud.config import load_config
from algorithms.fake_cloud.generate_dataset import generate_scene_dataset
from algorithms.fake_cloud.pointcloud import read_ply


@pytest.fixture(scope="session")
def generated_scene(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_dir = tmp_path_factory.mktemp("fake_cloud_scene") / "sample_scene"
    generate_scene_dataset(output_dir=output_dir, seed=0)
    return output_dir


def test_scene_generation(generated_scene: Path) -> None:
    scene_path = generated_scene / "scene.json"
    assert scene_path.exists()

    payload = json.loads(scene_path.read_text(encoding="utf-8"))
    verticals = [b for b in payload["boards"] if not b["is_bottom"]]

    assert payload["bottom_board"] == "bottom_board"
    assert 2 <= payload["vertical_board_count"] <= 3
    assert 2 <= len(verticals) <= 3

    yaws = []
    for board in verticals:
        tf = np.array(board["transform_world_from_board"], dtype=float)
        z_axis = tf[:3, 2]
        assert np.allclose(z_axis, np.array([0.0, 0.0, 1.0]), atol=1e-6)
        yaws.append(float(board["yaw_deg"]))

    for i in range(len(yaws)):
        for j in range(i + 1, len(yaws)):
            delta = abs((yaws[i] - yaws[j]) % 180.0)
            assert min(delta, abs(delta - 90.0)) <= 1e-6


def test_point_count_ranges(generated_scene: Path) -> None:
    assert not (generated_scene / "clean").exists()
    assert not (generated_scene / "noisy").exists()

    view_paths = sorted(generated_scene.glob("view*.ply"))
    assert 2 <= len(view_paths) <= 3

    counts = [len(read_ply(p)) for p in view_paths]
    for count in counts:
        assert 2_000 <= count <= 300_000

    merged = read_ply(generated_scene / "merged_without_pose_error.ply")
    assert 10_000 <= len(merged) <= 1_000_000


def test_merged_bbox_ranges(generated_scene: Path) -> None:
    merged = read_ply(generated_scene / "merged_without_pose_error.ply")
    mins = merged.min(axis=0)
    maxs = merged.max(axis=0)
    extent = maxs - mins

    assert 0.5 <= extent[0] <= 2.2
    assert 0.4 <= extent[1] <= 1.8
    assert 0.1 <= extent[2] <= 1.3


def test_cam_info_json_has_error_poses(generated_scene: Path) -> None:
    cam_info_path = generated_scene / "cam_info.json"
    assert cam_info_path.exists()

    payload = json.loads(cam_info_path.read_text(encoding="utf-8"))
    ply_paths = sorted(generated_scene.glob("view*.ply"))
    assert len(ply_paths) >= 2

    scene_payload = json.loads((generated_scene / "scene.json").read_text(encoding="utf-8"))
    target_world = np.array(scene_payload["bottom_plane_center_world"], dtype=float)
    config = load_config(generated_scene / "resolved_config.json")
    cameras = sample_cameras(config.camera, target_world)

    assert len(cameras) == len(ply_paths)

    for idx, cam in enumerate(cameras, start=1):
        view_name = f"view{idx}"
        tf_key = f"transform_world_from_camera_{view_name}"
        rot_key = f"rotation_world_from_camera_{view_name}"

        assert tf_key in payload
        assert rot_key in payload

        tf = np.array(payload[tf_key], dtype=float)
        rot = np.array(payload[rot_key], dtype=float)

        assert tf.shape == (4, 4)
        assert rot.shape == (3, 3)

        should_be_identity = rot.T @ rot
        assert np.allclose(should_be_identity, np.eye(3), atol=5e-3)

        nominal_tf = np.eye(4)
        nominal_tf[:3, :3] = cam.rotation_world_from_camera_cv
        nominal_tf[:3, 3] = cam.position_world

        if config.noise.enable_pose_noise:
            assert not np.allclose(tf, nominal_tf, atol=1e-9)
        else:
            assert np.allclose(tf, nominal_tf, atol=1e-9)


def test_pose_noise_effect(generated_scene: Path) -> None:
    without_err = read_ply(generated_scene / "merged_without_pose_error.ply")
    with_err = read_ply(generated_scene / "merged_with_pose_error.ply")

    assert without_err.shape[0] > 0
    assert with_err.shape[0] > 0

    if without_err.shape == with_err.shape:
        mean_diff = np.mean(np.linalg.norm(without_err - with_err, axis=1))
        assert mean_diff > 1e-5
    else:
        assert without_err.shape[0] == with_err.shape[0]

    extent = with_err.max(axis=0) - with_err.min(axis=0)
    assert np.all(extent > 0.01)
    assert np.all(extent < 3.0)


def test_visualization_image(generated_scene: Path) -> None:
    img_path = generated_scene / "structure_preview.png"
    assert img_path.exists()

    image = Image.open(img_path)
    arr = np.array(image)
    assert arr.shape[0] > 0 and arr.shape[1] > 0

    flat = arr.reshape(-1, arr.shape[-1])
    unique_colors = np.unique(flat, axis=0)
    assert unique_colors.shape[0] >= 4
