from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from algorithms.fake_cloud.camera_sampler import sample_cameras
from algorithms.fake_cloud.config import load_config, save_config
from algorithms.fake_cloud.noise_model import apply_depth_noise, perturb_camera_pose
from algorithms.fake_cloud.pointcloud import (
    depth_to_points_camera,
    intrinsics_from_fovy,
    transform_points_world,
    write_ply,
)
from algorithms.fake_cloud.renderer import OffscreenRenderer
from algorithms.fake_cloud.scene_builder import build_mjcf, generate_scene, save_scene_metadata
from algorithms.fake_cloud.visualize import save_structure_preview


def _camera_payload(
    view_name: str,
    position_world: np.ndarray,
    rotation_world_from_camera: np.ndarray,
    width: int,
    height: int,
    fov_y_deg: float,
) -> dict:
    transform = np.eye(4)
    transform[:3, :3] = rotation_world_from_camera
    transform[:3, 3] = position_world
    return {
        "view_name": view_name,
        "position_world": position_world.tolist(),
        "rotation_world_from_camera": rotation_world_from_camera.tolist(),
        "transform_world_from_camera": transform.tolist(),
        "fov_y_deg": fov_y_deg,
        "width": width,
        "height": height,
    }


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def generate_scene_dataset(
    output_dir: str | Path,
    seed: int | None = None,
    config_path: str | Path | None = None,
) -> Path:
    config = load_config(config_path)
    if seed is not None:
        config.seed = seed

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.seed)
    scene = generate_scene(config.scene, rng)
    cameras = sample_cameras(config.camera, scene.bottom_plane_center_world)

    save_scene_metadata(scene, output_root / "scene.json")
    _save_json(output_root / "config_used.json", config.to_dict())

    mjcf_xml = build_mjcf(scene, cameras, config.camera.fov_y_deg)
    renderer = OffscreenRenderer(
        mjcf_xml,
        width=config.camera.width,
        height=config.camera.height,
    )

    preview_rgb = renderer.render_rgb(cameras[0].name)
    save_structure_preview(preview_rgb, output_root / "structure_preview.png")

    intr = intrinsics_from_fovy(
        width=config.camera.width,
        height=config.camera.height,
        fov_y_deg=config.camera.fov_y_deg,
    )

    merged_without_pose_error_parts: list[np.ndarray] = []
    merged_with_pose_error_parts: list[np.ndarray] = []
    cam_info: dict[str, object] = {
        "width": config.camera.width,
        "height": config.camera.height,
        "fov_y_deg": config.camera.fov_y_deg,
        "includes_pose_error": bool(config.noise.enable_pose_noise),
    }

    for idx, cam in enumerate(cameras, start=1):
        view_name = f"view{idx}"
        depth = renderer.render_depth_meters(cam.name)
        noisy_depth = apply_depth_noise(depth, config.noise, rng)

        points_cam = depth_to_points_camera(
            noisy_depth,
            intr,
            config.camera.depth_range_min_m,
            config.camera.depth_range_max_m,
        )

        points_world_nominal = transform_points_world(
            points_cam,
            cam.rotation_world_from_camera_cv,
            cam.position_world,
        )
        write_ply(output_root / f"{view_name}.ply", points_world_nominal)

        merged_without_pose_error_parts.append(points_world_nominal)

        noisy_rot, noisy_pos = perturb_camera_pose(
            cam.rotation_world_from_camera_cv,
            cam.position_world,
            config.noise,
            rng,
        )
        # Pose noise is injected here before converting this view cloud to world
        # coordinates for the merged point cloud with camera pose error.
        points_world_pose_error = transform_points_world(points_cam, noisy_rot, noisy_pos)
        merged_with_pose_error_parts.append(points_world_pose_error)

        noisy_pose_payload = _camera_payload(
            view_name=view_name,
            position_world=noisy_pos,
            rotation_world_from_camera=noisy_rot,
            width=config.camera.width,
            height=config.camera.height,
            fov_y_deg=config.camera.fov_y_deg,
        )
        cam_info[f"position_world_{view_name}"] = noisy_pose_payload["position_world"]
        cam_info[f"rotation_world_from_camera_{view_name}"] = noisy_pose_payload[
            "rotation_world_from_camera"
        ]
        cam_info[f"transform_world_from_camera_{view_name}"] = noisy_pose_payload[
            "transform_world_from_camera"
        ]

    merged_without_pose_error = (
        np.concatenate(merged_without_pose_error_parts, axis=0)
        if merged_without_pose_error_parts
        else np.zeros((0, 3), dtype=float)
    )
    merged_with_pose_error = (
        np.concatenate(merged_with_pose_error_parts, axis=0)
        if merged_with_pose_error_parts
        else np.zeros((0, 3), dtype=float)
    )

    write_ply(output_root / "merged_without_pose_error.ply", merged_without_pose_error)
    write_ply(output_root / "merged_with_pose_error.ply", merged_with_pose_error)
    _save_json(output_root / "cam_info.json", cam_info)

    renderer.close()
    save_config(config, output_root / "resolved_config.json")
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic MuJoCo point-cloud dataset for ship-part scenes."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sample_scene"),
        help="Output folder for generated files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON config path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = generate_scene_dataset(
        output_dir=args.output,
        seed=args.seed,
        config_path=args.config,
    )
    print(f"Dataset generated at: {output_dir}")


if __name__ == "__main__":
    main()
