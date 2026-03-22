from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    fov_y_deg: float = 45.5
    fps: int = 30
    depth_range_min_m: float = 0.6
    depth_range_max_m: float = 8.0
    num_views: int = 3
    ring_radius_m: float = 1.25
    ring_height_m: float = 0.45
    first_view_deg: float = 20.0
    view_step_deg: float = 30.0


@dataclass
class SceneConfig:
    bottom_size_m: tuple[float, float, float] = (1.2, 0.8, 0.03)
    vertical_board_count_min: int = 2
    vertical_board_count_max: int = 3
    vertical_length_range_m: tuple[float, float] = (0.40, 0.85)
    vertical_height_range_m: tuple[float, float] = (0.30, 0.60)
    vertical_thickness_range_m: tuple[float, float] = (0.02, 0.045)


@dataclass
class NoiseConfig:
    enable_depth_noise: bool = True
    depth_noise_a_m: float = 0.0010
    depth_noise_b_m: float = 0.0020
    enable_pose_noise: bool = True
    pose_translation_sigma_m: float = 0.003
    pose_rotation_sigma_deg: float = 0.25


@dataclass
class DatasetConfig:
    seed: int = 0
    camera: CameraConfig = field(default_factory=CameraConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "DatasetConfig":
        camera_data = data.get("camera", {})
        scene_data = data.get("scene", {})
        noise_data = data.get("noise", {})

        scene = SceneConfig(
            bottom_size_m=tuple(scene_data.get("bottom_size_m", SceneConfig.bottom_size_m)),
            vertical_board_count_min=scene_data.get(
                "vertical_board_count_min", SceneConfig.vertical_board_count_min
            ),
            vertical_board_count_max=scene_data.get(
                "vertical_board_count_max", SceneConfig.vertical_board_count_max
            ),
            vertical_length_range_m=tuple(
                scene_data.get("vertical_length_range_m", SceneConfig.vertical_length_range_m)
            ),
            vertical_height_range_m=tuple(
                scene_data.get("vertical_height_range_m", SceneConfig.vertical_height_range_m)
            ),
            vertical_thickness_range_m=tuple(
                scene_data.get(
                    "vertical_thickness_range_m", SceneConfig.vertical_thickness_range_m
                )
            ),
        )

        num_views = int(camera_data.get("num_views", CameraConfig.num_views))
        if "view_step_deg" in camera_data:
            view_step_deg = camera_data["view_step_deg"]
        elif "ring_span_deg" in camera_data and num_views > 1:
            view_step_deg = camera_data["ring_span_deg"] / float(num_views - 1)
        else:
            view_step_deg = CameraConfig.view_step_deg

        camera = CameraConfig(
            width=camera_data.get("width", CameraConfig.width),
            height=camera_data.get("height", CameraConfig.height),
            fov_y_deg=camera_data.get("fov_y_deg", CameraConfig.fov_y_deg),
            fps=camera_data.get("fps", CameraConfig.fps),
            depth_range_min_m=camera_data.get(
                "depth_range_min_m", CameraConfig.depth_range_min_m
            ),
            depth_range_max_m=camera_data.get(
                "depth_range_max_m", CameraConfig.depth_range_max_m
            ),
            num_views=num_views,
            ring_radius_m=camera_data.get("ring_radius_m", CameraConfig.ring_radius_m),
            ring_height_m=camera_data.get("ring_height_m", CameraConfig.ring_height_m),
            first_view_deg=camera_data.get("first_view_deg", CameraConfig.first_view_deg),
            view_step_deg=view_step_deg,
        )

        noise = NoiseConfig(
            enable_depth_noise=noise_data.get(
                "enable_depth_noise", NoiseConfig.enable_depth_noise
            ),
            depth_noise_a_m=noise_data.get("depth_noise_a_m", NoiseConfig.depth_noise_a_m),
            depth_noise_b_m=noise_data.get("depth_noise_b_m", NoiseConfig.depth_noise_b_m),
            enable_pose_noise=noise_data.get("enable_pose_noise", NoiseConfig.enable_pose_noise),
            pose_translation_sigma_m=noise_data.get(
                "pose_translation_sigma_m", NoiseConfig.pose_translation_sigma_m
            ),
            pose_rotation_sigma_deg=noise_data.get(
                "pose_rotation_sigma_deg", NoiseConfig.pose_rotation_sigma_deg
            ),
        )

        return DatasetConfig(
            seed=data.get("seed", DatasetConfig.seed),
            camera=camera,
            scene=scene,
            noise=noise,
        )


def load_config(config_path: str | Path | None = None) -> DatasetConfig:
    if config_path is None:
        return DatasetConfig()

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return DatasetConfig.from_dict(data)


def save_config(config: DatasetConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)
