from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


class OffscreenRenderer:
    def __init__(self, mjcf_xml: str, width: int, height: int):
        self.model = mujoco.MjModel.from_xml_string(mjcf_xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, width=width, height=height)

    def render_depth_meters(self, camera_name: str) -> np.ndarray:
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=camera_name)
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()
        return depth_to_meters(depth, self.model)

    def render_rgb(self, camera_name: str) -> np.ndarray:
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera=camera_name)
        rgb = self.renderer.render()
        return rgb

    def close(self) -> None:
        self.renderer.close()


def depth_to_meters(depth: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
    max_depth = float(np.nanmax(depth))
    min_depth = float(np.nanmin(depth))
    if 0.0 <= min_depth and max_depth <= 1.0:
        near = model.vis.map.znear * model.stat.extent
        far = model.vis.map.zfar * model.stat.extent
        depth = near / (1.0 - depth * (1.0 - near / far))

    depth = np.where(np.isfinite(depth), depth, np.nan)
    return depth
