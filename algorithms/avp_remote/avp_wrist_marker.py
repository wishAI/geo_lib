import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent
MODULE_ROOT_STR = str(MODULE_ROOT)
if MODULE_ROOT_STR in sys.path:
    sys.path.remove(MODULE_ROOT_STR)
sys.path.insert(0, MODULE_ROOT_STR)

import numpy as np
import omni.kit.app

from asset_setup import prepare_landau_inputs
from avp_config import (
    ASSET_PRIM,
    AVP_SNAPSHOT_PATH,
    AVP_USD_PATH,
)
from tracking_source import TrackingSourceError, TrackingStream

try:
    from isaacsim import SimulationApp
except ImportError:
    SimulationApp = None

simulation_app = SimulationApp({"headless": False}) if SimulationApp is not None else None
APP = omni.kit.app.get_app()

from omni.isaac.core import World
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from pxr import Gf, Sdf, UsdLux

from avp_marker_visualizer import HandMarkerSetVisualizer, HandStyle, MarkerStyle, MarkerVisualizer
from avp_transform_utils import TransformOptions, build_xyz_transform, to_usd_world
from avp_tracking_schema import HAND_JOINT_NAMES
from usd_utils import apply_gray_override

# Snapshot and bridge payloads carry AVP transforms with translation in the
# last column. Transpose them into USD's authored row-major convention first.
AVP_TO_USD_OPTIONS = TransformOptions(
    column_major=True,
    pretransform=None,
    posttransform=build_xyz_transform(
        (0.0, 0.0, 180.0),
        (0.0, -0.13, 0.13),
        scale_xyz=(0.6, 0.6, 0.6),
    ),
)


@dataclass(frozen=True)
class RenderConfig:
    load_usd_asset: bool = True
    render_head_marker: bool = True
    print_debug: bool = True
    print_status_every_sec: float = 1.0
    sanitize_dome_light_textures: bool = True
    left_joint_root: str = "/World/AVP/LeftHandJoints"
    right_joint_root: str = "/World/AVP/RightHandJoints"
    head_marker_path: str = "/World/AVP/Head"
    left_hand_style: HandStyle = field(
        default_factory=lambda: HandStyle(
            wrist=MarkerStyle(mode="axes", color=(0.15, 0.75, 1.0)),
            joint=MarkerStyle(mode="sphere", color=(0.15, 0.75, 1.0)),
        )
    )
    right_hand_style: HandStyle = field(
        default_factory=lambda: HandStyle(
            wrist=MarkerStyle(mode="axes", color=(1.0, 0.45, 0.15)),
            joint=MarkerStyle(mode="sphere", color=(1.0, 0.45, 0.15)),
        )
    )
    head_style: MarkerStyle = field(
        default_factory=lambda: MarkerStyle(
            mode="sphere",
            color=(1.0, 1.0, 0.2),
            radius=0.035,
        )
    )


RENDER_CONFIG = RenderConfig()

def _to_usd_world(mat):
    return to_usd_world(mat, options=AVP_TO_USD_OPTIONS)


def _asset_path_to_str(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    path = getattr(value, "path", None)
    if path:
        return str(path)
    resolved = getattr(value, "resolvedPath", None)
    if resolved:
        return str(resolved)
    return str(value)


def _sanitize_dome_light_textures(stage):
    fixed_count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdLux.DomeLight):
            continue
        dome = UsdLux.DomeLight(prim)
        tex_attr = dome.GetTextureFileAttr()
        if not tex_attr:
            continue
        texture_file = _asset_path_to_str(tex_attr.Get()).strip()
        if not texture_file:
            continue

        try:
            tex_attr.Set(Sdf.AssetPath(""))
        except Exception:
            tex_attr.Set("")
        dome.GetColorAttr().Set(Gf.Vec3f(0.08, 0.08, 0.08))
        if not dome.GetIntensityAttr().HasAuthoredValue():
            dome.GetIntensityAttr().Set(200.0)
        fixed_count += 1

    if fixed_count:
        print(
            f"[AVP] Cleared texture files on {fixed_count} DomeLight prim(s) "
            "to avoid missing relative texture errors."
        )


class AvpJointSession:
    def __init__(self, tracking_source="bridge", snapshot_path=AVP_SNAPSHOT_PATH):
        self.world, self.stage = self._setup_world_and_stage()
        self.stream = TrackingStream(tracking_source=tracking_source, snapshot_path=snapshot_path)

        self.left_markers = HandMarkerSetVisualizer.for_hand(
            self.stage,
            RENDER_CONFIG.left_joint_root,
            HAND_JOINT_NAMES,
            style=RENDER_CONFIG.left_hand_style,
            label="left_hand",
            print_debug=RENDER_CONFIG.print_debug,
        )
        self.right_markers = HandMarkerSetVisualizer.for_hand(
            self.stage,
            RENDER_CONFIG.right_joint_root,
            HAND_JOINT_NAMES,
            style=RENDER_CONFIG.right_hand_style,
            label="right_hand",
            print_debug=RENDER_CONFIG.print_debug,
        )
        self.head_marker = MarkerVisualizer(
            self.stage,
            RENDER_CONFIG.head_marker_path,
            style=RENDER_CONFIG.head_style,
            label="head",
            print_debug=RENDER_CONFIG.print_debug,
        )

        self.last_status_print = 0.0

    def _setup_world_and_stage(self):
        world = None
        if simulation_app is not None:
            world = World(stage_units_in_meters=1.0)
            world.scene.add_default_ground_plane()
        prepare_landau_inputs(refresh=False)
        if RENDER_CONFIG.load_usd_asset:
            add_reference_to_stage(usd_path=str(AVP_USD_PATH), prim_path=ASSET_PRIM)
        stage = get_current_stage()
        apply_gray_override(stage, ASSET_PRIM)
        if RENDER_CONFIG.sanitize_dome_light_textures:
            _sanitize_dome_light_textures(stage)
        return world, stage

    def _print_status(self, frame, now):
        if not RENDER_CONFIG.print_debug:
            return
        if now - self.last_status_print <= RENDER_CONFIG.print_status_every_sec:
            return

        left_arm = frame.get("left_arm")
        right_arm = frame.get("right_arm")
        left_count = int(left_arm.shape[0]) if left_arm is not None else 0
        right_count = int(right_arm.shape[0]) if right_arm is not None else 0

        print(
            "[AVP] frame "
            f"left_joints={left_count} "
            f"right_joints={right_count} "
            f"left_pinch={frame.get('left_pinch_distance')} "
            f"right_pinch={frame.get('right_pinch_distance')}"
        )

        if left_arm is not None:
            print(f"[AVP] left wrist translation (m): {_to_usd_world(left_arm[0])[:3, 3]}")
        if right_arm is not None:
            print(f"[AVP] right wrist translation (m): {_to_usd_world(right_arm[0])[:3, 3]}")

        self.last_status_print = now

    def _to_usd_stack(self, joint_stack):
        if joint_stack is None:
            return None
        return np.stack([_to_usd_world(mat) for mat in joint_stack], axis=0)

    def _apply_frame(self, frame, now):
        self.left_markers.update(self._to_usd_stack(frame.get("left_arm")), now=now)
        self.right_markers.update(self._to_usd_stack(frame.get("right_arm")), now=now)

        if RENDER_CONFIG.render_head_marker:
            self.head_marker.update(_to_usd_world(frame.get("head")), now)

    def run(self):
        if self.world is not None:
            self.world.reset()
        while (simulation_app.is_running() if simulation_app is not None else APP.is_running()):
            now = time.time()
            frame = self.stream.get_tracking_frame()
            if frame is not None:
                self._apply_frame(frame, now)
                self._print_status(frame, now)
            if self.world is not None:
                self.world.step(render=True)
            else:
                APP.update()
        if simulation_app is not None:
            simulation_app.close()


def main():
    parser = argparse.ArgumentParser(description="Render AVP hand markers from bridge stream or snapshot payload.")
    parser.add_argument(
        "--tracking-source",
        choices=("bridge", "snapshot"),
        default="bridge",
        help="Tracking source: live bridge data or a single snapshot file.",
    )
    parser.add_argument(
        "--snapshot-path",
        default=str(AVP_SNAPSHOT_PATH),
        help="Snapshot file path used when --tracking-source snapshot.",
    )
    args = parser.parse_args()

    try:
        AvpJointSession(
            tracking_source=args.tracking_source,
            snapshot_path=Path(args.snapshot_path).expanduser(),
        ).run()
    except TrackingSourceError as exc:
        print(f"[AVP] {exc}")
        simulation_app.close()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
