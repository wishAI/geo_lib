import argparse
import json
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from avp_snapshot_io import SnapshotIOError, load_snapshot_payload
from config import (
    ASSET_PRIM,
    AVP_IP,
    AVP_SNAPSHOT_PATH,
    AVP_USD_PATH,
    BRIDGE_HOST,
    BRIDGE_PORT,
    USE_ZMQ,
    ZMQ_CONNECT_ENDPOINT,
    ZMQ_SUB_TOPICS,
)


def _print_error(context, err):
    print(f"[AVP] {context}: {type(err).__name__}: {err}")


class TrackingSourceError(Exception):
    pass


try:
    import zmq

    HAS_ZMQ = True
except Exception as e:
    zmq = None
    HAS_ZMQ = False
    _print_error("zmq import failed", e)


from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from pxr import Gf, Sdf, UsdLux

from avp_marker_visualizer import HandMarkerSetVisualizer, HandStyle, MarkerStyle, MarkerVisualizer
from avp_transform_utils import TransformOptions, build_xyz_transform, to_usd_world
from avp_tracking_schema import HAND_JOINT_NAMES, extract_tracking_frame
from usd_utils import apply_gray_override

try:
    from avp_stream import VisionProStreamer

    HAS_AVP_STREAM = True
except Exception as e:
    VisionProStreamer = None
    HAS_AVP_STREAM = False
    _print_error("avp_stream import failed", e)


# Use direct Vision Pro stream only when avp_stream is available and AVP_IP is set.
USE_AVP_STREAM = False
PRETRANSFORM_TRANSLATE_XYZ_M = (0.0, 0.0, 10.0)

# Vision Pro matrices are usually column-major (translation in last column).
# USD expects row-major when passed into Gf.Matrix4d(list-of-lists).
AVP_TO_USD_OPTIONS = TransformOptions(
    column_major=True,
    # pretransform is left-multiplied, so translation here is in world-space.
    # pretransform=build_xyz_transform((0.0, 0.0, 0.0), PRETRANSFORM_TRANSLATE_XYZ_M),
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


class TrackingStream:
    def __init__(self, tracking_source="bridge", snapshot_path=AVP_SNAPSHOT_PATH):
        if tracking_source not in ("bridge", "snapshot"):
            raise TrackingSourceError(f"Unsupported tracking source: {tracking_source}")

        self.tracking_source = tracking_source
        self.snapshot_path = Path(snapshot_path).expanduser()
        self.streamer = None
        self.bridge_sock = None
        self.snapshot_frame = None
        self._setup()

    def _make_bridge_udp_socket(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((BRIDGE_HOST, BRIDGE_PORT))
        sock.setblocking(False)
        return ("udp", sock)

    def _make_bridge_zmq_socket(self):
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.connect(ZMQ_CONNECT_ENDPOINT)
        for topic in ZMQ_SUB_TOPICS:
            sock.setsockopt(zmq.SUBSCRIBE, topic)
        return ("zmq", sock)

    def _setup(self):
        if self.tracking_source == "snapshot":
            try:
                payload = load_snapshot_payload(self.snapshot_path)
            except SnapshotIOError as e:
                raise TrackingSourceError(f"Snapshot mode failed: {e}") from e
            self.snapshot_frame = extract_tracking_frame(payload)
            print(f"[AVP] Using snapshot tracking from {self.snapshot_path}")
            return

        if HAS_AVP_STREAM and USE_AVP_STREAM and AVP_IP:
            self.streamer = VisionProStreamer(ip=AVP_IP)
            self.streamer.start_webrtc()
            print(f"[AVP] Using direct avp_stream to Vision Pro at {AVP_IP}")
            return

        if USE_ZMQ and HAS_ZMQ:
            self.bridge_sock = self._make_bridge_zmq_socket()
            print(f"[AVP] Listening bridge via ZMQ on {ZMQ_CONNECT_ENDPOINT}")
            return

        self.bridge_sock = self._make_bridge_udp_socket()
        print(f"[AVP] Listening bridge via UDP on {BRIDGE_HOST}:{BRIDGE_PORT}")

    def _recv_bridge(self):
        kind, sock = self.bridge_sock
        try:
            if kind == "zmq":
                parts = sock.recv_multipart(flags=zmq.NOBLOCK)
                if len(parts) < 2:
                    return None
                payload = parts[1]
            else:
                payload, _ = sock.recvfrom(65535)
        except BlockingIOError:
            return None
        except Exception:
            return None

        if not payload:
            return None

        try:
            return json.loads(payload.decode("utf-8"))
        except Exception as e:
            _print_error("bridge json decode failed", e)
            return None

    def get_tracking_frame(self):
        if self.tracking_source == "snapshot":
            return self.snapshot_frame

        raw = self.streamer.get_latest() if self.streamer is not None else self._recv_bridge()
        if raw is None:
            return None
        return extract_tracking_frame(raw)


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
        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()
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
        self.world.reset()
        while simulation_app.is_running():
            now = time.time()
            frame = self.stream.get_tracking_frame()
            if frame is not None:
                self._apply_frame(frame, now)
                self._print_status(frame, now)
            self.world.step(render=True)
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
