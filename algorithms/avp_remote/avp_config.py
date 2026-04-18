import os
from pathlib import Path

from asset_paths import (
    default_dex_retargeting_python_path,
    default_g1_urdf_path,
    default_h1_2_urdf_path,
    landau_mesh_root,
    landau_skeleton_json_path,
    landau_urdf_path,
    landau_usd_path,
)

REPO_ROOT = Path(__file__).resolve().parent


def _env_str(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _env_int(name, default):
    value = _env_str(name, None)
    if value is None:
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)


def _env_float(name, default):
    value = _env_str(name, None)
    if value is None:
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def _env_bool(name, default):
    value = _env_str(name, None)
    if value is None:
        return bool(default)

    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _env_xyz(name, default):
    value = _env_str(name, None)
    if value is None:
        return tuple(float(item) for item in default)
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        return tuple(float(item) for item in default)
    try:
        return tuple(float(part) for part in parts)
    except ValueError:
        return tuple(float(item) for item in default)


def _env_path(name, default):
    value = _env_str(name, None)
    if value is None:
        return Path(default).expanduser()

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


DEFAULT_AVP_IP = "192.168.2.206"
AVP_IP = _env_str("AVP_IP", DEFAULT_AVP_IP)

BRIDGE_HOST = _env_str("BRIDGE_HOST", "127.0.0.1")
BRIDGE_PORT = _env_int("BRIDGE_PORT", 45800)
USE_ZMQ = _env_bool("USE_ZMQ", True)
SEND_HZ = _env_int("SEND_HZ", 60)

ZMQ_BIND = True
ZMQ_BIND_ENDPOINT = f"tcp://*:{BRIDGE_PORT}"
ZMQ_CONNECT_ENDPOINT = f"tcp://{BRIDGE_HOST}:{BRIDGE_PORT}"
ZMQ_TOPIC_TRACKING = b"tracking"
ZMQ_TOPIC_WRIST = b"wrist"
ZMQ_SUB_TOPICS = (ZMQ_TOPIC_TRACKING, ZMQ_TOPIC_WRIST)

ASSET_PRIM = "/World/MyAsset"

ISAAC_SIM_SH_PATH = _env_path(
    "ISAAC_SIM_SH_PATH",
    Path("/home/wishai/vscode/IsaacLab/_isaac_sim/isaac-sim.sh"),
)

# AVP_USD_PATH is intentionally shared by both visualization scripts.
AVP_USD_PATH = _env_path("AVP_USD_PATH", landau_usd_path())
LOAD_USD_PATH = _env_path("AVP_USD_PATH", landau_usd_path())
AVP_URDF_PATH = _env_path("AVP_URDF_PATH", landau_urdf_path())
AVP_SKELETON_JSON_PATH = _env_path("AVP_SKELETON_JSON_PATH", landau_skeleton_json_path())
AVP_MESH_ROOT = _env_path("AVP_MESH_ROOT", landau_mesh_root())
AVP_ROBOT_XML_PATH = _env_path("AVP_ROBOT_XML_PATH", REPO_ROOT / "google_robot/robot.xml")
AVP_SNAPSHOT_PATH = _env_path("AVP_SNAPSHOT_PATH", REPO_ROOT / "avp_snapshot.json")
AVP_H1_2_URDF_PATH = _env_path("AVP_H1_2_URDF_PATH", default_h1_2_urdf_path())
AVP_G1_URDF_PATH = _env_path("AVP_G1_URDF_PATH", default_g1_urdf_path())
AVP_DEX_RETARGET_PYTHON = _env_path("AVP_DEX_RETARGET_PYTHON", default_dex_retargeting_python_path())

AVP_TRACKING_ROTATE_XYZ = _env_xyz("AVP_TRACKING_ROTATE_XYZ", (0.0, 0.0, 180.0))
AVP_TRACKING_TRANSLATE_XYZ = _env_xyz("AVP_TRACKING_TRANSLATE_XYZ", (0.0, -0.13, 0.13))
AVP_TRACKING_SCALE_XYZ = _env_xyz("AVP_TRACKING_SCALE_XYZ", (0.6, 0.6, 0.6))

RAW_VISUAL_OFFSET_XYZ = _env_xyz("AVP_RAW_VISUAL_OFFSET_XYZ", (-0.8, 0.0, 0.0))
SOLVED_VISUAL_OFFSET_XYZ = _env_xyz("AVP_SOLVED_VISUAL_OFFSET_XYZ", (0.8, 0.0, 0.0))
BASELINE_VISUAL_OFFSET_XYZ = _env_xyz("AVP_BASELINE_VISUAL_OFFSET_XYZ", (2.4, 0.0, 0.0))
BASELINE_VISUAL_YAW_DEGREES = _env_float("AVP_BASELINE_VISUAL_YAW_DEGREES", -90.0)

AVP_HAND_RADIUS_CAP_SCALE = _env_float("AVP_HAND_RADIUS_CAP_SCALE", 1.15)
