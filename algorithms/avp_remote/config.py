import os
from pathlib import Path

from asset_paths import (
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
