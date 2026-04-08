import importlib
import os
import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))


ENV_KEYS = (
    "AVP_IP",
    "BRIDGE_HOST",
    "BRIDGE_PORT",
    "USE_ZMQ",
    "SEND_HZ",
    "ISAAC_SIM_SH_PATH",
    "AVP_USD_PATH",
    "AVP_URDF_PATH",
    "AVP_SKELETON_JSON_PATH",
    "AVP_MESH_ROOT",
    "AVP_ROBOT_XML_PATH",
    "AVP_SNAPSHOT_PATH",
    "AVP_G1_URDF_PATH",
)


def _load_config_with_env(overrides=None):
    saved_env = {key: os.environ.get(key) for key in ENV_KEYS}
    try:
        for key in ENV_KEYS:
            os.environ.pop(key, None)
        for key, value in (overrides or {}).items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        sys.modules.pop("config", None)
        return importlib.import_module("config")
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        sys.modules.pop("config", None)


class TestConfig(unittest.TestCase):
    def test_defaults_when_env_unset(self):
        cfg = _load_config_with_env()
        self.assertEqual(cfg.AVP_IP, cfg.DEFAULT_AVP_IP)
        self.assertEqual(cfg.BRIDGE_HOST, "127.0.0.1")
        self.assertEqual(cfg.BRIDGE_PORT, 45800)
        self.assertTrue(cfg.USE_ZMQ)
        self.assertEqual(cfg.SEND_HZ, 60)
        self.assertEqual(cfg.ISAAC_SIM_SH_PATH, Path("/home/wishai/vscode/IsaacLab/_isaac_sim/isaac-sim.sh"))
        self.assertEqual(cfg.AVP_USD_PATH, cfg.REPO_ROOT / "inputs" / "landau_v10" / "landau_v10.usdc")
        self.assertEqual(cfg.LOAD_USD_PATH, cfg.REPO_ROOT / "inputs" / "landau_v10" / "landau_v10.usdc")
        self.assertEqual(
            cfg.AVP_URDF_PATH,
            cfg.REPO_ROOT / "inputs" / "landau_v10" / "landau_v10_parallel_mesh.urdf",
        )
        self.assertEqual(
            cfg.AVP_SKELETON_JSON_PATH,
            cfg.REPO_ROOT / "inputs" / "landau_v10" / "landau_v10_skeleton.json",
        )
        self.assertEqual(
            cfg.AVP_MESH_ROOT,
            cfg.REPO_ROOT / "inputs" / "landau_v10" / "mesh_collision_stl",
        )
        self.assertEqual(cfg.AVP_ROBOT_XML_PATH, cfg.REPO_ROOT / "google_robot/robot.xml")
        self.assertEqual(cfg.AVP_SNAPSHOT_PATH, cfg.REPO_ROOT / "avp_snapshot.json")
        self.assertEqual(
            cfg.AVP_G1_URDF_PATH,
            cfg.REPO_ROOT.parent.parent / "helper_repos" / "xr_teleoperate_shallow" / "assets" / "h1_2" / "h1_2.urdf",
        )

    def test_env_overrides_for_string_int_bool_and_paths(self):
        cfg = _load_config_with_env(
            {
                "AVP_IP": "10.0.0.11",
                "BRIDGE_HOST": "0.0.0.0",
                "BRIDGE_PORT": "60001",
                "USE_ZMQ": "false",
                "SEND_HZ": "120",
                "ISAAC_SIM_SH_PATH": "~/custom_isaac/isaac-sim.sh",
                "AVP_USD_PATH": "assets/custom_scene.usdc",
                "AVP_URDF_PATH": "assets/custom_robot.urdf",
                "AVP_SKELETON_JSON_PATH": "assets/custom_skeleton.json",
                "AVP_MESH_ROOT": "assets/mesh_collision_stl",
                "AVP_ROBOT_XML_PATH": "google_robot/scene.xml",
                "AVP_SNAPSHOT_PATH": "debug/snapshot.json",
                "AVP_G1_URDF_PATH": "debug/custom_g1.urdf",
            }
        )
        self.assertEqual(cfg.AVP_IP, "10.0.0.11")
        self.assertEqual(cfg.BRIDGE_HOST, "0.0.0.0")
        self.assertEqual(cfg.BRIDGE_PORT, 60001)
        self.assertFalse(cfg.USE_ZMQ)
        self.assertEqual(cfg.SEND_HZ, 120)
        self.assertEqual(cfg.ISAAC_SIM_SH_PATH, Path("~/custom_isaac/isaac-sim.sh").expanduser())
        self.assertEqual(cfg.AVP_USD_PATH, cfg.REPO_ROOT / "assets/custom_scene.usdc")
        self.assertEqual(cfg.LOAD_USD_PATH, cfg.REPO_ROOT / "assets/custom_scene.usdc")
        self.assertEqual(cfg.AVP_URDF_PATH, cfg.REPO_ROOT / "assets/custom_robot.urdf")
        self.assertEqual(cfg.AVP_SKELETON_JSON_PATH, cfg.REPO_ROOT / "assets/custom_skeleton.json")
        self.assertEqual(cfg.AVP_MESH_ROOT, cfg.REPO_ROOT / "assets/mesh_collision_stl")
        self.assertEqual(cfg.AVP_ROBOT_XML_PATH, cfg.REPO_ROOT / "google_robot/scene.xml")
        self.assertEqual(cfg.AVP_SNAPSHOT_PATH, cfg.REPO_ROOT / "debug/snapshot.json")
        self.assertEqual(
            cfg.AVP_G1_URDF_PATH,
            cfg.REPO_ROOT / "debug/custom_g1.urdf",
        )

    def test_invalid_int_or_bool_falls_back_to_defaults(self):
        cfg = _load_config_with_env({"BRIDGE_PORT": "bad", "SEND_HZ": "oops", "USE_ZMQ": "maybe"})
        self.assertEqual(cfg.BRIDGE_PORT, 45800)
        self.assertEqual(cfg.SEND_HZ, 60)
        self.assertTrue(cfg.USE_ZMQ)


if __name__ == "__main__":
    unittest.main()
