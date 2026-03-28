import importlib
import os
import sys
import unittest


ENV_KEYS = (
    "AVP_IP",
    "BRIDGE_HOST",
    "BRIDGE_PORT",
    "USE_ZMQ",
    "SEND_HZ",
    "AVP_USD_PATH",
    "AVP_ROBOT_XML_PATH",
    "AVP_SNAPSHOT_PATH",
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
        self.assertEqual(cfg.AVP_USD_PATH, cfg.REPO_ROOT / "landau_v8.usdc")
        self.assertEqual(cfg.LOAD_USD_PATH, cfg.REPO_ROOT / "landau_v10.usdc")
        self.assertEqual(cfg.AVP_ROBOT_XML_PATH, cfg.REPO_ROOT / "google_robot/robot.xml")
        self.assertEqual(cfg.AVP_SNAPSHOT_PATH, cfg.REPO_ROOT / "avp_snapshot.json")

    def test_env_overrides_for_string_int_bool_and_paths(self):
        cfg = _load_config_with_env(
            {
                "AVP_IP": "10.0.0.11",
                "BRIDGE_HOST": "0.0.0.0",
                "BRIDGE_PORT": "60001",
                "USE_ZMQ": "false",
                "SEND_HZ": "120",
                "AVP_USD_PATH": "assets/custom_scene.usdc",
                "AVP_ROBOT_XML_PATH": "google_robot/scene.xml",
                "AVP_SNAPSHOT_PATH": "debug/snapshot.json",
            }
        )
        self.assertEqual(cfg.AVP_IP, "10.0.0.11")
        self.assertEqual(cfg.BRIDGE_HOST, "0.0.0.0")
        self.assertEqual(cfg.BRIDGE_PORT, 60001)
        self.assertFalse(cfg.USE_ZMQ)
        self.assertEqual(cfg.SEND_HZ, 120)
        self.assertEqual(cfg.AVP_USD_PATH, cfg.REPO_ROOT / "assets/custom_scene.usdc")
        self.assertEqual(cfg.LOAD_USD_PATH, cfg.REPO_ROOT / "assets/custom_scene.usdc")
        self.assertEqual(cfg.AVP_ROBOT_XML_PATH, cfg.REPO_ROOT / "google_robot/scene.xml")
        self.assertEqual(cfg.AVP_SNAPSHOT_PATH, cfg.REPO_ROOT / "debug/snapshot.json")

    def test_invalid_int_or_bool_falls_back_to_defaults(self):
        cfg = _load_config_with_env({"BRIDGE_PORT": "bad", "SEND_HZ": "oops", "USE_ZMQ": "maybe"})
        self.assertEqual(cfg.BRIDGE_PORT, 45800)
        self.assertEqual(cfg.SEND_HZ, 60)
        self.assertTrue(cfg.USE_ZMQ)


if __name__ == "__main__":
    unittest.main()
