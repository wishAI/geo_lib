import math
import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from asset_paths import default_dex_retargeting_python_path, default_g1_urdf_path, landau_urdf_path
from avp_g1_pose import load_joint_limits
from avp_snapshot_io import load_snapshot_payload
from avp_tracking_schema import extract_tracking_frame
from dex_hand_retargeting import DexHandRetargetingClient


def _helper_ready() -> bool:
    helper_python = default_dex_retargeting_python_path()
    return helper_python.exists()


class TestDexHandRetargetingClient(unittest.TestCase):
    @unittest.skipUnless(_helper_ready(), "requires the dex-retargeting helper venv")
    @unittest.skipUnless(default_g1_urdf_path().exists(), "requires cloned H1_2 helper repo")
    def test_snapshot_retargeting_returns_landau_and_h1_2_hand_targets(self):
        snapshot_path = MODULE_ROOT / "avp_snapshot.json"
        payload = load_snapshot_payload(snapshot_path)
        frame = extract_tracking_frame(payload)
        client = DexHandRetargetingClient(
            helper_python=default_dex_retargeting_python_path(),
            landau_urdf_path=landau_urdf_path(),
            snapshot_path=snapshot_path,
            baseline_urdf_path=default_g1_urdf_path(),
        )

        try:
            result = client.retarget_frame(frame)
        finally:
            client.close()

        for joint_name in ("thumb1_l", "index1_l", "middle2_r"):
            self.assertIn(joint_name, result["landau"])
            self.assertTrue(math.isfinite(result["landau"][joint_name]))

        for joint_name in (
            "L_thumb_proximal_pitch_joint",
            "L_index_proximal_joint",
            "R_middle_proximal_joint",
        ):
            self.assertIn(joint_name, result["h1_2"])
            self.assertTrue(math.isfinite(result["h1_2"][joint_name]))

        landau_limits = load_joint_limits(landau_urdf_path())
        for joint_name in ("thumb1_l", "thumb2_l", "index1_l", "ring2_r"):
            lower, upper = landau_limits[joint_name]
            self.assertGreaterEqual(result["landau"][joint_name], lower - 1.0e-5)
            self.assertLessEqual(result["landau"][joint_name], upper + 1.0e-5)

        h1_2_limits = load_joint_limits(default_g1_urdf_path())
        for joint_name in (
            "L_thumb_proximal_yaw_joint",
            "L_thumb_proximal_pitch_joint",
            "L_index_proximal_joint",
            "R_middle_proximal_joint",
        ):
            lower, upper = h1_2_limits[joint_name]
            self.assertGreaterEqual(result["h1_2"][joint_name], lower - 1.0e-5)
            self.assertLessEqual(result["h1_2"][joint_name], upper + 1.0e-5)


if __name__ == "__main__":
    unittest.main()
