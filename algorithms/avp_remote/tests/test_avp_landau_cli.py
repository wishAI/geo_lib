import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from avp_landau_cli import parse_args


class TestAvpLandauCli(unittest.TestCase):
    def test_baseline_is_disabled_by_default(self):
        args = parse_args([])
        self.assertFalse(args.enable_baseline)

    def test_baseline_can_be_enabled_explicitly(self):
        args = parse_args(["--baseline"])
        self.assertTrue(args.enable_baseline)

    def test_solved_markers_are_disabled_by_default(self):
        args = parse_args([])
        self.assertFalse(args.show_solved_markers)

    def test_no_g1_alias_disables_baseline(self):
        args = parse_args(["--baseline", "--no-g1"])
        self.assertFalse(args.enable_baseline)

    def test_baseline_path_aliases_share_destination(self):
        args = parse_args(["--g1-urdf-path", "/tmp/custom_baseline.urdf"])
        self.assertEqual(args.baseline_urdf_path, "/tmp/custom_baseline.urdf")


if __name__ == "__main__":
    unittest.main()
