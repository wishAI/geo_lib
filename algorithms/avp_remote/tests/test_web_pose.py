import copy
import json
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web_pose import solve, validate_tracking
from landau_retarget import LandauUpperBodyRetargeter
from asset_paths import landau_urdf_path, landau_skeleton_json_path


class WebPoseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads((ROOT / 'avp_snapshot.json').read_text())
        cls.retarget = LandauUpperBodyRetargeter(urdf_path=landau_urdf_path(),
            skeleton_json_path=landau_skeleton_json_path(), snapshot_path=ROOT / 'avp_snapshot.json',
            use_trac_ik=False)

    def test_webxr_25_joints_match_native_fingers_without_forearms(self):
        webxr = copy.deepcopy(self.payload)
        for side in ('left', 'right'):
            webxr[side + '_arm'] = webxr[side + '_arm'][:25]
        expected = solve(self.retarget, self.payload)['pose']
        actual = solve(self.retarget, webxr)['pose']
        self.assertEqual(expected, actual)
        self.assertGreater(abs(actual['index2_l']), .1)

    def test_snapshot_is_deterministic_and_missing_hands_do_not_leak(self):
        first = solve(self.retarget, self.payload)['pose']
        missing = solve(self.retarget, {'head': self.payload['head']})['pose']
        self.assertNotIn('shoulder_l', missing)
        self.assertEqual(first, solve(self.retarget, self.payload)['pose'])
        for name, value in first.items():
            spec = self.retarget.joint_specs[name]
            self.assertTrue(np.isfinite(value))
            self.assertLessEqual(value, spec.upper)
            self.assertGreaterEqual(value, spec.lower)

    def test_reject_invalid_or_singular_transforms(self):
        for matrix in (np.zeros((4,4)), np.full((4,4), float('nan')), np.eye(3)):
            with self.assertRaises(ValueError):
                validate_tracking({'head': matrix.tolist()})
        with self.assertRaises(ValueError):
            validate_tracking({})
        with self.assertRaises(ValueError):
            validate_tracking({'left_arm': [np.eye(4).tolist()] * 24})

    @unittest.skipUnless((ROOT / 'outputs/web_scene/scene.json').exists(), 'Prepare browser scene first')
    def test_browser_bake_matches_worker_pose(self):
        scene = json.loads((ROOT / 'outputs/web_scene/scene.json').read_text())
        for name, value in solve(self.retarget, self.payload)['pose'].items():
            self.assertAlmostEqual(value, scene['snapshotPose'][name], places=6)


if __name__ == '__main__':
    unittest.main()
