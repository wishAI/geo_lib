from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from compare_urdf_pose_offline import load_records_from_json
from pose_diagnostics import arm_pose_symmetry_report


class PoseDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.outputs_dir = MODULE_ROOT / 'outputs'
        self.skeleton_json_path = self.outputs_dir / 'landau_v10_skeleton.json'
        _, self.records = load_records_from_json(self.skeleton_json_path)

    def test_open_arms_usd_symmetry_is_tight(self) -> None:
        report = arm_pose_symmetry_report(self.records, 'open_arms')
        metrics_by_left = {metric['left']: metric for metric in report['usd_pair_metrics']}

        self.assertLess(metrics_by_left['forearm_stretch_l']['mirror_position_error_m'], 1e-3)
        self.assertLess(metrics_by_left['hand_l']['mirror_position_error_m'], 1e-3)
        self.assertLess(metrics_by_left['hand_l']['mirror_rotation_error_deg'], 1.0)

    def test_open_arms_mesh_urdf_matches_usd_on_arm_chain(self) -> None:
        report = arm_pose_symmetry_report(
            self.records,
            'open_arms',
            urdf_path=self.outputs_dir / 'landau_v10_parallel_mesh.urdf',
        )

        for metric in report['usd_vs_urdf_pair_metrics']:
            self.assertLess(metric['left_position_error_m'], 1e-4)
            self.assertLess(metric['right_position_error_m'], 1e-4)
            self.assertLess(metric['left_rotation_error_deg'], 0.5)
            self.assertLess(metric['right_rotation_error_deg'], 0.5)

    def test_progressive_scan_stays_symmetric_through_full_open_arm_chain(self) -> None:
        report = arm_pose_symmetry_report(self.records, 'open_arms')
        scan_by_name = {step['step']: step for step in report['progressive_scan']}
        full_chain = scan_by_name['full_arm_chain']
        max_pos = max(metric['mirror_position_error_m'] for metric in full_chain['pair_metrics'])
        max_rot = max(metric['mirror_rotation_error_deg'] for metric in full_chain['pair_metrics'])

        self.assertLess(max_pos, 1e-3)
        self.assertLess(max_rot, 1.0)


if __name__ == '__main__':
    unittest.main()
