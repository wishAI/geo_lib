from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from compare_urdf_pose_offline import load_records_from_json
from pose_diagnostics import animation_clip_balance_report, arm_pose_symmetry_report, mirror_matrix_from_records, root_relative_world_map
from skeleton_common import build_pose_preset


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

    def test_walk_and_walk_right_keyframes_are_mirrored_across_arm_and_leg_links(self) -> None:
        mirror = mirror_matrix_from_records(self.records)
        walk_world = root_relative_world_map(self.records, build_pose_preset(self.records, 'walk'))
        walk_right_world = root_relative_world_map(self.records, build_pose_preset(self.records, 'walk_right'))

        for left_name, right_name in (
            ('arm_stretch_l', 'arm_stretch_r'),
            ('forearm_stretch_l', 'forearm_stretch_r'),
            ('hand_l', 'hand_r'),
            ('thigh_stretch_l', 'thigh_stretch_r'),
            ('foot_l', 'foot_r'),
        ):
            walk_left = walk_world[left_name][:3, 3]
            mirrored_walk_right = mirror @ walk_right_world[right_name][:3, 3]
            walk_right = walk_world[right_name][:3, 3]
            mirrored_walk_left = mirror @ walk_right_world[left_name][:3, 3]
            self.assertLess(float(((walk_left - mirrored_walk_right) ** 2).sum() ** 0.5), 1e-3)
            self.assertLess(float(((walk_right - mirrored_walk_left) ** 2).sum() ** 0.5), 1e-3)

    def test_walk_cycle_keeps_left_and_right_hands_similarly_active(self) -> None:
        report = animation_clip_balance_report(self.records, 'walk_cycle', sample_count=120)
        hand_metrics = next(metric for metric in report['pair_metrics'] if metric['left'] == 'hand_l')

        self.assertGreater(hand_metrics['left_path_length_m'], 0.03)
        self.assertGreater(hand_metrics['right_path_length_m'], 0.03)
        self.assertLess(hand_metrics['path_length_ratio'], 1.35)


if __name__ == '__main__':
    unittest.main()
