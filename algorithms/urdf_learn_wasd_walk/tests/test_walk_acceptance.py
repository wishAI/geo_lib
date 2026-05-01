from __future__ import annotations

from algorithms.urdf_learn_wasd_walk.walk_acceptance import default_walking_pose_thresholds, walking_pose_failures


def test_game_thresholds_scale_to_ten_meter_acceptance() -> None:
    thresholds = default_walking_pose_thresholds("game", target_distance_m=10.0)
    assert thresholds["min_control_root_height"] == 0.153
    assert thresholds["min_contact_switches_per_side"] == 12
    assert thresholds["min_touchdown_step_length_mean"] == 0.05


def test_walking_pose_failures_pass_on_complete_gait_metrics() -> None:
    thresholds = default_walking_pose_thresholds("fwd_only", target_distance_m=10.0)
    failures = walking_pose_failures(
        {
            "min_control_root_height": 0.18,
            "single_support_ratio": 0.12,
            "double_support_ratio": 0.5,
            "flight_ratio": 0.05,
            "contact_switches": [13, 13],
            "swing_clearance": [0.03, 0.03],
            "touchdown_step_length_mean": [0.06, 0.06],
            "touchdown_root_straddle_mean": [0.02, 0.02],
            "non_support_contact_step_sum": 1,
            "forward_displacement": 9.4,
        },
        thresholds,
    )
    assert failures == []


def test_walking_pose_failures_report_weight_transfer_deficits() -> None:
    thresholds = default_walking_pose_thresholds("fwd_only", target_distance_m=10.0)
    failures = walking_pose_failures(
        {
            "min_control_root_height": 0.09,
            "single_support_ratio": 0.02,
            "double_support_ratio": 0.97,
            "flight_ratio": 0.25,
            "contact_switches": [4, 5],
            "swing_clearance": [0.0, 0.0],
            "touchdown_step_length_mean": [0.01, 0.01],
            "touchdown_root_straddle_mean": [0.0, 0.0],
            "non_support_contact_step_sum": 6,
            "forward_displacement": 1.2,
        },
        thresholds,
    )
    assert any("min_control_root_height" in failure for failure in failures)
    assert any("touchdown_root_straddle_mean" in failure for failure in failures)
    assert any("forward_displacement" in failure for failure in failures)
