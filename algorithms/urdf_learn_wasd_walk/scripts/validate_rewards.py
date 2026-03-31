from __future__ import annotations

from algorithms.urdf_learn_wasd_walk.reward_probe import build_validation_report, validate_reward_report


def main() -> None:
    report = build_validation_report()
    validate_reward_report(report)

    print("[REWARD] validation passed")
    print(f"[REWARD] lin perfect/bad: {report.lin_tracking_perfect:.4f} / {report.lin_tracking_bad:.4f}")
    print(f"[REWARD] yaw perfect/bad: {report.yaw_tracking_perfect:.4f} / {report.yaw_tracking_bad:.4f}")
    print(f"[REWARD] orientation flat/tilted: {report.orientation_flat:.4f} / {report.orientation_tilted:.4f}")
    print(f"[REWARD] action rate zero/large: {report.action_rate_zero:.4f} / {report.action_rate_large:.4f}")


if __name__ == "__main__":
    main()
