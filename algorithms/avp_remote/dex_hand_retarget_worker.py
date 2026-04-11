from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parent
MODULE_ROOT_STR = str(MODULE_ROOT)
_BLOCKED_SITE_PATH_TOKENS = (
    "/_isaac_sim/",
    "/isaac-sim/",
)


def _sanitize_sys_path() -> None:
    sys.path[:] = [
        path
        for path in sys.path
        if not any(token in path for token in _BLOCKED_SITE_PATH_TOKENS)
    ]


_sanitize_sys_path()
if MODULE_ROOT_STR in sys.path:
    sys.path.remove(MODULE_ROOT_STR)
sys.path.insert(0, MODULE_ROOT_STR)

from dex_retargeting.retargeting_config import RetargetingConfig

from avp_snapshot_io import load_snapshot_payload
from avp_tracking_schema import HAND_JOINT_NAMES, extract_tracking_frame
from avp_transform_utils import TransformOptions, build_xyz_transform, to_usd_world
from avp_g1_pose import load_joint_limits
from dex_hand_specs import (
    DexVectorTargetSpec,
    align_human_local_positions,
    build_h1_2_target_specs,
    build_landau_target_specs,
    human_local_position_map,
    reference_vectors,
    scaling_factor,
    urdf_local_positions,
)


HAND_JOINT_INDEX = {name: index for index, name in enumerate(HAND_JOINT_NAMES)}
TRACKING_OPTIONS = TransformOptions(
    column_major=False,
    pretransform=build_xyz_transform(
        (0.0, 0.0, 180.0),
        (0.0, -0.13, 0.13),
        scale_xyz=(0.6, 0.6, 0.6),
    ).T,
    posttransform=None,
)


@dataclass
class WorkerTarget:
    spec: DexVectorTargetSpec
    robot_local_positions: dict[str, np.ndarray]
    scaling_factor: float
    retargeting: object
    fixed_qpos: np.ndarray
    joint_name_to_index: dict[str, int]
    joint_limits: dict[str, tuple[float, float]]


def _log(message: str) -> None:
    print(f"[AVP-DEX] {message}", file=sys.stderr, flush=True)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dex-retargeting in a helper Python environment.")
    parser.add_argument("--landau-urdf", required=True)
    parser.add_argument("--snapshot-path", required=True)
    parser.add_argument("--baseline-urdf", "--h1-2-urdf", "--g1-urdf", dest="baseline_urdf", default=None)
    return parser.parse_args(argv)


def _tracking_stack_world(stack) -> np.ndarray | None:
    if stack is None:
        return None
    return np.stack(
        [np.asarray(to_usd_world(mat, options=TRACKING_OPTIONS), dtype=float) for mat in stack],
        axis=0,
    )


def _frame_hand_local_positions(frame, side: str) -> np.ndarray | None:
    stack = _tracking_stack_world(frame.get(f"{side}_arm"))
    if stack is None or len(stack) != len(HAND_JOINT_NAMES):
        return None
    wrist_inv = np.linalg.inv(stack[HAND_JOINT_INDEX["wrist"]])
    return np.stack([(wrist_inv @ stack[index])[:3, 3] for index in range(len(HAND_JOINT_NAMES))], axis=0)


def _build_worker_target(spec: DexVectorTargetSpec, snapshot_local_positions: np.ndarray) -> WorkerTarget:
    robot_local = urdf_local_positions(spec)
    scale = scaling_factor(spec, robot_local, human_local_position_map(snapshot_local_positions))
    config = RetargetingConfig.from_dict(spec.build_config(scale, low_pass_alpha=0.0))
    retargeting = config.build()
    fixed_qpos = np.zeros(len(retargeting.optimizer.fixed_joint_names), dtype=np.float32)
    joint_name_to_index = {name: index for index, name in enumerate(retargeting.joint_names)}
    return WorkerTarget(
        spec=spec,
        robot_local_positions=robot_local,
        scaling_factor=scale,
        retargeting=retargeting,
        fixed_qpos=fixed_qpos,
        joint_name_to_index=joint_name_to_index,
        joint_limits=load_joint_limits(spec.urdf_path),
    )


class DexHandRetargetingWorker:
    def __init__(self, *, landau_urdf: Path, snapshot_path: Path, baseline_urdf: Path | None) -> None:
        snapshot_payload = load_snapshot_payload(snapshot_path)
        snapshot_frame = extract_tracking_frame(snapshot_payload)
        self.targets: list[WorkerTarget] = []

        for side, spec in build_landau_target_specs(landau_urdf).items():
            snapshot_local = _frame_hand_local_positions(snapshot_frame, side)
            if snapshot_local is None:
                raise RuntimeError(f"Snapshot is missing {side} hand data for dex-retargeting")
            self.targets.append(_build_worker_target(spec, snapshot_local))

        if baseline_urdf is not None and baseline_urdf.exists():
            for side, spec in build_h1_2_target_specs(baseline_urdf).items():
                snapshot_local = _frame_hand_local_positions(snapshot_frame, side)
                if snapshot_local is None:
                    raise RuntimeError(f"Snapshot is missing {side} hand data for H1_2 dex-retargeting")
                self.targets.append(_build_worker_target(spec, snapshot_local))

        _log(
            "Loaded dex-retargeting targets: "
            + ", ".join(f"{target.spec.group}:{target.spec.side}" for target in self.targets),
        )

    def reset(self) -> None:
        for target in self.targets:
            target.retargeting.reset()

    def retarget(self, request: dict[str, object]) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {"landau": {}, "h1_2": {}}
        side_local_positions: dict[str, dict[str, np.ndarray]] = {}
        for side in ("left", "right"):
            local_positions = request.get(f"{side}_local_positions")
            if local_positions is None:
                continue
            side_local_positions[side] = human_local_position_map(np.asarray(local_positions, dtype=float))

        for target in self.targets:
            human_local = side_local_positions.get(target.spec.side)
            if human_local is None:
                continue
            aligned = align_human_local_positions(target.spec, human_local, target.robot_local_positions)
            ref_value = reference_vectors(target.spec, aligned)
            qpos = target.retargeting.retarget(ref_value, fixed_qpos=target.fixed_qpos)
            result[target.spec.group].update(
                {
                    pose_name: float(
                        np.clip(
                            qpos[target.joint_name_to_index[target_joint_name]],
                            target.joint_limits[pose_name][0],
                            target.joint_limits[pose_name][1],
                        ),
                    )
                    for pose_name, target_joint_name in zip(
                        target.spec.resolved_pose_joint_names,
                        target.spec.target_joint_names,
                        strict=False,
                    )
                },
            )
        return result


def _emit(message: dict[str, object]) -> None:
    print(json.dumps(message), flush=True)


def main() -> None:
    args = _parse_args(sys.argv[1:])
    baseline_urdf = Path(args.baseline_urdf).expanduser().resolve() if args.baseline_urdf else None
    worker = DexHandRetargetingWorker(
        landau_urdf=Path(args.landau_urdf).expanduser().resolve(),
        snapshot_path=Path(args.snapshot_path).expanduser().resolve(),
        baseline_urdf=baseline_urdf,
    )
    _emit({"status": "ready"})

    for line in sys.stdin:
        message = line.strip()
        if not message:
            continue
        try:
            request = json.loads(message)
            command = request.get("command", "retarget")
            if command == "reset":
                worker.reset()
                _emit({"status": "ok"})
            elif command == "close":
                _emit({"status": "ok"})
                return
            elif command == "retarget":
                _emit({"status": "ok", "result": worker.retarget(request)})
            else:
                _emit({"status": "error", "message": f"Unknown command: {command}"})
        except Exception as exc:  # pragma: no cover - integration path
            _log(f"Worker request failed: {exc}")
            _emit({"status": "error", "message": str(exc)})


if __name__ == "__main__":
    main()
