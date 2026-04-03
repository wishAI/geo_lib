from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .config import ALGORITHM_ROOT, get_builtin_config, list_builtin_configs, load_config_from_json
from .solver import IkFastSolver, pose_error
from .urdf_utils import sanitize_urdf_for_mujoco


@dataclass(slots=True)
class BenchmarkSummary:
    solver_name: str
    samples: int
    success_count: int
    solve_times_ms: list[float]
    position_errors: list[float]
    rotation_errors: list[float]

    def to_dict(self) -> dict[str, object]:
        return {
            "solver_name": self.solver_name,
            "samples": self.samples,
            "success_count": self.success_count,
            "success_rate": self.success_count / self.samples if self.samples else 0.0,
            "mean_solve_ms": float(np.mean(self.solve_times_ms)) if self.solve_times_ms else None,
            "median_solve_ms": float(np.median(self.solve_times_ms)) if self.solve_times_ms else None,
            "max_solve_ms": float(np.max(self.solve_times_ms)) if self.solve_times_ms else None,
            "mean_position_error": float(np.mean(self.position_errors)) if self.position_errors else None,
            "max_position_error": float(np.max(self.position_errors)) if self.position_errors else None,
            "mean_rotation_error": float(np.mean(self.rotation_errors)) if self.rotation_errors else None,
            "max_rotation_error": float(np.max(self.rotation_errors)) if self.rotation_errors else None,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark an IKFast wrapper against pytracik using headless MuJoCo poses.")
    parser.add_argument("--config", default="ur5_helper_sample", help=f"Built-in config name or path to a JSON config. Built-ins: {', '.join(list_builtin_configs())}")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seed-mode", choices=("truth", "zero"), default="truth")
    parser.add_argument("--skip-pytracik", action="store_true")
    parser.add_argument("--output", type=Path, default=ALGORITHM_ROOT / "outputs" / "benchmark_latest.json")
    args = parser.parse_args()

    config = get_builtin_config(args.config) if args.config in list_builtin_configs() else load_config_from_json(args.config)
    benchmark = run_benchmark(config, samples=args.samples, seed=args.seed, seed_mode=args.seed_mode, compare_pytracik=not args.skip_pytracik)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(benchmark, indent=2))
    print(json.dumps(benchmark, indent=2))


def run_benchmark(
    config,
    *,
    samples: int,
    seed: int,
    seed_mode: str,
    compare_pytracik: bool,
) -> dict[str, object]:
    solver = IkFastSolver(config)
    mujoco_runner = MujocoChain.from_solver(solver)
    rng = np.random.default_rng(seed)

    ikfast_raw_summary = BenchmarkSummary("ikfast_raw", samples, 0, [], [], [])
    ikfast_bounded_summary = BenchmarkSummary("ikfast_bounded", samples, 0, [], [], [])
    pytracik_summary = BenchmarkSummary("pytracik", samples, 0, [], [], [])
    pytracik_solver = None
    if compare_pytracik:
        from trac_ik import TracIK

        pytracik_solver = TracIK(
            base_link_name=solver.config.base_link,
            tip_link_name=solver.config.tip_link,
            urdf_path=str(mujoco_runner.sanitized_urdf_path),
        )

    for _ in range(samples):
        target_joint_values = sample_joint_values(rng, solver.lower_bounds, solver.upper_bounds, solver.continuous_mask)
        target_position, target_rotation = mujoco_runner.forward_kinematics(target_joint_values)
        seed_values = target_joint_values if seed_mode == "truth" else np.zeros_like(target_joint_values)

        start = time.perf_counter_ns()
        ikfast_raw_joint_values = solver.ik_values(
            target_position,
            target_rotation,
            seed_joint_values=seed_values,
        )
        ikfast_raw_summary.solve_times_ms.append((time.perf_counter_ns() - start) / 1e6)
        if ikfast_raw_joint_values is not None:
            actual_position, actual_rotation = mujoco_runner.forward_kinematics(ikfast_raw_joint_values)
            pos_err, rot_err = pose_error(target_position, target_rotation, actual_position, actual_rotation)
            ikfast_raw_summary.success_count += 1
            ikfast_raw_summary.position_errors.append(pos_err)
            ikfast_raw_summary.rotation_errors.append(rot_err)

        start = time.perf_counter_ns()
        ikfast_bounded_result = solver.ik(
            target_position,
            target_rotation,
            seed_joint_values=seed_values,
        )
        ikfast_bounded_summary.solve_times_ms.append((time.perf_counter_ns() - start) / 1e6)
        if ikfast_bounded_result.success and ikfast_bounded_result.joint_values is not None:
            actual_position, actual_rotation = mujoco_runner.forward_kinematics(ikfast_bounded_result.joint_values)
            pos_err, rot_err = pose_error(target_position, target_rotation, actual_position, actual_rotation)
            ikfast_bounded_summary.success_count += 1
            ikfast_bounded_summary.position_errors.append(pos_err)
            ikfast_bounded_summary.rotation_errors.append(rot_err)

        if pytracik_solver is not None:
            start = time.perf_counter_ns()
            pytracik_joint_values = pytracik_solver.ik(
                target_position,
                target_rotation,
                seed_jnt_values=seed_values,
            )
            pytracik_summary.solve_times_ms.append((time.perf_counter_ns() - start) / 1e6)
            if pytracik_joint_values is not None:
                actual_position, actual_rotation = mujoco_runner.forward_kinematics(np.asarray(pytracik_joint_values, dtype=np.float64))
                pos_err, rot_err = pose_error(target_position, target_rotation, actual_position, actual_rotation)
                pytracik_summary.success_count += 1
                pytracik_summary.position_errors.append(pos_err)
                pytracik_summary.rotation_errors.append(rot_err)

    results: dict[str, object] = {
        "config": {
            "name": solver.config.name,
            "urdf_path": str(solver.config.urdf_path),
            "base_link": solver.config.base_link,
            "tip_link": solver.config.tip_link,
        },
        "samples": samples,
        "seed": seed,
        "seed_mode": seed_mode,
        "ikfast_raw": ikfast_raw_summary.to_dict(),
        "ikfast_bounded": ikfast_bounded_summary.to_dict(),
    }
    if pytracik_solver is not None:
        results["pytracik"] = pytracik_summary.to_dict()
    return results


@dataclass(slots=True)
class MujocoChain:
    sanitized_urdf_path: Path
    model: mujoco.MjModel
    data: mujoco.MjData
    joint_qpos_indices: tuple[int, ...]
    tip_body_id: int
    tip_offset_transform: np.ndarray
    chain_body_ids: tuple[int, ...]

    @classmethod
    def from_solver(cls, solver: IkFastSolver) -> "MujocoChain":
        sanitized_urdf_path = ALGORITHM_ROOT / "outputs" / "mujoco" / f"{solver.config.name}_kinematics.urdf"
        sanitize_urdf_for_mujoco(solver.config.urdf_path, sanitized_urdf_path)
        model = mujoco.MjModel.from_xml_path(str(sanitized_urdf_path))
        data = mujoco.MjData(model)
        joint_qpos_indices = []
        for joint_name in solver.active_joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"MuJoCo URDF import did not produce joint {joint_name!r}.")
            joint_qpos_indices.append(int(model.jnt_qposadr[joint_id]))
        tip_body_id, tip_offset_transform = _resolve_tip_body(model, solver)
        chain_body_ids = _resolve_chain_body_ids(model, solver)
        return cls(
            sanitized_urdf_path=sanitized_urdf_path,
            model=model,
            data=data,
            joint_qpos_indices=tuple(joint_qpos_indices),
            tip_body_id=int(tip_body_id),
            tip_offset_transform=tip_offset_transform,
            chain_body_ids=chain_body_ids,
        )

    def forward_kinematics(self, joint_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[:] = 0.0
        for qpos_index, joint_value in zip(self.joint_qpos_indices, joint_values):
            self.data.qpos[qpos_index] = float(joint_value)
        mujoco.mj_forward(self.model, self.data)
        position = np.asarray(self.data.xpos[self.tip_body_id], dtype=np.float64).copy()
        rotation = np.asarray(self.data.xmat[self.tip_body_id], dtype=np.float64).reshape(3, 3).copy()
        if not np.allclose(self.tip_offset_transform, np.eye(4)):
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = rotation
            pose[:3, 3] = position
            pose = pose @ self.tip_offset_transform
            position = pose[:3, 3].copy()
            rotation = pose[:3, :3].copy()
        return position, rotation

    def chain_points(self, joint_values: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = 0.0
        for qpos_index, joint_value in zip(self.joint_qpos_indices, joint_values):
            self.data.qpos[qpos_index] = float(joint_value)
        mujoco.mj_forward(self.model, self.data)
        points = [np.zeros(3, dtype=np.float64)]
        for body_id in self.chain_body_ids:
            points.append(np.asarray(self.data.xpos[body_id], dtype=np.float64).copy())
        tip_position, _ = self.forward_kinematics(joint_values)
        points.append(tip_position)
        deduped = [points[0]]
        for point in points[1:]:
            if np.linalg.norm(point - deduped[-1]) > 1e-9:
                deduped.append(point)
        return np.asarray(deduped, dtype=np.float64)


def sample_joint_values(
    rng: np.random.Generator,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    continuous_mask: np.ndarray,
) -> np.ndarray:
    values = rng.uniform(lower_bounds, upper_bounds)
    values[continuous_mask] = rng.uniform(lower_bounds[continuous_mask], upper_bounds[continuous_mask])
    return values.astype(np.float64)


def _resolve_tip_body(model: mujoco.MjModel, solver: IkFastSolver) -> tuple[int, np.ndarray]:
    tip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, solver.config.tip_link)
    if tip_body_id >= 0:
        return int(tip_body_id), np.eye(4, dtype=np.float64)

    child_to_joint = {joint.child: joint for joint in solver.chain.joints}
    current_link = solver.config.tip_link
    tip_offset = np.eye(4, dtype=np.float64)
    while current_link != solver.config.base_link:
        joint = child_to_joint[current_link]
        if joint.joint_type != "fixed":
            raise ValueError(
                f"MuJoCo URDF import did not retain tip link {solver.config.tip_link!r}, "
                f"and the missing segment includes active joint {joint.name!r}."
            )
        tip_offset = _origin_transform(joint.origin_xyz, joint.origin_rpy) @ tip_offset
        current_link = joint.parent
        tip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, current_link)
        if tip_body_id >= 0:
            return int(tip_body_id), tip_offset
    raise ValueError(f"MuJoCo URDF import did not produce any body on the chain to {solver.config.tip_link!r}.")


def _resolve_chain_body_ids(model: mujoco.MjModel, solver: IkFastSolver) -> tuple[int, ...]:
    body_ids: list[int] = []
    for joint in solver.chain.joints:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, joint.child)
        if body_id >= 0 and (not body_ids or body_ids[-1] != int(body_id)):
            body_ids.append(int(body_id))
    return tuple(body_ids)


def _origin_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rpy_to_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


if __name__ == "__main__":
    main()
