from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import RobotArmConfig
from .ikfast_library import IkFastLibrary
from .urdf_utils import JointLimit, UrdfChain, resolve_chain


@dataclass(slots=True)
class SolveResult:
    status: str
    joint_values: np.ndarray | None
    position_error: float | None
    rotation_error: float | None
    position_bound: float
    rotation_bound: float
    candidate_count: int

    @property
    def success(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "joint_values": None if self.joint_values is None else self.joint_values.tolist(),
            "position_error": self.position_error,
            "rotation_error": self.rotation_error,
            "position_bound": self.position_bound,
            "rotation_bound": self.rotation_bound,
            "candidate_count": self.candidate_count,
        }


class IkFastSolver:
    def __init__(self, config: RobotArmConfig):
        self.config = config.resolved()
        self.chain = resolve_chain(
            self.config.urdf_path,
            self.config.base_link,
            self.config.tip_link,
            continuous_joint_limits=self.config.continuous_joint_limits,
        )
        self.active_joint_names = self.chain.active_joint_names
        self.active_joint_limits = self.chain.active_joint_limits
        self.continuous_mask = np.asarray([limit.continuous for limit in self.active_joint_limits], dtype=bool)
        self.lower_bounds = np.asarray([limit.lower for limit in self.active_joint_limits], dtype=np.float64)
        self.upper_bounds = np.asarray([limit.upper for limit in self.active_joint_limits], dtype=np.float64)
        self.library = IkFastLibrary.from_config(self.config)
        if self.library.num_joints != len(self.active_joint_names):
            raise ValueError(
                f"IKFast joint count ({self.library.num_joints}) does not match URDF active joint count "
                f"({len(self.active_joint_names)})."
            )
        self.urdf_base_to_ikfast = np.asarray(self.config.urdf_base_to_ikfast, dtype=np.float64)
        self.urdf_tip_to_ikfast = np.asarray(self.config.urdf_tip_to_ikfast, dtype=np.float64)
        self.ikfast_to_urdf_base = np.linalg.inv(self.urdf_base_to_ikfast)
        self.ikfast_to_urdf_tip = np.linalg.inv(self.urdf_tip_to_ikfast)
        self.joint_name_to_index = {name: idx for idx, name in enumerate(self.active_joint_names)}
        self.free_parameter_indices = self._resolve_free_parameter_indices()
        self.reference_configuration = 0.5 * (self.lower_bounds + self.upper_bounds)
        self.reference_configuration[self.continuous_mask] = 0.0

    def fk(self, joint_values: np.ndarray) -> np.ndarray:
        translation, rotation = self.fk_pose(joint_values)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation
        pose[:3, 3] = translation
        return pose

    def fk_pose(self, joint_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw_translation, raw_rotation = self.fk_pose_raw(joint_values)
        return self._ikfast_pose_to_urdf(raw_translation, raw_rotation)

    def fk_pose_raw(self, joint_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(joint_values, dtype=np.float64)
        if values.shape != (self.library.num_joints,):
            raise ValueError(f"Expected fk input shape {(self.library.num_joints,)}, received {values.shape}.")
        return self.library.compute_fk(values)

    def ik_values(
        self,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        *,
        seed_joint_values: np.ndarray | None = None,
        free_joint_values: np.ndarray | None = None,
        max_solutions: int | None = None,
    ) -> np.ndarray | None:
        target_position_arr = np.asarray(target_position, dtype=np.float64)
        target_rotation_arr = np.asarray(target_rotation, dtype=np.float64)
        ikfast_target_position, ikfast_target_rotation = self._urdf_pose_to_ikfast(
            target_position_arr,
            target_rotation_arr,
        )
        solution, _ = self._select_joint_solution(
            target_position=ikfast_target_position,
            target_rotation=ikfast_target_rotation,
            seed_joint_values=None if seed_joint_values is None else np.asarray(seed_joint_values, dtype=np.float64),
            free_joint_values=None if free_joint_values is None else np.asarray(free_joint_values, dtype=np.float64),
            max_solutions=max_solutions or self.config.max_solutions,
        )
        return solution

    def ik(
        self,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        *,
        seed_joint_values: np.ndarray | None = None,
        free_joint_values: np.ndarray | None = None,
        position_tolerance: float | None = None,
        rotation_tolerance: float | None = None,
        max_solutions: int | None = None,
    ) -> SolveResult:
        pos_tol = float(self.config.position_tolerance if position_tolerance is None else position_tolerance)
        rot_tol = float(self.config.rotation_tolerance if rotation_tolerance is None else rotation_tolerance)
        target_position_arr = np.asarray(target_position, dtype=np.float64)
        target_rotation_arr = np.asarray(target_rotation, dtype=np.float64)
        ikfast_target_position, ikfast_target_rotation = self._urdf_pose_to_ikfast(
            target_position_arr,
            target_rotation_arr,
        )
        solution, candidate_count = self._select_joint_solution(
            target_position=ikfast_target_position,
            target_rotation=ikfast_target_rotation,
            seed_joint_values=None if seed_joint_values is None else np.asarray(seed_joint_values, dtype=np.float64),
            free_joint_values=None if free_joint_values is None else np.asarray(free_joint_values, dtype=np.float64),
            max_solutions=max_solutions or self.config.max_solutions,
        )
        if solution is None:
            return SolveResult(
                status="no",
                joint_values=None,
                position_error=None,
                rotation_error=None,
                position_bound=pos_tol,
                rotation_bound=rot_tol,
                candidate_count=candidate_count,
            )
        fk_position, fk_rotation = self.fk_pose(solution)
        position_error, rotation_error = pose_error(
            target_position_arr,
            target_rotation_arr,
            fk_position,
            fk_rotation,
        )
        if position_error > pos_tol or rotation_error > rot_tol:
            return SolveResult(
                status="no",
                joint_values=None,
                position_error=None,
                rotation_error=None,
                position_bound=pos_tol,
                rotation_bound=rot_tol,
                candidate_count=candidate_count,
            )
        return SolveResult(
            status="ok",
            joint_values=solution,
            position_error=position_error,
            rotation_error=rotation_error,
            position_bound=pos_tol,
            rotation_bound=rot_tol,
            candidate_count=candidate_count,
        )

    def _resolve_free_parameter_indices(self) -> tuple[int, ...]:
        if self.library.num_free_parameters == 0:
            return ()
        if self.library.free_parameter_indices:
            indices = tuple(int(index) for index in self.library.free_parameter_indices)
            if self.config.free_joint_names:
                expected_names = tuple(self.active_joint_names[index] for index in indices)
                if tuple(self.config.free_joint_names) != expected_names:
                    raise ValueError(
                        f"Configured free joints {self.config.free_joint_names!r} do not match the IKFast library "
                        f"free parameters {expected_names!r}."
                    )
            return indices
        if not self.config.free_joint_names:
            raise ValueError(
                "IKFast library reports free parameters but did not expose free indices. "
                "Provide free_joint_names in the config."
            )
        return tuple(self.joint_name_to_index[name] for name in self.config.free_joint_names)

    def _select_joint_solution(
        self,
        *,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        seed_joint_values: np.ndarray | None,
        free_joint_values: np.ndarray | None,
        max_solutions: int,
    ) -> tuple[np.ndarray | None, int]:
        if target_position.shape != (3,):
            raise ValueError(f"Expected target position shape (3,), received {target_position.shape}.")
        if target_rotation.shape != (3, 3):
            raise ValueError(f"Expected target rotation shape (3, 3), received {target_rotation.shape}.")
        if seed_joint_values is not None and seed_joint_values.shape != (self.library.num_joints,):
            raise ValueError(
                f"Expected seed shape {(self.library.num_joints,)}, received {seed_joint_values.shape}."
            )
        free_values = self._resolve_free_values(seed_joint_values, free_joint_values)
        candidates = self.library.compute_ik(
            target_position,
            target_rotation,
            free_parameters=free_values,
            max_solutions=max_solutions,
        )
        if len(candidates) == 0:
            return None, 0
        reference = seed_joint_values if seed_joint_values is not None else self.reference_configuration
        valid_candidates: list[np.ndarray] = []
        for candidate in candidates:
            adjusted = self._adjust_continuous_candidate(candidate, reference)
            adjusted = self._clip_to_close_limits(adjusted)
            if self._within_limits(adjusted):
                valid_candidates.append(adjusted)
        if not valid_candidates:
            return None, int(len(candidates))
        distances = [joint_distance(candidate, reference, self.continuous_mask) for candidate in valid_candidates]
        best_index = int(np.argmin(np.asarray(distances, dtype=np.float64)))
        return valid_candidates[best_index], int(len(candidates))

    def _urdf_pose_to_ikfast(self, position: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation
        pose[:3, 3] = position
        transformed = self.urdf_base_to_ikfast @ pose @ self.urdf_tip_to_ikfast
        return transformed[:3, 3].copy(), transformed[:3, :3].copy()

    def _ikfast_pose_to_urdf(self, position: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation
        pose[:3, 3] = position
        transformed = self.ikfast_to_urdf_base @ pose @ self.ikfast_to_urdf_tip
        return transformed[:3, 3].copy(), transformed[:3, :3].copy()

    def _resolve_free_values(
        self,
        seed_joint_values: np.ndarray | None,
        free_joint_values: np.ndarray | None,
    ) -> np.ndarray:
        if self.library.num_free_parameters == 0:
            return np.empty(0, dtype=np.float64)
        if free_joint_values is not None:
            values = np.asarray(free_joint_values, dtype=np.float64)
            if values.shape != (self.library.num_free_parameters,):
                raise ValueError(
                    f"Expected free joint values shape {(self.library.num_free_parameters,)}, received {values.shape}."
                )
            return values
        reference = seed_joint_values if seed_joint_values is not None else self.reference_configuration
        return np.asarray([reference[index] for index in self.free_parameter_indices], dtype=np.float64)

    def _adjust_continuous_candidate(self, candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
        adjusted = np.asarray(candidate, dtype=np.float64).copy()
        two_pi = 2.0 * math.pi
        for index, is_continuous in enumerate(self.continuous_mask):
            if not is_continuous:
                continue
            lower = self.lower_bounds[index]
            upper = self.upper_bounds[index]
            base_value = adjusted[index]
            min_k = math.floor((lower - base_value) / two_pi) - 1
            max_k = math.ceil((upper - base_value) / two_pi) + 1
            feasible_values = [
                base_value + (two_pi * k)
                for k in range(min_k, max_k + 1)
                if (lower - 1e-8) <= base_value + (two_pi * k) <= (upper + 1e-8)
            ]
            if feasible_values:
                adjusted[index] = min(feasible_values, key=lambda value: abs(value - reference[index]))
        return adjusted

    def _clip_to_close_limits(self, candidate: np.ndarray) -> np.ndarray:
        clipped = candidate.copy()
        for index, limit in enumerate(self.active_joint_limits):
            if clipped[index] < limit.lower and clipped[index] >= (limit.lower - 1e-8):
                clipped[index] = limit.lower
            if clipped[index] > limit.upper and clipped[index] <= (limit.upper + 1e-8):
                clipped[index] = limit.upper
        return clipped

    def _within_limits(self, candidate: np.ndarray) -> bool:
        return bool(np.all(candidate >= (self.lower_bounds - 1e-8)) and np.all(candidate <= (self.upper_bounds + 1e-8)))


def joint_distance(candidate: np.ndarray, reference: np.ndarray, continuous_mask: np.ndarray) -> float:
    diffs = np.asarray(candidate - reference, dtype=np.float64)
    for index, is_continuous in enumerate(continuous_mask):
        if is_continuous:
            diffs[index] = math.atan2(math.sin(diffs[index]), math.cos(diffs[index]))
    return float(np.linalg.norm(diffs))


def pose_error(
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    actual_position: np.ndarray,
    actual_rotation: np.ndarray,
) -> tuple[float, float]:
    position_error = float(np.linalg.norm(np.asarray(target_position) - np.asarray(actual_position)))
    delta = np.asarray(target_rotation, dtype=np.float64).T @ np.asarray(actual_rotation, dtype=np.float64)
    trace = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    rotation_error = float(math.acos(trace))
    return position_error, rotation_error
