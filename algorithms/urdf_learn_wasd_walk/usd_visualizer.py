from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .asset_setup import prepare_landau_inputs


@dataclass(frozen=True)
class SkeletonRecord:
    index: int
    name: str
    parent_index: int
    axis: np.ndarray
    local_matrix: np.ndarray


def _orthonormalize(rotation: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(rotation)
    fixed = u @ vh
    if np.linalg.det(fixed) < 0:
        u[:, -1] *= -1.0
        fixed = u @ vh
    return fixed


def axis_angle_matrix(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    axis_arr = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis_arr))
    if norm < 1.0e-8 or abs(angle_rad) < 1.0e-10:
        return np.eye(3, dtype=float)
    x, y, z = axis_arr / norm
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def quat_wxyz_from_matrix(matrix: np.ndarray) -> np.ndarray:
    trace = float(matrix[0, 0] + matrix[1, 1] + matrix[2, 2])
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (matrix[2, 1] - matrix[1, 2]) * s
        y = (matrix[0, 2] - matrix[2, 0]) * s
        z = (matrix[1, 0] - matrix[0, 1]) * s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=float)
    return quat / np.linalg.norm(quat)


def quat_wxyz_to_matrix(quat_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(value) for value in quat_wxyz)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1.0e-12:
        return np.eye(3, dtype=float)
    w /= norm
    x /= norm
    y /= norm
    z /= norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rigid_transform(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation, dtype=float)
    transform[:3, 3] = np.asarray(translation, dtype=float)
    return transform


def inverse_rigid_transform(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=float)
    rotation = np.asarray(transform[:3, :3], dtype=float)
    translation = np.asarray(transform[:3, 3], dtype=float)
    rotation_inv = rotation.T
    result[:3, :3] = rotation_inv
    result[:3, 3] = -(rotation_inv @ translation)
    return result


def load_skeleton_records(skeleton_json_path: Path) -> list[SkeletonRecord]:
    payload = json.loads(Path(skeleton_json_path).read_text())
    raw_records = payload.get("records", ())
    if not raw_records:
        raise RuntimeError(f"Skeleton JSON has no records: {skeleton_json_path}")
    records = []
    for raw in raw_records:
        records.append(
            SkeletonRecord(
                index=int(raw["index"]),
                name=str(raw["name"]),
                parent_index=int(raw["parent_index"]),
                axis=np.asarray(raw.get("axis", (0.0, 0.0, 1.0)), dtype=float),
                local_matrix=np.asarray(raw["local_matrix"], dtype=float),
            )
        )
    records.sort(key=lambda record: record.index)
    return records


def apply_joint_positions_to_local_matrices(
    records: Sequence[SkeletonRecord], pose_by_name: dict[str, float]
) -> list[np.ndarray]:
    posed = [record.local_matrix.copy() for record in records]
    for record in records:
        angle = pose_by_name.get(record.name)
        if angle is None:
            continue
        updated = posed[record.index].copy()
        updated[:3, :3] = _orthonormalize(updated[:3, :3] @ axis_angle_matrix(record.axis, float(angle)))
        posed[record.index] = updated
    return posed


def _find_first_skeleton(stage, root_prim_path: str):
    from pxr import Usd, UsdSkel

    root_prim = stage.GetPrimAtPath(root_prim_path)
    for prim in Usd.PrimRange(root_prim):
        candidate = UsdSkel.Skeleton(prim)
        if candidate and candidate.GetPrim().IsValid():
            return candidate
    return None


def _trs_arrays_from_local_matrices(matrices: Sequence[np.ndarray]):
    from pxr import Gf

    translations = []
    rotations = []
    scales = []
    for matrix in matrices:
        rotation = np.asarray(matrix[:3, :3], dtype=float)
        scale = np.linalg.norm(rotation, axis=0)
        safe_scale = np.where(scale < 1.0e-8, 1.0, scale)
        pure_rotation = rotation / safe_scale
        quat_wxyz = quat_wxyz_from_matrix(pure_rotation)
        translations.append(Gf.Vec3f(float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3])))
        rotations.append(
            Gf.Quatf(
                float(quat_wxyz[0]),
                Gf.Vec3f(float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])),
            )
        )
        scales.append(Gf.Vec3h(float(scale[0]), float(scale[1]), float(scale[2])))
    return translations, rotations, scales


def _apply_pose_to_usd_skeleton(stage, skeleton, local_matrices: Sequence[np.ndarray]) -> str:
    from pxr import UsdSkel

    binding_prim = skeleton.GetPrim()
    current = skeleton.GetPrim()
    while current and current.IsValid():
        if current.IsA(UsdSkel.Root):
            binding_prim = current
            break
        current = current.GetParent()

    anim = UsdSkel.Animation.Define(stage, binding_prim.GetPath().AppendChild("ParallelPoseAnim"))
    anim.GetJointsAttr().Set(skeleton.GetJointsAttr().Get())
    translations, rotations, scales = _trs_arrays_from_local_matrices(local_matrices)
    anim.GetTranslationsAttr().Set(translations)
    anim.GetRotationsAttr().Set(rotations)
    anim.GetScalesAttr().Set(scales)
    binding = UsdSkel.BindingAPI.Apply(binding_prim)
    if not binding.GetSkeletonRel().GetTargets():
        binding.CreateSkeletonRel().SetTargets([skeleton.GetPath()])
    binding.CreateAnimationSourceRel().SetTargets([anim.GetPrim().GetPath()])
    if binding_prim != skeleton.GetPrim():
        UsdSkel.BindingAPI.Apply(skeleton.GetPrim()).CreateAnimationSourceRel().SetTargets([anim.GetPrim().GetPath()])
    return str(anim.GetPrim().GetPath())


def _set_xform_matrix(stage, prim_path: str, matrix: np.ndarray) -> None:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    transform_op = xformable.AddTransformOp(opSuffix="livePose", precision=UsdGeom.XformOp.PrecisionDouble)
    transform_op.Set(Gf.Matrix4d(matrix.T.tolist()))


def _set_visibility(stage, prim_path: str, visible: bool) -> None:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return
    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()


class LandauUsdVisualizer:
    """Mirror the live Landau URDF articulation onto the copied colored USD asset."""

    def __init__(
        self,
        stage,
        env_prim_path: str,
        *,
        usd_path: Path | None = None,
        skeleton_json_path: Path | None = None,
    ) -> None:
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import UsdGeom

        prepared = prepare_landau_inputs(refresh=False)
        self.stage = stage
        self.env_prim_path = env_prim_path
        self.urdf_prim_path = f"{env_prim_path}/Robot"
        self.visual_root_path = f"{env_prim_path}/LandauVisual"
        self.visual_asset_path = f"{self.visual_root_path}/Asset"
        self.usd_path = Path(usd_path or prepared.usd_path).resolve()
        self.skeleton_json_path = Path(skeleton_json_path or prepared.skeleton_json_path).resolve()
        self.records = load_skeleton_records(self.skeleton_json_path)

        root_records = [record for record in self.records if record.parent_index < 0]
        if len(root_records) != 1:
            raise RuntimeError(
                f"Expected exactly one root skeleton record in {self.skeleton_json_path}, found {len(root_records)}."
            )
        self._root_rest_inverse = inverse_rigid_transform(root_records[0].local_matrix)

        UsdGeom.Xform.Define(stage, self.visual_root_path)
        add_reference_to_stage(usd_path=str(self.usd_path), prim_path=self.visual_asset_path)
        self.skeleton = _find_first_skeleton(stage, self.visual_asset_path)
        if self.skeleton is None:
            raise RuntimeError(f"Referenced USD did not expose a skeleton under {self.visual_asset_path}")
        self._skeleton_root_name = root_records[0].name

    def set_urdf_visibility(self, visible: bool) -> None:
        _set_visibility(self.stage, self.urdf_prim_path, visible)

    def sync_from_robot(self, robot, env_index: int = 0) -> None:
        joint_pos = robot.data.joint_pos[env_index].detach().cpu().numpy()

        pose_by_name = {joint_name: float(angle) for joint_name, angle in zip(robot.joint_names, joint_pos, strict=False)}
        local_matrices = apply_joint_positions_to_local_matrices(self.records, pose_by_name)
        _apply_pose_to_usd_skeleton(self.stage, self.skeleton, local_matrices)

        if self._skeleton_root_name in robot.body_names:
            body_ids, _ = robot.find_bodies([self._skeleton_root_name], preserve_order=True)
            body_id = body_ids[0]
            root_pos = robot.data.body_pos_w[env_index, body_id].detach().cpu().numpy()
            root_quat = robot.data.body_quat_w[env_index, body_id].detach().cpu().numpy()
        else:
            root_pos = robot.data.root_pos_w[env_index].detach().cpu().numpy()
            root_quat = robot.data.root_quat_w[env_index].detach().cpu().numpy()

        robot_root_world = rigid_transform(quat_wxyz_to_matrix(root_quat), root_pos)
        visual_root_world = robot_root_world @ self._root_rest_inverse
        _set_xform_matrix(self.stage, self.visual_root_path, visual_root_world)
