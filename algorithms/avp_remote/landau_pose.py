from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from urdf_kinematics import (
    joint_local_transform,
    load_urdf_joint_specs,
    world_map_from_joint_specs,
)


@dataclass(frozen=True)
class SkeletonRecord:
    index: int
    name: str
    parent_index: int
    axis: np.ndarray
    limits: tuple[float, float]
    local_matrix: np.ndarray


@dataclass(frozen=True)
class UrdfVisualMeshRecord:
    mesh_name: str
    link_name: str
    mesh_path: Path
    origin_matrix: np.ndarray


def _orthonormalize(rotation: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(rotation)
    fixed = u @ vh
    if np.linalg.det(fixed) < 0.0:
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


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return axis_angle_matrix((1.0, 0.0, 0.0), roll) @ axis_angle_matrix((0.0, 1.0, 0.0), pitch) @ axis_angle_matrix((0.0, 0.0, 1.0), yaw)


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


def load_urdf_visual_mesh_records(urdf_path: Path) -> dict[str, tuple[UrdfVisualMeshRecord, ...]]:
    root = ET.parse(urdf_path).getroot()
    urdf_dir = Path(urdf_path).resolve().parent
    records: dict[str, list[UrdfVisualMeshRecord]] = {}
    for link_el in root.findall("link"):
        link_name = link_el.attrib.get("name")
        if not link_name:
            continue
        for visual_index, visual_el in enumerate(link_el.findall("visual")):
            mesh_el = visual_el.find("./geometry/mesh")
            if mesh_el is None:
                continue
            filename = mesh_el.attrib.get("filename")
            if not filename:
                continue
            origin_el = visual_el.find("origin")
            xyz = (0.0, 0.0, 0.0)
            rpy = (0.0, 0.0, 0.0)
            if origin_el is not None:
                xyz_text = origin_el.attrib.get("xyz")
                rpy_text = origin_el.attrib.get("rpy")
                if xyz_text:
                    xyz = tuple(float(value) for value in xyz_text.split())
                if rpy_text:
                    rpy = tuple(float(value) for value in rpy_text.split())
            records.setdefault(link_name, []).append(
                UrdfVisualMeshRecord(
                    mesh_name=f"visual_{visual_index:02d}",
                    link_name=link_name,
                    mesh_path=(urdf_dir / filename).resolve(),
                    origin_matrix=rigid_transform(rpy_matrix(*rpy), xyz),
                )
            )
    return {
        link_name: tuple(link_records)
        for link_name, link_records in records.items()
    }


def load_stl_mesh_arrays(mesh_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = Path(mesh_path).read_bytes()
    if len(data) >= 84:
        triangle_count = int.from_bytes(data[80:84], byteorder="little", signed=False)
        expected_size = 84 + triangle_count * 50
        if expected_size == len(data):
            triangles = np.frombuffer(
                data,
                dtype=np.dtype(
                    [
                        ("normal", "<f4", (3,)),
                        ("v0", "<f4", (3,)),
                        ("v1", "<f4", (3,)),
                        ("v2", "<f4", (3,)),
                        ("attr", "<u2"),
                    ]
                ),
                count=triangle_count,
                offset=84,
            )
            vertices = np.stack((triangles["v0"], triangles["v1"], triangles["v2"]), axis=1).reshape(-1, 3).astype(float)
            face_counts = np.full(triangle_count, 3, dtype=np.int32)
            face_indices = np.arange(triangle_count * 3, dtype=np.int32)
            return vertices, face_counts, face_indices

    vertices_list: list[list[float]] = []
    for raw_line in data.decode("utf-8", errors="ignore").splitlines():
        stripped = raw_line.strip()
        if not stripped.startswith("vertex "):
            continue
        _, x_text, y_text, z_text = stripped.split()
        vertices_list.append([float(x_text), float(y_text), float(z_text)])
    if not vertices_list or len(vertices_list) % 3 != 0:
        raise RuntimeError(f"Unable to parse STL triangles from {mesh_path}")
    vertices = np.asarray(vertices_list, dtype=float)
    triangle_count = vertices.shape[0] // 3
    face_counts = np.full(triangle_count, 3, dtype=np.int32)
    face_indices = np.arange(triangle_count * 3, dtype=np.int32)
    return vertices, face_counts, face_indices


def _author_raw_mesh(stage, prim_path: str, mesh_path: Path) -> None:
    from pxr import Vt, UsdGeom

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    points, face_counts, face_indices = load_stl_mesh_arrays(mesh_path)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(face_counts))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(face_indices))
    mesh.GetSubdivisionSchemeAttr().Set("none")
    mesh.CreateDisplayColorAttr([(0.65, 0.65, 0.68)])


def load_skeleton_records(skeleton_json_path: Path) -> list[SkeletonRecord]:
    payload = json.loads(Path(skeleton_json_path).read_text())
    raw_records = payload.get("records", ())
    if not raw_records:
        raise RuntimeError(f"Skeleton JSON has no records: {skeleton_json_path}")
    records = []
    for raw in raw_records:
        limit_values = raw.get("limits", (-math.pi, math.pi))
        records.append(
            SkeletonRecord(
                index=int(raw["index"]),
                name=str(raw["name"]),
                parent_index=int(raw["parent_index"]),
                axis=np.asarray(raw.get("axis", (0.0, 0.0, 1.0)), dtype=float),
                limits=(float(limit_values[0]), float(limit_values[1])),
                local_matrix=np.asarray(raw["local_matrix"], dtype=float),
            )
        )
    records.sort(key=lambda record: record.index)
    return records


def apply_joint_positions_to_local_matrices(
    records: Sequence[SkeletonRecord],
    pose_by_name: dict[str, float],
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


def world_matrices_from_local(records: Sequence[SkeletonRecord], local_matrices: Sequence[np.ndarray]) -> list[np.ndarray]:
    world = [np.eye(4, dtype=float) for _ in records]
    for record in records:
        local = np.asarray(local_matrices[record.index], dtype=float)
        if record.parent_index < 0:
            world[record.index] = local
        else:
            world[record.index] = world[record.parent_index] @ local
    return world


def world_map_from_pose(records: Sequence[SkeletonRecord], pose_by_name: dict[str, float]) -> dict[str, np.ndarray]:
    local_matrices = apply_joint_positions_to_local_matrices(records, pose_by_name)
    world_matrices = world_matrices_from_local(records, local_matrices)
    return {
        record.name: world_matrices[record.index]
        for record in records
    }


def local_matrices_from_world_map(
    records: Sequence[SkeletonRecord],
    world_by_name: dict[str, np.ndarray],
) -> list[np.ndarray]:
    local_matrices: list[np.ndarray] = []
    for record in records:
        world_transform = world_by_name.get(record.name)
        if world_transform is None:
            local_matrices.append(record.local_matrix.copy())
            continue
        if record.parent_index < 0:
            local_matrices.append(np.asarray(world_transform, dtype=float))
            continue
        parent_name = records[record.parent_index].name
        parent_world = world_by_name.get(parent_name)
        if parent_world is None:
            local_matrices.append(record.local_matrix.copy())
            continue
        local_matrices.append(inverse_rigid_transform(parent_world) @ world_transform)
    return local_matrices


def root_height_offset(records: Sequence[SkeletonRecord], clearance: float = 0.02) -> float:
    min_z = min(float(record.local_matrix[2, 3]) for record in records if record.parent_index < 0)
    if len(records) > 1:
        world_matrices = world_matrices_from_local(records, [record.local_matrix for record in records])
        min_z = min(min_z, min(float(matrix[2, 3]) for matrix in world_matrices))
    return -min_z + clearance


def _find_first_skeleton(stage, root_prim_path: str):
    from pxr import Usd, UsdSkel

    root_prim = stage.GetPrimAtPath(root_prim_path)
    for prim in Usd.PrimRange(root_prim):
        candidate = UsdSkel.Skeleton(prim)
        if candidate and candidate.GetPrim().IsValid():
            return candidate
    return None


def _sanitize_dome_lights(stage, root_prim_path: str) -> int:
    from pxr import Gf, Sdf, Usd, UsdLux

    fixed_count = 0
    root_prim = stage.GetPrimAtPath(root_prim_path)
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdLux.DomeLight):
            continue
        dome = UsdLux.DomeLight(prim)
        texture_attr = dome.GetTextureFileAttr()
        if texture_attr:
            try:
                texture_attr.Set(Sdf.AssetPath(""))
            except Exception:
                texture_attr.Set("")
        dome.GetColorAttr().Set(Gf.Vec3f(0.08, 0.08, 0.08))
        if not dome.GetIntensityAttr().HasAuthoredValue():
            dome.GetIntensityAttr().Set(200.0)
        fixed_count += 1
    return fixed_count


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
    anim, translations_attr, rotations_attr, scales_attr = _bind_pose_animation(stage, skeleton)
    _set_animation_pose(translations_attr, rotations_attr, scales_attr, local_matrices)
    return str(anim.GetPrim().GetPath())


def _bind_pose_animation(stage, skeleton, anim_name: str = "AvpPoseAnim"):
    from pxr import UsdSkel

    binding_prim = skeleton.GetPrim()
    current = skeleton.GetPrim()
    while current and current.IsValid():
        if current.IsA(UsdSkel.Root):
            binding_prim = current
            break
        current = current.GetParent()

    anim = UsdSkel.Animation.Define(stage, binding_prim.GetPath().AppendChild(anim_name))
    anim.GetJointsAttr().Set(skeleton.GetJointsAttr().Get())
    binding = UsdSkel.BindingAPI.Apply(binding_prim)
    if not binding.GetSkeletonRel().GetTargets():
        binding.CreateSkeletonRel().SetTargets([skeleton.GetPath()])
    binding.CreateAnimationSourceRel().SetTargets([anim.GetPrim().GetPath()])
    if binding_prim != skeleton.GetPrim():
        UsdSkel.BindingAPI.Apply(skeleton.GetPrim()).CreateAnimationSourceRel().SetTargets([anim.GetPrim().GetPath()])
    return (
        anim,
        anim.GetTranslationsAttr(),
        anim.GetRotationsAttr(),
        anim.GetScalesAttr(),
    )


def _set_animation_pose(translations_attr, rotations_attr, scales_attr, local_matrices: Sequence[np.ndarray]) -> None:
    translations, rotations, scales = _trs_arrays_from_local_matrices(local_matrices)
    translations_attr.Set(translations)
    rotations_attr.Set(rotations)
    scales_attr.Set(scales)


def _set_xform_matrix(stage, prim_path: str, matrix: np.ndarray) -> None:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    xformable = UsdGeom.Xformable(prim)
    transform_op = None
    for existing in xformable.GetOrderedXformOps():
        if existing.GetOpType() == UsdGeom.XformOp.TypeTransform:
            transform_op = existing
            break
    if transform_op is None:
        transform_op = xformable.AddTransformOp(opSuffix="livePose", precision=UsdGeom.XformOp.PrecisionDouble)
    transform_op.Set(Gf.Matrix4d(matrix.T.tolist()))


def _set_root_translate(stage, prim_path: str, xyz: Sequence[float]) -> None:
    matrix = np.eye(4, dtype=float)
    matrix[:3, 3] = np.asarray(xyz, dtype=float)
    _set_xform_matrix(stage, prim_path, matrix)


def set_visibility(stage, prim_path: str, visible: bool) -> None:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return
    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()


class LandauUsdPoseDriver:
    def __init__(
        self,
        stage,
        *,
        urdf_path: Path | None = None,
        usd_path: Path,
        skeleton_json_path: Path,
        visual_root_path: str = "/World/LandauVisual",
        visual_asset_path: str = "/World/LandauVisual/Asset",
        root_offset_xyz: Sequence[float] = (0.0, 0.0, 0.0),
        anim_name: str = "AvpPoseAnim",
    ) -> None:
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import UsdGeom

        self.stage = stage
        self.visual_root_path = visual_root_path
        self.visual_asset_path = visual_asset_path
        self.urdf_path = Path(urdf_path).resolve() if urdf_path is not None else None
        self.usd_path = Path(usd_path).resolve()
        self.skeleton_json_path = Path(skeleton_json_path).resolve()
        self.records = load_skeleton_records(self.skeleton_json_path)
        self.urdf_joint_specs = load_urdf_joint_specs(self.urdf_path) if self.urdf_path is not None else None
        self.root_offset_xyz = np.asarray(root_offset_xyz, dtype=float)
        self.anim_name = anim_name

        root_records = [record for record in self.records if record.parent_index < 0]
        if len(root_records) != 1:
            raise RuntimeError(
                f"Expected exactly one root skeleton record in {self.skeleton_json_path}, found {len(root_records)}."
            )

        self.root_record = root_records[0]
        self.root_z_offset = root_height_offset(self.records)
        self.root_translate_xyz = self.root_offset_xyz + np.array((0.0, 0.0, self.root_z_offset), dtype=float)
        UsdGeom.Xform.Define(stage, self.visual_root_path)
        add_reference_to_stage(usd_path=str(self.usd_path), prim_path=self.visual_asset_path)
        self.skeleton = _find_first_skeleton(stage, self.visual_asset_path)
        if self.skeleton is None:
            raise RuntimeError(f"Referenced USD did not expose a skeleton under {self.visual_asset_path}")
        _sanitize_dome_lights(stage, self.visual_asset_path)
        self.anim, self.translations_attr, self.rotations_attr, self.scales_attr = _bind_pose_animation(
            stage,
            self.skeleton,
            anim_name=self.anim_name,
        )

        _set_root_translate(stage, self.visual_root_path, self.root_translate_xyz)

    def apply_pose(self, pose_by_name: dict[str, float]) -> None:
        if self.urdf_joint_specs is None:
            local_matrices = apply_joint_positions_to_local_matrices(self.records, pose_by_name)
        else:
            world_by_name = world_map_from_joint_specs(self.urdf_joint_specs, pose_by_name, pose_key="child_link")
            local_matrices = local_matrices_from_world_map(self.records, world_by_name)
        _set_animation_pose(
            self.translations_attr,
            self.rotations_attr,
            self.scales_attr,
            local_matrices,
        )


class LandauRawMeshPoseDriver:
    def __init__(
        self,
        stage,
        *,
        urdf_path: Path,
        skeleton_json_path: Path,
        visual_root_path: str = "/World/LandauRaw",
        root_offset_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        from pxr import UsdGeom

        self.stage = stage
        self.urdf_path = Path(urdf_path).resolve()
        self.skeleton_json_path = Path(skeleton_json_path).resolve()
        self.visual_root_path = visual_root_path
        self.root_offset_xyz = np.asarray(root_offset_xyz, dtype=float)
        self.records = load_skeleton_records(self.skeleton_json_path)
        self.urdf_joint_specs = load_urdf_joint_specs(self.urdf_path)
        self.mesh_records = load_urdf_visual_mesh_records(self.urdf_path)
        self.link_prim_paths: dict[str, str] = {}
        self.root_link_names = self._compute_root_link_names()

        self.root_z_offset = root_height_offset(self.records)
        self.root_translate_xyz = self.root_offset_xyz + np.array((0.0, 0.0, self.root_z_offset), dtype=float)
        UsdGeom.Xform.Define(stage, self.visual_root_path)

        for root_link in self.root_link_names:
            link_path = f"{self.visual_root_path}/{root_link}"
            UsdGeom.Xform.Define(stage, link_path)
            self.link_prim_paths[root_link] = link_path
            self._author_link_meshes(root_link)

        pending_specs = dict(self.urdf_joint_specs)
        while pending_specs:
            progressed = False
            for joint_name in list(pending_specs):
                spec = pending_specs[joint_name]
                parent_path = self.link_prim_paths.get(spec.parent_link)
                if parent_path is None:
                    continue
                link_path = f"{parent_path}/{spec.child_link}"
                UsdGeom.Xform.Define(stage, link_path)
                self.link_prim_paths[spec.child_link] = link_path
                self._author_link_meshes(spec.child_link)
                pending_specs.pop(joint_name)
                progressed = True
            if not progressed:
                unresolved = ", ".join(sorted(pending_specs))
                raise RuntimeError(f"Unable to build URDF link hierarchy for {self.urdf_path}: {unresolved}")

        _set_root_translate(stage, self.visual_root_path, self.root_translate_xyz)

    def _compute_root_link_names(self) -> tuple[str, ...]:
        parent_links = {spec.parent_link for spec in self.urdf_joint_specs.values()}
        child_links = {spec.child_link for spec in self.urdf_joint_specs.values()}
        roots = sorted(parent_links - child_links)
        return tuple(roots) if roots else ("base_link",)

    def _author_link_meshes(self, link_name: str) -> None:
        from pxr import UsdGeom

        link_path = self.link_prim_paths[link_name]
        for mesh_record in self.mesh_records.get(link_name, ()):
            mesh_xform_path = f"{link_path}/{mesh_record.mesh_name}"
            mesh_prim_path = f"{mesh_xform_path}/Mesh"
            UsdGeom.Xform.Define(self.stage, mesh_xform_path)
            if not self.stage.GetPrimAtPath(mesh_prim_path).IsValid():
                _author_raw_mesh(self.stage, mesh_prim_path, mesh_record.mesh_path)
            _set_xform_matrix(self.stage, mesh_xform_path, mesh_record.origin_matrix)

    def apply_pose(self, pose_by_name: dict[str, float]) -> None:
        identity = np.eye(4, dtype=float)
        for root_link in self.root_link_names:
            _set_xform_matrix(self.stage, self.link_prim_paths[root_link], identity)
        for spec in self.urdf_joint_specs.values():
            angle = float(pose_by_name.get(spec.child_link, 0.0))
            _set_xform_matrix(self.stage, self.link_prim_paths[spec.child_link], joint_local_transform(spec, angle))
