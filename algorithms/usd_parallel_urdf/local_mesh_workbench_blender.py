from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import bmesh
import bpy


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PACKAGE_RELATIVE = Path("algorithms/usd_parallel_urdf/outputs/urdf_packages/landau_v10")


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Build watertight per-link STL assets locally with Blender.")
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / PACKAGE_RELATIVE)
    parser.add_argument("--method", choices=("lowpoly_surface", "convex_hull", "obb"), default="lowpoly_surface")
    parser.add_argument("--target-face-ratio", type=float, default=0.25)
    parser.add_argument("--max-faces", type=int, default=1200)
    parser.add_argument("--max-hull-faces", type=int, default=48)
    parser.add_argument("--target-hull-points", type=int, default=24)
    parser.add_argument("--min-thickness", type=float, default=0.004)
    parser.add_argument("--part", action="append", default=[])
    args = parser.parse_args(argv)
    if not 0.01 <= args.target_face_ratio <= 1.0:
        parser.error("--target-face-ratio must be between 0.01 and 1.0")
    if args.max_faces < 12 or args.max_hull_faces < 12:
        parser.error("face budgets must be at least 12")
    if args.min_thickness <= 0:
        parser.error("--min-thickness must be positive")
    return args


def _source_candidates(output_dir: Path) -> list[Path]:
    local = output_dir / "source_mesh_stl" / "landau_v10"
    cloud_package = Path.home() / "Nextcloud" / "Projects" / "geo_lib" / "remote_outputs" / PACKAGE_RELATIVE
    return [local, cloud_package / "source_mesh_stl" / "landau_v10"]


def _select_source_dir(args: argparse.Namespace, output_dir: Path) -> Path:
    candidates = [args.source_dir.expanduser().resolve()] if args.source_dir else _source_candidates(output_dir)
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.stl")):
            return candidate
    raise FileNotFoundError("No source STL directory is available on this Mac. Pull or build the source meshes first.")


def _clear_scene() -> None:
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _active_mesh() -> bpy.types.Object:
    objects = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not objects:
        raise RuntimeError("STL import did not create a mesh")
    active = objects[0]
    if len(objects) > 1:
        bpy.context.view_layer.objects.active = active
        for obj in objects:
            obj.select_set(True)
        bpy.ops.object.join()
    bpy.context.view_layer.objects.active = active
    active.select_set(True)
    return active


def _import_stl(path: Path) -> bpy.types.Object:
    _clear_scene()
    bpy.ops.wm.stl_import(
        filepath=str(path),
        global_scale=1.0,
        use_scene_unit=False,
        use_facet_normal=True,
        forward_axis="Y",
        up_axis="Z",
        use_mesh_validate=True,
    )
    obj = _active_mesh()
    obj.name = path.stem
    return obj


def _mesh_bounds(obj: bpy.types.Object) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    points = [vertex.co for vertex in obj.data.vertices]
    if not points:
        raise RuntimeError("Mesh has no vertices")
    return (
        tuple(min(point[axis] for point in points) for axis in range(3)),
        tuple(max(point[axis] for point in points) for axis in range(3)),
    )


def _edit_cleanup(obj: bpy.types.Object) -> None:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=1e-7)
    bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=False)
    bpy.ops.object.mode_set(mode="OBJECT")


def _solidify_open_surface(obj: bpy.types.Object, thickness: float) -> None:
    """Give open skinned patches volume before voxel remeshing.

    Per-link source STLs are often connected skin patches rather than closed
    solids.  Voxel-remeshing those zero-thickness patches directly can retain
    only fragments of the patch.  Solidifying first preserves the complete
    source silhouette and also preserves separate meaningful shells.
    """
    if _topology(obj)["watertight"]:
        return
    modifier = obj.modifiers.new(name="Open surface thickness", type="SOLIDIFY")
    modifier.thickness = thickness
    modifier.offset = 0.0
    modifier.use_even_offset = True
    modifier.use_quality_normals = True
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def _triangulate(obj: bpy.types.Object) -> None:
    modifier = obj.modifiers.new(name="Triangulate", type="TRIANGULATE")
    modifier.keep_custom_normals = True
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def _decimate(obj: bpy.types.Object, target_faces: int) -> None:
    _triangulate(obj)
    face_count = len(obj.data.polygons)
    if face_count <= target_faces:
        return
    modifier = obj.modifiers.new(name="Closed surface decimation", type="DECIMATE")
    modifier.decimate_type = "COLLAPSE"
    modifier.ratio = max(0.001, min(1.0, target_faces / face_count))
    modifier.use_collapse_triangulate = True
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    _triangulate(obj)


def _close_small_boundary_loops(obj: bpy.types.Object, max_boundary_edges: int = 128) -> bool:
    """Close tiny voxel/decimator cracks without replacing the visible shells."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    boundary_edges = [edge for edge in bm.edges if edge.is_boundary]
    if not boundary_edges or len(boundary_edges) > max_boundary_edges:
        bm.free()
        return not boundary_edges
    bmesh.ops.holes_fill(bm, edges=boundary_edges, sides=0)
    if bm.faces:
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    _triangulate(obj)
    return True


def _make_bounds_box(
    obj: bpy.types.Object,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    min_thickness: float,
) -> None:
    lower, upper = bounds
    center = tuple((lower[i] + upper[i]) * 0.5 for i in range(3))
    size = tuple(max(upper[i] - lower[i], min_thickness) for i in range(3))
    vertices = [
        (center[0] + sx * size[0] * 0.5, center[1] + sy * size[1] * 0.5, center[2] + sz * size[2] * 0.5)
        for sx, sy, sz in ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))
    ]
    faces = [
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    ]
    mesh = bpy.data.meshes.new(f"{obj.name}_bounds")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    old_mesh = obj.data
    obj.data = mesh
    bpy.data.meshes.remove(old_mesh)


def _convex_hull(obj: bpy.types.Object) -> None:
    _edit_cleanup(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.convex_hull(delete_unused=True, use_existing_faces=False)
    bpy.ops.object.mode_set(mode="OBJECT")


def _convex_hull_loose_components(obj: bpy.types.Object) -> bpy.types.Object:
    """Close every remeshed shell independently instead of merging the part."""
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")
    components = [item for item in bpy.context.selected_objects if item.type == "MESH"]
    valid = []
    for component in components:
        if len(component.data.vertices) < 4:
            bpy.data.objects.remove(component, do_unlink=True)
            continue
        bpy.ops.object.select_all(action="DESELECT")
        component.select_set(True)
        bpy.context.view_layer.objects.active = component
        try:
            _convex_hull(component)
            valid.append(component)
        except RuntimeError:
            bpy.data.objects.remove(component, do_unlink=True)
    if not valid:
        raise RuntimeError(f"No solid components remain for {obj.name}")
    bpy.ops.object.select_all(action="DESELECT")
    for component in valid:
        component.select_set(True)
    bpy.context.view_layer.objects.active = valid[0]
    if len(valid) > 1:
        bpy.ops.object.join()
    return valid[0]


def _voxel_surface(
    obj: bpy.types.Object,
    target_faces: int,
    min_thickness: float,
) -> tuple[bpy.types.Object, str | None]:
    lower, upper = _mesh_bounds(obj)
    longest = max(upper[i] - lower[i] for i in range(3))
    resolution = max(32, min(96, int(round(math.sqrt(max(target_faces, 48) / 6.0) * 4.0))))
    if longest > 0.3:
        # Large detailed links (notably the head and ears) need the requested
        # thickness scale; a coarse longest-axis voxel erases their silhouette.
        voxel_size = max(min(min_thickness * 0.5, longest / resolution), longest / 128.0, 1e-5)
    else:
        voxel_size = max(min_thickness * 0.5, longest / resolution, 1e-5)
    _solidify_open_surface(obj, max(voxel_size * 1.25, min_thickness * 0.5))
    obj.data.remesh_voxel_size = voxel_size
    obj.data.remesh_voxel_adaptivity = 0.0
    obj.data.use_remesh_fix_poles = True
    obj.data.use_remesh_preserve_volume = True
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.voxel_remesh()
    _edit_cleanup(obj)
    _close_small_boundary_loops(obj)
    remesh_backup = obj.data.copy()
    surface_fallback = None
    _decimate(obj, target_faces)
    # Blender's collapse decimator can leave a few isolated vertices/edges even
    # when every face shell is closed.  Remove those harmless loose elements
    # before the manifold check so a complete remesh is not replaced by a hull.
    _edit_cleanup(obj)
    _close_small_boundary_loops(obj)
    _edit_cleanup(obj)
    if not _topology(obj)["watertight"]:
        # Retry from the pre-decimation surface so a failed collapse cannot
        # merge/discard shells before the topology fallback sees them.
        failed_mesh = obj.data
        obj.data = remesh_backup
        bpy.data.meshes.remove(failed_mesh)
        obj = _convex_hull_loose_components(obj)
        _decimate(obj, target_faces)
        _edit_cleanup(obj)
        surface_fallback = "per_shell_convex_hull_topology_guard"
    else:
        bpy.data.meshes.remove(remesh_backup)
    return obj, surface_fallback


def _bounds_retention(
    source: tuple[tuple[float, float, float], tuple[float, float, float]],
    generated: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> dict:
    source_lower, source_upper = source
    generated_lower, generated_upper = generated
    extent_ratios = []
    center_offsets = []
    for axis in range(3):
        source_extent = source_upper[axis] - source_lower[axis]
        generated_extent = generated_upper[axis] - generated_lower[axis]
        if source_extent <= 1e-9:
            continue
        extent_ratios.append(generated_extent / source_extent)
        source_center = (source_lower[axis] + source_upper[axis]) * 0.5
        generated_center = (generated_lower[axis] + generated_upper[axis]) * 0.5
        center_offsets.append(abs(generated_center - source_center) / source_extent)
    return {
        "min_extent_ratio": min(extent_ratios, default=1.0),
        "max_center_offset_ratio": max(center_offsets, default=0.0),
        "shape_retained": min(extent_ratios, default=1.0) >= 0.82 and max(center_offsets, default=0.0) <= 0.18,
    }


def _topology(obj: bpy.types.Object) -> dict:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    boundary_edges = sum(1 for edge in bm.edges if edge.is_boundary)
    non_manifold_edges = sum(1 for edge in bm.edges if not edge.is_manifold)
    vertices = len(bm.verts)
    faces = len(bm.faces)
    remaining = set(bm.verts)
    component_count = 0
    while remaining:
        component_count += 1
        stack = [remaining.pop()]
        while stack:
            vertex = stack.pop()
            for edge in vertex.link_edges:
                other = edge.other_vert(vertex)
                if other in remaining:
                    remaining.remove(other)
                    stack.append(other)
    bm.free()
    return {
        "vertex_count": vertices,
        "face_count": faces,
        "boundary_edge_count": boundary_edges,
        "non_manifold_edge_count": non_manifold_edges,
        "component_count": component_count,
        "watertight": non_manifold_edges == 0 and faces >= 4,
    }


def _ensure_watertight(
    obj: bpy.types.Object,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    min_thickness: float,
) -> tuple[dict, str | None]:
    topology = _topology(obj)
    if topology["watertight"]:
        return topology, None
    try:
        _convex_hull(obj)
        _triangulate(obj)
        topology = _topology(obj)
        if topology["watertight"]:
            return topology, "convex_hull"
    except RuntimeError:
        pass
    _make_bounds_box(obj, bounds, min_thickness)
    topology = _topology(obj)
    if not topology["watertight"]:
        raise RuntimeError(f"Unable to produce a watertight mesh for {obj.name}")
    return topology, "bounds_box"


def _export_stl(obj: bpy.types.Object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.stl_export(
        filepath=str(path),
        ascii_format=False,
        export_selected_objects=True,
        global_scale=1.0,
        use_scene_unit=False,
        forward_axis="Y",
        up_axis="Z",
        apply_modifiers=True,
    )


def _load_summary_template(output_dir: Path, source_dir: Path) -> dict:
    name = "landau_v10_mesh_collision_summary.json"
    candidates = [output_dir / name, source_dir.parents[1] / name]
    available = [path for path in candidates if path.is_file()]
    if not available:
        return {"links": {}}
    path = max(available, key=lambda item: item.stat().st_mtime)
    return json.loads(path.read_text(encoding="utf-8"))


def _refresh_local_package(output_dir: Path, source_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    local_source = output_dir / "source_mesh_stl" / "landau_v10"
    local_source.mkdir(parents=True, exist_ok=True)
    if source_dir.resolve() != local_source.resolve():
        for path in source_dir.glob("*.stl"):
            shutil.copy2(path, local_source / path.name)
    urdf_name = "landau_v10_parallel_mesh.urdf"
    remote_urdf = source_dir.parents[1] / urdf_name
    local_urdf = output_dir / urdf_name
    if remote_urdf.is_file() and remote_urdf.resolve() != local_urdf.resolve():
        shutil.copy2(remote_urdf, local_urdf)
    if local_urdf.is_file():
        os.utime(local_urdf, None)


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    source_dir = _select_source_dir(args, output_dir)
    source_paths = sorted(source_dir.glob("*.stl"))
    if args.part:
        requested = set(args.part)
        source_paths = [path for path in source_paths if path.stem in requested]
        missing = requested - {path.stem for path in source_paths}
        if missing:
            raise FileNotFoundError(f"Unknown source parts: {', '.join(sorted(missing))}")
    if not source_paths:
        raise FileNotFoundError("No source STL files were selected")

    _refresh_local_package(output_dir, source_dir)
    local_source_dir = output_dir / "source_mesh_stl" / "landau_v10"
    mesh_dir = output_dir / "mesh_collision_stl" / "landau_v10"
    summary = _load_summary_template(output_dir, source_dir)
    links = summary.setdefault("links", {})
    started = time.monotonic()

    for index, original_path in enumerate(source_paths, 1):
        source_path = local_source_dir / original_path.name
        obj = _import_stl(source_path)
        source_faces = len(obj.data.polygons)
        bounds = _mesh_bounds(obj)
        fallback = None
        shape_fallback = None
        if args.method == "lowpoly_surface":
            target_faces = min(args.max_faces, max(48, int(round(source_faces * args.target_face_ratio))))
            obj, shape_fallback = _voxel_surface(obj, target_faces, args.min_thickness)
            method = "blender_voxel_surface"
        elif args.method == "convex_hull":
            target_faces = min(args.max_hull_faces, max(12, args.target_hull_points * 2))
            _convex_hull(obj)
            _decimate(obj, target_faces)
            method = "blender_convex_hull"
        else:
            target_faces = 12
            _make_bounds_box(obj, bounds, args.min_thickness)
            method = "bounds_box"

        retention = _bounds_retention(bounds, _mesh_bounds(obj))
        remesh_topology = _topology(obj)
        if args.method == "lowpoly_surface" and not remesh_topology["watertight"]:
            obj = _convex_hull_loose_components(obj)
            _decimate(obj, target_faces)
            shape_fallback = "per_shell_convex_hull_topology_guard"
        elif not retention["shape_retained"]:
            obj = _import_stl(source_path)
            _convex_hull(obj)
            _decimate(obj, target_faces)
            shape_fallback = "convex_hull_shape_guard"
        topology, topology_fallback = _ensure_watertight(obj, bounds, args.min_thickness)
        fallback = topology_fallback or shape_fallback
        retention = _bounds_retention(bounds, _mesh_bounds(obj))
        if not retention["shape_retained"]:
            _make_bounds_box(obj, bounds, args.min_thickness)
            topology = _topology(obj)
            retention = _bounds_retention(bounds, _mesh_bounds(obj))
            fallback = "bounds_box_shape_guard"
        output_path = mesh_dir / source_path.name
        _export_stl(obj, output_path)
        details = dict(links.get(source_path.stem, {}))
        details.update({
            "stl_path": str(output_path),
            "source_stl_path": str(source_path),
            "seed_triangle_count": source_faces,
            "mesh_triangle_count": topology["face_count"],
            "method": method,
            "fallback": fallback,
            **retention,
            **topology,
        })
        links[source_path.stem] = details
        print(
            f"[LOCAL MESH] {index:02d}/{len(source_paths):02d} {source_path.stem}: "
            f"{source_faces} → {topology['face_count']} triangles, watertight={topology['watertight']}",
            flush=True,
        )

    summary.update({
        "mesh_output_dir": str(mesh_dir),
        "mesh_simplify_mode": args.method,
        "build_backend": f"local_blender_{bpy.app.version_string}",
        "max_hull_faces": args.max_hull_faces,
        "target_hull_points": args.target_hull_points,
        "config": {
            "min_thickness": args.min_thickness,
            "lowpoly_default": {
                "target_face_ratio": args.target_face_ratio,
                "max_faces": args.max_faces,
            },
        },
    })
    summary_path = output_dir / "landau_v10_mesh_collision_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[LOCAL MESH] wrote {len(source_paths)} watertight parts in {time.monotonic() - started:.2f}s", flush=True)
    print(f"[LOCAL MESH] summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
