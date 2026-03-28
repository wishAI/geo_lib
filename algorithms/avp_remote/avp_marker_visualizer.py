import re
from dataclasses import dataclass, field

import numpy as np

from pxr import Gf, Tf, Usd, UsdGeom


def _normalize_abs_prim_path(path_value, field_name):
    raw = str(path_value or "").strip()
    if not raw:
        raise ValueError(f"{field_name} must be a non-empty absolute USD prim path")
    if not raw.startswith("/"):
        raise ValueError(f"{field_name} must start with '/', got: {path_value!r}")
    parts = [part for part in raw.split("/") if part]
    if not parts:
        raise ValueError(f"{field_name} must target a prim below root '/', got: {path_value!r}")
    return "/" + "/".join(parts)


def _join_prim_path(parent_path, child_token):
    parent = _normalize_abs_prim_path(parent_path, "parent_path").rstrip("/")
    child = str(child_token or "").strip().strip("/")
    if not child:
        raise ValueError("child_token must be non-empty for USD prim path join")
    return f"{parent}/{child}"


def _make_valid_prim_token(value, fallback):
    text = str(value or "")
    make_identifier = getattr(Tf, "MakeValidIdentifier", None)
    if callable(make_identifier):
        try:
            text = make_identifier(text)
        except Exception:
            pass
    text = re.sub(r"[^A-Za-z0-9_]", "_", text)
    if not text:
        text = fallback
    if not (text[0].isalpha() or text[0] == "_"):
        text = f"_{text}"
    return text


@dataclass(frozen=True)
class MarkerStyle:
    mode: str = "axes"
    color: tuple[float, float, float] = (1.0, 0.1, 0.1)
    radius: float = 0.005
    axis_length: float = 0.08
    axis_radius: float = 0.0075
    cylinder_length: float = 0.05
    cylinder_radius: float = 0.004

    def __post_init__(self):
        if self.mode not in ("axes", "sphere", "cylinder"):
            raise ValueError(f"Unsupported marker mode: {self.mode}")


@dataclass(frozen=True)
class HandStyle:
    wrist_index: int = 0
    wrist: MarkerStyle = field(
        default_factory=lambda: MarkerStyle(
            mode="axes",
            color=(1.0, 0.1, 0.1),
            radius=0.005,
            axis_length=0.07,
            axis_radius=0.006,
            cylinder_length=0.03,
            cylinder_radius=0.004,
        )
    )
    joint: MarkerStyle = field(
        default_factory=lambda: MarkerStyle(
            mode="sphere",
            color=(0.2, 0.8, 1.0),
            radius=0.005,
            axis_length=0.07,
            axis_radius=0.006,
            cylinder_length=0.03,
            cylinder_radius=0.004,
        )
    )

    def __post_init__(self):
        if self.joint.mode not in ("sphere", "cylinder"):
            raise ValueError("joint marker mode must be either 'sphere' or 'cylinder'")


class MarkerVisualizer:
    _AXES = (
        ("AxisX", {"rotate": Gf.Vec3f(0.0, 90.0, 0.0), "color": (1.0, 0.1, 0.1)}),
        ("AxisY", {"rotate": Gf.Vec3f(-90.0, 0.0, 0.0), "color": (0.1, 1.0, 0.1)}),
        ("AxisZ", {"rotate": Gf.Vec3f(0.0, 0.0, 0.0), "color": (0.1, 0.3, 1.0)}),
    )

    def __init__(
        self,
        stage,
        marker_root,
        *,
        style=None,
        label="marker",
        print_debug=True,
    ):
        style = style if style is not None else MarkerStyle()
        self.stage = stage
        self.marker_root = _normalize_abs_prim_path(marker_root, "marker_root")
        self.style = style
        self.mode = style.mode
        self.color = style.color
        self.radius = style.radius
        self.axis_length = style.axis_length
        self.axis_radius = style.axis_radius
        self.cylinder_length = style.cylinder_length
        self.cylinder_radius = style.cylinder_radius
        self.label = label
        self.print_debug = print_debug
        self.last_world_print = 0.0
        self.marker_prim = self._ensure_marker()

    def print_status(self):
        if not self.print_debug:
            return
        if not self.marker_prim or not self.marker_prim.IsValid():
            print(f"[AVP] {self.label} prim invalid at {self.marker_root}")
            return
        visibility = UsdGeom.Imageable(self.marker_prim).GetVisibilityAttr().Get()
        print(f"[AVP] {self.label} prim ok at {self.marker_root}, visibility={visibility}")

    def update(self, mat_world, now):
        if mat_world is None:
            return
        self._set_matrix(mat_world)
        if not self.print_debug:
            return
        if now - self.last_world_print > 1.0:
            marker_world = self._get_world_matrix()
            if marker_world is not None:
                print(f"[AVP] {self.label} world translation (m): {marker_world[:3, 3]}")
            self.last_world_print = now

    def _apply_display(self, gprim, color):
        if not gprim:
            return
        gprim.CreateDisplayColorAttr([color])
        gprim.CreateDisplayOpacityAttr([1.0])

    def _ensure_marker(self):
        root = UsdGeom.Xform.Define(self.stage, self.marker_root)

        if self.mode in ("axes", "sphere"):
            sphere_path = _join_prim_path(self.marker_root, "Sphere")
            sphere_prim = self.stage.GetPrimAtPath(sphere_path)
            if not sphere_prim or not sphere_prim.IsValid():
                sphere = UsdGeom.Sphere.Define(self.stage, sphere_path)
            else:
                sphere = UsdGeom.Sphere(sphere_prim)
            sphere.GetRadiusAttr().Set(float(self.radius))
            self._apply_display(UsdGeom.Gprim(sphere.GetPrim()), self.color)

        if self.mode == "axes":
            for axis_name, axis_cfg in self._AXES:
                axis_path = _join_prim_path(self.marker_root, axis_name)
                axis_prim = self.stage.GetPrimAtPath(axis_path)
                if not axis_prim or not axis_prim.IsValid():
                    axis = UsdGeom.Cylinder.Define(self.stage, axis_path)
                else:
                    axis = UsdGeom.Cylinder(axis_prim)
                axis.GetHeightAttr().Set(float(self.axis_length))
                axis.GetRadiusAttr().Set(float(self.axis_radius))

                xform = UsdGeom.Xformable(axis.GetPrim())
                xform.ClearXformOpOrder()
                rotate = xform.AddRotateXYZOp()
                rotate.Set(axis_cfg["rotate"])
                translate = xform.AddTranslateOp()
                translate.Set(Gf.Vec3d(0.0, 0.0, self.axis_length * 0.5))

                self._apply_display(UsdGeom.Gprim(axis.GetPrim()), axis_cfg["color"])
        elif self.mode == "cylinder":
            cyl_path = _join_prim_path(self.marker_root, "Direction")
            cyl_prim = self.stage.GetPrimAtPath(cyl_path)
            if not cyl_prim or not cyl_prim.IsValid():
                cyl = UsdGeom.Cylinder.Define(self.stage, cyl_path)
            else:
                cyl = UsdGeom.Cylinder(cyl_prim)
            cyl.GetHeightAttr().Set(float(self.cylinder_length))
            cyl.GetRadiusAttr().Set(float(self.cylinder_radius))
            xform = UsdGeom.Xformable(cyl.GetPrim())
            xform.ClearXformOpOrder()
            translate = xform.AddTranslateOp()
            translate.Set(Gf.Vec3d(0.0, 0.0, self.cylinder_length * 0.5))
            self._apply_display(UsdGeom.Gprim(cyl.GetPrim()), self.color)

        return root.GetPrim()

    def _set_matrix(self, mat4):
        xformable = UsdGeom.Xformable(self.marker_prim)
        ops = xformable.GetOrderedXformOps()
        op = None
        for existing in ops:
            if existing.GetOpType() == UsdGeom.XformOp.TypeTransform:
                op = existing
                break
        if op is None:
            op = xformable.AddTransformOp()
        op.Set(Gf.Matrix4d(mat4.tolist()))

    def _get_world_matrix(self):
        prim = self.stage.GetPrimAtPath(self.marker_root)
        if not prim or not prim.IsValid():
            return None
        xformable = UsdGeom.Xformable(prim)
        mat = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return np.array(mat, dtype=float)


class HandMarkerSetVisualizer:
    def __init__(
        self,
        stage,
        marker_root,
        joint_names,
        *,
        style=None,
        label="hand",
        print_debug=False,
    ):
        style = style if style is not None else HandStyle()

        marker_root_path = _normalize_abs_prim_path(marker_root, "marker_root")
        self.markers = []
        self.style = style
        self.joint_names = tuple(joint_names)
        for index, joint_name in enumerate(self.joint_names):
            is_wrist = index == style.wrist_index
            joint_token = _make_valid_prim_token(
                f"j{index:02d}_{joint_name}",
                fallback=f"joint_{index:02d}",
            )
            marker_path = _join_prim_path(marker_root_path, joint_token)
            marker_style = style.wrist if is_wrist else style.joint
            marker = MarkerVisualizer(
                stage,
                marker_path,
                style=marker_style,
                label=f"{label}:{joint_name}",
                print_debug=print_debug and is_wrist,
            )
            self.markers.append(marker)

    def update(self, joint_mats_world, now=0.0):
        if joint_mats_world is None:
            return
        for marker, mat in zip(self.markers, joint_mats_world):
            marker.update(mat, now)

    @classmethod
    def for_hand(
        cls,
        stage,
        marker_root,
        joint_names,
        *,
        style=None,
        label="hand",
        print_debug=False,
    ):
        style = style if style is not None else HandStyle()
        return cls(
            stage,
            marker_root,
            joint_names,
            style=style,
            label=label,
            print_debug=print_debug,
        )
