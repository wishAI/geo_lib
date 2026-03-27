from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw


_COMMAND_RE = re.compile(r'[MmLlHhVvCcQqAaZz]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?')
_TRANSFORM_RE = re.compile(r'([a-zA-Z]+)\(([^)]*)\)')


@dataclass(frozen=True)
class SemanticLayout:
    width_m: float
    height_m: float
    resolution_m: float
    wall_grid: np.ndarray
    furniture_grid: np.ndarray

    @property
    def occupied_grid(self) -> np.ndarray:
        return np.logical_or(self.wall_grid, self.furniture_grid)

    @property
    def shape(self) -> tuple[int, int]:
        return self.wall_grid.shape

    def preview_rgb(self) -> np.ndarray:
        img = np.full((*self.shape, 3), 255, dtype=np.uint8)
        img[self.furniture_grid] = np.array([219, 191, 146], dtype=np.uint8)
        img[self.wall_grid] = np.array([72, 79, 84], dtype=np.uint8)
        return img


@dataclass(frozen=True)
class RectGeom:
    kind: str
    x_center_m: float
    y_center_m: float
    width_m: float
    depth_m: float
    height_m: float


@dataclass(frozen=True)
class SceneSpec:
    width_m: float
    height_m: float
    floor_thickness_m: float
    wall_geoms: list[RectGeom]
    furniture_geoms: list[RectGeom]

    @property
    def all_geoms(self) -> list[RectGeom]:
        return [*self.wall_geoms, *self.furniture_geoms]

    @property
    def bbox(self) -> dict[str, list[float]]:
        return {
            'min': [-self.width_m / 2.0, -self.height_m / 2.0, 0.0],
            'max': [self.width_m / 2.0, self.height_m / 2.0, 2.6],
            'extent': [self.width_m, self.height_m, 2.6],
        }


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


def _parse_float_list(value: str) -> list[float]:
    return [float(token) for token in re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', value)]


def _transform_matrix(spec: str | None) -> np.ndarray:
    matrix = np.eye(3, dtype=float)
    if not spec:
        return matrix
    for name, params in _TRANSFORM_RE.findall(spec):
        values = _parse_float_list(params)
        name = name.lower().strip()
        if name == 'matrix':
            a, b, c, d, e, f = values
            current = np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]], dtype=float)
        elif name == 'translate':
            tx = values[0]
            ty = values[1] if len(values) > 1 else 0.0
            current = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=float)
        elif name == 'scale':
            sx = values[0]
            sy = values[1] if len(values) > 1 else sx
            current = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        elif name == 'rotate':
            angle = math.radians(values[0])
            c = math.cos(angle)
            s = math.sin(angle)
            rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
            if len(values) == 3:
                cx, cy = values[1], values[2]
                current = (
                    np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]], dtype=float)
                    @ rotation
                    @ np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], dtype=float)
                )
            else:
                current = rotation
        else:
            continue
        matrix = matrix @ current
    return matrix


def _apply_transform(point: tuple[float, float], matrix: np.ndarray) -> tuple[float, float]:
    vec = np.array([point[0], point[1], 1.0], dtype=float)
    res = matrix @ vec
    return float(res[0]), float(res[1])


def _parse_fill(fill: str | None) -> tuple[int, int, int] | None:
    if not fill or fill.lower() == 'none':
        return None
    fill = fill.strip()
    if fill.startswith('#') and len(fill) == 7:
        return int(fill[1:3], 16), int(fill[3:5], 16), int(fill[5:7], 16)
    return None


def _luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = [channel / 255.0 for channel in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _sample_quadratic(p0, p1, p2, count=14):
    points = []
    for step in range(1, count + 1):
        t = step / count
        mt = 1.0 - t
        x = mt * mt * p0[0] + 2.0 * mt * t * p1[0] + t * t * p2[0]
        y = mt * mt * p0[1] + 2.0 * mt * t * p1[1] + t * t * p2[1]
        points.append((x, y))
    return points


def _sample_cubic(p0, p1, p2, p3, count=18):
    points = []
    for step in range(1, count + 1):
        t = step / count
        mt = 1.0 - t
        x = mt**3 * p0[0] + 3.0 * mt * mt * t * p1[0] + 3.0 * mt * t * t * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3.0 * mt * mt * t * p1[1] + 3.0 * mt * t * t * p2[1] + t**3 * p3[1]
        points.append((x, y))
    return points


def _vector_angle(u: np.ndarray, v: np.ndarray) -> float:
    dot = float(np.dot(u, v))
    det = float(u[0] * v[1] - u[1] * v[0])
    return math.atan2(det, dot)


def _sample_arc(start, rx, ry, rotation_deg, large_arc, sweep, end):
    if rx == 0.0 or ry == 0.0:
        return [end]
    phi = math.radians(rotation_deg % 360.0)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    x1, y1 = start
    x2, y2 = end
    dx = (x1 - x2) / 2.0
    dy = (y1 - y2) / 2.0
    x1p = cos_phi * dx + sin_phi * dy
    y1p = -sin_phi * dx + cos_phi * dy
    rx = abs(rx)
    ry = abs(ry)
    scale = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if scale > 1.0:
        factor = math.sqrt(scale)
        rx *= factor
        ry *= factor
    numerator = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    denominator = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    factor = 0.0 if denominator == 0.0 else math.sqrt(max(0.0, numerator / denominator))
    if large_arc == sweep:
        factor = -factor
    cxp = factor * (rx * y1p) / ry
    cyp = factor * (-ry * x1p) / rx
    cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2.0
    cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2.0
    start_vec = np.array([(x1p - cxp) / rx, (y1p - cyp) / ry], dtype=float)
    end_vec = np.array([(-x1p - cxp) / rx, (-y1p - cyp) / ry], dtype=float)
    theta_1 = _vector_angle(np.array([1.0, 0.0], dtype=float), start_vec)
    delta = _vector_angle(start_vec, end_vec)
    if not sweep and delta > 0.0:
        delta -= 2.0 * math.pi
    elif sweep and delta < 0.0:
        delta += 2.0 * math.pi
    count = max(8, int(abs(delta) / (math.pi / 14.0)))
    points = []
    for step in range(1, count + 1):
        theta = theta_1 + delta * (step / count)
        ct = math.cos(theta)
        st = math.sin(theta)
        x = cx + rx * ct * cos_phi - ry * st * sin_phi
        y = cy + rx * ct * sin_phi + ry * st * cos_phi
        points.append((x, y))
    return points


def _sample_path(d: str) -> list[list[tuple[float, float]]]:
    tokens = _COMMAND_RE.findall(d.replace(',', ' '))
    index = 0
    cmd = ''
    current = (0.0, 0.0)
    start = (0.0, 0.0)
    subpaths = []
    active = []

    def ensure(point):
        nonlocal active, start
        if not active:
            start = point
            active = [point]
            subpaths.append(active)

    while index < len(tokens):
        token = tokens[index]
        if token.isalpha():
            cmd = token
            index += 1
        absolute = cmd.isupper()
        op = cmd.upper()
        if op == 'M':
            x = float(tokens[index])
            y = float(tokens[index + 1])
            index += 2
            point = (x, y) if absolute else (current[0] + x, current[1] + y)
            current = point
            start = point
            active = [point]
            subpaths.append(active)
            cmd = 'L' if absolute else 'l'
        elif op == 'L':
            x = float(tokens[index])
            y = float(tokens[index + 1])
            index += 2
            point = (x, y) if absolute else (current[0] + x, current[1] + y)
            ensure(current)
            active.append(point)
            current = point
        elif op == 'H':
            x = float(tokens[index])
            index += 1
            point = (x, current[1]) if absolute else (current[0] + x, current[1])
            ensure(current)
            active.append(point)
            current = point
        elif op == 'V':
            y = float(tokens[index])
            index += 1
            point = (current[0], y) if absolute else (current[0], current[1] + y)
            ensure(current)
            active.append(point)
            current = point
        elif op == 'Q':
            x1 = float(tokens[index])
            y1 = float(tokens[index + 1])
            x = float(tokens[index + 2])
            y = float(tokens[index + 3])
            index += 4
            control = (x1, y1) if absolute else (current[0] + x1, current[1] + y1)
            point = (x, y) if absolute else (current[0] + x, current[1] + y)
            ensure(current)
            active.extend(_sample_quadratic(current, control, point))
            current = point
        elif op == 'C':
            x1 = float(tokens[index])
            y1 = float(tokens[index + 1])
            x2 = float(tokens[index + 2])
            y2 = float(tokens[index + 3])
            x = float(tokens[index + 4])
            y = float(tokens[index + 5])
            index += 6
            control1 = (x1, y1) if absolute else (current[0] + x1, current[1] + y1)
            control2 = (x2, y2) if absolute else (current[0] + x2, current[1] + y2)
            point = (x, y) if absolute else (current[0] + x, current[1] + y)
            ensure(current)
            active.extend(_sample_cubic(current, control1, control2, point))
            current = point
        elif op == 'A':
            rx = float(tokens[index])
            ry = float(tokens[index + 1])
            rotation = float(tokens[index + 2])
            large_arc = int(float(tokens[index + 3]))
            sweep = int(float(tokens[index + 4]))
            x = float(tokens[index + 5])
            y = float(tokens[index + 6])
            index += 7
            point = (x, y) if absolute else (current[0] + x, current[1] + y)
            ensure(current)
            active.extend(_sample_arc(current, rx, ry, rotation, large_arc, sweep, point))
            current = point
        elif op == 'Z':
            if active and active[-1] != start:
                active.append(start)
            current = start
            cmd = ''
        else:
            raise ValueError(f'Unsupported path command: {cmd}')
    return [subpath for subpath in subpaths if len(subpath) >= 3]


def _collect_polygons(element: ET.Element, inherited: np.ndarray, out: list[tuple[tuple[int, int, int], list[tuple[float, float]]]]) -> None:
    transform = inherited @ _transform_matrix(element.attrib.get('transform'))
    tag = element.tag.split('}')[-1]
    fill = _parse_fill(element.attrib.get('fill'))
    if tag == 'path' and fill is not None:
        for polygon in _sample_path(element.attrib['d']):
            out.append((fill, [_apply_transform(point, transform) for point in polygon]))
    elif tag == 'rect' and fill is not None:
        x = float(element.attrib.get('x', '0'))
        y = float(element.attrib.get('y', '0'))
        width = float(element.attrib['width'])
        height = float(element.attrib['height'])
        polygon = [(x, y), (x + width, y), (x + width, y + height), (x, y + height), (x, y)]
        out.append((fill, [_apply_transform(point, transform) for point in polygon]))
    elif tag in {'circle', 'ellipse'} and fill is not None:
        if tag == 'circle':
            cx = float(element.attrib['cx'])
            cy = float(element.attrib['cy'])
            rx = float(element.attrib['r'])
            ry = rx
        else:
            cx = float(element.attrib['cx'])
            cy = float(element.attrib['cy'])
            rx = float(element.attrib['rx'])
            ry = float(element.attrib['ry'])
        polygon = []
        for step in range(36):
            theta = 2.0 * math.pi * step / 36.0
            polygon.append((cx + rx * math.cos(theta), cy + ry * math.sin(theta)))
        polygon.append(polygon[0])
        out.append((fill, [_apply_transform(point, transform) for point in polygon]))
    for child in element:
        _collect_polygons(child, transform, out)


def _filter_small(mask: np.ndarray, minimum_pixels: int) -> np.ndarray:
    result = mask.copy()
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for row in range(height):
        for col in range(width):
            if not result[row, col] or visited[row, col]:
                continue
            stack = [(row, col)]
            component = []
            visited[row, col] = True
            while stack:
                cr, cc = stack.pop()
                component.append((cr, cc))
                for dr, dc in neighbors:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and result[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            if len(component) < minimum_pixels:
                for cr, cc in component:
                    result[cr, cc] = False
    return result


def load_semantic_layout(svg_path: str | Path, meters_per_pixel: float = 0.01, resolution_m: float = 0.02) -> SemanticLayout:
    root = ET.fromstring(Path(svg_path).read_text(encoding='utf-8'))
    width_px = float(root.attrib['width'])
    height_px = float(root.attrib['height'])
    width_m = width_px * meters_per_pixel
    height_m = height_px * meters_per_pixel
    target_width = max(64, int(round(width_m / resolution_m)))
    target_height = max(64, int(round(height_m / resolution_m)))
    base_size = (int(round(width_px)), int(round(height_px)))
    wall_img = Image.new('L', base_size, 0)
    furniture_img = Image.new('L', base_size, 0)
    view_box = _parse_float_list(root.attrib.get('viewBox', f'0 0 {width_px} {height_px}'))
    if len(view_box) >= 4:
        inherited = np.array([[1.0, 0.0, -view_box[0]], [0.0, 1.0, -view_box[1]], [0.0, 0.0, 1.0]], dtype=float)
    else:
        inherited = np.eye(3, dtype=float)
    polygons = []
    _collect_polygons(root, inherited, polygons)
    wall_draw = ImageDraw.Draw(wall_img)
    furniture_draw = ImageDraw.Draw(furniture_img)
    for fill, polygon in polygons:
        if len(polygon) < 3:
            continue
        luminance = _luminance(fill)
        if luminance > 0.96:
            continue
        draw = wall_draw if luminance < 0.46 else furniture_draw
        draw.polygon(polygon, fill=255)
    wall_grid = np.array(wall_img.resize((target_width, target_height), resample=Image.Resampling.BILINEAR), dtype=np.uint8) > 96
    furniture_grid = np.array(furniture_img.resize((target_width, target_height), resample=Image.Resampling.BILINEAR), dtype=np.uint8) > 96
    wall_grid = _filter_small(wall_grid, minimum_pixels=12)
    furniture_grid = _filter_small(np.logical_and(furniture_grid, np.logical_not(wall_grid)), minimum_pixels=8)
    return SemanticLayout(width_m=width_m, height_m=height_m, resolution_m=resolution_m, wall_grid=wall_grid, furniture_grid=furniture_grid)


def _block_reduce(mask: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return mask.copy()
    height, width = mask.shape
    pad_h = (stride - height % stride) % stride
    pad_w = (stride - width % stride) % stride
    padded = np.pad(mask, ((0, pad_h), (0, pad_w)), constant_values=False)
    reduced = padded.reshape(padded.shape[0] // stride, stride, padded.shape[1] // stride, stride)
    return np.any(reduced, axis=(1, 3))


def _rectangles_from_mask(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    work = mask.copy()
    height, width = work.shape
    rects = []
    for row in range(height):
        col = 0
        while col < width:
            if not work[row, col]:
                col += 1
                continue
            span_w = 1
            while col + span_w < width and work[row, col + span_w]:
                span_w += 1
            span_h = 1
            while row + span_h < height and np.all(work[row + span_h, col : col + span_w]):
                span_h += 1
            work[row : row + span_h, col : col + span_w] = False
            rects.append((col, row, span_w, span_h))
            col += span_w
    return rects


def _rect_geom(rect, resolution_m, width_m, height_m, kind, geom_height) -> RectGeom:
    col, row, span_w, span_h = rect
    center_x = -width_m / 2.0 + (col + span_w / 2.0) * resolution_m
    center_y = height_m / 2.0 - (row + span_h / 2.0) * resolution_m
    return RectGeom(kind=kind, x_center_m=center_x, y_center_m=center_y, width_m=span_w * resolution_m, depth_m=span_h * resolution_m, height_m=geom_height)


def build_scene_spec(layout: SemanticLayout, geom_stride: int = 6, wall_height_m: float = 2.4, furniture_height_m: float = 0.65, floor_thickness_m: float = 0.05) -> SceneSpec:
    wall_mask = _block_reduce(layout.wall_grid, geom_stride)
    furniture_mask = _block_reduce(layout.furniture_grid, geom_stride)
    resolution_m = layout.resolution_m * geom_stride
    wall_geoms = [_rect_geom(rect, resolution_m, layout.width_m, layout.height_m, 'wall', wall_height_m) for rect in _rectangles_from_mask(wall_mask)]
    furniture_geoms = [_rect_geom(rect, resolution_m, layout.width_m, layout.height_m, 'furniture', furniture_height_m) for rect in _rectangles_from_mask(furniture_mask)]
    return SceneSpec(width_m=layout.width_m, height_m=layout.height_m, floor_thickness_m=floor_thickness_m, wall_geoms=wall_geoms, furniture_geoms=furniture_geoms)


def _world_to_grid(layout: SemanticLayout, x_m: float, y_m: float) -> tuple[int, int]:
    col = int(np.clip((x_m + layout.width_m / 2.0) / layout.resolution_m, 0, layout.shape[1] - 1))
    row = int(np.clip((layout.height_m / 2.0 - y_m) / layout.resolution_m, 0, layout.shape[0] - 1))
    return row, col


def _grid_to_world(layout: SemanticLayout, row: int, col: int) -> tuple[float, float]:
    x_m = -layout.width_m / 2.0 + (col + 0.5) * layout.resolution_m
    y_m = layout.height_m / 2.0 - (row + 0.5) * layout.resolution_m
    return x_m, y_m


def _inflate(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    inflated = mask.copy()
    if radius_cells <= 0:
        return inflated
    height, width = mask.shape
    for row, col in np.argwhere(mask):
        for rr in range(max(0, row - radius_cells), min(height, row + radius_cells + 1)):
            for cc in range(max(0, col - radius_cells), min(width, col + radius_cells + 1)):
                if (rr - row) ** 2 + (cc - col) ** 2 <= radius_cells ** 2:
                    inflated[rr, cc] = True
    return inflated


def _nearest_free(mask: np.ndarray, row: int, col: int) -> tuple[int, int]:
    if not mask[row, col]:
        return row, col
    for radius in range(1, max(mask.shape)):
        for rr in range(max(0, row - radius), min(mask.shape[0], row + radius + 1)):
            for cc in range(max(0, col - radius), min(mask.shape[1], col + radius + 1)):
                if not mask[rr, cc]:
                    return rr, cc
    return row, col


def select_robot_start(layout: SemanticLayout, robot_radius_m: float = 0.18) -> Pose2D:
    safe_mask = _inflate(layout.occupied_grid, max(1, int(math.ceil(robot_radius_m / layout.resolution_m))))
    row, col = _nearest_free(safe_mask, safe_mask.shape[0] // 2, safe_mask.shape[1] // 2)
    start_x, start_y = _grid_to_world(layout, row, col)
    return Pose2D(start_x, start_y, 0.0)


def build_mjcf(scene: SceneSpec, robot_start: Pose2D) -> str:
    geoms = [
        f"    <geom name='floor' type='box' pos='0 0 {-scene.floor_thickness_m / 2.0:.4f}' size='{scene.width_m / 2.0:.4f} {scene.height_m / 2.0:.4f} {scene.floor_thickness_m / 2.0:.4f}' rgba='0.95 0.95 0.95 1'/>"
    ]
    for index, geom in enumerate(scene.wall_geoms):
        geoms.append(
            f"    <geom name='wall_{index:03d}' type='box' pos='{geom.x_center_m:.4f} {geom.y_center_m:.4f} {geom.height_m / 2.0:.4f}' size='{geom.width_m / 2.0:.4f} {geom.depth_m / 2.0:.4f} {geom.height_m / 2.0:.4f}' rgba='0.28 0.31 0.33 1'/>"
        )
    for index, geom in enumerate(scene.furniture_geoms):
        geoms.append(
            f"    <geom name='furniture_{index:03d}' type='box' pos='{geom.x_center_m:.4f} {geom.y_center_m:.4f} {geom.height_m / 2.0:.4f}' size='{geom.width_m / 2.0:.4f} {geom.depth_m / 2.0:.4f} {geom.height_m / 2.0:.4f}' rgba='0.83 0.74 0.57 1'/>"
        )
    xml = "<mujoco model='svg_scene_builder'>\n"
    xml += "  <compiler angle='radian'/>\n"
    xml += "  <option timestep='0.02' gravity='0 0 -9.81'/>\n"
    xml += "  <worldbody>\n"
    xml += "    <light name='sun' pos='0 0 8' dir='0 0 -1'/>\n"
    xml += '\n'.join(geoms) + '\n'
    xml += "    <body name='robot' pos='0 0 0.12'>\n"
    xml += "      <joint name='robot_x' type='slide' axis='1 0 0'/>\n"
    xml += "      <joint name='robot_y' type='slide' axis='0 1 0'/>\n"
    xml += "      <joint name='robot_yaw' type='hinge' axis='0 0 1'/>\n"
    xml += "      <geom name='robot_base' type='cylinder' pos='0 0 0' size='0.16 0.06' rgba='0.16 0.52 0.69 1'/>\n"
    xml += "      <site name='lidar_origin' pos='0 0 0.02' size='0.015'/>\n"
    xml += "    </body>\n"
    xml += "  </worldbody>\n"
    xml += "  <keyframe>\n"
    xml += f"    <key name='start' qpos='{robot_start.x:.4f} {robot_start.y:.4f} {robot_start.yaw:.4f}'/>\n"
    xml += "  </keyframe>\n"
    xml += '</mujoco>\n'
    return xml


def detect_ros2_environment() -> dict[str, object]:
    ros_root = Path('/opt/ros')
    distros = sorted(path.name for path in ros_root.glob('*') if path.is_dir()) if ros_root.exists() else []
    return {
        'ros_root_exists': ros_root.exists(),
        'installed_distros': distros,
        'ros2_in_path': shutil.which('ros2') is not None,
    }


def _save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def save_scene_package(output_dir: Path, layout: SemanticLayout, scene: SceneSpec, start_pose: Pose2D, scene_xml: str, svg_path: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / 'semantic_layout.npz',
        width_m=np.array(layout.width_m, dtype=float),
        height_m=np.array(layout.height_m, dtype=float),
        resolution_m=np.array(layout.resolution_m, dtype=float),
        wall_grid=layout.wall_grid.astype(np.uint8),
        furniture_grid=layout.furniture_grid.astype(np.uint8),
    )
    _save_image(output_dir / 'layout_preview.png', layout.preview_rgb())
    (output_dir / 'scene.xml').write_text(scene_xml, encoding='utf-8')
    shutil.copyfile(svg_path, output_dir / 'source_svg.svg')
    start_payload = {'x': start_pose.x, 'y': start_pose.y, 'yaw': start_pose.yaw}
    (output_dir / 'start_pose.json').write_text(json.dumps(start_payload, indent=2), encoding='utf-8')
    summary = {
        'bbox': scene.bbox,
        'wall_geom_count': len(scene.wall_geoms),
        'furniture_geom_count': len(scene.furniture_geoms),
        'total_geom_count': len(scene.all_geoms),
        'layout_shape': [layout.shape[0], layout.shape[1]],
        'resolution_m': layout.resolution_m,
        'start_pose': start_payload,
    }
    (output_dir / 'scene_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    package = {
        'stage': 'svg_scene_builder',
        'version': 1,
        'source_svg': 'source_svg.svg',
        'scene_xml': 'scene.xml',
        'semantic_layout': 'semantic_layout.npz',
        'layout_preview': 'layout_preview.png',
        'start_pose': 'start_pose.json',
        'scene_summary': 'scene_summary.json',
        'ros2_environment': detect_ros2_environment(),
        'robot_radius_m': 0.18,
        'bbox': scene.bbox,
        'width_m': layout.width_m,
        'height_m': layout.height_m,
        'resolution_m': layout.resolution_m,
    }
    (output_dir / 'scene_package.json').write_text(json.dumps(package, indent=2), encoding='utf-8')
    return package


def run_scene_builder(output_dir: str | Path, svg_path: str | Path | None = None, map_resolution_m: float = 0.02) -> dict[str, object]:
    base_dir = Path(__file__).resolve().parent
    svg_file = Path(svg_path) if svg_path is not None else base_dir / 'svg_room_map.svg'
    layout = load_semantic_layout(svg_file, resolution_m=map_resolution_m)
    scene = build_scene_spec(layout)
    start_pose = select_robot_start(layout)
    scene_xml = build_mjcf(scene, start_pose)
    package = save_scene_package(Path(output_dir), layout, scene, start_pose, scene_xml, svg_file)
    return {
        'scene_output_dir': str(Path(output_dir)),
        'wall_geom_count': len(scene.wall_geoms),
        'furniture_geom_count': len(scene.furniture_geoms),
        'total_geom_count': len(scene.all_geoms),
        'layout_shape': [layout.shape[0], layout.shape[1]],
        'start_pose': {'x': start_pose.x, 'y': start_pose.y, 'yaw': start_pose.yaw},
        'bbox': scene.bbox,
        'scene_package': package,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a MuJoCo scene package from an SVG floor plan')
    parser.add_argument('--svg', type=Path, default=Path(__file__).resolve().parent / 'svg_room_map.svg')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'outputs' / 'sample_scene')
    parser.add_argument('--map-resolution', type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_scene_builder(output_dir=args.output, svg_path=args.svg, map_resolution_m=args.map_resolution)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
