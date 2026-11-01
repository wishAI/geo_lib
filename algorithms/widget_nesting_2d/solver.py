from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field, replace
from functools import cached_property
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, Sequence

from shapely.affinity import rotate as shapely_rotate
from shapely.affinity import translate as shapely_translate
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.geometry.polygon import orient

try:
    from .problem import ProblemSpec, WidgetSpec, load_problem, save_solution
except ImportError:  # pragma: no cover - script mode
    from problem import ProblemSpec, WidgetSpec, load_problem, save_solution


ALL_CORNERS = ("lower_left", "lower_right", "upper_left", "upper_right")
DEFAULT_CORNERS = ("lower_left",)


@dataclass(frozen=True)
class SolverConfig:
    rotation_mode: str = "free"
    rotation_step_degrees: float = 15.0
    beam_width: int = 6
    population_size: int = 12
    generations: int = 6
    elite_count: int = 4
    mutation_rate: float = 0.20
    max_candidates_per_item: int = 16
    max_item_anchor_points: int = 6
    max_free_space_anchor_points: int = 8
    compaction_passes: int = 2
    placement_tolerance: float = 1e-6
    seed: int = 20260403
    candidate_mode: str = "nfp"
    constructive_start: bool = True
    preferred_corners: tuple[str, ...] = DEFAULT_CORNERS

    def __post_init__(self):
        for key in ("beam_width", "population_size", "generations", "elite_count",
                    "max_candidates_per_item", "max_item_anchor_points", "max_free_space_anchor_points"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.rotation_mode not in ("free", "discrete"):
            raise ValueError("rotation_mode must be free or discrete")
        if not math.isfinite(self.rotation_step_degrees) or self.rotation_step_degrees <= 0:
            raise ValueError("rotation_step_degrees must be finite and positive")
        if not math.isfinite(self.placement_tolerance) or self.placement_tolerance <= 0:
            raise ValueError("placement_tolerance must be finite and positive")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("mutation_rate must be in [0, 1]")
        if self.candidate_mode not in ("contact", "nfp"):
            raise ValueError("candidate_mode must be contact or nfp")
        if not self.preferred_corners or any(c not in ALL_CORNERS for c in self.preferred_corners):
            raise ValueError("invalid preferred_corners")

    @classmethod
    def from_problem(cls, problem: ProblemSpec, overrides: dict[str, Any] | None = None) -> "SolverConfig":
        merged = dict(problem.config)
        if overrides:
            merged.update({key: value for key, value in overrides.items() if value is not None})
        preferred_corners = tuple(merged.get("preferred_corners", DEFAULT_CORNERS))
        return cls(
            rotation_mode=str(merged.get("rotation_mode", "free")),
            constructive_start=bool(merged.get("constructive_start", True)),
            candidate_mode=str(merged.get("candidate_mode", "nfp")),
            rotation_step_degrees=float(merged.get("rotation_step_degrees", 15.0)),
            beam_width=int(merged.get("beam_width", 6)),
            population_size=int(merged.get("population_size", 12)),
            generations=int(merged.get("generations", 6)),
            elite_count=int(merged.get("elite_count", 4)),
            mutation_rate=float(merged.get("mutation_rate", 0.20)),
            max_candidates_per_item=int(merged.get("max_candidates_per_item", 24)),
            max_item_anchor_points=int(merged.get("max_item_anchor_points", 12)),
            max_free_space_anchor_points=int(merged.get("max_free_space_anchor_points", 18)),
            compaction_passes=int(merged.get("compaction_passes", 3)),
            placement_tolerance=float(merged.get("placement_tolerance", 1e-6)),
            seed=int(merged.get("seed", 20260403)),
            preferred_corners=preferred_corners or DEFAULT_CORNERS,
        )


@dataclass(frozen=True)
class BoardRuntime:
    board_id: str
    polygon: Polygon
    buffered_polygon: Polygon
    area: float
    bounds: tuple[float, float, float, float]


@dataclass(frozen=True)
class ItemRuntime:
    instance_id: str
    widget_id: str
    polygon_local: Polygon
    area: float
    hole_area: float
    bbox_area: float
    rotation_variants: tuple["RotationVariant", ...]
    free_rotation: bool = False


@dataclass(frozen=True)
class RotationVariant:
    angle_degrees: float
    polygon: Polygon
    bounds: tuple[float, float, float, float]
    anchor_limit: int = 12

    @cached_property
    def anchor_points(self) -> tuple[tuple[float, float], ...]:
        # Constructors only use bounds; defer boundary sampling until NFP needs it.
        return tuple(_item_anchor_points(self.polygon, limit=self.anchor_limit))


@dataclass(frozen=True)
class Placement:
    item_instance_id: str
    widget_id: str
    board_id: str
    board_index: int
    rotation_degrees: float
    polygon: Polygon
    centroid_x: float
    centroid_y: float
    polygon_local: Polygon | None = None

    @property
    def area(self) -> float:
        return float(self.polygon.area)


@dataclass(frozen=True)
class BoardState:
    occupied: Any = field(default_factory=GeometryCollection)
    free_space: Any | None = None
    placements: tuple[Placement, ...] = ()


@dataclass(frozen=True)
class LayoutState:
    board_states: tuple[BoardState, ...]
    placed: tuple[Placement, ...] = ()
    skipped_item_ids: tuple[str, ...] = ()
    placed_area: float = 0.0
    skipped_area: float = 0.0

    @property
    def placed_count(self) -> int:
        return len(self.placed)


@dataclass(frozen=True)
class SolutionResult:
    order: tuple[str, ...]
    placements: tuple[Placement, ...]
    skipped_item_ids: tuple[str, ...]
    placed_area: float
    skipped_area: float
    max_rest_rectangle_area: float
    sum_rest_rectangle_area: float
    board_metrics: tuple[dict[str, Any], ...]
    search_stats: dict[str, Any]


def _iter_polygons(geometry: Any) -> Iterable[Polygon]:
    if geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
        return
    if isinstance(geometry, MultiPolygon):
        for poly in geometry.geoms:
            if poly.area > 1e-9:
                yield poly
        return
    if isinstance(geometry, GeometryCollection):
        for geom in geometry.geoms:
            if isinstance(geom, Polygon) and geom.area > 1e-9:
                yield geom


def _bounds_area(bounds: tuple[float, float, float, float]) -> float:
    min_x, min_y, max_x, max_y = bounds
    return max(0.0, max_x - min_x) * max(0.0, max_y - min_y)


def _corner_xy(bounds: tuple[float, float, float, float], corner: str) -> tuple[float, float]:
    min_x, min_y, max_x, max_y = bounds
    mapping = {
        "lower_left": (min_x, min_y),
        "lower_right": (max_x, min_y),
        "upper_left": (min_x, max_y),
        "upper_right": (max_x, max_y),
    }
    return mapping[corner]


def _movement_axes(corner: str) -> tuple[tuple[int, int], tuple[int, int]]:
    mapping = {
        "lower_left": ((-1, 0), (0, -1)),
        "lower_right": ((1, 0), (0, -1)),
        "upper_left": ((-1, 0), (0, 1)),
        "upper_right": ((1, 0), (0, 1)),
    }
    return mapping[corner]


def _dedupe_points(points: Iterable[tuple[float, float]], *, decimals: int = 6) -> list[tuple[float, float]]:
    seen: set[tuple[float, float]] = set()
    unique: list[tuple[float, float]] = []
    for x, y in points:
        key = (round(float(x), decimals), round(float(y), decimals))
        if key in seen:
            continue
        seen.add(key)
        unique.append((float(x), float(y)))
    return unique


def _boundary_points(polygon: Polygon) -> list[tuple[float, float]]:
    points = list(polygon.exterior.coords[:-1])
    for ring in polygon.interiors:
        points.extend(ring.coords[:-1])
    return _dedupe_points((float(x), float(y)) for x, y in points)


def _bounds_corners(bounds: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    min_x, min_y, max_x, max_y = bounds
    return [(min_x, min_y), (max_x, min_y), (min_x, max_y), (max_x, max_y)]


def _sample_points(points: Sequence[tuple[float, float]], limit: int) -> list[tuple[float, float]]:
    if limit <= 0:
        return []
    unique = _dedupe_points(points)
    if len(unique) <= limit:
        return unique

    candidate_lists = [
        sorted(unique, key=lambda pt: (pt[0] + pt[1], pt[0], pt[1])),
        sorted(unique, key=lambda pt: (pt[0] - pt[1], pt[0], pt[1])),
        sorted(unique, key=lambda pt: (-pt[0] + pt[1], -pt[0], pt[1])),
        sorted(unique, key=lambda pt: (-(pt[0] + pt[1]), -pt[0], -pt[1])),
    ]

    chosen: list[tuple[float, float]] = []
    chosen_keys: set[tuple[float, float]] = set()
    for directional_points in zip(*candidate_lists):
        for point in directional_points:
            key = (round(point[0], 6), round(point[1], 6))
            if key in chosen_keys:
                continue
            chosen.append(point)
            chosen_keys.add(key)
            if len(chosen) >= limit:
                return chosen

    stride = max(1, len(unique) // max(1, limit - len(chosen)))
    for idx in range(0, len(unique), stride):
        point = unique[idx]
        key = (round(point[0], 6), round(point[1], 6))
        if key in chosen_keys:
            continue
        chosen.append(point)
        chosen_keys.add(key)
        if len(chosen) >= limit:
            break
    return chosen


def _localize_polygon(polygon: Polygon) -> Polygon:
    centroid = polygon.centroid
    localized = shapely_translate(polygon, xoff=-centroid.x, yoff=-centroid.y)
    return orient(localized, sign=1.0)


def _allowed_angles(widget: WidgetSpec, config: SolverConfig) -> tuple[float, ...]:
    if widget.allowed_angles_degrees:
        raw = widget.allowed_angles_degrees
    else:
        step = widget.rotation_step_degrees or config.rotation_step_degrees
        count = max(1, int(math.ceil(360.0 / step)))
        raw = tuple(idx * step for idx in range(count))

    deduped: list[float] = []
    seen: set[float] = set()
    for angle in raw:
        canonical = round(float(angle) % 360.0, 6)
        if canonical in seen:
            continue
        seen.add(canonical)
        deduped.append(canonical)
    return tuple(deduped)


def _free_rotation(widget: WidgetSpec, config: SolverConfig) -> bool:
    # Explicit manufacturing/orientation constraints always take precedence.
    return (config.rotation_mode == "free" and not widget.allowed_angles_degrees
            and widget.rotation_step_degrees is None)


def _edge_directions(polygon, limit=3):
    edges = []
    for ring in (polygon.exterior, *polygon.interiors):
        points = list(ring.coords)
        for (x, y), (u, v) in zip(points, points[1:]):
            edges.append(((u-x)**2 + (v-y)**2, math.degrees(math.atan2(v-y, u-x)) % 180))
    directions = []
    for _, angle in sorted(edges, reverse=True):
        if not any(abs((angle-a+90) % 180-90) < 1e-7 for a in directions):
            directions.append(angle)
        if len(directions) == limit:
            break
    return directions


def _free_seed_angles(polygon, boards):
    # Geometry-derived angles are real-valued, not snapped to an angular grid.
    # Bound their count independently of outline complexity and instance count.
    targets = [0., 90.]
    for board in boards[:4]:
        targets.extend(_edge_directions(board.polygon, 2))
    aligned = [(target-edge+turn) % 360 for edge in _edge_directions(polygon)
               for target in targets for turn in (0, 180)]
    return tuple(dict.fromkeys(round(a, 9) for a in [0., 90., 180., 270., *aligned, *range(0, 360, 30)]))


def _board_fit_angles(polygon, boards, variants, tolerance):
    """Refine empty-sheet feasibility when no seed orientation fits its bounds.

    Only exceptional tight fits pay this cost. Golden-section searches minimize
    bounding-box overflow within bounded angular intervals without constructing
    hundreds of Shapely polygons. This remains a heuristic for irregular sheets.
    """
    points = list(polygon.convex_hull.exterior.coords)[:-1]
    result = []
    for board in boards:
        x0, y0, x1, y1 = board.bounds
        width, height = x1-x0, y1-y0
        if polygon.area > board.area+tolerance or any(
                v.bounds[2]-v.bounds[0] <= width+tolerance and
                v.bounds[3]-v.bounds[1] <= height+tolerance for v in variants):
            continue
        def overflow(angle):
            c, t = math.cos(math.radians(angle)), math.sin(math.radians(angle))
            xs = [x*c-y*t for x, y in points]
            ys = [x*t+y*c for x, y in points]
            return max((max(xs)-min(xs))/width, (max(ys)-min(ys))/height)
        ratio = (math.sqrt(5)-1)/2
        for start in range(0, 180, 30):
            lo, hi = float(start), float(start+30)
            a, b = hi-ratio*(hi-lo), lo+ratio*(hi-lo)
            fa, fb = overflow(a), overflow(b)
            for _ in range(36):
                if fa < fb:
                    hi, b, fb = b, a, fa
                    a = hi-ratio*(hi-lo); fa = overflow(a)
                else:
                    lo, a, fa = a, b, fb
                    b = lo+ratio*(hi-lo); fb = overflow(b)
            angle = a if fa < fb else b
            if overflow(angle) <= 1+tolerance/max(width, height):
                result.extend((angle, angle+180))
    return result


def _rotation_variant(polygon, angle, config):
    rotated = shapely_rotate(polygon, angle, origin=(0., 0.))
    return RotationVariant(angle, rotated, rotated.bounds, config.max_item_anchor_points)


def _build_runtime(problem: ProblemSpec, config: SolverConfig) -> tuple[list[BoardRuntime], dict[str, ItemRuntime]]:
    boards: list[BoardRuntime] = []
    for board in problem.boards:
        polygon = board.polygon.to_polygon(name=f"board:{board.board_id}")
        boards.append(
            BoardRuntime(
                board_id=board.board_id,
                polygon=polygon,
                buffered_polygon=polygon.buffer(config.placement_tolerance),
                area=float(polygon.area),
                bounds=polygon.bounds,
            )
        )

    items: dict[str, ItemRuntime] = {}
    for widget in problem.widgets:
        polygon = widget.polygon.to_polygon(name=f"widget:{widget.widget_id}")
        local_polygon = _localize_polygon(polygon)
        min_x, min_y, max_x, max_y = local_polygon.bounds
        bbox_area = max(0.0, max_x - min_x) * max(0.0, max_y - min_y)
        hole_area = sum(Polygon(ring).area for ring in widget.polygon.holes)
        free_rotation = _free_rotation(widget, config)
        angles = _free_seed_angles(local_polygon, boards) if free_rotation else _allowed_angles(widget, config)
        rotation_variants = tuple(_rotation_variant(local_polygon, a, config) for a in angles)
        if free_rotation:
            fit_angles = _board_fit_angles(local_polygon, boards, rotation_variants, config.placement_tolerance)
            rotation_variants += tuple(_rotation_variant(local_polygon, a, config) for a in fit_angles)
        for index in range(widget.quantity):
            item_id = f"{widget.widget_id}#{index + 1}"
            items[item_id] = ItemRuntime(
                instance_id=item_id,
                widget_id=widget.widget_id,
                polygon_local=local_polygon,
                area=float(local_polygon.area),
                hole_area=float(hole_area),
                bbox_area=float(bbox_area),
                rotation_variants=rotation_variants,
                free_rotation=free_rotation,
            )
    return boards, items


def _board_rest_rectangles(board: BoardRuntime, occupied: Any) -> list[dict[str, Any]]:
    if occupied.is_empty:
        if board.polygon.equals(box(*board.bounds)):
            return [{"kind": "board_free", "area": board.area, "bounds": list(board.bounds)}]
        return []  # No proven rectangle for an irregular untouched board.

    board_min_x, board_min_y, board_max_x, board_max_y = board.bounds
    occ_min_x, occ_min_y, occ_max_x, occ_max_y = occupied.bounds
    occ_min_x = max(board_min_x, occ_min_x)
    occ_min_y = max(board_min_y, occ_min_y)
    occ_max_x = min(board_max_x, occ_max_x)
    occ_max_y = min(board_max_y, occ_max_y)

    raw_rectangles = [
        ("left_strip", (board_min_x, board_min_y, occ_min_x, board_max_y)),
        ("right_strip", (occ_max_x, board_min_y, board_max_x, board_max_y)),
        ("bottom_strip", (board_min_x, board_min_y, board_max_x, occ_min_y)),
        ("top_strip", (board_min_x, occ_max_y, board_max_x, board_max_y)),
        ("lower_left", (board_min_x, board_min_y, occ_min_x, occ_min_y)),
        ("lower_right", (occ_max_x, board_min_y, board_max_x, occ_min_y)),
        ("upper_left", (board_min_x, occ_max_y, occ_min_x, board_max_y)),
        ("upper_right", (occ_max_x, occ_max_y, board_max_x, board_max_y)),
    ]

    results: list[dict[str, Any]] = []
    for kind, bounds in raw_rectangles:
        area = _bounds_area(bounds)
        if area <= 1e-9:
            continue
        rect = box(*bounds)
        if board.polygon.covers(rect):
            results.append({"kind": kind, "area": float(area), "bounds": list(bounds)})
    return results


def _board_metrics(board: BoardRuntime, board_state: BoardState) -> dict[str, Any]:
    rest_rectangles = _board_rest_rectangles(board, board_state.occupied)
    max_rest = max((entry["area"] for entry in rest_rectangles), default=0.0)
    sum_rest = sum(entry["area"] for entry in rest_rectangles)
    occupied_area = sum(placement.area for placement in board_state.placements)
    return {
        "board_id": board.board_id,
        "board_area": board.area,
        "used_area": occupied_area,
        "utilization": 0.0 if board.area <= 1e-9 else occupied_area / board.area,
        "placement_count": len(board_state.placements),
        "occupied_bounds": [] if board_state.occupied.is_empty else list(board_state.occupied.bounds),
        "max_rest_rectangle_area": max_rest,
        "sum_rest_rectangle_area": sum_rest,
        "rest_rectangles": rest_rectangles,
    }


def _state_rank_key(state: LayoutState, boards: Sequence[BoardRuntime]) -> tuple[float, ...]:
    metrics = [_board_metrics(board, board_state) for board, board_state in zip(boards, state.board_states)]
    max_rest = max((entry["max_rest_rectangle_area"] for entry in metrics), default=0.0)
    sum_rest = sum(entry["sum_rest_rectangle_area"] for entry in metrics)
    used_boards = sum(1 for board_state in state.board_states if board_state.placements)
    occupied_bbox_area = sum(_bounds_area(board_state.occupied.bounds) for board_state in state.board_states if not board_state.occupied.is_empty)
    return (
        round(state.placed_area, 6),
        round(max_rest, 6),
        round(sum_rest, 6),
        -float(used_boards),
        -round(occupied_bbox_area, 6),
        -round(state.skipped_area, 6),
        float(state.placed_count),
        -float(len(state.skipped_item_ids)),
    )


def _state_signature(state: LayoutState) -> tuple[Any, ...]:
    placements = tuple(
        (
            placement.item_instance_id,
            placement.board_id,
            round(placement.rotation_degrees, 3),
            round(placement.centroid_x, 3),
            round(placement.centroid_y, 3),
        )
        for placement in state.placed
    )
    return placements + (("__skipped__",) + tuple(state.skipped_item_ids),)


def _trim_beam(states: Sequence[LayoutState], boards: Sequence[BoardRuntime], limit: int) -> list[LayoutState]:
    unique: dict[tuple[Any, ...], LayoutState] = {}
    for state in states:
        signature = _state_signature(state)
        prior = unique.get(signature)
        if prior is None or _state_rank_key(state, boards) > _state_rank_key(prior, boards):
            unique[signature] = state
    ranked = sorted(unique.values(), key=lambda state: _state_rank_key(state, boards), reverse=True)
    return ranked[:limit]


def _initial_state(boards: Sequence[BoardRuntime]) -> LayoutState:
    return LayoutState(board_states=tuple(BoardState(free_space=b.polygon) for b in boards))


def _board_free_space(board: BoardRuntime, board_state: BoardState) -> Any:
    if board_state.free_space is not None:
        return board_state.free_space
    if board_state.occupied.is_empty:
        return board.polygon
    return board.polygon.difference(board_state.occupied)


def _is_valid_placement(geometry: Polygon, board: BoardRuntime, occupied: Any, tolerance: float) -> bool:
    if not board.buffered_polygon.covers(geometry):
        return False
    if occupied.is_empty:
        return True
    return geometry.intersection(occupied).area <= tolerance


def _max_shift(
    geometry: Polygon,
    direction: tuple[int, int],
    *,
    board: BoardRuntime,
    occupied: Any,
    tolerance: float,
) -> Polygon:
    dx, dy = direction
    if dx == 0 and dy == 0:
        return geometry

    board_min_x, board_min_y, board_max_x, board_max_y = board.bounds
    max_dimension = max(board_max_x - board_min_x, board_max_y - board_min_y, 1.0)
    low = 0.0
    high = 1.0

    while high <= max_dimension * 2.0:
        shifted = shapely_translate(geometry, xoff=dx * high, yoff=dy * high)
        if _is_valid_placement(shifted, board, occupied, tolerance):
            low = high
            high *= 2.0
        else:
            break

    for _ in range(18):
        mid = (low + high) / 2.0
        shifted = shapely_translate(geometry, xoff=dx * mid, yoff=dy * mid)
        if _is_valid_placement(shifted, board, occupied, tolerance):
            low = mid
        else:
            high = mid
    return shapely_translate(geometry, xoff=dx * low, yoff=dy * low)


def _compact_geometry(
    geometry: Polygon,
    *,
    board: BoardRuntime,
    occupied: Any,
    corner: str,
    passes: int,
    tolerance: float,
) -> Polygon:
    compacted = geometry
    axis_a, axis_b = _movement_axes(corner)
    for _ in range(max(1, passes)):
        compacted = _max_shift(compacted, axis_a, board=board, occupied=occupied, tolerance=tolerance)
        compacted = _max_shift(compacted, axis_b, board=board, occupied=occupied, tolerance=tolerance)
    return orient(compacted, sign=1.0)


def _item_anchor_points(geometry: Polygon, limit: int) -> list[tuple[float, float]]:
    boundary = _sample_points(_boundary_points(geometry), max(0, limit - 6))
    points = list(boundary)
    points.extend(_bounds_corners(geometry.bounds))
    centroid = geometry.centroid
    points.append((centroid.x, centroid.y))
    points.append((geometry.representative_point().x, geometry.representative_point().y))
    return _sample_points(points, limit)


def _free_space_anchor_points(geometry: Polygon, limit: int) -> list[tuple[float, float]]:
    boundary = _sample_points(_boundary_points(geometry), max(0, limit - 5))
    points = list(boundary)
    points.extend(_bounds_corners(geometry.bounds))
    rep = geometry.representative_point()
    points.append((rep.x, rep.y))
    return _sample_points(points, limit)


def _candidate_key(board_id: str, angle: float, geometry: Polygon) -> tuple[Any, ...]:
    centroid = geometry.centroid
    bounds = geometry.bounds
    return (
        board_id,
        round(angle, 3),
        round(centroid.x, 3),
        round(centroid.y, 3),
        round(bounds[0], 3),
        round(bounds[1], 3),
        round(bounds[2], 3),
        round(bounds[3], 3),
    )


def _placement_rank_key(board: BoardRuntime, occupied_after: Any, geometry: Polygon) -> tuple[float, ...]:
    rest_rectangles = _board_rest_rectangles(board, occupied_after)
    max_rest = max((entry["area"] for entry in rest_rectangles), default=0.0)
    sum_rest = sum(entry["area"] for entry in rest_rectangles)
    occupied_bbox_area = _bounds_area(occupied_after.bounds)
    centroid = geometry.centroid
    best_corner_distance = min(
        math.dist((centroid.x, centroid.y), _corner_xy(board.bounds, corner))
        for corner in ALL_CORNERS
    )
    return (
        round(max_rest, 6),
        round(sum_rest, 6),
        -round(occupied_bbox_area, 6),
        -round(best_corner_distance, 6),
    )


def _rectangular_candidate_rank(board_bounds, occupied_bounds, bounds, centroid):
    """Same placement score on a rectangle, without constructing a union.

    The score depends only on occupied bounds and the moving centroid. This
    allows exact score pruning before expensive polygon translations/tests.
    """
    x0, y0, x1, y1 = board_bounds
    a, b, c, d = bounds
    if occupied_bounds is not None:
        u, v, w, z = occupied_bounds
        a, b, c, d = min(a, u), min(b, v), max(c, w), max(d, z)
    bbox_area = (c-a)*(d-b)
    left, right = max(0., a-x0), max(0., x1-c)
    bottom, top = max(0., b-y0), max(0., y1-d)
    width, height = x1-x0, y1-y0
    areas = (left*height, right*height, bottom*width, top*width,
             left*bottom, right*bottom, left*top, right*top)
    areas = [area for area in areas if area > 1e-9]
    distance = min(math.hypot(centroid[0]-x, centroid[1]-y)
                   for x in (x0, x1) for y in (y0, y1))
    return (round(max(areas, default=0.), 6), round(sum(areas), 6),
            -round(bbox_area, 6), -round(distance, 6))


def _build_placement(
    item: ItemRuntime,
    board: BoardRuntime,
    board_index: int,
    angle: float,
    geometry: Polygon,
) -> Placement:
    centroid = geometry.centroid
    return Placement(
        item_instance_id=item.instance_id,
        widget_id=item.widget_id,
        board_id=board.board_id,
        board_index=board_index,
        rotation_degrees=angle,
        polygon=orient(geometry, sign=1.0),
        centroid_x=float(centroid.x),
        centroid_y=float(centroid.y),
        polygon_local=next((v.polygon for v in item.rotation_variants if v.angle_degrees == angle), None),
    )


def _candidate_geometries(
    rotated_item: Polygon,
    item_anchor_points: Sequence[tuple[float, float]],
    free_component: Polygon,
    *,
    max_free_space_anchor_points: int,
) -> list[Polygon]:
    return [shapely_translate(rotated_item, xoff=x, yoff=y) for x, y in
            _candidate_offsets(rotated_item.bounds, item_anchor_points, free_component,
                               max_free_space_anchor_points)]


def _candidate_offsets(bounds, anchors, component, limit):
    free_points = _free_space_anchor_points(component, limit=limit)
    return [(tx-ax, ty-ay) for tx, ty in free_points for ax, ay in anchors] + [
        (fx-ix, fy-iy) for (fx, fy), (ix, iy) in
        zip(_bounds_corners(component.bounds), _bounds_corners(bounds))]


def _expired(deadline: float | None) -> bool:
    return deadline is not None and time.perf_counter() >= deadline


def _find_contact_candidates(
    item: ItemRuntime,
    state: LayoutState,
    boards: Sequence[BoardRuntime],
    config: SolverConfig,
    deadline: float | None = None,
) -> list[Placement]:
    candidates: list[tuple[tuple[float, ...], Placement]] = []
    seen: set[tuple[Any, ...]] = set()

    for board_index, board in enumerate(boards):
        board_state = state.board_states[board_index]
        free_space = _board_free_space(board, board_state)
        for free_component in _iter_polygons(free_space):
            if free_component.area + config.placement_tolerance < item.area:
                continue

            comp_min_x, comp_min_y, comp_max_x, comp_max_y = free_component.bounds
            comp_width = comp_max_x - comp_min_x
            comp_height = comp_max_y - comp_min_y

            for variant in item.rotation_variants:
                rotated = variant.polygon
                angle = variant.angle_degrees
                rot_min_x, rot_min_y, rot_max_x, rot_max_y = variant.bounds
                if (rot_max_x - rot_min_x) > comp_width + config.placement_tolerance or (
                    rot_max_y - rot_min_y
                ) > comp_height + config.placement_tolerance:
                    continue

                for seed_geometry in _candidate_geometries(
                    rotated,
                    variant.anchor_points,
                    free_component,
                    max_free_space_anchor_points=config.max_free_space_anchor_points,
                ):
                    for corner in config.preferred_corners:
                        if _expired(deadline):
                            candidates.sort(key=lambda entry: entry[0], reverse=True)
                            return [p for _, p in candidates[:config.max_candidates_per_item]]
                        geometry = _compact_geometry(
                            seed_geometry,
                            board=board,
                            occupied=board_state.occupied,
                            corner=corner,
                            passes=config.compaction_passes,
                            tolerance=config.placement_tolerance,
                        )
                        if not _is_valid_placement(geometry, board, board_state.occupied, config.placement_tolerance):
                            continue
                        key = _candidate_key(board.board_id, angle, geometry)
                        if key in seen:
                            continue
                        seen.add(key)
                        occupied_after = geometry if board_state.occupied.is_empty else board_state.occupied.union(geometry)
                        placement = _build_placement(item, board, board_index, angle, geometry)
                        candidates.append((_placement_rank_key(board, occupied_after, geometry), placement))

    candidates.sort(key=lambda entry: entry[0], reverse=True)
    return [placement for _, placement in candidates[: config.max_candidates_per_item]]


def _find_item_candidates(item, state, boards, config, deadline=None):
    candidates = _find_angle_candidates(item, state, boards, config, deadline)
    if not item.free_rotation or not candidates or _expired(deadline):
        return candidates
    # Bounded local angular pattern search; translations are re-solved at each
    # angle, so refinement can move around contacts rather than just spin in place.
    def rank(p):
        occupied = state.board_states[p.board_index].occupied
        return _placement_rank_key(boards[p.board_index], occupied.union(p.polygon), p.polygon)
    best = max(candidates, key=rank)
    best_rank = rank(best)
    seen = {round(v.angle_degrees % 360, 9) for v in item.rotation_variants}
    for step in (7.5, 3.75, 1.875, .9375, .46875):
        if _expired(deadline):
            break
        angles = [(best.rotation_degrees + sign*step) % 360 for sign in (-1, 1)]
        angles = [a for a in angles if round(a, 9) not in seen]
        seen.update(round(a, 9) for a in angles)
        variants = tuple(_rotation_variant(item.polygon_local, a, config) for a in angles)
        refined = _find_angle_candidates(replace(item, rotation_variants=variants), state, boards, config, deadline)
        candidates.extend(refined)
        if refined:
            trial = max(refined, key=rank)
            trial_rank = rank(trial)
            if trial_rank > best_rank:
                best, best_rank = trial, trial_rank
    candidates.sort(key=rank, reverse=True)
    return candidates[:config.max_candidates_per_item]


def _find_angle_candidates(item, state, boards, config, deadline=None):
    if config.candidate_mode == "contact":
        return _find_contact_candidates(item, state, boards, config, deadline)
    try:
        from .nfp import translation_candidates
    except ImportError:  # script mode
        from nfp import translation_candidates
    candidates, seen = [], set()
    for board_index, board in enumerate(boards):
        bs = state.board_states[board_index]
        rectangular = board.polygon.equals(box(*board.bounds))
        occupied_bounds = None if bs.occupied.is_empty else bs.occupied.bounds
        free_components = tuple(_iter_polygons(_board_free_space(board, bs)))
        fixed = [(p.polygon_local if p.polygon_local is not None else
                  shapely_translate(p.polygon, xoff=-p.centroid_x, yoff=-p.centroid_y),
                  p.centroid_x, p.centroid_y) for p in bs.placements]
        for variant in item.rotation_variants:
            if _expired(deadline):
                candidates.sort(key=lambda entry: entry[0], reverse=True)
                return [p for _, p in candidates[:config.max_candidates_per_item]]
            points = translation_candidates(board.polygon, variant.polygon, fixed, config.placement_tolerance)
            # Contact fallback retains exact hole fits lost by closed-set polygon
            # operations at a zero-area inner-fit region.
            for component in free_components:
                points.extend(_candidate_offsets(variant.bounds, variant.anchor_points, component,
                                                 config.max_free_space_anchor_points))
            points = _dedupe_points(points, decimals=9)
            if rectangular:
                a, b, c, d = variant.bounds
                centroid = variant.polygon.centroid
                cx, cy = centroid.x, centroid.y
                ranked = [(_rectangular_candidate_rank(board.bounds, occupied_bounds,
                           (a+x, b+y, c+x, d+y), (cx+x, cy+y)), x, y) for x, y in points]
                ranked.sort(key=lambda entry: entry[0], reverse=True)
            else:
                ranked = [(None, x, y) for x, y in points]
            for fast_rank, x, y in ranked:
                if _expired(deadline):
                    break
                if fast_rank is not None and len(candidates) >= config.max_candidates_per_item:
                    if fast_rank <= candidates[-1][0]:
                        break
                geometry = shapely_translate(variant.polygon, xoff=x, yoff=y)
                if not _is_valid_placement(geometry, board, bs.occupied, config.placement_tolerance):
                    continue
                key = _candidate_key(board.board_id, variant.angle_degrees, geometry)
                if key in seen:
                    continue
                seen.add(key)
                placement = _build_placement(item, board, board_index, variant.angle_degrees, geometry)
                if fast_rank is None:
                    occupied = geometry if bs.occupied.is_empty else bs.occupied.union(geometry)
                    fast_rank = _placement_rank_key(board, occupied, geometry)
                candidates.append((fast_rank, placement))
                candidates.sort(key=lambda entry: entry[0], reverse=True)
                del candidates[config.max_candidates_per_item:]
    candidates.sort(key=lambda entry: entry[0], reverse=True)
    return [p for _, p in candidates[:config.max_candidates_per_item]]


def _apply_placement(state: LayoutState, placement: Placement) -> LayoutState:
    board_states = list(state.board_states)
    board_state = board_states[placement.board_index]
    occupied = placement.polygon if board_state.occupied.is_empty else board_state.occupied.union(placement.polygon)
    prior_free_space = board_state.free_space
    free_space = None if prior_free_space is None else prior_free_space.difference(placement.polygon)
    board_states[placement.board_index] = BoardState(
        occupied=occupied,
        free_space=free_space,
        placements=board_state.placements + (placement,),
    )
    return LayoutState(
        board_states=tuple(board_states),
        placed=state.placed + (placement,),
        skipped_item_ids=state.skipped_item_ids,
        placed_area=state.placed_area + placement.area,
        skipped_area=state.skipped_area,
    )


def _skip_item(state: LayoutState, item: ItemRuntime) -> LayoutState:
    return LayoutState(
        board_states=state.board_states,
        placed=state.placed,
        skipped_item_ids=state.skipped_item_ids + (item.instance_id,),
        placed_area=state.placed_area,
        skipped_area=state.skipped_area + item.area,
    )


def _evaluate_order(
    order: tuple[str, ...],
    *,
    items: dict[str, ItemRuntime],
    boards: Sequence[BoardRuntime],
    config: SolverConfig,
    deadline: float | None = None,
    initial_state: LayoutState | None = None,
) -> tuple[LayoutState, dict[str, int]]:
    beam = [initial_state or _initial_state(boards)]
    stats = {"retained_candidates": 0, "states_expanded": 0}

    for index, item_id in enumerate(order):
        if _expired(deadline):
            for remaining in order[index:]:
                beam = [_skip_item(state, items[remaining]) for state in beam]
            break
        item = items[item_id]
        next_states: list[LayoutState] = []
        for state in beam:
            stats["states_expanded"] += 1
            candidates = _find_item_candidates(item, state, boards, config, deadline)
            stats["retained_candidates"] += len(candidates)
            for placement in candidates:
                next_states.append(_apply_placement(state, placement))
            next_states.append(_skip_item(state, item))
        beam = _trim_beam(next_states, boards, config.beam_width)
        if not beam:
            beam = [_initial_state(boards)]
            break

    best = max(beam, key=lambda state: _state_rank_key(state, boards))
    return best, stats


def _ordered_crossover(parent_a: tuple[str, ...], parent_b: tuple[str, ...], rng: random.Random) -> tuple[str, ...]:
    if len(parent_a) < 2:
        return parent_a
    left, right = sorted(rng.sample(range(len(parent_a)), 2))
    child: list[str | None] = [None] * len(parent_a)
    child[left : right + 1] = parent_a[left : right + 1]
    fill = [item for item in parent_b if item not in child]
    fill_index = 0
    for index, value in enumerate(child):
        if value is None:
            child[index] = fill[fill_index]
            fill_index += 1
    return tuple(value for value in child if value is not None)


def _mutate(order: tuple[str, ...], rng: random.Random, rate: float) -> tuple[str, ...]:
    values = list(order)
    if len(values) < 2:
        return order
    if rng.random() <= rate:
        i, j = rng.sample(range(len(values)), 2)
        values[i], values[j] = values[j], values[i]
    if len(values) >= 4 and rng.random() <= rate / 2.0:
        start, end = sorted(rng.sample(range(len(values)), 2))
        values[start : end + 1] = reversed(values[start : end + 1])
    return tuple(values)


def _canonical_order(order, items):
    counts = {}
    result = []
    for item_id in order:
        widget_id = items[item_id].widget_id
        counts[widget_id] = counts.get(widget_id, 0) + 1
        result.append(f"{widget_id}#{counts[widget_id]}")
    return tuple(result)


def _build_initial_population(item_ids: list[str], items: dict[str, ItemRuntime], config: SolverConfig) -> list[tuple[str, ...]]:
    area_desc = tuple(sorted(item_ids, key=lambda item_id: (items[item_id].area, items[item_id].hole_area), reverse=True))
    area_asc = tuple(sorted(item_ids, key=lambda item_id: (items[item_id].area, items[item_id].hole_area)))
    hole_desc = tuple(sorted(item_ids, key=lambda item_id: (items[item_id].hole_area, items[item_id].area), reverse=True))
    compact_first = tuple(
        sorted(
            item_ids,
            key=lambda item_id: (
                items[item_id].area / max(items[item_id].bbox_area, 1e-9),
                items[item_id].area,
            ),
            reverse=True,
        )
    )

    population = list(dict.fromkeys(_canonical_order(o, items) for o in
                                    (area_desc, area_asc, hole_desc, compact_first)))
    rng = random.Random(config.seed)
    for _ in range(config.population_size * 20):
        if len(population) >= config.population_size:
            break
        shuffled = item_ids[:]
        rng.shuffle(shuffled)
        order = _canonical_order(tuple(shuffled), items)
        if order not in population:
            population.append(order)
    return population[: config.population_size]


def _state_from_placements(placements, boards, items):
    """Assemble a constructive incumbent without repeated global polygon unions."""
    from shapely.ops import unary_union
    states = []
    for b in boards:
        placed = tuple(p for p in placements if p.board_id == b.board_id)
        occupied = unary_union([p.polygon for p in placed])
        states.append(BoardState(occupied, b.polygon.difference(occupied), placed))
    return LayoutState(tuple(states), tuple(placements), (), sum(p.area for p in placements), 0.)


def _refine_placed_rotations(state, boards, items, config, deadline):
    """Remove/reinsert boundary pieces, retaining every incumbent on failure.

    Unlike restarting from an empty sheet, this spends the angular budget on
    improving the already complete scene. Never accept a loss of placed area.
    """
    def rank_state(candidate):
        # Equal outer bounds need not freeze the interior: moving parts toward
        # the lower-left corner can unlock later boundary compaction.
        potential = sum(p.centroid_x/(boards[p.board_index].bounds[2]-boards[p.board_index].bounds[0]) +
                        p.centroid_y/(boards[p.board_index].bounds[3]-boards[p.board_index].bounds[1])
                        for p in candidate.placed)
        return (*_state_rank_key(candidate, boards), -round(potential, 9))
    rank = rank_state(state)
    improved = 0
    order = sorted(state.placed, key=lambda p: p.polygon.bounds[2], reverse=True)
    for old in order:
        if _expired(deadline):
            break
        item = items[old.item_instance_id]
        if not item.free_rotation:
            continue
        if all(abs((v.angle_degrees-old.rotation_degrees+180)%360-180) > 1e-8
               for v in item.rotation_variants):
            item = replace(item, rotation_variants=(
                _rotation_variant(item.polygon_local, old.rotation_degrees, config), *item.rotation_variants))
        fixed = [p for p in state.placed if p.item_instance_id != old.item_instance_id]
        base = _state_from_placements(fixed, boards, items)
        base = replace(base, skipped_item_ids=state.skipped_item_ids, skipped_area=state.skipped_area)
        candidates = _find_item_candidates(item, base, boards, config, deadline)
        best, best_rank = state, rank
        for placement in candidates:
            trial = _apply_placement(base, placement)
            trial_rank = rank_state(trial)
            if trial_rank > best_rank:
                best, best_rank = trial, trial_rank
        if best is not state:
            state, rank = best, best_rank
            improved += 1
    return state, improved


def _compact_angle_items(items):
    # Keep the four best angular envelopes, including arbitrary edge angles.
    # This is an alternative construction, never a forced rotation of a layout.
    return {k: replace(i, rotation_variants=tuple(sorted(i.rotation_variants,
                    key=lambda v: _bounds_area(v.bounds))[:4])) if i.free_rotation else i
            for k, i in items.items()}


def _strip_envelope_layouts(boards, pools, deadline, tolerance):
    """Try narrower working sheets to preserve a reusable edge strip."""
    try:
        from .constructive import envelope_layouts
    except ImportError:  # script mode
        from constructive import envelope_layouts
    if not all(b.polygon.equals(box(*b.bounds)) for b in boards):
        return
    deadline = min(deadline, time.perf_counter()+.15) if deadline else time.perf_counter()+.15
    for fraction in (.85, .7, .55, .4):
        for axis in (0, 1):
            if _expired(deadline):
                return
            working = []
            for board in boards:
                a, b, c, d = board.bounds
                bounds = (a, b, a+(c-a)*fraction if axis == 0 else c,
                          b+(d-b)*fraction if axis == 1 else d)
                polygon = box(*bounds)
                working.append(replace(board, polygon=polygon, bounds=bounds,
                                       area=polygon.area, buffered_polygon=polygon.buffer(tolerance)))
            for pool in pools:
                yield from envelope_layouts(working, pool, deadline, tolerance)


def solve_problem(problem: ProblemSpec, config: SolverConfig | None = None, *,
                  time_limit_seconds: float | None = None) -> SolutionResult:
    """Return a feasible incumbent at a cooperative deadline, if supplied.

    An individual GEOS operation cannot be interrupted; elapsed time can exceed
    the limit. Callers needing a hard limit must also isolate the process.
    """
    started = time.perf_counter()
    if time_limit_seconds is not None and (not math.isfinite(time_limit_seconds) or time_limit_seconds <= 0):
        raise ValueError("time_limit_seconds must be finite and positive")
    deadline = None if time_limit_seconds is None else started + time_limit_seconds
    effective_config = config or SolverConfig.from_problem(problem)
    boards, items = _build_runtime(problem, effective_config)
    item_ids = list(items)
    population = _build_initial_population(item_ids, items, effective_config)
    rng = random.Random(effective_config.seed)
    cache: dict[tuple[str, ...], tuple[LayoutState, dict[str, int]]] = {}

    best_order = population[0]
    best_state: LayoutState | None = None
    best_metrics: tuple[float, ...] | None = None
    aggregate_stats = {"orders_evaluated": 0, "retained_candidates": 0, "states_expanded": 0}

    # Build a cheap full-scene incumbent before exhaustive polygon candidates.
    # On small scenes retain the existing hole-aware search and its scoring.
    warm_full = False
    if effective_config.constructive_start and effective_config.candidate_mode == 'nfp' and (len(items) >= 32 or any(i.free_rotation for i in items.values())):
        try:
            from .constructive import envelope_layouts
        except ImportError:  # script mode
            from constructive import envelope_layouts
        # A large number of different envelopes fragments the free-rectangle
        # list; reserve most time for the bitset polygon constructor in that case.
        envelope_cap = .5 if len({i.widget_id for i in items.values()}) >= 64 else 4.
        warm_deadline = started + (min(envelope_cap, time_limit_seconds * .4) if time_limit_seconds else envelope_cap)
        # Get a cheap cardinal incumbent first; richer angular searches must
        # improve it rather than spending the entire budget on extra variants.
        quick_items = {k: replace(i, rotation_variants=tuple(
            v for v in i.rotation_variants if abs(v.angle_degrees % 90) < 1e-8))
            if i.free_rotation else i for k, i in items.items()}
        pools = (quick_items, _compact_angle_items(items), items) if any(i.free_rotation for i in items.values()) else (items,)
        attempts = 0
        layouts = chain.from_iterable(envelope_layouts(boards, pool, warm_deadline,
                                       effective_config.placement_tolerance) for pool in pools)
        if len(items) <= 32 and any(i.free_rotation for i in items.values()):
            layouts = chain(layouts, _strip_envelope_layouts(boards, pools, warm_deadline,
                                                            effective_config.placement_tolerance))
        for raw_placements in layouts:
            attempts += 1
            placements = [_build_placement(items[key], boards[bi], bi, variant.angle_degrees,
                                          shapely_translate(variant.polygon, xoff=x, yoff=y))
                          for key, bi, variant, x, y in raw_placements]
            state = _state_from_placements(placements, boards, items)
            ids = {p.item_instance_id for p in placements}
            for key in item_ids:
                if key not in ids:
                    state = _skip_item(state, items[key])
            key = _state_rank_key(state, boards)
            if best_metrics is None or key > best_metrics:
                best_state, best_metrics = state, key
                best_order = tuple(p.item_instance_id for p in state.placed) + state.skipped_item_ids
            if not state.skipped_item_ids:
                warm_full = True
                # Continue through the angular restarts even after a cardinal
                # full fit. Only the best complete objective survives.
        aggregate_stats['constructive_attempts'] = attempts
        aggregate_stats['constructive_seconds'] = time.perf_counter() - started
        aggregate_stats['constructive_placed'] = best_state.placed_count if best_state else 0
        tight_holes = (any(i.hole_area for i in items.values()) and
                       sum(i.area for i in items.values()) >= .95*sum(b.area for b in boards))
        filled_sheet = warm_full and best_state.placed_area >= sum(b.area for b in boards)-effective_config.placement_tolerance
        # If just one of many pieces is missing, preserve time for direct
        # insertion into the existing layout before a broader restart search.
        almost_full = (not warm_full and len(items) > 6 and best_state is not None
                       and best_state.placed_count == len(items)-1)
        if (1 < len(items) <= 32 and any(i.free_rotation for i in items.values())
                and not tight_holes and not filled_sheet and not almost_full and not _expired(deadline)):
            try:
                from .raster_start import raster_search
            except ImportError:  # script mode
                from raster_start import raster_search
            joint_cap = .35 if len(items) > 6 and not warm_full else .9
            joint_deadline = min(deadline, started + min(joint_cap, time_limit_seconds*.45)) if deadline else started+joint_cap
            attempts = 0
            for raw in raster_search(boards, items, joint_deadline, seed=effective_config.seed):
                attempts += 1
                placements = [_build_placement(items[k], boards[bi], bi, v.angle_degrees,
                                              shapely_translate(v.polygon, xoff=x, yoff=y))
                              for k, bi, v, x, y in raw]
                state = _state_from_placements(placements, boards, items)
                ids = {p.item_instance_id for p in placements}
                for k in item_ids:
                    if k not in ids:
                        state = _skip_item(state, items[k])
                key = _state_rank_key(state, boards)
                if best_metrics is None or key > best_metrics:
                    best_state, best_metrics = state, key
                    best_order = tuple(p.item_instance_id for p in state.placed) + state.skipped_item_ids
            aggregate_stats['joint_rotation_candidates'] = attempts
            warm_full = best_state is not None and not best_state.skipped_item_ids
        if (not warm_full or 1 < len(items) <= 32) and not _expired(deadline):
            try:
                from .raster_start import raster_layout
            except ImportError:  # script mode
                from raster_start import raster_layout
            raster_deadline = min(deadline, started + time_limit_seconds * .75) if deadline else started + 8.
            raster_attempts = [(quick_items, 512, 'area'), (items, 512, 'area'),
                               (quick_items, 512, 'long_side'), (quick_items, 512, 'bbox'),
                               (quick_items, 768, 'area'), (items, 768, 'height')]
            for pool, resolution, mode in raster_attempts:
                if _expired(raster_deadline):
                    break
                raw = raster_layout(boards, pool, raster_deadline, resolution=resolution, order_mode=mode)
                placements = [_build_placement(items[k], boards[bi], bi, v.angle_degrees,
                                              shapely_translate(v.polygon, xoff=x, yoff=y))
                              for k, bi, v, x, y in raw]
                state = _state_from_placements(placements, boards, items)
                ids = {p.item_instance_id for p in placements}
                for k in item_ids:
                    if k not in ids:
                        state = _skip_item(state, items[k])
                if best_metrics is None or _state_rank_key(state, boards) > best_metrics:
                    best_state, best_metrics = state, _state_rank_key(state, boards)
                    best_order = tuple(p.item_instance_id for p in state.placed) + state.skipped_item_ids
                aggregate_stats['raster_placed'] = best_state.placed_count
                if not best_state.skipped_item_ids:
                    warm_full = True
                    break
        if best_state and best_state.skipped_item_ids and not _expired(deadline):
            repair_order = best_state.skipped_item_ids
            initial = LayoutState(best_state.board_states, best_state.placed, (), best_state.placed_area, 0.)
            repair_deadline = min(deadline, started + time_limit_seconds * .75) if deadline else started + 6.
            repaired, stats = _evaluate_order(repair_order, items=items, boards=boards,
                                             config=effective_config, deadline=repair_deadline, initial_state=initial)
            aggregate_stats['repair_placed'] = repaired.placed_count - best_state.placed_count
            aggregate_stats['repair_states_expanded'] = stats['states_expanded']
            if _state_rank_key(repaired, boards) > best_metrics:
                best_state, best_metrics = repaired, _state_rank_key(repaired, boards)
                best_order = tuple(p.item_instance_id for p in repaired.placed) + repaired.skipped_item_ids
            warm_full = not best_state.skipped_item_ids
        if warm_full:
            aggregate_stats['termination_reason'] = 'constructive_full_fit'
            aggregate_stats['first_full_fit_seconds'] = time.perf_counter() - started

    search_deadline = deadline
    refine_small = len(items) <= 32 and any(i.free_rotation for i in items.values())

    for _generation in range(0 if warm_full else effective_config.generations):
        evaluated: list[tuple[tuple[float, ...], tuple[str, ...], LayoutState]] = []
        for order in population:
            if best_state is not None and _expired(search_deadline):
                break
            cached = cache.get(order)
            if cached is None:
                state, stats = _evaluate_order(order, items=items, boards=boards, config=effective_config, deadline=search_deadline)
                cache[order] = (state, stats)
                aggregate_stats["retained_candidates"] += stats["retained_candidates"]
                aggregate_stats["states_expanded"] += stats["states_expanded"]
            else:
                state, stats = cached
            aggregate_stats["orders_evaluated"] += 1
            key = _state_rank_key(state, boards)
            evaluated.append((key, order, state))
            if best_metrics is None or key > best_metrics:
                best_metrics = key
                best_state = state
                best_order = order

        if _expired(search_deadline) or _generation + 1 == effective_config.generations:
            break
        evaluated.sort(key=lambda entry: entry[0], reverse=True)
        elite_count = min(effective_config.elite_count, max(1, len(evaluated) // 2))
        elites = [order for _, order, _ in evaluated[: elite_count]]
        next_population = elites[:]
        parent_pool = [order for _, order, _ in evaluated[: max(effective_config.elite_count * 2, len(elites))]]
        for _ in range(effective_config.population_size * 20):
            if len(next_population) >= effective_config.population_size:
                break
            parent_a = rng.choice(parent_pool)
            parent_b = rng.choice(parent_pool)
            child = _ordered_crossover(parent_a, parent_b, rng)
            child = _canonical_order(_mutate(child, rng, effective_config.mutation_rate), items)
            if child not in next_population:
                next_population.append(child)
        population = next_population

    assert best_state is not None

    if refine_small and not _expired(deadline):
        polish_deadline = min(deadline, time.perf_counter()+.5) if deadline else time.perf_counter()+.5
        before = time.perf_counter()
        best_state, improvements = _refine_placed_rotations(best_state, boards, items, effective_config, polish_deadline)
        aggregate_stats['rotation_refinement_seconds'] = time.perf_counter()-before
        aggregate_stats['rotation_refinement_improvements'] = improvements
        best_order = tuple(p.item_instance_id for p in best_state.placed) + best_state.skipped_item_ids
        if warm_full:
            aggregate_stats['termination_reason'] = 'constructive_full_fit_with_angular_repair'


    board_metrics = tuple(_board_metrics(board, board_state) for board, board_state in zip(boards, best_state.board_states))
    max_rest = max((entry["max_rest_rectangle_area"] for entry in board_metrics), default=0.0)
    sum_rest = sum(entry["sum_rest_rectangle_area"] for entry in board_metrics)
    aggregate_stats["time_limit_seconds"] = time_limit_seconds
    aggregate_stats["budget_exhausted"] = _expired(deadline)
    aggregate_stats["solve_seconds"] = time.perf_counter() - started
    aggregate_stats["non_cardinal_placements"] = sum(abs((p.rotation_degrees+45)%90-45) > 1e-4 for p in best_state.placed)
    aggregate_stats["rotation_mode"] = effective_config.rotation_mode
    aggregate_stats["free_rotation_instances"] = sum(i.free_rotation for i in items.values())
    aggregate_stats["candidate_mode"] = effective_config.candidate_mode
    aggregate_stats["unique_orders_evaluated"] = len(cache)
    aggregate_stats["beam_width"] = effective_config.beam_width
    aggregate_stats["population_size"] = effective_config.population_size
    aggregate_stats["generations"] = effective_config.generations

    return SolutionResult(
        order=best_order,
        placements=best_state.placed,
        skipped_item_ids=best_state.skipped_item_ids,
        placed_area=best_state.placed_area,
        skipped_area=best_state.skipped_area,
        max_rest_rectangle_area=max_rest,
        sum_rest_rectangle_area=sum_rest,
        board_metrics=board_metrics,
        search_stats=aggregate_stats,
    )


def _polygon_to_json(polygon: Polygon) -> dict[str, Any]:
    return {
        "shell": [[float(x), float(y)] for x, y in polygon.exterior.coords[:-1]],
        "holes": [[[float(x), float(y)] for x, y in ring.coords[:-1]] for ring in polygon.interiors],
    }


def solution_to_dict(problem: ProblemSpec, solution: SolutionResult) -> dict[str, Any]:
    total_widget_area = sum(
        widget.polygon.to_polygon(name=f"widget:{widget.widget_id}").area * widget.quantity for widget in problem.widgets
    )
    total_board_area = sum(board.polygon.to_polygon(name=f"board:{board.board_id}").area for board in problem.boards)
    return {
        "units": problem.units,
        "score": {
            "placed_area": solution.placed_area,
            "skipped_area": solution.skipped_area,
            "placed_ratio_vs_requested": 0.0 if total_widget_area <= 1e-9 else solution.placed_area / total_widget_area,
            "board_utilization_ratio": 0.0 if total_board_area <= 1e-9 else solution.placed_area / total_board_area,
            "max_rest_rectangle_area": solution.max_rest_rectangle_area,
            "sum_rest_rectangle_area": solution.sum_rest_rectangle_area,
        },
        "placements": [
            {
                "item_instance_id": placement.item_instance_id,
                "widget_id": placement.widget_id,
                "board_id": placement.board_id,
                "rotation_degrees": placement.rotation_degrees,
                "centroid": [placement.centroid_x, placement.centroid_y],
                "polygon": _polygon_to_json(placement.polygon),
            }
            for placement in solution.placements
        ],
        "skipped_item_ids": list(solution.skipped_item_ids),
        "best_order": list(solution.order),
        "board_metrics": list(solution.board_metrics),
        "search_stats": solution.search_stats,
    }


def validate_solution(problem: ProblemSpec, solution: SolutionResult, *, tolerance: float = 1e-6, config: SolverConfig | None = None) -> None:
    boards = {
        board.board_id: board.polygon.to_polygon(name=f"board:{board.board_id}")
        for board in problem.boards
    }

    effective = config or SolverConfig.from_problem(problem)
    _, items = _build_runtime(problem, effective)
    placed_ids = [p.item_instance_id for p in solution.placements]
    all_ids = placed_ids + list(solution.skipped_item_ids)
    if len(set(all_ids)) != len(all_ids) or set(all_ids) != set(items):
        raise AssertionError("placed/skipped identities must partition requested instances")
    if not math.isclose(sum(p.area for p in solution.placements), solution.placed_area, rel_tol=0, abs_tol=tolerance):
        raise AssertionError("placed area accounting mismatch")
    if not math.isclose(sum(items[i].area for i in solution.skipped_item_ids), solution.skipped_area, rel_tol=0, abs_tol=tolerance):
        raise AssertionError("skipped area accounting mismatch")
    for p in solution.placements:
        item = items[p.item_instance_id]
        if p.widget_id != item.widget_id:
            raise AssertionError("widget identity mismatch")
        if not 0 <= p.board_index < len(problem.boards) or problem.boards[p.board_index].board_id != p.board_id:
            raise AssertionError("board identity mismatch")
        if not math.isfinite(p.rotation_degrees):
            raise AssertionError("rotation must be finite")
        if not item.free_rotation and not any(
                abs((v.angle_degrees-p.rotation_degrees+180)%360-180) < 1e-6 for v in item.rotation_variants):
            raise AssertionError("rotation is not allowed")
        expected = shapely_translate(shapely_rotate(item.polygon_local, p.rotation_degrees, origin=(0, 0)),
                                     xoff=p.centroid_x, yoff=p.centroid_y)
        if not expected.buffer(tolerance).covers(p.polygon) or not p.polygon.buffer(tolerance).covers(expected):
            raise AssertionError("widget geometry was changed")

    placements_by_board: dict[str, list[Placement]] = {board_id: [] for board_id in boards}
    for placement in solution.placements:
        placements_by_board[placement.board_id].append(placement)
        if not boards[placement.board_id].buffer(tolerance).covers(placement.polygon):
            raise AssertionError(f"{placement.item_instance_id} is outside board {placement.board_id}")

    from shapely.strtree import STRtree
    for board_id, placements in placements_by_board.items():
        polygons = [p.polygon for p in placements]
        tree = STRtree(polygons)
        for i, left in enumerate(placements):
            for j in tree.query(left.polygon):
                if j <= i:
                    continue
                right = placements[j]
                overlap_area = left.polygon.intersection(right.polygon).area
                if overlap_area > tolerance:
                    raise AssertionError(
                        f"{left.item_instance_id} overlaps {right.item_instance_id} on {board_id}: {overlap_area}"
                    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve a 2D widget nesting problem.")
    parser.add_argument("--input", required=True, help="Path to problem JSON")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Override RNG seed")
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--rotation-mode", choices=["free", "discrete"], default=None)
    parser.add_argument("--rotation-step-degrees", type=float, default=None)
    parser.add_argument("--time-limit-seconds", type=float, default=2.0)
    parser.add_argument("--candidate-mode", choices=["nfp", "contact"], default=None)
    parser.add_argument("--mujoco-debug", action="store_true", help="Render an optional MuJoCo debug view")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    problem = load_problem(args.input)
    config = SolverConfig.from_problem(
        problem,
        overrides={
            "candidate_mode": args.candidate_mode,
            "seed": args.seed,
            "beam_width": args.beam_width,
            "population_size": args.population_size,
            "generations": args.generations,
            "rotation_mode": args.rotation_mode,
            "rotation_step_degrees": args.rotation_step_degrees,
        },
    )
    solution = solve_problem(problem, config=config, time_limit_seconds=args.time_limit_seconds)
    validate_solution(problem, solution, tolerance=config.placement_tolerance, config=config)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    solution_dict = solution_to_dict(problem, solution)
    save_solution(output_dir / "solution.json", solution_dict)

    try:
        from .render import render_solution_image
    except ImportError:  # pragma: no cover - script mode
        from render import render_solution_image

    render_path = render_solution_image(problem, solution, output_dir / "nesting_layout.png")

    mujoco_debug_path = None
    if args.mujoco_debug:
        try:
            from .mujoco_debug import render_mujoco_debug
        except ImportError:  # pragma: no cover - script mode
            from mujoco_debug import render_mujoco_debug
        mujoco_debug_path = render_mujoco_debug(problem, solution, output_dir / "mujoco_debug.png")

    summary = {
        "solution_json": str(output_dir / "solution.json"),
        "render_image": str(render_path),
        "score": solution_dict["score"],
        "skipped_item_ids": solution_dict["skipped_item_ids"],
        "search_stats": solution.search_stats,
    }
    if mujoco_debug_path is not None:
        summary["mujoco_debug_image"] = str(mujoco_debug_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
