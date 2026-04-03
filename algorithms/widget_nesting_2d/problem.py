from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.polygon import orient
from shapely.validation import make_valid


def _as_point_list(value: Any, *, key: str) -> list[tuple[float, float]]:
    if not isinstance(value, list) or len(value) < 3:
        raise ValueError(f"{key} must be a list of at least three [x, y] points")

    points: list[tuple[float, float]] = []
    for idx, raw in enumerate(value):
        if (
            not isinstance(raw, list)
            or len(raw) != 2
            or not isinstance(raw[0], (int, float))
            or not isinstance(raw[1], (int, float))
        ):
            raise ValueError(f"{key}[{idx}] must be [x, y]")
        points.append((float(raw[0]), float(raw[1])))
    return points


def _normalize_polygon(geometry: Polygon, *, name: str) -> Polygon:
    if geometry.is_empty:
        raise ValueError(f"{name} resolved to an empty polygon")

    fixed = make_valid(geometry)
    if isinstance(fixed, Polygon):
        result = fixed
    elif isinstance(fixed, MultiPolygon):
        parts = [part for part in fixed.geoms if isinstance(part, Polygon) and part.area > 1e-9]
        if len(parts) != 1:
            raise ValueError(f"{name} must resolve to exactly one polygon, got {len(parts)} parts")
        result = parts[0]
    elif isinstance(fixed, GeometryCollection):
        parts = [part for part in fixed.geoms if isinstance(part, Polygon) and part.area > 1e-9]
        if len(parts) != 1:
            raise ValueError(f"{name} must resolve to exactly one polygon, got {len(parts)} polygon parts")
        result = parts[0]
    else:
        raise ValueError(f"{name} must be a polygon")

    if result.area <= 1e-9:
        raise ValueError(f"{name} area must be positive")

    return orient(result, sign=1.0)


@dataclass(frozen=True)
class PolygonSpec:
    shell: tuple[tuple[float, float], ...]
    holes: tuple[tuple[tuple[float, float], ...], ...] = ()

    @classmethod
    def from_json(cls, raw: dict[str, Any], *, key: str) -> "PolygonSpec":
        if not isinstance(raw, dict):
            raise ValueError(f"{key} must be an object")
        if "shell" not in raw:
            raise ValueError(f"{key} must include shell")
        shell = tuple(_as_point_list(raw["shell"], key=f"{key}.shell"))
        holes_raw = raw.get("holes", [])
        if holes_raw is None:
            holes_raw = []
        if not isinstance(holes_raw, list):
            raise ValueError(f"{key}.holes must be a list")
        holes = tuple(tuple(_as_point_list(hole, key=f"{key}.holes[{idx}]")) for idx, hole in enumerate(holes_raw))
        return cls(shell=shell, holes=holes)

    def to_polygon(self, *, name: str) -> Polygon:
        return _normalize_polygon(Polygon(self.shell, holes=self.holes), name=name)

    def to_json(self) -> dict[str, Any]:
        return {
            "shell": [[x, y] for x, y in self.shell],
            "holes": [[[x, y] for x, y in ring] for ring in self.holes],
        }


@dataclass(frozen=True)
class BoardSpec:
    board_id: str
    polygon: PolygonSpec

    @classmethod
    def from_json(cls, raw: dict[str, Any], *, index: int) -> "BoardSpec":
        if not isinstance(raw, dict):
            raise ValueError(f"boards[{index}] must be an object")
        board_id = raw.get("id")
        if not isinstance(board_id, str) or not board_id:
            raise ValueError(f"boards[{index}].id must be a non-empty string")
        polygon = PolygonSpec.from_json(raw.get("polygon"), key=f"boards[{index}].polygon")
        return cls(board_id=board_id, polygon=polygon)

    def to_json(self) -> dict[str, Any]:
        return {"id": self.board_id, "polygon": self.polygon.to_json()}


@dataclass(frozen=True)
class WidgetSpec:
    widget_id: str
    quantity: int
    polygon: PolygonSpec
    allowed_angles_degrees: tuple[float, ...] = ()
    rotation_step_degrees: float | None = None

    @classmethod
    def from_json(cls, raw: dict[str, Any], *, index: int) -> "WidgetSpec":
        if not isinstance(raw, dict):
            raise ValueError(f"widgets[{index}] must be an object")
        widget_id = raw.get("id")
        if not isinstance(widget_id, str) or not widget_id:
            raise ValueError(f"widgets[{index}].id must be a non-empty string")
        quantity = raw.get("quantity")
        if not isinstance(quantity, int) or quantity <= 0:
            raise ValueError(f"widgets[{index}].quantity must be a positive integer")
        polygon = PolygonSpec.from_json(raw.get("polygon"), key=f"widgets[{index}].polygon")
        allowed_angles_raw = raw.get("allowed_angles_degrees", [])
        if allowed_angles_raw is None:
            allowed_angles_raw = []
        if not isinstance(allowed_angles_raw, list):
            raise ValueError(f"widgets[{index}].allowed_angles_degrees must be a list")
        allowed_angles = tuple(float(value) for value in allowed_angles_raw)
        step_raw = raw.get("rotation_step_degrees")
        rotation_step = None if step_raw is None else float(step_raw)
        if rotation_step is not None and rotation_step <= 0:
            raise ValueError(f"widgets[{index}].rotation_step_degrees must be > 0")
        return cls(
            widget_id=widget_id,
            quantity=quantity,
            polygon=polygon,
            allowed_angles_degrees=allowed_angles,
            rotation_step_degrees=rotation_step,
        )

    def to_json(self) -> dict[str, Any]:
        data = {
            "id": self.widget_id,
            "quantity": self.quantity,
            "polygon": self.polygon.to_json(),
        }
        if self.allowed_angles_degrees:
            data["allowed_angles_degrees"] = list(self.allowed_angles_degrees)
        if self.rotation_step_degrees is not None:
            data["rotation_step_degrees"] = self.rotation_step_degrees
        return data


@dataclass(frozen=True)
class ProblemSpec:
    units: str
    boards: tuple[BoardSpec, ...]
    widgets: tuple[WidgetSpec, ...]
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "ProblemSpec":
        if not isinstance(raw, dict):
            raise ValueError("problem JSON must be an object")
        units = raw.get("units", "unit")
        if not isinstance(units, str) or not units:
            raise ValueError("units must be a non-empty string")
        boards_raw = raw.get("boards")
        widgets_raw = raw.get("widgets")
        if not isinstance(boards_raw, list) or not boards_raw:
            raise ValueError("boards must be a non-empty list")
        if not isinstance(widgets_raw, list) or not widgets_raw:
            raise ValueError("widgets must be a non-empty list")
        boards = tuple(BoardSpec.from_json(entry, index=i) for i, entry in enumerate(boards_raw))
        widgets = tuple(WidgetSpec.from_json(entry, index=i) for i, entry in enumerate(widgets_raw))
        config = raw.get("config", {})
        if not isinstance(config, dict):
            raise ValueError("config must be an object when provided")
        return cls(units=units, boards=boards, widgets=widgets, config=dict(config))

    def to_json(self) -> dict[str, Any]:
        return {
            "units": self.units,
            "boards": [board.to_json() for board in self.boards],
            "widgets": [widget.to_json() for widget in self.widgets],
            "config": self.config,
        }


def load_problem(path: str | Path) -> ProblemSpec:
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return ProblemSpec.from_json(raw)


def _round_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isfinite(value):
            return round(value, 6)
        return value
    if isinstance(value, dict):
        return {key: _round_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_round_value(val) for val in value]
    return value


def save_solution(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(_round_value(payload), fh, indent=2, sort_keys=True)
        fh.write("\n")
    return output_path
