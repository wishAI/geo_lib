from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BENCHMARK_ROOT = Path(__file__).resolve().parent / "inputs" / "public_benchmarks"
JAGUA_ASSET_BASE = "https://github.com/JeroenGar/jagua-rs/blob/main/assets"


@dataclass(frozen=True)
class CaseDefinition:
    case_id: str
    description: str
    problem: dict[str, Any]


def _load_benchmark(name: str) -> dict[str, Any]:
    path = BENCHMARK_ROOT / name
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _normalize_ring(points: list[list[float]], *, scale: float = 1.0) -> list[list[float]]:
    if len(points) >= 2 and points[0] == points[-1]:
        points = points[:-1]
    return [[round(float(x) * scale, 6), round(float(y) * scale, 6)] for x, y in points]


def _benchmark_widget(
    benchmark_name: str,
    *,
    item_id: int,
    quantity: int,
    scale: float = 1.0,
    widget_id: str | None = None,
) -> dict[str, Any]:
    benchmark = _load_benchmark(benchmark_name)
    items = {int(item["id"]): item for item in benchmark["items"]}
    item = items[item_id]
    widget = {
        "id": widget_id or f"{Path(benchmark_name).stem}_{item_id}",
        "quantity": quantity,
        "allowed_angles_degrees": [float(angle) for angle in item.get("allowed_orientations", [0.0, 180.0])],
        "polygon": {
            "shell": _normalize_ring(item["shape"]["data"], scale=scale),
        },
    }
    return widget


def _rect_board(board_id: str, width: float, height: float) -> dict[str, Any]:
    return {
        "id": board_id,
        "polygon": {
            "shell": [[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]]
        },
    }


def build_case_library() -> dict[str, CaseDefinition]:
    cases: list[CaseDefinition] = []

    cases.append(
        CaseDefinition(
            case_id="single_widget_fill",
            description="Synthetic repeated single widget that should fully occupy the board.",
            problem={
                "units": "mm",
                "source": {"kind": "synthetic"},
                "boards": [_rect_board("board_main", 100.0, 100.0)],
                "widgets": [
                    {
                        "id": "tile",
                        "quantity": 4,
                        "allowed_angles_degrees": [0.0, 90.0],
                        "polygon": {"shell": [[-25.0, -25.0], [25.0, -25.0], [25.0, 25.0], [-25.0, 25.0]]},
                    }
                ],
                "config": {"rotation_step_degrees": 90.0, "beam_width": 2, "population_size": 2, "generations": 1},
            },
        )
    )

    cases.append(
        CaseDefinition(
            case_id="hole_reuse",
            description="Synthetic frame-with-hole case where a smaller widget can be nested into the hole.",
            problem={
                "units": "mm",
                "source": {"kind": "synthetic"},
                "boards": [_rect_board("board_main", 110.0, 70.0)],
                "widgets": [
                    {
                        "id": "frame",
                        "quantity": 1,
                        "allowed_angles_degrees": [0.0, 90.0],
                        "polygon": {
                            "shell": [[-50.0, -30.0], [50.0, -30.0], [50.0, 30.0], [-50.0, 30.0]],
                            "holes": [[[-15.0, -12.0], [15.0, -12.0], [15.0, 12.0], [-15.0, 12.0]]],
                        },
                    },
                    {
                        "id": "square",
                        "quantity": 1,
                        "allowed_angles_degrees": [0.0, 45.0, 90.0],
                        "polygon": {"shell": [[-9.0, -9.0], [9.0, -9.0], [9.0, 9.0], [-9.0, 9.0]]},
                    },
                ],
                "config": {"rotation_step_degrees": 45.0, "beam_width": 3, "population_size": 4, "generations": 2},
            },
        )
    )

    cases.append(
        CaseDefinition(
            case_id="shirts_combo",
            description="Literature-derived clothing-pattern mix from the jagua-rs shirts benchmark with enough space to encourage clustering and a large leftover strip.",
            problem={
                "units": "benchmark_unit",
                "source": {
                    "kind": "benchmark_subset",
                    "benchmark": "shirts.json",
                    "url": f"{JAGUA_ASSET_BASE}/shirts.json",
                    "note": "Subset of literature-derived shirt-pattern pieces from jagua-rs assets.",
                },
                "boards": [_rect_board("board_main", 40.0, 20.0)],
                "widgets": [
                    _benchmark_widget("shirts.json", item_id=0, quantity=2),
                    _benchmark_widget("shirts.json", item_id=1, quantity=2),
                    _benchmark_widget("shirts.json", item_id=2, quantity=2),
                    _benchmark_widget("shirts.json", item_id=5, quantity=2),
                ],
                "config": {
                    "beam_width": 3,
                    "population_size": 4,
                    "generations": 2,
                    "rotation_step_degrees": 180.0,
                    "preferred_corners": ["lower_left", "upper_left"],
                    "max_candidates_per_item": 10,
                    "max_item_anchor_points": 4,
                    "max_free_space_anchor_points": 5,
                },
            },
        )
    )

    cases.append(
        CaseDefinition(
            case_id="trousers_shortage",
            description="Literature-derived trouser-pattern subset with intentional board shortage to test area-maximizing partial fill.",
            problem={
                "units": "benchmark_unit",
                "source": {
                    "kind": "benchmark_subset",
                    "benchmark": "trousers.json",
                    "url": f"{JAGUA_ASSET_BASE}/trousers.json",
                    "note": "Subset of literature-derived trouser-pattern pieces from jagua-rs assets.",
                },
                "boards": [_rect_board("board_main", 100.0, 40.0)],
                "widgets": [
                    _benchmark_widget("trousers.json", item_id=0, quantity=1),
                    _benchmark_widget("trousers.json", item_id=1, quantity=2),
                    _benchmark_widget("trousers.json", item_id=2, quantity=1),
                    _benchmark_widget("trousers.json", item_id=3, quantity=1),
                    _benchmark_widget("trousers.json", item_id=6, quantity=2),
                ],
                "config": {
                    "beam_width": 3,
                    "population_size": 4,
                    "generations": 2,
                    "rotation_step_degrees": 180.0,
                    "preferred_corners": ["lower_left", "lower_right"],
                    "max_candidates_per_item": 10,
                    "max_item_anchor_points": 4,
                    "max_free_space_anchor_points": 5,
                },
            },
        )
    )

    cases.append(
        CaseDefinition(
            case_id="swim_curved_mix",
            description="Literature-derived irregular curved swimwear pieces from the jagua-rs swim benchmark.",
            problem={
                "units": "scaled_benchmark_unit",
                "source": {
                    "kind": "benchmark_subset",
                    "benchmark": "swim.json",
                    "url": f"{JAGUA_ASSET_BASE}/swim.json",
                    "note": "Scaled subset of curved pieces from the swim benchmark in jagua-rs assets.",
                },
                "boards": [_rect_board("board_main", 360.0, 220.0)],
                "widgets": [
                    _benchmark_widget("swim.json", item_id=1, quantity=2, scale=0.18, widget_id="swim_1"),
                    _benchmark_widget("swim.json", item_id=7, quantity=1, scale=0.18, widget_id="swim_7"),
                    _benchmark_widget("swim.json", item_id=8, quantity=1, scale=0.18, widget_id="swim_8"),
                ],
                "config": {
                    "beam_width": 3,
                    "population_size": 4,
                    "generations": 2,
                    "rotation_step_degrees": 30.0,
                    "preferred_corners": ["lower_left", "upper_left"],
                    "max_candidates_per_item": 12,
                    "max_item_anchor_points": 5,
                    "max_free_space_anchor_points": 6,
                },
            },
        )
    )

    return {case.case_id: case for case in cases}
