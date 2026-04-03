from __future__ import annotations

from shapely.geometry import Polygon

from algorithms.widget_nesting_2d.problem import ProblemSpec
from algorithms.widget_nesting_2d.solver import SolverConfig, solve_problem, validate_solution


def _rectangle(width: float, height: float) -> dict[str, list[list[float]]]:
    half_w = width / 2.0
    half_h = height / 2.0
    return {"shell": [[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]]}


def _problem(raw: dict) -> ProblemSpec:
    return ProblemSpec.from_json(raw)


def _config(**overrides: float | int) -> SolverConfig:
    base = {
        "rotation_step_degrees": 15,
        "beam_width": 2,
        "population_size": 2,
        "generations": 1,
        "seed": 20260403,
    }
    base.update(overrides)
    return SolverConfig(**base)


def test_places_small_widget_inside_large_hole_when_needed() -> None:
    problem = _problem(
        {
            "units": "mm",
            "boards": [{"id": "board", "polygon": {"shell": [[0, 0], [110, 0], [110, 70], [0, 70]]}}],
            "widgets": [
                {
                    "id": "frame",
                    "quantity": 1,
                    "allowed_angles_degrees": [0, 90],
                    "polygon": {
                        "shell": [[-50, -30], [50, -30], [50, 30], [-50, 30]],
                        "holes": [[[-15, -12], [15, -12], [15, 12], [-15, 12]]],
                    },
                },
                {
                    "id": "square",
                    "quantity": 1,
                    "allowed_angles_degrees": [0, 45, 90],
                    "polygon": _rectangle(18, 18),
                },
            ],
        }
    )

    solution = solve_problem(problem, config=_config(rotation_step_degrees=45))
    validate_solution(problem, solution)
    assert len(solution.placements) == 2

    frame = next(placement for placement in solution.placements if placement.widget_id == "frame")
    square = next(placement for placement in solution.placements if placement.widget_id == "square")
    hole_polygon = Polygon(frame.polygon.interiors[0])
    assert hole_polygon.buffer(1e-6).covers(square.polygon)


def test_never_overlaps_and_stays_within_board() -> None:
    problem = _problem(
        {
            "units": "mm",
            "boards": [{"id": "board", "polygon": {"shell": [[0, 0], [180, 0], [180, 130], [0, 130]]}}],
            "widgets": [
                {
                    "id": "hook",
                    "quantity": 2,
                    "polygon": {
                        "shell": [[-26, -26], [6, -26], [6, -8], [24, -8], [24, 8], [-8, 8], [-8, 26], [-26, 26]]
                    },
                },
                {
                    "id": "tab",
                    "quantity": 2,
                    "polygon": {"shell": [[-24, -16], [8, -16], [8, -4], [22, -4], [22, 16], [-24, 16]]},
                },
                {
                    "id": "square",
                    "quantity": 2,
                    "allowed_angles_degrees": [0, 45, 90],
                    "polygon": _rectangle(16, 16),
                },
            ],
        }
    )

    solution = solve_problem(problem, config=_config(rotation_step_degrees=30))
    validate_solution(problem, solution)
    assert len(solution.skipped_item_ids) == 0


def test_chooses_more_area_when_board_cannot_fit_everything() -> None:
    problem = _problem(
        {
            "units": "mm",
            "boards": [{"id": "board", "polygon": {"shell": [[0, 0], [100, 0], [100, 100], [0, 100]]}}],
            "widgets": [
                {
                    "id": "big_plate",
                    "quantity": 1,
                    "allowed_angles_degrees": [0, 90],
                    "polygon": _rectangle(100, 70),
                },
                {
                    "id": "medium_plate",
                    "quantity": 4,
                    "allowed_angles_degrees": [0, 90],
                    "polygon": _rectangle(50, 40),
                },
            ],
        }
    )

    solution = solve_problem(problem, config=_config(rotation_step_degrees=90, population_size=10, generations=6))
    validate_solution(problem, solution)
    assert solution.placed_area >= 7999.0
    assert any(item_id.startswith("big_plate#") for item_id in solution.skipped_item_ids)


def test_full_fit_preserves_large_corner_rectangle() -> None:
    problem = _problem(
        {
            "units": "mm",
            "boards": [{"id": "board", "polygon": {"shell": [[0, 0], [100, 0], [100, 100], [0, 100]]}}],
            "widgets": [
                {
                    "id": "little",
                    "quantity": 4,
                    "allowed_angles_degrees": [0, 90],
                    "polygon": _rectangle(10, 10),
                }
            ],
        }
    )

    solution = solve_problem(problem, config=_config(rotation_step_degrees=90))
    validate_solution(problem, solution)
    assert len(solution.skipped_item_ids) == 0
    assert solution.max_rest_rectangle_area >= 8000.0


def test_repeated_single_widget_can_fill_board() -> None:
    problem = _problem(
        {
            "units": "mm",
            "boards": [{"id": "board", "polygon": {"shell": [[0, 0], [100, 0], [100, 100], [0, 100]]}}],
            "widgets": [
                {
                    "id": "tile",
                    "quantity": 4,
                    "allowed_angles_degrees": [0, 90],
                    "polygon": _rectangle(50, 50),
                }
            ],
        }
    )

    solution = solve_problem(problem, config=_config(rotation_step_degrees=90))
    validate_solution(problem, solution)
    assert solution.placed_area >= 9999.0
    assert len(solution.skipped_item_ids) == 0
    assert solution.max_rest_rectangle_area <= 1e-6
