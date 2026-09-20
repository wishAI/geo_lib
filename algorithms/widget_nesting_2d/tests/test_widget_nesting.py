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


def test_small_population_and_zero_mutation_terminate():
    problem = _problem({'boards': [{'id': 'b', 'polygon': _rectangle(10, 10)}],
                        'widgets': [{'id': 'w', 'quantity': 1, 'polygon': _rectangle(2, 2),
                                     'allowed_angles_degrees': [0]}]})
    solution = solve_problem(problem, SolverConfig(population_size=12, generations=3, mutation_rate=0))
    validate_solution(problem, solution)
    assert len(solution.placements) == 1
    assert solution.search_stats['unique_orders_evaluated'] == 1


def test_nfp_slanted_three_edge_contact():
    side = 40 / 9
    problem = _problem({'boards': [{'id': 'triangle', 'polygon': {'shell': [[0, 0], [10, 0], [3, 8]]}}],
                        'widgets': [{'id': 'square', 'quantity': 1, 'polygon': _rectangle(side, side),
                                     'allowed_angles_degrees': [0]}]})
    solution = solve_problem(problem, _config(candidate_mode='nfp'))
    validate_solution(problem, solution)
    assert len(solution.placements) == 1
    assert abs(solution.placements[0].polygon.bounds[0] - 5/3) < 1e-6


def test_sampling_covers_extreme_directions():
    from algorithms.widget_nesting_2d.solver import _sample_points
    points = [(0, 0), (1, 0), (2, 0), (0, 10), (10, 0), (10, 10)]
    assert set(_sample_points(points, 4)) == {(0, 0), (0, 10), (10, 0), (10, 10)}
    assert _sample_points(points, 0) == []


def test_subunit_compaction():
    from shapely.geometry import GeometryCollection, box
    from algorithms.widget_nesting_2d.solver import _build_runtime, _max_shift
    problem = _problem({'boards': [{'id': 'b', 'polygon': _rectangle(10, 10)}],
                        'widgets': [{'id': 'w', 'quantity': 1, 'polygon': _rectangle(1, 1)}]})
    boards, _ = _build_runtime(problem, _config())
    moved = _max_shift(box(-4.5, 0, -3.5, 1), (-1, 0), board=boards[0],
                       occupied=GeometryCollection(), tolerance=1e-6)
    assert abs(moved.bounds[0] + 5) < 1e-5


def test_scaled_exact_tilings():
    for scale in (0.001, 1, 10000):
        problem = _problem({'boards': [{'id': 'b', 'polygon': _rectangle(10*scale, 10*scale)}],
                            'widgets': [{'id': 'tile', 'quantity': 4, 'polygon': _rectangle(5*scale, 5*scale),
                                         'allowed_angles_degrees': [0]}]})
        solution = solve_problem(problem, _config())
        validate_solution(problem, solution)
        assert len(solution.placements) == 4


def test_validator_rejects_duplicate_instance_and_changed_shape():
    from dataclasses import replace
    import pytest
    from shapely.geometry import box
    problem = _problem({'boards': [{'id': 'b', 'polygon': _rectangle(10, 10)}],
                        'widgets': [{'id': 'tile', 'quantity': 1, 'polygon': _rectangle(2, 2),
                                     'allowed_angles_degrees': [0]}]})
    solution = solve_problem(problem, _config())
    with pytest.raises(AssertionError, match='identities'):
        validate_solution(problem, replace(solution, placements=solution.placements*2))
    wrong = replace(solution.placements[0], polygon=box(0, 0, 1, 1))
    with pytest.raises(AssertionError):
        validate_solution(problem, replace(solution, placements=(wrong,), placed_area=1))


def test_invalid_input_ids_and_nonfinite_coordinates():
    import pytest
    raw = {'boards': [{'id': 'b', 'polygon': _rectangle(10, 10)}],
           'widgets': [{'id': 'w', 'quantity': 1, 'polygon': _rectangle(1, 1)}]*2}
    with pytest.raises(ValueError, match='unique'):
        _problem(raw)
    raw['widgets'] = raw['widgets'][:1]
    raw['widgets'][0]['polygon']['shell'][0][0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        _problem(raw)


def test_genetic_population_breeds_with_default_elite_count():
    raw = {'boards': [{'id': 'b', 'polygon': _rectangle(10, 10)}],
           'widgets': [{'id': str(i), 'quantity': 1, 'allowed_angles_degrees': [0],
                        'polygon': _rectangle(i+1, 1)} for i in range(4)]}
    solution = solve_problem(_problem(raw), _config(population_size=4, generations=3, mutation_rate=1))
    assert solution.search_stats['unique_orders_evaluated'] > 4


def test_nfp_obstacles_match_independent_overlap_predicate():
    import random
    from shapely.affinity import translate
    from shapely.geometry import Point, box
    from algorithms.widget_nesting_2d.nfp import _obstacle, clear_caches, cache_info
    fixed = Polygon([(0,0),(10,0),(10,10),(0,10)], holes=[[(2,2),(8,2),(8,8),(2,8)]])
    moving = Polygon([(0,0),(3,0),(3,1),(1,1),(1,3),(0,3)])
    clear_caches()
    obstacle = _obstacle(fixed.wkb, moving.wkb)
    rng = random.Random(1234)
    for _ in range(200):
        x, y = rng.uniform(-5,15), rng.uniform(-5,15)
        collision = fixed.intersection(translate(moving, xoff=x, yoff=y)).area > 1e-8
        assert obstacle.contains(Point(x,y)) == collision
    _obstacle(fixed.wkb, moving.wkb)
    assert cache_info()['misses'] == 1
    assert cache_info()['hits'] == 1


def test_expired_budget_returns_fully_accounted_incumbent():
    from unittest.mock import patch
    problem = _problem({'boards': [{'id': 'b', 'polygon': _rectangle(10, 10)}],
                        'widgets': [{'id': 'tile', 'quantity': 4, 'polygon': _rectangle(2, 2),
                                     'allowed_angles_degrees': [0]}]})
    # Deterministic deadline exhaustion before the first placement, not a flaky
    # assertion about how quickly this computer happens to run.
    with patch('algorithms.widget_nesting_2d.solver.time.perf_counter', side_effect=[0] + [2]*100):
        solution = solve_problem(problem, _config(), time_limit_seconds=1)
    validate_solution(problem, solution)
    assert solution.search_stats['budget_exhausted']
    assert len(solution.skipped_item_ids) == 4


def test_budget_preserves_partial_placement():
    from unittest.mock import patch
    from algorithms.widget_nesting_2d.solver import _find_item_candidates
    problem = _problem({'boards': [{'id': 'b', 'polygon': _rectangle(10, 10)}],
                        'widgets': [{'id': 'tile', 'quantity': 4, 'polygon': _rectangle(2, 2),
                                     'allowed_angles_degrees': [0]}]})
    clock = [0]
    def first_item_then_expire(*args, **kwargs):
        candidates = _find_item_candidates(*args, **kwargs)
        clock[0] = 2
        return candidates
    with patch('algorithms.widget_nesting_2d.solver.time.perf_counter', side_effect=lambda:clock[0]), \
         patch('algorithms.widget_nesting_2d.solver._find_item_candidates', side_effect=first_item_then_expire):
        solution = solve_problem(problem, _config(beam_width=1), time_limit_seconds=1)
    validate_solution(problem, solution)
    assert len(solution.placements) == 1
    assert len(solution.skipped_item_ids) == 3
