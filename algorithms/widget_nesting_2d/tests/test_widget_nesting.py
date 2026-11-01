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


def test_free_rotation_fits_non_grid_angle_and_discrete_mode_cannot():
    from shapely.affinity import rotate
    from shapely.geometry import box
    # Neither cardinal nor 15-degree sampling can fit this tight sheet.
    tilted = rotate(box(0, 0, 10, 2), 17.321, origin=(0, 0))
    problem = _problem({'boards': [{'id': 'b', 'polygon': _rectangle(10.0001, 2.0001)}],
                        'widgets': [{'id': 'bar', 'quantity': 1,
                                     'polygon': {'shell': [list(p) for p in tilted.exterior.coords[:-1]]}}],
                        'config': {'rotation_step_degrees': 90}})
    free = solve_problem(problem, _config(), time_limit_seconds=1)
    validate_solution(problem, free)
    assert len(free.placements) == 1
    assert abs(free.placements[0].rotation_degrees % 90 - 72.679) < 1e-6
    discrete_cfg = _config(rotation_mode='discrete', rotation_step_degrees=15)
    discrete = solve_problem(problem, discrete_cfg, time_limit_seconds=1)
    validate_solution(problem, discrete, config=discrete_cfg)
    assert not discrete.placements


def test_free_validator_accepts_arbitrary_angles_but_enforces_shape_and_restrictions():
    from dataclasses import replace
    import pytest
    from shapely.affinity import rotate
    from algorithms.widget_nesting_2d.benchmark import _result_from_placements
    from algorithms.widget_nesting_2d.solver import Placement, _localize_polygon
    problem = _problem({'boards': [{'id': 'b', 'polygon': _rectangle(20, 20)}],
                        'widgets': [{'id': 'w', 'quantity': 1, 'polygon': _rectangle(3, 2)}]})
    cfg = _config()
    local = _localize_polygon(problem.widgets[0].polygon.to_polygon(name='w'))
    polygon = rotate(local, 12.3456789, origin=(0, 0))
    p = Placement('w#1', 'w', 'b', 0, 12.3456789, polygon, 0, 0)
    s = _result_from_placements(problem, cfg, [p], 'test', 0)
    validate_solution(problem, s, config=cfg)
    with pytest.raises(AssertionError, match='geometry'):
        validate_solution(problem, replace(s, placements=(replace(p, rotation_degrees=11),)), config=cfg)
    for widget in (replace(problem.widgets[0], allowed_angles_degrees=(0, 90)),
                   replace(problem.widgets[0], rotation_step_degrees=90)):
        restricted = replace(problem, widgets=(widget,))
        with pytest.raises(AssertionError, match='rotation is not allowed'):
            validate_solution(restricted, s, config=cfg)


def test_free_angular_refinement_improves_stock_without_grid_snapping():
    from algorithms.widget_nesting_2d.solver import (
        _build_runtime, _initial_state, _find_item_candidates, _find_angle_candidates,
        _placement_rank_key)
    problem = _problem({'boards': [{'id': 'b', 'polygon': _rectangle(20, 20)}],
                        'widgets': [{'id': 'triangle', 'quantity': 1,
                                     'polygon': {'shell': [[0, 0], [7, 1], [1, 4]]}}]})
    cfg = _config(constructive_start=False)
    boards, items = _build_runtime(problem, cfg)
    from dataclasses import replace
    # Isolate local refinement from the geometry-aligned seed improvement.
    item = replace(items['triangle#1'], rotation_variants=(items['triangle#1'].rotation_variants[0],))
    state = _initial_state(boards)
    seeds = _find_angle_candidates(item, state, boards, cfg)
    refined = _find_item_candidates(item, state, boards, cfg)
    rank = lambda p: _placement_rank_key(boards[0], p.polygon, p.polygon)
    assert max(map(rank, refined)) >= max(map(rank, seeds))
    # The refinement search tests angles outside the original seed set.
    assert any(all(abs(p.rotation_degrees-v.angle_degrees) > 1e-7
                   for v in item.rotation_variants) for p in refined)


def test_free_rotation_finds_narrow_feasible_interval_between_seed_angles():
    import math
    angle = math.radians(17.321)
    width = 10*math.cos(angle) + math.sin(angle) + .0001
    height = 10*math.sin(angle) + math.cos(angle) + .0001
    problem = _problem({'boards': [{'id':'b', 'polygon':_rectangle(width, height)}],
                        'widgets': [{'id':'bar', 'quantity':1, 'polygon':_rectangle(10, 1)}]})
    solution = solve_problem(problem, time_limit_seconds=1)
    validate_solution(problem, solution)
    assert len(solution.placements) == 1
    assert min(abs(solution.placements[0].rotation_degrees % 180-a)
               for a in (17.321, 180-17.321)) < .01


def test_free_angular_restart_is_used_on_ordinary_clothing():
    from algorithms.widget_nesting_2d.case_library import build_case_library
    from algorithms.widget_nesting_2d.benchmark import benchmark_problem
    raw = benchmark_problem(build_case_library()['swim_curved_mix'].problem, 'free')
    problem = _problem(raw)
    solution = solve_problem(problem, time_limit_seconds=2)
    validate_solution(problem, solution)
    assert len(solution.placements) == 4
    assert solution.max_rest_rectangle_area > 7000
    assert any(abs((p.rotation_degrees+45)%90-45) > 1 for p in solution.placements)


def test_boundary_reinsertion_improves_real_shirts_and_preserves_failed_replacement(monkeypatch):
    from dataclasses import replace
    from shapely.affinity import translate
    from algorithms.widget_nesting_2d import solver
    from algorithms.widget_nesting_2d.case_library import build_case_library
    from algorithms.widget_nesting_2d.benchmark import benchmark_problem
    from algorithms.widget_nesting_2d.constructive import envelope_layouts
    problem = _problem(benchmark_problem(build_case_library()['shirts_combo'].problem, 'free'))
    cfg = _config(); boards, items = solver._build_runtime(problem, cfg)
    cardinal = {k: replace(i, rotation_variants=tuple(v for v in i.rotation_variants
                          if abs(v.angle_degrees % 90) < 1e-8)) for k, i in items.items()}
    raw = next(envelope_layouts(boards, cardinal, None, cfg.placement_tolerance))
    placements = [solver._build_placement(items[k],boards[bi],bi,v.angle_degrees,
                  translate(v.polygon,xoff=x,yoff=y)) for k,bi,v,x,y in raw]
    state = solver._state_from_placements(placements, boards, items)
    improved, changes = solver._refine_placed_rotations(state, boards, items, cfg, None)
    assert len(improved.placed) == len(state.placed) == 8
    assert changes > 0
    assert solver._state_rank_key(improved, boards) > solver._state_rank_key(state, boards)
    assert any(abs((p.rotation_degrees+45)%90-45) > 1 for p in improved.placed)
    monkeypatch.setattr(solver, '_find_item_candidates', lambda *a, **kw: [])
    unchanged, changes = solver._refine_placed_rotations(state, boards, items, cfg, None)
    assert unchanged is state
    assert changes == 0


def test_free_rotation_budget_keeps_exact_hole_placement_search():
    from algorithms.widget_nesting_2d.case_library import build_case_library
    from algorithms.widget_nesting_2d.benchmark import benchmark_problem
    problem = _problem(benchmark_problem(build_case_library()['exact_hole_fill'].problem, 'free'))
    solution = solve_problem(problem, time_limit_seconds=2)
    validate_solution(problem, solution)
    assert len(solution.placements) == 2
    assert abs(solution.placed_area-100) < 1e-6


def test_strip_masks_conservatively_cover_rotated_concavities_and_holes():
    import shapely
    import numpy as np
    from shapely.affinity import rotate, scale
    from shapely.geometry import box
    from algorithms.widget_nesting_2d.raster_start import _mask
    shapes = [box(0, 0, 9, 8).difference(box(2, 2, 7, 6)),
              Polygon([(0, 0), (8, 0), (8, 2), (2, 2), (2, 8), (0, 8)])]
    for shape in shapes:
        for angle in (0, 17.321, 113.7):
            for factor in (.01, 1, 100):
                polygon = scale(rotate(shape, angle), factor, factor)
                step = max(polygon.bounds[2]-polygon.bounds[0], polygon.bounds[3]-polygon.bounds[1])/47
                width, height, rows, _, _ = _mask(polygon, step)
                a, b, _, _ = polygon.bounds
                xx, yy = np.meshgrid(np.arange(width), np.arange(height))
                cells = shapely.box(a+xx*step, b+yy*step, a+(xx+1)*step, b+(yy+1)*step)
                positive = shapely.area(shapely.intersection(cells, polygon)) > 1e-12*factor*factor
                assert all(rows[y] & (1 << x) for y, x in zip(*np.nonzero(positive)))


def test_numeric_candidate_rank_matches_polygon_union_score():
    import random
    from shapely.geometry import box, GeometryCollection
    from algorithms.widget_nesting_2d import solver
    rng = random.Random(123)
    board = solver.BoardRuntime('b', box(-20, -10, 40, 60), box(-20, -10, 40, 60),
                                4200, (-20, -10, 40, 60))
    for index in range(50):
        x, y = rng.uniform(-18, 30), rng.uniform(-8, 50)
        moving = box(x, y, x+2, y+3)
        fixed = GeometryCollection() if index == 0 else box(-15, -5, 10, 20)
        actual = solver._placement_rank_key(board, fixed.union(moving), moving)
        fast = solver._rectangular_candidate_rank(board.bounds, None if fixed.is_empty else fixed.bounds,
                                                  moving.bounds, (moving.centroid.x, moving.centroid.y))
        assert fast == actual


def test_joint_order_and_rotation_search_fits_tight_curved_parts():
    from shapely.affinity import translate
    from algorithms.widget_nesting_2d import solver
    from algorithms.widget_nesting_2d.case_library import build_case_library
    from algorithms.widget_nesting_2d.benchmark import benchmark_problem
    from algorithms.widget_nesting_2d.raster_start import raster_search
    problem = _problem(benchmark_problem(build_case_library()['swim_curved_mix_tight'].problem, 'free'))
    cfg = _config(seed=42)
    boards, items = solver._build_runtime(problem, cfg)
    full = None
    for raw in raster_search(boards, items, None, seed=42):
        if len(raw) == len(items):
            full = raw
            break
    assert full is not None
    polygons = [translate(v.polygon, xoff=x, yoff=y) for _, _, v, x, y in full]
    assert all(boards[bi].polygon.covers(poly) for (_, bi, _, _, _), poly in zip(full, polygons))
    assert all(a.intersection(b).area < 1e-8 for i, a in enumerate(polygons) for b in polygons[i+1:])
    assert any(abs((v.angle_degrees+45)%90-45) > 1 for _, _, v, _, _ in full)
