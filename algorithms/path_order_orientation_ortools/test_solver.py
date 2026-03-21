from __future__ import annotations

from pathlib import Path

from algorithms.path_order_orientation_ortools.baseline import random_select_best
from algorithms.path_order_orientation_ortools.generate_example import write_example_json
from algorithms.path_order_orientation_ortools.render import render_solution_image
from algorithms.path_order_orientation_ortools.solver import load_instance, score_solution, solve_with_ortools

RENDER_DIR = Path(__file__).resolve().parent / "rendered_results"
TEST_SEED = 20260321
TEST_NUM_PATHS = 20


def _seeded_random_instance(tmp_path: Path) -> dict:
    """Create a reproducible random-layout JSON instance and load it."""
    json_path = tmp_path / "seeded_random_paths.json"
    write_example_json(json_path, num_paths=TEST_NUM_PATHS, seed=TEST_SEED)
    return load_instance(json_path)


def _validate_minimal_solution_shape(solution: dict, expected_len: int) -> None:
    """Both solvers must return only sequence + reversed flags (plus metadata)."""
    assert "sequence" in solution
    assert len(solution["sequence"]) == expected_len

    for step in solution["sequence"]:
        assert set(step.keys()) == {"path_id", "reversed"}
        assert isinstance(step["path_id"], str)
        assert isinstance(step["reversed"], bool)

    # Cost must come from shared scoring, not solver-specific output.
    assert "total_connection_length" not in solution


def test_example_json_loads_and_has_required_sections(tmp_path: Path) -> None:
    instance = _seeded_random_instance(tmp_path)

    assert "points" in instance
    assert "distance_matrix" in instance
    assert "paths" in instance
    assert len(instance["paths"]) == TEST_NUM_PATHS


def test_ortools_solution_is_complete_and_consistent(tmp_path: Path) -> None:
    instance = _seeded_random_instance(tmp_path)
    ortools_solution = solve_with_ortools(instance, time_limit_sec=30.0)

    _validate_minimal_solution_shape(ortools_solution, expected_len=TEST_NUM_PATHS)

    path_ids = [p["id"] for p in instance["paths"]]
    seq_path_ids = [s["path_id"] for s in ortools_solution["sequence"]]
    assert set(seq_path_ids) == set(path_ids)

    scored = score_solution(instance, ortools_solution)
    assert len(scored["transitions"]) == len(path_ids) - 1

    transition_sum = sum(float(t["cost"]) for t in scored["transitions"])
    assert abs(transition_sum - float(scored["total_connection_length"])) <= 1e-6


def test_ortools_not_worse_and_preferably_better_than_random_baseline(tmp_path: Path) -> None:
    instance = _seeded_random_instance(tmp_path)

    # Fixed sample budget + fixed seed keeps baseline deterministic.
    random_solution = random_select_best(instance, samples=400, seed=2026)
    ortools_solution = solve_with_ortools(instance, time_limit_sec=30.0)

    _validate_minimal_solution_shape(random_solution, expected_len=TEST_NUM_PATHS)
    _validate_minimal_solution_shape(ortools_solution, expected_len=TEST_NUM_PATHS)

    # Both methods are judged by exactly the same scoring function.
    random_scored = score_solution(instance, random_solution)
    ortools_scored = score_solution(instance, ortools_solution)

    # Render both results to images at true coordinate scale.
    ortools_img = render_solution_image(
        instance=instance,
        solution=ortools_solution,
        output_path=RENDER_DIR / "ortools_result.png",
        title="OR-Tools Optimized Result (Seeded Random 20 Paths)",
    )
    random_img = render_solution_image(
        instance=instance,
        solution=random_solution,
        output_path=RENDER_DIR / "random_select_best_result.png",
        title="Random Select Best Result (Seeded Random 20 Paths)",
    )

    assert ortools_scored["total_connection_length"] <= random_scored["total_connection_length"]

    # This generated instance is designed so OR-Tools should strictly win.
    assert ortools_scored["total_connection_length"] < random_scored["total_connection_length"]

    assert ortools_img.exists()
    assert random_img.exists()
    assert ortools_img.stat().st_size > 0
    assert random_img.stat().st_size > 0
