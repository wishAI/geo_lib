from __future__ import annotations

from pathlib import Path

from algorithms.path_order_orientation_ortools.baseline import random_select_best
from algorithms.path_order_orientation_ortools.render import render_solution_image
from algorithms.path_order_orientation_ortools.solver import load_instance, solve_with_ortools

EXAMPLE_PATH = Path(__file__).resolve().parent / "example_paths.json"
RENDER_DIR = Path(__file__).resolve().parent / "rendered_results"


def test_example_json_loads_and_has_required_sections() -> None:
    instance = load_instance(EXAMPLE_PATH)

    assert "points" in instance
    assert "distance_matrix" in instance
    assert "paths" in instance
    assert len(instance["paths"]) >= 2


def test_ortools_solution_is_complete_and_consistent() -> None:
    instance = load_instance(EXAMPLE_PATH)
    result = solve_with_ortools(instance, time_limit_sec=15.0)

    path_ids = [p["id"] for p in instance["paths"]]

    assert len(result["order"]) == len(path_ids)
    assert set(result["order"]) == set(path_ids)
    assert len(result["transitions"]) == len(path_ids) - 1

    transition_sum = sum(t["cost"] for t in result["transitions"])
    assert transition_sum == result["total_transition_cost"]


def test_ortools_not_worse_and_preferably_better_than_random_baseline() -> None:
    instance = load_instance(EXAMPLE_PATH)

    # A moderate fixed budget keeps the baseline fast and deterministic.
    random_result = random_select_best(instance, samples=400, seed=2026)
    ortools_result = solve_with_ortools(instance, time_limit_sec=15.0)

    # Render both results to images at true coordinate scale.
    ortools_img = render_solution_image(
        instance=instance,
        result=ortools_result,
        output_path=RENDER_DIR / "ortools_result.png",
        title="OR-Tools Optimized Result",
    )
    random_img = render_solution_image(
        instance=instance,
        result=random_result,
        output_path=RENDER_DIR / "random_select_best_result.png",
        title="Random Select Best Result",
    )

    assert ortools_result["total_transition_cost"] <= random_result["total_transition_cost"]

    # This example is intentionally constructed so OR-Tools should strictly win.
    assert ortools_result["total_transition_cost"] < random_result["total_transition_cost"]

    assert ortools_img.exists()
    assert random_img.exists()
    assert ortools_img.stat().st_size > 0
    assert random_img.stat().st_size > 0
