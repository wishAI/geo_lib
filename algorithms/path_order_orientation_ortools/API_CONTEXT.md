# API Context: Path Order + Orientation Optimization

## Purpose

Find the best visiting order of paths and orientation per path to minimize transition-only Euclidean travel cost between consecutive paths.

## Dependencies

- Python 3.10+
- `ortools`
- `matplotlib` (image rendering)
- `pytest` (tests only)

## Data Contract

Input JSON object keys:

- `points`: list of points with unique `id`
- `distance_matrix`: dict-of-dicts where `distance_matrix[a][b]` exists for all point IDs
- `paths`: list of path tasks
  - each path has `id`, `start`, `end`

## Core Rule

Both solvers return only a minimal solution representation:

- `sequence`: ordered list of `{ "path_id": str, "reversed": bool }`

Scoring for comparison is always computed by the same function:

- `score_solution(instance, solution) -> dict`
  - uses Euclidean distance between points from `points[].x` and `points[].y`

This ensures OR-Tools and random baseline are judged under identical logic.

## Public Functions

From `solver.py`:

- `load_instance(json_path) -> dict`
  - loads and validates basic schema
- `solve_with_ortools(instance, time_limit_sec=30.0, num_workers=8) -> dict`
  - returns minimal solution (`sequence` only + metadata)
- `score_solution(instance, solution) -> dict`
  - uses Euclidean distance between points from `points[].x` and `points[].y`
  - shared evaluator returning
  - `total_connection_length`
  - `transitions`
  - `sequence_detailed`
  - `explanation`
- `evaluate_assignment(instance, order_indices, orientation_by_path_idx) -> float`
  - low-level transition-only evaluator used internally
- `build_sequence_from_assignment(instance, order_indices, orientation_by_path_idx) -> list`
  - converts internal assignment to minimal sequence format

From `baseline.py`:

- `random_select_best(instance, samples=1500, seed=2026) -> dict`
  - random sampling baseline, returns minimal solution format

From `render.py`:

- `render_solution_image(instance, solution, output_path, title=None) -> Path`
  - renders one solution image with fixed-scale coordinates
  - internally calls `score_solution()` to compute transitions and total length
  - blue lines = input paths, dark green lines = generated transitions

## Minimal Solution Schema

Shared by OR-Tools and random baseline:

- `method`: method name
- `sequence`: list of
  - `path_id`
  - `reversed`

Optional metadata fields may be present (e.g., solver status, sample count),
but scoring fields are not part of minimal solver outputs.

## Determinism

- OR-Tools solve is deterministic for this small instance in practice.
- Baseline randomness is controlled by explicit `seed`.
- Tests use fixed seeds and fixed sampling budgets.

## Integration Notes

- Euclidean distance is always computed from map coordinates (`x`, `y`).
- Internal path lengths are intentionally excluded from objective.
- For floating distances, solver scales costs to integers before CP-SAT optimization.
- Test-time images are saved under `rendered_results/` in this folder.
- To port this module, copy the folder and preserve imports between `solver.py`, `baseline.py`, and `render.py`.
