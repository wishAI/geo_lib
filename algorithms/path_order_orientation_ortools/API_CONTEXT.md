# API Context: Path Order + Orientation Optimization

## Purpose

Find the best visiting order of paths and best orientation per path to minimize transition-only travel cost between consecutive paths.

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

## Public Functions

From `solver.py`:

- `load_instance(json_path) -> dict`
  - loads and validates basic schema
- `solve_with_ortools(instance, time_limit_sec=30.0, num_workers=8) -> dict`
  - returns optimized order + orientations + total transition cost + readable explanation
- `evaluate_assignment(instance, order_indices, orientation_by_path_idx) -> float`
  - evaluates transition-only cost for a complete assignment
- `build_result_from_assignment(instance, order_indices, orientation_by_path_idx, method) -> dict`
  - builds standardized result payload

From `baseline.py`:

- `random_select_best(instance, samples=1500, seed=2026) -> dict`
  - random sampling baseline with reproducible seed

From `render.py`:

- `render_solution_image(instance, result, output_path, title=None) -> Path`
  - renders one solution image with fixed-scale coordinates
  - blue lines = input paths, dark green lines = generated transitions

## Solver Output Schema

Key fields from both solver and baseline:

- `method`: solver method name
- `order`: ordered list of path IDs
- `orientation_by_path`: mapping `{path_id: "forward"|"reverse"}`
- `sequence`: detailed per-step list (position, entry/exit)
- `transitions`: transition details between consecutive steps
- `total_transition_cost`: objective in original units
- `explanation`: readable multi-line explanation

Extra fields from OR-Tools solver:

- `solver_objective`
- `solver_status`

## Determinism

- OR-Tools solve is deterministic for this small example in practice.
- Baseline randomness is controlled by explicit `seed`.
- Tests use fixed seeds and fixed sampling budget.

## Integration Notes

- Keep all distances in the same unit system.
- Internal path lengths are intentionally excluded from objective.
- For floating distances, solver scales costs to integers before CP-SAT optimization.
- Test-time images are saved under `rendered_results/` in this folder.
- To port this module, copy the folder and preserve imports between `solver.py`, `baseline.py`, and `render.py`.
