# Path Order + Orientation Optimization (OR-Tools)

This module solves the following transition-cost optimization problem:

- You must visit every path exactly once.
- Each path can be traversed either `forward` (`start -> end`) or `reverse` (`end -> start`).
- Internal path length is fixed and ignored.
- Only transition cost between consecutive paths is optimized:
  - `cost(A, B) = euclidean(exit_of_A_orientation, entry_of_B_orientation)`

The solver jointly optimizes:
- visiting order of paths
- orientation of each path

## Important Design Rule

Both methods (`solve_with_ortools` and `random_select_best`) return only:
- ordered path sequence
- whether each path is reversed

They do **not** compute final score internally for comparison.

A single shared function in `solver.py` is used for judging both methods:
- `score_solution(instance, solution)`
- it always computes Euclidean distances from point coordinates (`x`, `y`)

This ensures both are evaluated under exactly the same scoring logic.

## Problem Formulation

We model each path as two orientation states.

- State `(i, 0)` means path `i` in forward orientation.
- State `(i, 1)` means path `i` in reverse orientation.

Then we solve a combinatorial optimization model with OR-Tools CP-SAT:

- select exactly one orientation state for each path
- build one sequence through selected states (open path)
- objective = sum of transition costs between consecutive selected states

Implementation uses `AddCircuit` with a dummy node so a cycle corresponds to an open sequence.

## JSON Format

Input file must contain:

```json
{
  "points": [
    {"id": "A", "x": 0, "y": 0},
    {"id": "B", "x": 1, "y": 0}
  ],
  "distance_matrix": {
    "A": {"A": 0, "B": 10},
    "B": {"A": 10, "B": 0}
  },
  "paths": [
    {"id": "P1", "start": "A", "end": "B"}
  ]
}
```

Notes:
- `distance_matrix` is kept for compatibility and schema completeness.
- Optimization/scoring uses Euclidean distance computed from `points[].x` and `points[].y`.

## Files

- `solver.py`: OR-Tools solver + shared scoring function.
- `baseline.py`: random baseline (`random_select_best`).
- `render.py`: scaled image renderer.
- `generate_example.py`: seeded-random example generator.
- `example_paths.json`: pre-generated example instance.
- `test_solver.py`: pytest unit tests.
- `API_CONTEXT.md`: integration contract for future agents.

## Setup

```bash
pyenv activate ptenv
pip install ortools pytest matplotlib
```

## Generate Example Data

```bash
cd algorithms/path_order_orientation_ortools
python generate_example.py --seed 20260321 --num_paths 20
```

This writes/overwrites `example_paths.json`.

## Run OR-Tools Solver

```bash
cd algorithms/path_order_orientation_ortools
python solver.py --input example_paths.json
```

CLI prints:
- minimal solution sequence (`path_id`, `reversed`)
- shared-score output (`total_connection_length`, transitions, explanation)

## Run Random Baseline

```bash
cd algorithms/path_order_orientation_ortools
python baseline.py --input example_paths.json --samples 400 --seed 2026
```

## Run Tests (Also Generates Images)

From repo root:

```bash
pyenv activate ptenv
pytest -q algorithms/path_order_orientation_ortools/test_solver.py
```

The test suite:
- builds a seeded-random 20-path instance
- scores OR-Tools and random baseline with the same `score_solution()` function
- writes two images to `algorithms/path_order_orientation_ortools/rendered_results/`:
  - `ortools_result.png`
  - `random_select_best_result.png`

In those images:
- blue lines = input paths
- dark green lines = generated transition connections
