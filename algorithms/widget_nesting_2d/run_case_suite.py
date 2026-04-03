from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

try:
    from .case_library import build_case_library
    from .problem import ProblemSpec, save_solution
    from .render import render_solution_image
    from .solver import SolverConfig, solve_problem, solution_to_dict, validate_solution
except ImportError:  # pragma: no cover - script mode
    from case_library import build_case_library
    from problem import ProblemSpec, save_solution
    from render import render_solution_image
    from solver import SolverConfig, solve_problem, solution_to_dict, validate_solution


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the widget nesting example suite.")
    parser.add_argument(
        "--output-root",
        default="algorithms/widget_nesting_2d/outputs",
        help="Directory where per-case outputs and index.json are written",
    )
    parser.add_argument("--case", action="append", default=[], help="Optional case id to run. Repeatable.")
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--rotation-step-degrees", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    case_library = build_case_library()
    selected_ids = args.case or list(case_library)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    index_entries: list[dict[str, Any]] = []
    for case_id in selected_ids:
        case = case_library[case_id]
        case_output_dir = output_root / case.case_id
        case_output_dir.mkdir(parents=True, exist_ok=True)
        save_solution(case_output_dir / "problem.json", case.problem)

        problem = ProblemSpec.from_json(case.problem)
        config = SolverConfig.from_problem(
            problem,
            overrides={
                "beam_width": args.beam_width,
                "population_size": args.population_size,
                "generations": args.generations,
                "rotation_step_degrees": args.rotation_step_degrees,
            },
        )
        start = time.perf_counter()
        solution = solve_problem(problem, config=config)
        elapsed = time.perf_counter() - start
        validate_solution(problem, solution, tolerance=config.placement_tolerance)

        solution_payload = solution_to_dict(problem, solution)
        save_solution(case_output_dir / "solution.json", solution_payload)
        render_solution_image(problem, solution, case_output_dir / "nesting_layout.png")

        index_entries.append(
            {
                "case_id": case.case_id,
                "description": case.description,
                "source": case.problem.get("source", {}),
                "elapsed_seconds": round(elapsed, 4),
                "score": solution_payload["score"],
                "skipped_item_ids": solution_payload["skipped_item_ids"],
                "problem_json": str(case_output_dir / "problem.json"),
                "solution_json": str(case_output_dir / "solution.json"),
                "layout_png": str(case_output_dir / "nesting_layout.png"),
            }
        )

    payload = {"cases": index_entries}
    save_solution(output_root / "index.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
