from __future__ import annotations

import argparse
import json
from pathlib import Path

from algorithms.path_order_orientation_ortools.render import render_solution_image
from algorithms.path_order_orientation_ortools.solver import load_instance, score_solution, solve_with_ortools


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the visual path-order example")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-limit", type=float, default=12.0)
    args = parser.parse_args()

    instance = load_instance(args.input)
    raw = solve_with_ortools(instance, time_limit_sec=args.time_limit)
    scored = score_solution(instance, raw)
    result = {**raw, **scored}
    args.output.mkdir(parents=True, exist_ok=True)
    solution_path = args.output / "solution.json"
    image_path = args.output / "solution.png"
    solution_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    render_solution_image(instance, raw, image_path, title="Geo Lab · OR-Tools path order")
    print(json.dumps({
        "solverStatus": raw["solver_status"],
        "totalConnectionLength": scored["total_connection_length"],
        "solution": str(solution_path),
        "preview": str(image_path),
    }, indent=2))


if __name__ == "__main__":
    main()
