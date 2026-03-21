from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

try:
    from .solver import build_result_from_assignment, evaluate_assignment, load_instance
except ImportError:  # Allows running as a script from this folder.
    from solver import build_result_from_assignment, evaluate_assignment, load_instance


def random_select_best(
    instance: Dict[str, Any],
    samples: int = 1500,
    seed: int = 2026,
) -> Dict[str, Any]:
    """Randomly sample many feasible solutions, return the best found."""
    if samples <= 0:
        raise ValueError("samples must be > 0")

    rng = random.Random(seed)
    n = len(instance["paths"])
    if n == 0:
        raise ValueError("paths must not be empty")

    base_order = list(range(n))

    best_order = None
    best_orientation = None
    best_cost = float("inf")

    for _ in range(samples):
        order = base_order[:]
        rng.shuffle(order)

        orientation_by_path_idx = {i: rng.randint(0, 1) for i in range(n)}
        cost = evaluate_assignment(instance, order, orientation_by_path_idx)

        if cost < best_cost:
            best_cost = cost
            best_order = order
            best_orientation = orientation_by_path_idx

    assert best_order is not None
    assert best_orientation is not None

    result = build_result_from_assignment(
        instance=instance,
        order_indices=best_order,
        orientation_by_path_idx=best_orientation,
        method="random_select_best",
    )
    result["samples"] = samples
    result["seed"] = seed
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random baseline for path order + orientation optimization")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parent / "example_paths.json"),
        help="Input JSON instance path",
    )
    parser.add_argument("--samples", type=int, default=1500, help="Number of random samples")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    instance = load_instance(args.input)
    result = random_select_best(instance, samples=args.samples, seed=args.seed)

    print(json.dumps(result, indent=2))
    print("\n" + result["explanation"])


if __name__ == "__main__":
    main()
