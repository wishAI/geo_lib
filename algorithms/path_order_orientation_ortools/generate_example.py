from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from .solver import FORWARD, REVERSE, get_oriented_endpoints
except ImportError:  # Allows running as a script from this folder.
    from solver import FORWARD, REVERSE, get_oriented_endpoints


def create_example_instance(num_paths: int = 8) -> Dict[str, Any]:
    """
    Build a small but non-trivial dataset.

    Design:
    - Each path has unique endpoints.
    - Most point-to-point distances are large.
    - A single intended sequence with alternating orientations has very cheap transitions,
      so OR-Tools should clearly beat the random baseline with fixed sampling budget.
    """
    if num_paths < 2:
        raise ValueError("num_paths must be at least 2")

    points: List[Dict[str, Any]] = []
    paths: List[Dict[str, str]] = []

    # Create 2 endpoints per path: S_i and E_i.
    for i in range(1, num_paths + 1):
        s_id = f"S{i}"
        e_id = f"E{i}"

        points.append({"id": s_id, "x": i * 10, "y": 0})
        points.append({"id": e_id, "x": i * 10, "y": 10})

        paths.append({"id": f"P{i}", "start": s_id, "end": e_id})

    point_ids = [p["id"] for p in points]

    # Start with a large, symmetric baseline matrix.
    distance_matrix: Dict[str, Dict[str, int]] = {pid: {} for pid in point_ids}
    for i, a in enumerate(point_ids):
        for j, b in enumerate(point_ids):
            if i == j:
                distance_matrix[a][b] = 0
            else:
                # Keep all generic transitions expensive.
                distance_matrix[a][b] = 80 + 7 * abs(i - j)

    # Intended best orientations: alternating forward/reverse.
    intended_orientations = [FORWARD if i % 2 == 0 else REVERSE for i in range(num_paths)]

    # Make only intended consecutive transitions very cheap.
    for i in range(num_paths - 1):
        left_path = paths[i]
        right_path = paths[i + 1]

        _, left_exit = get_oriented_endpoints(left_path, intended_orientations[i])
        right_entry, _ = get_oriented_endpoints(right_path, intended_orientations[i + 1])

        distance_matrix[left_exit][right_entry] = 1
        distance_matrix[right_entry][left_exit] = 1

    return {
        "points": points,
        "distance_matrix": distance_matrix,
        "paths": paths,
    }


def write_example_json(output_path: str | Path, num_paths: int = 8) -> Path:
    out = Path(output_path)
    instance = create_example_instance(num_paths=num_paths)
    out.write_text(json.dumps(instance, indent=2), encoding="utf-8")
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate example JSON for path order + orientation optimization")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "example_paths.json"),
        help="Where to write the JSON instance",
    )
    parser.add_argument("--num_paths", type=int, default=8, help="Number of paths to generate")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out = write_example_json(args.output, num_paths=args.num_paths)
    print(f"Wrote example instance to: {out}")


if __name__ == "__main__":
    main()
