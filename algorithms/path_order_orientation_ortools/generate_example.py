from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.dist(a, b)


def _random_endpoint_pair(
    rng: random.Random,
    xy_min: float,
    xy_max: float,
    min_sep: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Generate two endpoints with a minimum separation."""
    for _ in range(200):
        x1 = rng.uniform(xy_min, xy_max)
        y1 = rng.uniform(xy_min, xy_max)
        x2 = rng.uniform(xy_min, xy_max)
        y2 = rng.uniform(xy_min, xy_max)
        if _euclidean((x1, y1), (x2, y2)) >= min_sep:
            return (x1, y1), (x2, y2)
    # Fallback if random sampling is unlucky.
    return (xy_min, xy_min), (xy_max, xy_max)


def create_example_instance(
    num_paths: int = 8,
    seed: int = 2026,
    xy_min: float = 0.0,
    xy_max: float = 100.0,
    min_endpoint_separation: float = 15.0,
) -> Dict[str, Any]:
    """
    Build a seeded-random dataset with Euclidean distance matrix.

    Distance matrix entries are true Euclidean distances between map points.
    """
    if num_paths < 2:
        raise ValueError("num_paths must be at least 2")
    if xy_max <= xy_min:
        raise ValueError("xy_max must be greater than xy_min")

    rng = random.Random(seed)
    points: List[Dict[str, Any]] = []
    paths: List[Dict[str, str]] = []
    point_xy: Dict[str, tuple[float, float]] = {}

    # Create 2 endpoints per path, with random coordinates.
    for i in range(1, num_paths + 1):
        s_id = f"S{i}"
        e_id = f"E{i}"

        (sx, sy), (ex, ey) = _random_endpoint_pair(
            rng=rng,
            xy_min=xy_min,
            xy_max=xy_max,
            min_sep=min_endpoint_separation,
        )

        sx = round(sx, 4)
        sy = round(sy, 4)
        ex = round(ex, 4)
        ey = round(ey, 4)

        points.append({"id": s_id, "x": sx, "y": sy})
        points.append({"id": e_id, "x": ex, "y": ey})

        point_xy[s_id] = (sx, sy)
        point_xy[e_id] = (ex, ey)

        paths.append({"id": f"P{i}", "start": s_id, "end": e_id})

    point_ids = [p["id"] for p in points]

    # Build a complete Euclidean distance matrix.
    distance_matrix: Dict[str, Dict[str, float]] = {pid: {} for pid in point_ids}
    for a in point_ids:
        for b in point_ids:
            if a == b:
                distance_matrix[a][b] = 0.0
            else:
                distance_matrix[a][b] = round(_euclidean(point_xy[a], point_xy[b]), 6)

    return {
        "points": points,
        "distance_matrix": distance_matrix,
        "paths": paths,
    }


def write_example_json(
    output_path: str | Path,
    num_paths: int = 8,
    seed: int = 2026,
) -> Path:
    out = Path(output_path)
    instance = create_example_instance(num_paths=num_paths, seed=seed)
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
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for reproducible geometry")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out = write_example_json(args.output, num_paths=args.num_paths, seed=args.seed)
    print(f"Wrote example instance to: {out}")


if __name__ == "__main__":
    main()
