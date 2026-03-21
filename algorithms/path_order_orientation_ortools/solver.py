from __future__ import annotations

import argparse
import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ortools.sat.python import cp_model

FORWARD = 0
REVERSE = 1


def load_instance(json_path: str | Path) -> Dict[str, Any]:
    """Load and lightly validate an instance JSON file."""
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for key in ["points", "distance_matrix", "paths"]:
        if key not in data:
            raise ValueError(f"Missing required key: {key}")

    point_ids = [p["id"] for p in data["points"]]
    point_id_set = set(point_ids)

    if len(point_ids) != len(point_id_set):
        raise ValueError("Point IDs must be unique")

    matrix = data["distance_matrix"]
    if not isinstance(matrix, dict):
        raise ValueError("distance_matrix must be a dict-of-dicts keyed by point IDs")

    for p in data["points"]:
        if "x" not in p or "y" not in p:
            raise ValueError(f"Point {p.get('id', p)} must include x and y for Euclidean distance")
        float(p["x"])
        float(p["y"])

    for pid in point_ids:
        if pid not in matrix:
            raise ValueError(f"distance_matrix is missing row for point: {pid}")
        for qid in point_ids:
            if qid not in matrix[pid]:
                raise ValueError(f"distance_matrix[{pid}] is missing column for point: {qid}")

    path_ids = [p["id"] for p in data["paths"]]
    if len(path_ids) != len(set(path_ids)):
        raise ValueError("Path IDs must be unique")

    for p in data["paths"]:
        for k in ["id", "start", "end"]:
            if k not in p:
                raise ValueError(f"Path {p} missing required key: {k}")
        if p["start"] not in point_id_set or p["end"] not in point_id_set:
            raise ValueError(f"Path {p['id']} references unknown point ID")

    return data


def orientation_name(orientation: int) -> str:
    return "forward" if orientation == FORWARD else "reverse"


def get_oriented_endpoints(path_obj: Dict[str, Any], orientation: int) -> Tuple[str, str]:
    """Return (entry_point, exit_point) for a chosen orientation."""
    if orientation == FORWARD:
        return path_obj["start"], path_obj["end"]
    if orientation == REVERSE:
        return path_obj["end"], path_obj["start"]
    raise ValueError(f"Unknown orientation: {orientation}")


def get_distance(distance_matrix: Dict[str, Dict[str, float]], a: str, b: str) -> float:
    return float(distance_matrix[a][b])


def _normalize_number(x: float) -> int | float:
    if abs(x - round(x)) < 1e-9:
        return int(round(x))
    return round(x, 6)


def _infer_scale(values: List[float], max_decimals: int = 6) -> int:
    max_places = 0
    for v in values:
        d = Decimal(str(v))
        places = max(0, -d.as_tuple().exponent)
        max_places = max(max_places, min(places, max_decimals))
    return 10**max_places


def get_euclidean_distance_matrix(instance: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Build or fetch cached Euclidean distance matrix from point coordinates."""
    cache_key = "__euclidean_distance_matrix_cache"
    cached = instance.get(cache_key)
    if isinstance(cached, dict):
        return cached

    coords: Dict[str, Tuple[float, float]] = {}
    for p in instance["points"]:
        coords[p["id"]] = (float(p["x"]), float(p["y"]))

    matrix: Dict[str, Dict[str, float]] = {pid: {} for pid in coords}
    for a, (ax, ay) in coords.items():
        for b, (bx, by) in coords.items():
            if a == b:
                matrix[a][b] = 0.0
            else:
                matrix[a][b] = math.dist((ax, ay), (bx, by))

    instance[cache_key] = matrix
    return matrix


def _path_id_to_index(instance: Dict[str, Any]) -> Dict[str, int]:
    return {p["id"]: i for i, p in enumerate(instance["paths"])}


def build_sequence_from_assignment(
    instance: Dict[str, Any],
    order_indices: List[int],
    orientation_by_path_idx: Dict[int, int],
) -> List[Dict[str, Any]]:
    """Build minimal solver output sequence: ordered path IDs + reversed bool."""
    paths = instance["paths"]
    sequence: List[Dict[str, Any]] = []

    for path_idx in order_indices:
        sequence.append(
            {
                "path_id": paths[path_idx]["id"],
                "reversed": orientation_by_path_idx[path_idx] == REVERSE,
            }
        )
    return sequence


def solution_to_assignment(
    instance: Dict[str, Any],
    solution: Dict[str, Any],
) -> Tuple[List[int], Dict[int, int]]:
    """Convert minimal solver output into index-based assignment representation."""
    if "sequence" not in solution or not isinstance(solution["sequence"], list):
        raise ValueError("solution must include list field: sequence")

    sequence = solution["sequence"]
    paths = instance["paths"]
    n = len(paths)

    if len(sequence) != n:
        raise ValueError(f"solution sequence length {len(sequence)} must equal number of paths {n}")

    id_to_idx = _path_id_to_index(instance)
    seen: set[str] = set()

    order_indices: List[int] = []
    orientation_by_path_idx: Dict[int, int] = {}

    for step in sequence:
        if "path_id" not in step or "reversed" not in step:
            raise ValueError("each sequence item must include path_id and reversed")

        path_id = step["path_id"]
        if path_id not in id_to_idx:
            raise ValueError(f"unknown path_id in solution sequence: {path_id}")
        if path_id in seen:
            raise ValueError(f"duplicate path_id in solution sequence: {path_id}")

        seen.add(path_id)
        path_idx = id_to_idx[path_id]
        order_indices.append(path_idx)

        reversed_flag = bool(step["reversed"])
        orientation_by_path_idx[path_idx] = REVERSE if reversed_flag else FORWARD

    return order_indices, orientation_by_path_idx


def evaluate_assignment(
    instance: Dict[str, Any],
    order_indices: List[int],
    orientation_by_path_idx: Dict[int, int],
) -> float:
    """Compute transition-only cost for a full order + orientation assignment."""
    paths = instance["paths"]
    matrix = get_euclidean_distance_matrix(instance)

    total = 0.0
    for i in range(len(order_indices) - 1):
        curr_idx = order_indices[i]
        next_idx = order_indices[i + 1]

        _, curr_exit = get_oriented_endpoints(paths[curr_idx], orientation_by_path_idx[curr_idx])
        next_entry, _ = get_oriented_endpoints(paths[next_idx], orientation_by_path_idx[next_idx])

        total += get_distance(matrix, curr_exit, next_entry)
    return total


def score_solution(instance: Dict[str, Any], solution: Dict[str, Any]) -> Dict[str, Any]:
    """
    Shared scoring function used for all methods.

    It computes total connection length and transition details from the same logic,
    so OR-Tools and random baseline are judged under identical scoring.
    """
    order_indices, orientation_by_path_idx = solution_to_assignment(instance, solution)

    paths = instance["paths"]
    matrix = get_euclidean_distance_matrix(instance)

    sequence_detailed = []
    for pos, path_idx in enumerate(order_indices, start=1):
        path_obj = paths[path_idx]
        orient = orientation_by_path_idx[path_idx]
        entry, exit_ = get_oriented_endpoints(path_obj, orient)
        sequence_detailed.append(
            {
                "position": pos,
                "path_id": path_obj["id"],
                "orientation": orientation_name(orient),
                "entry_point": entry,
                "exit_point": exit_,
            }
        )

    transitions = []
    total = 0.0
    for i in range(len(sequence_detailed) - 1):
        left = sequence_detailed[i]
        right = sequence_detailed[i + 1]
        cost = get_distance(matrix, left["exit_point"], right["entry_point"])
        total += cost
        transitions.append(
            {
                "from_path": left["path_id"],
                "from_exit": left["exit_point"],
                "to_path": right["path_id"],
                "to_entry": right["entry_point"],
                "cost": _normalize_number(cost),
            }
        )

    total_normalized = _normalize_number(total)
    explanation = build_explanation(sequence_detailed, transitions, total_normalized)

    return {
        "sequence_detailed": sequence_detailed,
        "transitions": transitions,
        "total_connection_length": total_normalized,
        "explanation": explanation,
    }


def build_explanation(sequence: List[Dict[str, Any]], transitions: List[Dict[str, Any]], total: int | float) -> str:
    lines = ["Chosen path sequence and orientations:"]
    for step in sequence:
        lines.append(
            f"  {step['position']}. {step['path_id']} ({step['orientation']}): "
            f"entry={step['entry_point']} -> exit={step['exit_point']}"
        )

    if transitions:
        lines.append("Transition costs:")
        for t in transitions:
            lines.append(
                f"  {t['from_path']}[{t['from_exit']}] -> "
                f"{t['to_path']}[{t['to_entry']}] = {t['cost']}"
            )
    else:
        lines.append("Transition costs: none (only one path).")

    lines.append(f"Total connection length: {total}")
    return "\n".join(lines)


def _build_scaled_state_transition_costs(instance: Dict[str, Any]) -> Tuple[List[List[int]], int]:
    """Transition costs between oriented states, scaled to integers for CP-SAT."""
    paths = instance["paths"]
    matrix = get_euclidean_distance_matrix(instance)
    num_states = 2 * len(paths)

    raw_costs = [[0.0 for _ in range(num_states)] for _ in range(num_states)]
    all_vals: List[float] = []

    for s in range(num_states):
        p_i = s // 2
        o_i = s % 2
        _, exit_i = get_oriented_endpoints(paths[p_i], o_i)

        for t in range(num_states):
            if s == t:
                raw_costs[s][t] = 0.0
                continue
            p_j = t // 2
            o_j = t % 2
            entry_j, _ = get_oriented_endpoints(paths[p_j], o_j)
            c = get_distance(matrix, exit_i, entry_j)
            raw_costs[s][t] = c
            all_vals.append(c)

    scale = _infer_scale(all_vals) if all_vals else 1
    scaled_costs = [[int(round(c * scale)) for c in row] for row in raw_costs]
    return scaled_costs, scale


def solve_with_ortools(
    instance: Dict[str, Any],
    time_limit_sec: float = 30.0,
    num_workers: int = 8,
) -> Dict[str, Any]:
    """
    Solve order + orientation optimization using CP-SAT and an AddCircuit model.

    Return format is intentionally minimal:
      - ordered path sequence
      - reversed flag per path in sequence

    Scoring is done separately by score_solution() for all methods.
    """
    paths = instance["paths"]
    n = len(paths)
    if n == 0:
        raise ValueError("paths must not be empty")

    num_states = 2 * n
    total_nodes = 1 + num_states  # +1 for dummy

    scaled_costs, scale = _build_scaled_state_transition_costs(instance)

    model = cp_model.CpModel()

    # y[s] indicates whether oriented state s is selected.
    y = [model.NewBoolVar(f"state_selected_{s}") for s in range(num_states)]

    arcs = []
    arc_var: Dict[Tuple[int, int], cp_model.IntVar] = {}

    # State node self-loops represent "not selected" in AddCircuit.
    for s in range(num_states):
        node = s + 1
        loop = model.NewBoolVar(f"loop_{node}")
        arcs.append((node, node, loop))
        model.Add(loop + y[s] == 1)

    # Exactly one orientation per original path.
    for p_idx in range(n):
        model.Add(y[2 * p_idx] + y[2 * p_idx + 1] == 1)

    # All directed arcs between distinct nodes (dummy + state nodes).
    for i in range(total_nodes):
        for j in range(total_nodes):
            if i == j:
                continue
            x = model.NewBoolVar(f"x_{i}_{j}")
            arcs.append((i, j, x))
            arc_var[(i, j)] = x

    model.AddCircuit(arcs)

    objective_terms = []
    for (i, j), var in arc_var.items():
        if i == 0 or j == 0:
            continue  # dummy arcs do not contribute to transition objective
        s_i = i - 1
        s_j = j - 1
        objective_terms.append(scaled_costs[s_i][s_j] * var)

    model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_search_workers = num_workers

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible solution found by OR-Tools")

    next_node: Dict[int, int] = {}
    for (i, j), var in arc_var.items():
        if solver.Value(var) == 1:
            next_node[i] = j

    # Extract ordered oriented states from dummy node.
    ordered_states: List[int] = []
    current = next_node[0]
    visited = set()
    while current != 0:
        if current in visited:
            raise RuntimeError("Cycle extraction failed: repeated node before returning to dummy")
        visited.add(current)
        ordered_states.append(current - 1)  # map node -> state index
        current = next_node[current]

    if len(ordered_states) != n:
        raise RuntimeError(
            f"Expected {n} selected states (one per path), got {len(ordered_states)}"
        )

    order_indices = [state // 2 for state in ordered_states]
    orientation_by_path_idx = {path_idx: (state % 2) for path_idx, state in zip(order_indices, ordered_states)}

    sequence = build_sequence_from_assignment(
        instance=instance,
        order_indices=order_indices,
        orientation_by_path_idx=orientation_by_path_idx,
    )

    objective_scaled = solver.ObjectiveValue()
    objective_raw = objective_scaled / scale

    return {
        "method": "ortools_cp_sat",
        "sequence": sequence,
        "solver_objective": _normalize_number(objective_raw),
        "solver_status": solver.StatusName(status),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve path order + orientation optimization with OR-Tools")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parent / "example_paths.json"),
        help="Input JSON instance path",
    )
    parser.add_argument(
        "--time_limit_sec",
        type=float,
        default=30.0,
        help="CP-SAT time limit in seconds",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    instance = load_instance(args.input)
    raw_solution = solve_with_ortools(instance, time_limit_sec=args.time_limit_sec)
    scored = score_solution(instance, raw_solution)

    output = {
        **raw_solution,
        "total_connection_length": scored["total_connection_length"],
        "transitions": scored["transitions"],
        "sequence_detailed": scored["sequence_detailed"],
        "explanation": scored["explanation"],
    }

    print(json.dumps(output, indent=2))
    print("\n" + output["explanation"])


if __name__ == "__main__":
    main()
