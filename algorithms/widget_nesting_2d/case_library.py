from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BENCHMARK_ROOT = Path(__file__).resolve().parent / "inputs" / "public_benchmarks"
JAGUA_ASSET_BASE = "https://github.com/JeroenGar/jagua-rs/blob/main/assets"


@dataclass(frozen=True)
class CaseDefinition:
    case_id: str
    description: str
    problem: dict[str, Any]
    witness: dict[str, Any] | None = None


def _load_benchmark(name: str) -> dict[str, Any]:
    path = BENCHMARK_ROOT / name
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _normalize_ring(points: list[list[float]], *, scale: float = 1.0) -> list[list[float]]:
    if len(points) >= 2 and points[0] == points[-1]:
        points = points[:-1]
    return [[round(float(x) * scale, 6), round(float(y) * scale, 6)] for x, y in points]


def _benchmark_widget(
    benchmark_name: str,
    *,
    item_id: int,
    quantity: int,
    scale: float = 1.0,
    widget_id: str | None = None,
) -> dict[str, Any]:
    benchmark = _load_benchmark(benchmark_name)
    items = {int(item["id"]): item for item in benchmark["items"]}
    item = items[item_id]
    widget = {
        "id": widget_id or f"{Path(benchmark_name).stem}_{item_id}",
        "quantity": quantity,
        "allowed_angles_degrees": [float(angle) for angle in item.get("allowed_orientations", [0.0, 180.0])],
        "polygon": {
            "shell": _normalize_ring(item["shape"]["data"], scale=scale),
        },
    }
    return widget


def _rect_board(board_id: str, width: float, height: float) -> dict[str, Any]:
    return {
        "id": board_id,
        "polygon": {
            "shell": [[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]]
        },
    }


def build_case_library(*, include_large: bool = False, include_diverse: bool = False) -> dict[str, CaseDefinition]:
    cases: list[CaseDefinition] = []

    cases.append(
        CaseDefinition(
            case_id="single_widget_fill",
            description="Synthetic repeated single widget that should fully occupy the board.",
            problem={
                "units": "mm",
                "source": {"kind": "synthetic"},
                "boards": [_rect_board("board_main", 100.0, 100.0)],
                "widgets": [
                    {
                        "id": "tile",
                        "quantity": 4,
                        "allowed_angles_degrees": [0.0, 90.0],
                        "polygon": {"shell": [[-25.0, -25.0], [25.0, -25.0], [25.0, 25.0], [-25.0, 25.0]]},
                    }
                ],
                "config": {"rotation_step_degrees": 90.0, "beam_width": 2, "population_size": 2, "generations": 1},
            },
        )
    )

    cases.append(
        CaseDefinition(
            case_id="hole_reuse",
            description="Synthetic frame-with-hole case where a smaller widget can be nested into the hole.",
            problem={
                "units": "mm",
                "source": {"kind": "synthetic"},
                "boards": [_rect_board("board_main", 110.0, 70.0)],
                "widgets": [
                    {
                        "id": "frame",
                        "quantity": 1,
                        "allowed_angles_degrees": [0.0, 90.0],
                        "polygon": {
                            "shell": [[-50.0, -30.0], [50.0, -30.0], [50.0, 30.0], [-50.0, 30.0]],
                            "holes": [[[-15.0, -12.0], [15.0, -12.0], [15.0, 12.0], [-15.0, 12.0]]],
                        },
                    },
                    {
                        "id": "square",
                        "quantity": 1,
                        "allowed_angles_degrees": [0.0, 45.0, 90.0],
                        "polygon": {"shell": [[-9.0, -9.0], [9.0, -9.0], [9.0, 9.0], [-9.0, 9.0]]},
                    },
                ],
                "config": {"rotation_step_degrees": 45.0, "beam_width": 3, "population_size": 4, "generations": 2},
            },
        )
    )

    cases.append(
        CaseDefinition(
            case_id="shirts_combo",
            description="Literature-derived clothing-pattern mix from the jagua-rs shirts benchmark with enough space to encourage clustering and a large leftover strip.",
            problem={
                "units": "benchmark_unit",
                "source": {
                    "kind": "benchmark_subset",
                    "benchmark": "shirts.json",
                    "url": f"{JAGUA_ASSET_BASE}/shirts.json",
                    "note": "Subset of literature-derived shirt-pattern pieces from jagua-rs assets.",
                },
                "boards": [_rect_board("board_main", 40.0, 20.0)],
                "widgets": [
                    _benchmark_widget("shirts.json", item_id=0, quantity=2),
                    _benchmark_widget("shirts.json", item_id=1, quantity=2),
                    _benchmark_widget("shirts.json", item_id=2, quantity=2),
                    _benchmark_widget("shirts.json", item_id=5, quantity=2),
                ],
                "config": {
                    "beam_width": 3,
                    "population_size": 4,
                    "generations": 2,
                    "rotation_step_degrees": 180.0,
                    "preferred_corners": ["lower_left", "upper_left"],
                    "max_candidates_per_item": 10,
                    "max_item_anchor_points": 4,
                    "max_free_space_anchor_points": 5,
                },
            },
        )
    )

    cases.append(
        CaseDefinition(
            case_id="trousers_shortage",
            description="Literature-derived trouser-pattern subset with intentional board shortage to test area-maximizing partial fill.",
            problem={
                "units": "benchmark_unit",
                "source": {
                    "kind": "benchmark_subset",
                    "benchmark": "trousers.json",
                    "url": f"{JAGUA_ASSET_BASE}/trousers.json",
                    "note": "Subset of literature-derived trouser-pattern pieces from jagua-rs assets.",
                },
                "boards": [_rect_board("board_main", 100.0, 40.0)],
                "widgets": [
                    _benchmark_widget("trousers.json", item_id=0, quantity=1),
                    _benchmark_widget("trousers.json", item_id=1, quantity=2),
                    _benchmark_widget("trousers.json", item_id=2, quantity=1),
                    _benchmark_widget("trousers.json", item_id=3, quantity=1),
                    _benchmark_widget("trousers.json", item_id=6, quantity=2),
                ],
                "config": {
                    "beam_width": 3,
                    "population_size": 4,
                    "generations": 2,
                    "rotation_step_degrees": 180.0,
                    "preferred_corners": ["lower_left", "lower_right"],
                    "max_candidates_per_item": 10,
                    "max_item_anchor_points": 4,
                    "max_free_space_anchor_points": 5,
                },
            },
        )
    )

    cases.append(
        CaseDefinition(
            case_id="swim_curved_mix",
            description="Literature-derived irregular curved swimwear pieces from the jagua-rs swim benchmark.",
            problem={
                "units": "scaled_benchmark_unit",
                "source": {
                    "kind": "benchmark_subset",
                    "benchmark": "swim.json",
                    "url": f"{JAGUA_ASSET_BASE}/swim.json",
                    "note": "Scaled subset of curved pieces from the swim benchmark in jagua-rs assets.",
                },
                "boards": [_rect_board("board_main", 360.0, 220.0)],
                "widgets": [
                    _benchmark_widget("swim.json", item_id=1, quantity=2, scale=0.18, widget_id="swim_1"),
                    _benchmark_widget("swim.json", item_id=7, quantity=1, scale=0.18, widget_id="swim_7"),
                    _benchmark_widget("swim.json", item_id=8, quantity=1, scale=0.18, widget_id="swim_8"),
                ],
                "config": {
                    "beam_width": 3,
                    "population_size": 4,
                    "generations": 2,
                    "rotation_step_degrees": 30.0,
                    "preferred_corners": ["lower_left", "upper_left"],
                    "max_candidates_per_item": 12,
                    "max_item_anchor_points": 5,
                    "max_free_space_anchor_points": 6,
                },
            },
        )
    )

    cases.extend(_extended_cases(cases))
    if include_large:
        cases.extend(_large_cases(cases))
    if include_diverse:
        from .diverse_cases import build_diverse_cases
        cases.extend(build_diverse_cases(cases, CaseDefinition))
    return {case.case_id: case for case in cases}


def _large_cases(base_cases):
    """Opt-in scaling cases: quantities grow without simplifying input geometry."""
    import copy
    import math
    from shapely.geometry import Polygon

    lookup = {c.case_id: c for c in base_cases}
    result = []
    cfg = {"beam_width": 2, "population_size": 12, "generations": 100000,
           "max_candidates_per_item": 12, "rotation_step_degrees": 90}
    for count in (100, 500, 1000):
        columns = math.ceil(math.sqrt(count))
        rows = math.ceil(count / columns)
        result.append(CaseDefinition(f"large_tiles_{count}",
            "Repeated rectangles with an explicit feasible grid and 10% slack on each board axis.", {
                "units": "test_unit", "source": {"kind": "synthetic_scaling", "known_feasible": True,
                    "grid_columns": columns, "grid_rows": rows, "axis_slack_ratio": 1.1},
                "boards": [_rect_board("board", columns * 11, rows * 6.6)],
                "widgets": [{"id": "tile", "quantity": count, "allowed_angles_degrees": [0,90,180,270],
                             "polygon": {"shell": [[0,0],[10,0],[10,6],[0,6]]}}], "config": dict(cfg)}))
    for base, count, load in (("shirts_combo",100,.65), ("shirts_combo",300,.65),
                              ("alphabet_abcdefghijklmnopqrstuvwxyz",260,.4)):
        raw = copy.deepcopy(lookup[base].problem)
        for i, widget in enumerate(raw['widgets']):
            widget['quantity'] = count // len(raw['widgets']) + (i < count % len(raw['widgets']))
            widget['allowed_angles_degrees'] = [0,90,180,270]
        area = sum(Polygon(w['polygon']['shell'], w['polygon'].get('holes', [])).area * w['quantity']
                   for w in raw['widgets'])
        height = math.sqrt(area / load / 1.5)
        raw['boards'] = [_rect_board('board', height * 1.5, height)]
        raw['config'] = dict(cfg)
        raw['source'].update(scaling_count=count, nominal_area_load=load,
                             note='Repeated source shapes; resized sheet and four orthogonal rotations. Feasibility not assumed.')
        name = f"large_{'shirts' if base == 'shirts_combo' else 'alphabet'}_{count}"
        result.append(CaseDefinition(name, f"{count} instances retaining original concavities and holes.", raw))
    return result


def _extended_cases(base_cases):
    """Deterministic stress cases; all geometry and provenance saved by runners."""
    import copy
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    from shapely.geometry import Polygon

    cfg = {"beam_width": 2, "population_size": 4, "generations": 3,
           "max_candidates_per_item": 12, "rotation_step_degrees": 90}
    def case(name, widgets, width, height, description):
        return CaseDefinition(name, description, {
            "units": "test_unit", "source": {"kind": "synthetic"},
            "boards": [_rect_board("board", width, height)], "widgets": widgets,
            "config": dict(cfg)})
    def widget(name, shell, quantity=1, holes=None, angles=None):
        return {"id": name, "quantity": quantity, "allowed_angles_degrees": angles or [0, 90, 180, 270],
                "polygon": {"shell": shell, "holes": holes or []}}
    rect = lambda w, h: [[0, 0], [w, 0], [w, h], [0, h]]
    cases = []
    for filename in ("complex_dual_board", "constrained_single_board"):
        path = Path(__file__).resolve().parent / "inputs" / f"{filename}.json"
        raw = json.loads(path.read_text())
        raw["config"] = dict(cfg)
        cases.append(CaseDefinition(filename, "Existing repository widget input with bounded benchmark search.", raw))
    cases.append(case("concave_interlock", [widget("L", [[0,0],[4,0],[4,1],[1,1],[1,4],[0,4]], 4)], 8, 4,
                      "Four concave L pieces with complementary orientations."))
    cases.append(case("exact_hole_fill", [widget("frame", rect(10,10), holes=[[[2,2],[8,2],[8,8],[2,8]]]),
                                          widget("plug", rect(6,6))], 10, 10, "Zero-clearance part-in-part fit."))
    cases.append(case("multiple_holes", [widget("frame", rect(20,10), holes=[[[2,2],[7,2],[7,8],[2,8]], [[12,2],[18,2],[18,8],[12,8]]]),
                                         widget("plug", rect(4,5), 2)], 20,10, "Two separate holes in one frame."))
    t = case("slanted_contact", [widget("square", rect(40/9,40/9), angles=[0])], 10, 8,
             "Square touches three triangular board edges without vertex alignment.")
    t.problem["boards"][0]["polygon"]["shell"] = [[0,0],[10,0],[3,8]]
    cases.append(t)
    t = case("board_cutout", [widget("tile", rect(3,3), 8)], 12,12, "Board has an unusable central hole.")
    t.problem["boards"][0]["polygon"]["holes"] = [[[4,4],[8,4],[8,8],[4,8]]]
    cases.append(t)
    cases.append(case("impossible_rotation", [widget("bar", rect(11,2), angles=[0])], 10,10,
                      "Impossible fixed orientation; correct result skips the item."))
    cases.append(case("knapsack", [widget("large", rect(10,7), angles=[0,90]),
                                  widget("medium", rect(5,4), 4, angles=[0,90])], 10,10,
                      "Omitting one large item permits greater total placed area."))
    # Matplotlib ships DejaVu Sans. Flatten its real vector outlines; symmetric
    # difference applies the even-odd fill rule and retains letter counters.
    font = FontProperties(family="DejaVu Sans")
    letters = {}
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        rings = TextPath((0,0), letter, size=20, prop=font).to_polygons()
        shape = Polygon()
        for ring in rings:
            shape = shape.symmetric_difference(Polygon(ring))
        if shape.geom_type != "Polygon" or not shape.is_valid:
            raise ValueError(f"Expected a single valid outline for {letter}")
        letters[letter] = widget(letter, [list(p) for p in shape.exterior.coords[:-1]],
                                 holes=[[list(p) for p in r.coords[:-1]] for r in shape.interiors])
    for group in ("ABCD", "EFGH", "IJKL", "MNOP", "QRST", "UVWXYZ", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        t = case("alphabet_" + group.lower(), [letters[x] for x in group],
                 62 if len(group) < 10 else 115, 25 if len(group) < 10 else 60,
                 "Real DejaVu Sans uppercase outlines, including concavities and counters.")
        t.problem["source"] = {"kind": "font_outlines", "font": "DejaVu Sans",
                                "url": "https://dejavu-fonts.github.io/", "flattening": "matplotlib TextPath.to_polygons"}
        cases.append(t)
    for base in base_cases[2:]:
        raw = copy.deepcopy(base.problem)
        # An explicit stress variant, not the original published benchmark size.
        raw["boards"][0]["polygon"]["shell"][1][0] *= 0.65
        raw["boards"][0]["polygon"]["shell"][2][0] *= 0.65
        raw["config"] = dict(cfg)
        cases.append(CaseDefinition(base.case_id + "_tight", "65% board-width stress variant of " + base.case_id, raw))
    return cases
