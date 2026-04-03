from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.patches import Rectangle

try:
    from .problem import ProblemSpec
    from .solver import SolutionResult
except ImportError:  # pragma: no cover - script mode
    from problem import ProblemSpec
    from solver import SolutionResult


PALETTE = [
    "#0b6e4f",
    "#c97c5d",
    "#3c6997",
    "#d17a22",
    "#7a306c",
    "#2d6a4f",
    "#d64045",
    "#6c757d",
]


def _draw_polygon(ax: Any, polygon: Any, *, facecolor: str, edgecolor: str, linewidth: float, alpha: float) -> None:
    shell = PolygonPatch(list(polygon.exterior.coords), closed=True, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha)
    ax.add_patch(shell)
    for ring in polygon.interiors:
        hole = PolygonPatch(list(ring.coords), closed=True, facecolor="white", edgecolor=edgecolor, linewidth=max(0.5, linewidth * 0.5), alpha=1.0)
        ax.add_patch(hole)


def render_solution_image(problem: ProblemSpec, solution: SolutionResult, output_path: str | Path) -> Path:
    boards = [board.polygon.to_polygon(name=f"board:{board.board_id}") for board in problem.boards]
    placements_by_board: dict[str, list[Any]] = {board.board_id: [] for board in problem.boards}
    for placement in solution.placements:
        placements_by_board[placement.board_id].append(placement)

    fig, axes = plt.subplots(1, len(problem.boards), figsize=(8 * len(problem.boards), 7), dpi=180)
    if len(problem.boards) == 1:
        axes = [axes]

    for board_index, (ax, board_spec, board_polygon, board_metric) in enumerate(zip(axes, problem.boards, boards, solution.board_metrics)):
        _draw_polygon(ax, board_polygon, facecolor="#f5f1e8", edgecolor="#2b2d42", linewidth=2.0, alpha=1.0)

        placements = sorted(placements_by_board[board_spec.board_id], key=lambda placement: placement.area, reverse=True)
        for placement_index, placement in enumerate(placements):
            color = PALETTE[placement_index % len(PALETTE)]
            _draw_polygon(ax, placement.polygon, facecolor=color, edgecolor="#1f2933", linewidth=1.2, alpha=0.92)
            ax.text(
                placement.centroid_x,
                placement.centroid_y,
                placement.item_instance_id,
                fontsize=7,
                ha="center",
                va="center",
                color="white",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "#1f2933", "edgecolor": "none", "alpha": 0.75},
            )

        if board_metric["rest_rectangles"]:
            best_rectangle = max(board_metric["rest_rectangles"], key=lambda entry: entry["area"])
            min_x, min_y, max_x, max_y = best_rectangle["bounds"]
            rect = Rectangle(
                (min_x, min_y),
                max_x - min_x,
                max_y - min_y,
                facecolor="#90be6d",
                edgecolor="#386641",
                linewidth=1.5,
                linestyle="--",
                alpha=0.25,
            )
            ax.add_patch(rect)

        min_x, min_y, max_x, max_y = board_polygon.bounds
        pad_x = max(10.0, (max_x - min_x) * 0.08)
        pad_y = max(10.0, (max_y - min_y) * 0.08)
        ax.set_xlim(min_x - pad_x, max_x + pad_x)
        ax.set_ylim(min_y - pad_y, max_y + pad_y)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"{board_spec.board_id}\nutil={board_metric['utilization']:.3f} | max rest rect={board_metric['max_rest_rectangle_area']:.1f}"
        )
        ax.set_xlabel(problem.units)
        ax.set_ylabel(problem.units)
        ax.grid(True, linestyle="--", alpha=0.20)

    placed_count = len(solution.placements)
    skipped_count = len(solution.skipped_item_ids)
    fig.suptitle(
        f"Widget nesting | placed area={solution.placed_area:.1f} | placed={placed_count} | skipped={skipped_count}",
        fontsize=14,
    )
    fig.tight_layout()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
