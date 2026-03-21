from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _point_map(instance: Dict[str, Any]) -> Dict[str, tuple[float, float]]:
    return {p["id"]: (float(p["x"]), float(p["y"])) for p in instance["points"]}


def render_solution_image(
    instance: Dict[str, Any],
    result: Dict[str, Any],
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """
    Render one solution image.

    Colors:
    - blue: original input paths
    - dark green: generated transition connections between paths
    """
    point_xy = _point_map(instance)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)

    # Draw fixed input paths in blue.
    for path in instance["paths"]:
        x1, y1 = point_xy[path["start"]]
        x2, y2 = point_xy[path["end"]]
        ax.plot([x1, x2], [y1, y2], color="blue", linewidth=2.0, alpha=0.9)

        # Label each path at the segment midpoint.
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        ax.text(mx, my + 0.6, path["id"], color="blue", fontsize=8, ha="center", va="bottom")

    # Draw generated transition connections in dark green.
    for tr in result.get("transitions", []):
        x1, y1 = point_xy[tr["from_exit"]]
        x2, y2 = point_xy[tr["to_entry"]]
        ax.plot([x1, x2], [y1, y2], color="darkgreen", linewidth=2.5, alpha=0.95)

        # Arrow indicates direction from previous path exit to next path entry.
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={"arrowstyle": "->", "color": "darkgreen", "lw": 1.4},
        )

    # Draw all points.
    xs = [xy[0] for xy in point_xy.values()]
    ys = [xy[1] for xy in point_xy.values()]
    ax.scatter(xs, ys, color="black", s=16, zorder=5)

    for pid, (x, y) in point_xy.items():
        ax.text(x, y - 0.9, pid, fontsize=7, color="black", ha="center", va="top")

    ax.set_aspect("equal", adjustable="box")

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad_x = max(2.0, (max_x - min_x) * 0.08)
    pad_y = max(2.0, (max_y - min_y) * 0.20)
    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)

    cost = result.get("total_transition_cost", "?")
    method = result.get("method", "solution")
    ax.set_title(title or f"{method} | transition cost = {cost}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle="--", alpha=0.25)

    legend_handles = [
        Line2D([0], [0], color="blue", lw=2.0, label="Input path"),
        Line2D([0], [0], color="darkgreen", lw=2.5, label="Generated connection"),
    ]
    ax.legend(handles=legend_handles, loc="upper left")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

    return out
