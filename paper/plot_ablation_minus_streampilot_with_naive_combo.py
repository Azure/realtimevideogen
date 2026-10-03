from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DPI = 300
PAPER_FIG_SIZE = (5.5, 2)
SECONDS_IN_HOUR = 60 * 60
MAX_FRONTIER_TTFF = 2 * SECONDS_IN_HOUR
MAX_FRONTIER_COST = 5_000
MAX_PLOT_TTFF = 1.5 * SECONDS_IN_HOUR
MAX_PLOT_COST = 225
REQUIRED_COLUMNS = ("ttff_s", "cost")
PRIMARY_LABEL = "StreamPilot"
LEGEND_COLUMNS = 2


@dataclass(frozen=True)
class CurveSpec:
    label: str
    filename: str


CURVE_SPECS = (
    CurveSpec("StreamPilot", "provisioning_streamwise.csv"),
    CurveSpec("No Spot", "provisioning_streamwise_no_spot.csv"),
    CurveSpec("No disaggregation", "provisioning_streamwise_no_disag.csv"),
    CurveSpec("No upscaler", "provisioning_streamwise_no_upscaler.csv"),
    CurveSpec("Naive", "provisioning_streamwise_naive.csv"),
    CurveSpec("Static Allocation", "provisioning_streamwise_naive_allocator.csv"),
    CurveSpec("Naive Combo", "provisioning_naive_combo.csv"),
    CurveSpec("StreamPilot (A100)", "provisioning_streamwise_A100.csv"),
    CurveSpec("Optimal", "provisioning_streamwise_milp.csv"),
)


def load_curve_points(data_path: Path) -> np.ndarray:
    if not data_path.is_file():
        raise FileNotFoundError(f"Ablation data file does not exist: {data_path}")

    try:
        data = pd.read_csv(data_path, comment="#")
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"Ablation data file is empty: {data_path}") from error

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(
            f"Ablation data file {data_path} is missing required columns: {', '.join(missing_columns)}"
        )

    metrics = data.loc[:, list(REQUIRED_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    points = metrics.to_numpy(dtype=float)
    if points.shape[0] == 0:
        raise ValueError(f"Ablation data file has no rows: {data_path}")
    if not np.isfinite(points).all():
        raise ValueError(f"Ablation data file contains non-numeric or non-finite metrics: {data_path}")
    if (points[:, 0] <= 0).any():
        raise ValueError(f"Ablation data file contains non-positive TTFF values: {data_path}")
    if (points[:, 1] < 0).any():
        raise ValueError(f"Ablation data file contains negative cost values: {data_path}")

    return points


def get_pareto_frontier(
    points: np.ndarray,
    max_y: float | None = None,
    max_x: float | None = None,
) -> np.ndarray:
    if points.size == 0:
        return points.copy()
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Pareto frontier points must have shape (n, 2)")

    sorted_points = points[np.lexsort((points[:, 1], points[:, 0]))]
    pareto_front = [sorted_points[0]]
    for point in sorted_points[1:]:
        if point[1] < pareto_front[-1][1]:
            pareto_front.append(point)

    extreme_point_start = np.array([pareto_front[0][0], max(points[:, 1])])
    extreme_point_end = np.array([max(points[:, 0]), pareto_front[-1][1]])
    pareto_front.append(extreme_point_start)
    pareto_front.append(extreme_point_end)

    if max_x is not None:
        candidate = np.array([max_x, min(points[:, 1])])
        if candidate[0] > pareto_front[-1][0] and candidate[1] <= pareto_front[-1][1]:
            pareto_front.append(candidate)
    if max_y is not None:
        candidate = np.array([min(points[:, 0]), max_y])
        if candidate[1] > pareto_front[0][1] and candidate[0] <= pareto_front[0][0]:
            pareto_front.append(candidate)

    frontier = np.array(pareto_front)
    frontier = frontier[np.lexsort((-frontier[:, 1], frontier[:, 0]))]
    _, unique_indices = np.unique(frontier, axis=0, return_index=True)
    return frontier[np.sort(unique_indices)]


def get_time_ticklabels(exclude_x: set[int] | None = None) -> tuple[list[int], list[str]]:
    excluded = exclude_x or set()
    ticks_seconds = [1, 2, 5, 10, 15, 30]
    minute_ticks = [1, 2, 5, 10, 20, 40]
    hour_ticks = [1, 3, 5, 8, 12]
    day_ticks = [1]

    ticks = (
        ticks_seconds
        + [minutes * 60 for minutes in minute_ticks]
        + [hours * SECONDS_IN_HOUR for hours in hour_ticks]
        + [days * 24 * SECONDS_IN_HOUR for days in day_ticks]
    )
    labels = (
        [f"{seconds:g}s" for seconds in ticks_seconds]
        + [f"{minutes:g}m" for minutes in minute_ticks]
        + [f"{hours:g}h" for hours in hour_ticks]
        + [f"{days:g}d" for days in day_ticks]
    )

    filtered = [(tick, label) for tick, label in zip(ticks, labels) if tick not in excluded]
    return [tick for tick, _ in filtered], [label for _, label in filtered]


def plot_ablation(data_dir: Path, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=PAPER_FIG_SIZE)
    try:
        for curve in CURVE_SPECS:
            points = load_curve_points(data_dir / curve.filename)
            pareto_front = get_pareto_frontier(
                points,
                max_y=MAX_FRONTIER_COST,
                max_x=MAX_FRONTIER_TTFF,
            )
            ax.plot(
                pareto_front[:, 0],
                pareto_front[:, 1],
                linewidth=3,
                label=curve.label,
                zorder=1 if curve.label == PRIMARY_LABEL else 0,
            )

        ticks, tick_labels = get_time_ticklabels(exclude_x={5 * 60})
        ax.set_xscale("log")
        ax.set_xticks(ticks, tick_labels)
        ax.set_xlim(10, MAX_PLOT_TTFF)
        ax.set_xlabel("TTFF", labelpad=-8)

        ax.set_ylim(0, MAX_PLOT_COST)
        ax.set_ylabel("Cost ($)")
        ax.legend(loc="upper right", fontsize=9, ncol=LEGEND_COLUMNS)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)

        fig.tight_layout(pad=0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    finally:
        plt.close(fig)


def parse_args() -> ArgumentParser:
    paper_dir = Path(__file__).resolve().parent
    parser = ArgumentParser(description="Plot the StreamPilot ablation with Naive Combo.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=paper_dir / "data",
        help="Directory containing the ablation CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=paper_dir / "figures" / "ablation_minus_streampilot_with_naive_combo.pdf",
        help="Output PDF path.",
    )
    return parser


if __name__ == "__main__":
    args = parse_args().parse_args()
    plot_ablation(args.data_dir, args.output)
