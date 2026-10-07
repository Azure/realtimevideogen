from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SIMULATOR_DIR = Path(__file__).resolve().parent.parent / "simulator"
sys.path.insert(0, str(SIMULATOR_DIR))
try:
    from constants import SECONDS_IN_HOUR
    from plot_utils import _get_time_ticklabels
    from plot_utils import get_color_map
    from utils import get_pareto_frontier_paper
finally:
    sys.path.pop(0)


DPI = 300
PAPER_FIG_SIZE = (5.5, 2.5)
MAX_COST = 225
MAX_TTFF = 10 * SECONDS_IN_HOUR
PODCAST_MINUTES = 10
REQUIRED_COLUMNS = (
    "num_a100",
    "num_h100",
    "num_h200",
    "num_gb200",
    "ttff_s",
    "cost",
)


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, comment="#")
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing_columns)}")
    return data


def get_datapoint(
    data: pd.DataFrame,
    num_a100: int,
    num_h100: int,
    num_h200: int,
    num_gb200: int,
) -> np.ndarray | None:
    filtered = data[
        (data["num_a100"] == num_a100)
        & (data["num_h100"] == num_h100)
        & (data["num_h200"] == num_h200)
        & (data["num_gb200"] == num_gb200)
    ][["ttff_s", "cost"]].sort_values(by=["ttff_s", "cost"])
    if filtered.empty:
        return None
    return filtered.to_numpy()[0]


def get_hardware_subsets(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    gpu_columns = ["num_a100", "num_h100", "num_h200", "num_gb200"]
    subsets: dict[str, pd.DataFrame] = {}
    single_hardware_mask = pd.Series(False, index=data.index)
    for label, column in zip(["A100", "H100", "H200", "GB200"], gpu_columns, strict=True):
        other_columns = [other for other in gpu_columns if other != column]
        mask = (data[column] > 0) & (data[other_columns] == 0).all(axis=1)
        subsets[label] = data[mask]
        single_hardware_mask |= mask
    subsets["Mixed"] = data[~single_hardware_mask]
    return subsets


def get_frontier(data: pd.DataFrame) -> np.ndarray:
    return get_pareto_frontier_paper(
        data[["ttff_s", "cost"]].to_numpy(),
        max_x=MAX_TTFF,
    )


def plot_eval_ttff_cost(
    stream_pilot_path: Path,
    optimal_path: Path,
    naive_spot_path: Path,
    naive_path: Path,
    ddit_path: Path,
    output_path: Path,
) -> None:
    stream_pilot = load_data(stream_pilot_path)
    optimal = load_data(optimal_path)
    combined = pd.concat([stream_pilot, optimal]).reset_index(drop=True)
    naive_spot = load_data(naive_spot_path)
    naive = load_data(naive_path)
    ddit = load_data(ddit_path)

    fig, ax = plt.subplots(figsize=PAPER_FIG_SIZE)
    colors = get_color_map()
    hardware_styles = {
        "A100": (colors[1], "o"),
        "H100": (colors[2], "^"),
        "H200": (colors[3], "s"),
        "GB200": (colors[4], "d"),
        "Mixed": (colors[5], "x"),
    }
    for label, data in get_hardware_subsets(combined).items():
        color, marker = hardware_styles[label]
        data.plot(
            x="ttff_s",
            y="cost",
            kind="scatter",
            ylim=(0, MAX_COST),
            s=80,
            alpha=0.3 if label == "Mixed" else 0.7,
            color=color,
            label=label,
            ax=ax,
            marker=marker,
            zorder=0 if label == "Mixed" else 1,
        )

    ax.text(
        12,
        22,
        "Better",
        ha="center",
        va="center",
        rotation=45,
        size=8,
        bbox={"boxstyle": "larrow,pad=0.2", "fc": "white", "ec": "black", "lw": 1},
    )

    for label, data, color, linestyle in [
        ("Frontier", combined, colors[0], "-"),
        ("Naive", naive_spot, colors[1], "--"),
        ("DDiT", ddit, colors[6], "--"),
    ]:
        frontier = get_frontier(data)
        ax.plot(
            frontier[:, 0],
            frontier[:, 1],
            color=color,
            linewidth=3,
            linestyle=linestyle,
            label=label,
            zorder=0,
        )

    datapoints = {
        "8xA100": get_datapoint(combined, 8, 0, 0, 0),
        "16xA100": get_datapoint(combined, 16, 0, 0, 0),
        "8xH100": get_datapoint(combined, 0, 8, 0, 0),
        "8xGB200": get_datapoint(combined, 0, 0, 0, 8),
        "200x\nGB200": get_datapoint(combined, 0, 0, 0, 200),
        "32xH200": get_datapoint(combined, 0, 0, 32, 0),
        "192xH100": get_datapoint(combined, 0, 192, 0, 0),
        "256xA100\n+64xH200": get_datapoint(combined, 256, 0, 64, 0),
        "1456xH100": get_datapoint(combined, 0, 1456, 0, 0),
    }
    for label, datapoint in datapoints.items():
        if datapoint is not None:
            ax.text(
                datapoint[0],
                datapoint[1] + 15,
                label,
                ha="center",
                va="center",
                bbox={"facecolor": "white", "alpha": 0.5, "edgecolor": "gray"},
                size=8,
            )

    naive_datapoint = get_datapoint(naive, 8, 0, 0, 0)
    naive_spot_datapoint = get_datapoint(naive_spot, 8, 0, 0, 0)
    if naive_datapoint is None or naive_spot_datapoint is None:
        raise ValueError("Naive datasets must contain the 8xA100 configuration")
    ax.plot(
        naive_spot_datapoint[0],
        naive_spot_datapoint[1],
        marker="o",
        markerfacecolor="none",
        linewidth=2,
        markeredgecolor=colors[1],
        markersize=8,
    )
    ax.text(
        naive_spot_datapoint[0] + 15,
        naive_spot_datapoint[1] + 15,
        "Naive\n8xA100",
        ha="center",
        va="center",
        bbox={"facecolor": "white", "alpha": 0.5, "edgecolor": "gray"},
        size=8,
    )

    ticks, tick_labels = _get_time_ticklabels()
    ax.set_xscale("log")
    ax.set_xticks(ticks, tick_labels)
    ax.set_xlim(8, MAX_TTFF)
    ax.set_ylim(0, MAX_COST)
    ax.set_xlabel(r"$TTFF_{eff}$")
    ax.set_ylabel("Cost ($)")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    cost_per_minute_ax = ax.twinx()
    cost_per_minute_ax.set_ylim(0, MAX_COST / PODCAST_MINUTES)
    cost_per_minute_ax.set_ylabel("Cost ($/minute)", rotation=-90, labelpad=15)

    fig.tight_layout(pad=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def parse_args() -> ArgumentParser:
    paper_dir = Path(__file__).resolve().parent
    data_dir = paper_dir / "data"
    parser = ArgumentParser(description="Plot effective TTFF against cost.")
    parser.add_argument("--stream-pilot-data", type=Path, default=data_dir / "provisioning_streamwise.csv")
    parser.add_argument("--optimal-data", type=Path, default=data_dir / "provisioning_streamwise_milp.csv")
    parser.add_argument("--naive-spot-data", type=Path, default=data_dir / "provisioning_naive_spot.csv")
    parser.add_argument("--naive-data", type=Path, default=data_dir / "provisioning_naive.csv")
    parser.add_argument(
        "--ddit-data",
        type=Path,
        default=data_dir / "llm" / "provisioning_ddit_spot_upscaler.csv",
    )
    parser.add_argument("--output", type=Path, default=paper_dir / "figures" / "eval_ttff_cost.pdf")
    return parser


if __name__ == "__main__":
    args = parse_args().parse_args()
    plot_eval_ttff_cost(
        stream_pilot_path=args.stream_pilot_data,
        optimal_path=args.optimal_data,
        naive_spot_path=args.naive_spot_data,
        naive_path=args.naive_data,
        ddit_path=args.ddit_data,
        output_path=args.output,
    )
