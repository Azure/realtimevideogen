from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd


SIMULATOR_DIR = Path(__file__).resolve().parent.parent / "simulator"
sys.path.insert(0, str(SIMULATOR_DIR))
try:
    from constants import SECONDS_IN_HOUR
    from plot_utils import _get_time_ticklabels
    from utils import get_pareto_frontier_paper
finally:
    sys.path.pop(0)


DPI = 300
PAPER_FIG_SIZE = (5.5, 2)
MAX_FRONTIER_TTFF = 10 * SECONDS_IN_HOUR
MAX_PLOT_TTFF = 10 * 60
MAX_COST = 85
SLIDE_SECONDS = 0.5


def load_frontier(path: Path, max_cost: float | None = None) -> np.ndarray:
    data = pd.read_csv(path, comment="#")
    missing_columns = sorted({"ttff_s", "cost"} - set(data.columns))
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing_columns)}")
    return get_pareto_frontier_paper(
        data[["ttff_s", "cost"]].to_numpy(),
        max_x=MAX_FRONTIER_TTFF,
        max_y=max_cost,
    )


def plot_quality_frontier(data_dir: Path, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=PAPER_FIG_SIZE)
    for quality in ["high", "medium", "low"]:
        frontier = load_frontier(
            data_dir / f"provisioning_streamwise_{quality}.csv",
            max_cost=100,
        )
        ax.plot(
            frontier[:, 0],
            frontier[:, 1],
            linewidth=3,
            label=quality.capitalize(),
        )

    adaptive_frontier = load_frontier(data_dir / "provisioning_streamwise_adaptive.csv")
    ax.plot(
        adaptive_frontier[:, 0],
        adaptive_frontier[:, 1],
        linewidth=3,
        linestyle="--",
        label="Adaptive",
    )
    ax.plot(
        adaptive_frontier[:, 0] - SLIDE_SECONDS,
        adaptive_frontier[:, 1],
        linewidth=3,
        linestyle="-.",
        label="+Slide",
    )

    ticks, tick_labels = _get_time_ticklabels(exclude_x=[15])
    ax.set_xscale("log")
    ax.set_xticks(ticks, tick_labels)
    ax.set_xlim(0.5, MAX_PLOT_TTFF)
    ax.set_xlabel(r"$TTFF_{eff}$", labelpad=-2)
    ax.set_ylabel("Cost ($)")
    ax.set_ylim(0, MAX_COST)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, which="major", linestyle="--", alpha=0.7)
    ax.grid(True, which="minor", linestyle=":", alpha=0.4)

    cost_per_minute_ax = ax.twinx()
    cost_per_minute_ax.set_ylim(0, MAX_COST / 10)
    cost_per_minute_ax.set_ylabel("Cost ($/minute)", rotation=-90, labelpad=15)

    fig.tight_layout(pad=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def parse_args() -> ArgumentParser:
    paper_dir = Path(__file__).resolve().parent
    parser = ArgumentParser(description="Plot effective TTFF and cost across quality levels.")
    parser.add_argument("--data-dir", type=Path, default=paper_dir / "data")
    parser.add_argument(
        "--output",
        type=Path,
        default=paper_dir / "figures" / "eval_ttff_cost_quality_frontier.pdf",
    )
    return parser


if __name__ == "__main__":
    args = parse_args().parse_args()
    plot_quality_frontier(args.data_dir, args.output)
