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
PAPER_FIG_SIZE = (5.5, 2)
MAX_TTFF = 10 * SECONDS_IN_HOUR
MAX_COST = 425


def load_frontier(path: Path) -> np.ndarray:
    data = pd.read_csv(path, comment="#")
    missing_columns = sorted({"ttff_s", "cost"} - set(data.columns))
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing_columns)}")
    return get_pareto_frontier_paper(
        data[["ttff_s", "cost"]].to_numpy(),
        max_y=5_000,
        max_x=MAX_TTFF,
    )


def get_llm_color(label: str) -> tuple[float, float, float]:
    colors = get_color_map()
    color_map = {
        "StreamPilot": colors[0],
        "Naive": colors[1],
        "HexGen": colors[2],
        "Helix": colors[3],
        "DDiT": colors[4],
    }
    clean_label = label.replace(" (Spot)", "")
    return color_map.get(clean_label, (0.0, 0.0, 0.0))


def plot_eval_llm(data_dir: Path, output_path: Path) -> None:
    curve_paths = {
        "StreamPilot": data_dir / "provisioning_streamwise.csv",
        "HexGen": data_dir / "llm" / "provisioning_hexgen.csv",
        "Helix": data_dir / "llm" / "provisioning_helix.csv",
        "DDiT": data_dir / "llm" / "provisioning_ddit.csv",
        "HexGen (Spot)": data_dir / "llm" / "provisioning_hexgen_spot.csv",
        "Helix (Spot)": data_dir / "llm" / "provisioning_helix_spot.csv",
        "DDiT (Spot)": data_dir / "llm" / "provisioning_ddit_spot.csv",
        "Naive": data_dir / "provisioning_naive.csv",
        "Naive (Spot)": data_dir / "provisioning_naive_spot.csv",
    }

    fig, ax = plt.subplots(figsize=PAPER_FIG_SIZE)
    for label, path in curve_paths.items():
        frontier = load_frontier(path)
        ax.plot(
            frontier[:, 0],
            frontier[:, 1],
            color=get_llm_color(label),
            linewidth=3,
            linestyle="--" if "Spot" in label else "-",
            label=label if "Spot" not in label else "",
            zorder=0,
        )

    ticks, tick_labels = _get_time_ticklabels(exclude_x=[5 * 60, 10 * 60])
    ax.set_xscale("log")
    ax.set_xticks(ticks, tick_labels)
    ax.set_xlim(8, MAX_TTFF)
    ax.set_xlabel(r"$TTFF_{eff}$", labelpad=-8)
    ax.set_ylabel("Cost ($)")
    ax.set_ylim(0, MAX_COST)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    fig.tight_layout(pad=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def parse_args() -> ArgumentParser:
    paper_dir = Path(__file__).resolve().parent
    parser = ArgumentParser(description="Plot LLM-inspired allocation baselines.")
    parser.add_argument("--data-dir", type=Path, default=paper_dir / "data")
    parser.add_argument("--output", type=Path, default=paper_dir / "figures" / "eval_llm.pdf")
    return parser


if __name__ == "__main__":
    args = parse_args().parse_args()
    plot_eval_llm(args.data_dir, args.output)
