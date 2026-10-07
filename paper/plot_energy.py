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
MAX_ENERGY_KWH = 16
MAX_FRONTIER_TTFF = 10 * SECONDS_IN_HOUR
MAX_PLOT_TTFF = 1.5 * SECONDS_IN_HOUR
WATT_SECONDS_PER_KWH = SECONDS_IN_HOUR * 1000
REQUIRED_COLUMNS = (
    "num_a100",
    "num_h100",
    "num_h200",
    "num_gb200",
    "ttff_s",
    "energy",
)


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, comment="#")
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing_columns)}")
    if (data["energy"] <= 0).any():
        raise ValueError(f"{path} must contain positive energy measurements")
    data["total_energy_kWh"] = data["energy"] / WATT_SECONDS_PER_KWH
    return data


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
        data[["ttff_s", "total_energy_kWh"]].to_numpy(),
        max_y=5000,
        max_x=MAX_FRONTIER_TTFF,
    )


def plot_energy(
    stream_pilot_path: Path,
    naive_path: Path,
    ddit_path: Path,
    output_path: Path,
) -> None:
    stream_pilot = load_data(stream_pilot_path)
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
    for label, data in get_hardware_subsets(stream_pilot).items():
        color, marker = hardware_styles[label]
        data.plot(
            x="ttff_s",
            y="total_energy_kWh",
            kind="scatter",
            s=80,
            alpha=0.3 if label == "Mixed" else 0.7,
            color=color,
            label=label,
            ax=ax,
            marker=marker,
            zorder=0 if label == "Mixed" else 1,
        )

    for label, data, color, linestyle in [
        ("Frontier", stream_pilot, colors[0], "-"),
        ("Naive", naive, colors[1], "--"),
        ("DDiT", ddit, colors[6], "--"),
    ]:
        frontier = get_frontier(data)
        ax.plot(
            frontier[:, 0],
            frontier[:, 1],
            linewidth=3,
            color=color,
            linestyle=linestyle,
            label=label,
            zorder=0,
        )

    ticks, tick_labels = _get_time_ticklabels()
    ax.set_xscale("log")
    ax.set_xticks(ticks, tick_labels)
    ax.set_xlim(8, MAX_PLOT_TTFF)
    ax.set_ylim(0, MAX_ENERGY_KWH)
    ax.set_xlabel(r"$TTFF_{eff}$", labelpad=-2)
    ax.set_ylabel("Energy (kWh)")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    fig.tight_layout(pad=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def parse_args() -> ArgumentParser:
    paper_dir = Path(__file__).resolve().parent
    data_dir = paper_dir / "data"
    parser = ArgumentParser(description="Plot effective TTFF against energy consumption.")
    parser.add_argument("--stream-pilot-data", type=Path, default=data_dir / "provisioning_streamwise.csv")
    parser.add_argument("--naive-data", type=Path, default=data_dir / "provisioning_naive.csv")
    parser.add_argument(
        "--ddit-data",
        type=Path,
        default=data_dir / "llm" / "provisioning_ddit_spot_upscaler.csv",
    )
    parser.add_argument("--output", type=Path, default=paper_dir / "figures" / "energy.pdf")
    return parser


if __name__ == "__main__":
    args = parse_args().parse_args()
    plot_energy(
        stream_pilot_path=args.stream_pilot_data,
        naive_path=args.naive_data,
        ddit_path=args.ddit_data,
        output_path=args.output,
    )
