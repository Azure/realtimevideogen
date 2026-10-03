from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PAPER_FIG_SIZE = (5.5, 2)
DPI = 300
Y_TICKS = [1, 10, 100, 1_000, 10_000, 100_000]
Y_TICK_LABELS = ["1", "10", "100", "1k", "10k", "100k"]


def plot_cost_qpm_barplot(data_path: Path, output_path: Path) -> None:
    df_qpm = pd.read_csv(data_path)
    stage_columns = [
        column for column in df_qpm.columns if column not in {"QPM", "Total"}
    ]

    fig, ax = plt.subplots(figsize=PAPER_FIG_SIZE)
    df_qpm.set_index("QPM")[stage_columns].plot.bar(
        stacked=True,
        logy=True,
        width=0.75,
        ylabel="Cost ($/hour)",
        xlabel="Queries per minute (QPM)",
        ax=ax,
    )

    x_labels = [
        "Single" if qpm == 0.5 else f"{qpm:g}" for qpm in df_qpm["QPM"]
    ]
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_ylim(1, None)
    ax.set_yticks(Y_TICKS)
    ax.set_yticklabels(Y_TICK_LABELS)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(
        ncols=1,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=9,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def parse_args() -> ArgumentParser:
    paper_dir = Path(__file__).resolve().parent
    parser = ArgumentParser(
        description="Plot per-stage hourly cost as stacked bars over QPM."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=paper_dir / "data" / "provisioning_qpm.csv",
        help="Input provisioning QPM CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=paper_dir / "figures" / "eval_cost_qpm_barplot.pdf",
        help="Output PDF path.",
    )
    return parser


if __name__ == "__main__":
    args = parse_args().parse_args()
    plot_cost_qpm_barplot(args.data, args.output)
