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
    from deadline_scheduling import get_cost_savings
    from deadline_scheduling import get_latency_matched_reference
    from deadline_scheduling import get_load_aware_fcfs_costs
finally:
    sys.path.pop(0)


PAPER_FIG_SIZE = (4.0, 3.6)
DPI = 300
LEGEND_FONT_SIZE = 9
LEGEND_ANCHOR = (1.01, 0.5)
STAGE_LEGEND_ANCHOR = (1.01, 0.38)
STAGE_Y_TICKS = [1, 10, 100, 1_000, 10_000, 100_000]
STAGE_Y_TICK_LABELS = ["1", "10", "100", "1k", "10k", "100k"]
TOTAL_Y_TICKS = [100, 1_000, 10_000, 100_000, 1_000_000]
TOTAL_Y_TICK_LABELS = ["100", "1k", "10k", "100k", "1M"]
BAR_WIDTH = 0.2
# (label, face color, hatch, alpha) for each total-cost bar series.
BAR_STYLES = (
    ("StreamPilot", "#404040", "", 1.0),
    ("StreamPilot (FCFS)", "#404040", "////", 0.55),
    ("DDiT (EDF)", "#b0b0b0", "xxxx", 1.0),
    ("DDiT (FCFS)", "#b0b0b0", "\\\\\\\\", 0.55),
)


def plot_stage_costs(
    ax: plt.Axes,
    qpm_values: np.ndarray,
    x_labels: list[str],
    stage_costs: pd.DataFrame,
) -> None:
    for stage in stage_costs.columns:
        ax.plot(qpm_values, stage_costs[stage], marker="o", label=stage)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1, None)
    ax.set_xticks(qpm_values, x_labels)
    ax.set_yticks(STAGE_Y_TICKS, STAGE_Y_TICK_LABELS)
    ax.set_ylabel("Cost ($/hour)")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(
        loc="center left",
        bbox_to_anchor=STAGE_LEGEND_ANCHOR,
        fontsize=LEGEND_FONT_SIZE,
    )


def plot_total_costs(
    ax: plt.Axes,
    x_labels: list[str],
    totals: list[np.ndarray],
    fcfs_savings_percent: list[np.ndarray],
) -> None:
    x = np.arange(len(x_labels))
    offsets = (np.arange(len(BAR_STYLES)) - (len(BAR_STYLES) - 1) / 2) * BAR_WIDTH
    for offset, series, (label, color, hatch, alpha) in zip(
        offsets, totals, BAR_STYLES, strict=True
    ):
        ax.bar(
            x + offset,
            series,
            BAR_WIDTH,
            color=color,
            alpha=alpha,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.35,
            label=label,
        )
    # Annotate FCFS bars with the StreamPilot cost saving relative to them.
    for offset, series, percents in zip(
        offsets[[1, 3]], totals[1::2], fcfs_savings_percent, strict=True
    ):
        for bar_x, total, percent in zip(x + offset, series, percents, strict=True):
            if percent <= 0:
                continue
            ax.text(
                bar_x,
                total * 1.15,
                f"{percent:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    ax.set_yscale("log")
    ax.set_yticks(TOTAL_Y_TICKS, TOTAL_Y_TICK_LABELS)
    ax.set_ylim(TOTAL_Y_TICKS[0], TOTAL_Y_TICKS[-1] * 10)
    ax.set_xticks(x, x_labels)
    ax.set_xlim(-0.5, len(x_labels) - 0.5)
    ax.set_ylabel("Cost ($/hour)")
    ax.set_xlabel("Queries per minute (QPM)")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(
        loc="center left",
        bbox_to_anchor=LEGEND_ANCHOR,
        fontsize=LEGEND_FONT_SIZE,
    )


def plot_fcfs_vs_edf(
    data_path: Path,
    stream_pilot_single_path: Path,
    ddit_single_path: Path,
    output_path: Path,
) -> None:
    df_qpm = pd.read_csv(data_path)
    stage_columns = [
        column for column in df_qpm.columns if column not in {"QPM", "Total"}
    ]
    edf_costs = df_qpm[stage_columns].to_numpy()
    plotted_edf_totals = edf_costs.sum(axis=1)
    edf_totals = (
        df_qpm["Total"].to_numpy()
        if "Total" in df_qpm
        else plotted_edf_totals
    )
    fcfs_costs = get_load_aware_fcfs_costs(edf_costs)
    fcfs_totals = fcfs_costs.sum(axis=1)
    savings, savings_percent = get_cost_savings(
        fcfs_totals,
        edf_totals,
    )

    df_stream_pilot_single = pd.read_csv(
        stream_pilot_single_path,
        comment="#",
    )
    df_ddit_single = pd.read_csv(
        ddit_single_path,
        comment="#",
    )
    (
        reference_ttff,
        reference_cost,
        ddit_reference_ttff,
        ddit_single_cost,
        ddit_met_reference_ttff,
    ) = get_latency_matched_reference(
        target_cost=edf_totals[0],
        reference_ttff=df_stream_pilot_single["ttff_s"].to_numpy(),
        reference_costs=df_stream_pilot_single["cost"].to_numpy(),
        candidate_ttff=df_ddit_single["ttff_s"].to_numpy(),
        candidate_costs=df_ddit_single["cost"].to_numpy(),
    )
    ddit_scale = ddit_single_cost / reference_cost
    ddit_costs = edf_costs * ddit_scale
    ddit_totals = ddit_costs.sum(axis=1)
    ddit_fcfs_costs = get_load_aware_fcfs_costs(ddit_costs)
    ddit_fcfs_totals = ddit_fcfs_costs.sum(axis=1)
    _, ddit_savings_percent = get_cost_savings(
        ddit_fcfs_totals,
        ddit_totals,
    )
    _, ddit_fcfs_vs_stream_pilot_percent = get_cost_savings(
        ddit_fcfs_totals,
        edf_totals,
    )

    qpm_values = df_qpm["QPM"].to_numpy()
    x_labels = [
        "Single" if qpm == 0.5 else f"{qpm:g}" for qpm in qpm_values
    ]
    fig, (stage_ax, total_ax) = plt.subplots(
        2,
        1,
        figsize=PAPER_FIG_SIZE,
        gridspec_kw={"hspace": 0.3},
    )
    plot_stage_costs(stage_ax, qpm_values, x_labels, df_qpm[stage_columns])
    plot_total_costs(
        total_ax,
        x_labels,
        [edf_totals, fcfs_totals, ddit_totals, ddit_fcfs_totals],
        [savings_percent, ddit_fcfs_vs_stream_pilot_percent],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(
        f"{'QPM':>8} {'StreamPilot FCFS':>17} "
        f"{'StreamPilot':>14} {'DDiT EDF':>12} "
        f"{'DDiT FCFS':>12} {'SP save':>10} {'DDiT save':>11}"
    )
    for (
        qpm_label,
        fcfs,
        stream_pilot,
        ddit,
        ddit_fcfs,
        saved_percent,
        ddit_saved_percent,
    ) in zip(
        x_labels,
        fcfs_totals,
        edf_totals,
        ddit_totals,
        ddit_fcfs_totals,
        savings_percent,
        ddit_savings_percent,
        strict=True,
    ):
        print(
            f"{qpm_label:>8} {fcfs:>17.2f} "
            f"{stream_pilot:>14.2f} {ddit:>12.2f} "
            f"{ddit_fcfs:>12.2f} "
            f"{saved_percent:>9.2f}% {ddit_saved_percent:>9.2f}%"
        )
    comparison = "met" if ddit_met_reference_ttff else "could not meet"
    print(
        f"DDiT {comparison} the {reference_ttff:.2f}s StreamPilot "
        f"reference TTFF; selected {ddit_reference_ttff:.2f}s DDiT "
        f"point at ${ddit_single_cost:.2f} vs ${reference_cost:.2f}."
    )


def parse_args() -> ArgumentParser:
    paper_dir = Path(__file__).resolve().parent
    parser = ArgumentParser(
        description="Compare homogeneous-SLO FCFS and StreamPilot costs."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=paper_dir / "data" / "provisioning_qpm.csv",
        help="StreamPilot provisioning QPM CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            paper_dir
            / "figures"
            / "eval_cost_qpm_barplot_fcfs_vs_edf.pdf"
        ),
        help="Output PDF path.",
    )
    parser.add_argument(
        "--stream-pilot-single-data",
        type=Path,
        default=paper_dir / "data" / "provisioning_streamwise.csv",
        help="StreamPilot single-request latency-cost results.",
    )
    parser.add_argument(
        "--ddit-single-data",
        type=Path,
        default=paper_dir / "data" / "llm" / "provisioning_ddit.csv",
        help="DDiT single-request latency-cost results.",
    )
    return parser


if __name__ == "__main__":
    args = parse_args().parse_args()
    plot_fcfs_vs_edf(
        data_path=args.data,
        stream_pilot_single_path=args.stream_pilot_single_data,
        ddit_single_path=args.ddit_single_data,
        output_path=args.output,
    )
