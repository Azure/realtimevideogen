from argparse import ArgumentParser
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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


PAPER_FIG_SIZE = (5.5, 2.5)
DPI = 300
Y_TICKS = [1, 10, 100, 1_000, 10_000, 100_000]
Y_TICK_LABELS = ["1", "10", "100", "1k", "10k", "100k"]
STAGE_LABELS = {
    "FramePack VAE": "VAE",
    "FantasyTalking VAE": "VAE",
    "Fantasy Talking VAE": "VAE",
}


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

    fig, ax = plt.subplots(figsize=PAPER_FIG_SIZE)
    x = np.arange(len(df_qpm))
    width = 0.21
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][
        : len(stage_columns)
    ]
    fcfs_bottom = np.zeros(len(df_qpm))
    edf_bottom = np.zeros(len(df_qpm))
    ddit_bottom = np.zeros(len(df_qpm))
    ddit_fcfs_bottom = np.zeros(len(df_qpm))

    for stage_index, (stage, color) in enumerate(
        zip(stage_columns, colors, strict=True)
    ):
        ax.bar(
            x - 1.5 * width,
            edf_costs[:, stage_index],
            width,
            bottom=edf_bottom,
            color=color,
            edgecolor="black",
            linewidth=0.35,
        )
        ax.bar(
            x - 0.5 * width,
            fcfs_costs[:, stage_index],
            width,
            bottom=fcfs_bottom,
            color=color,
            edgecolor="black",
            linewidth=0.35,
            hatch="////",
            alpha=0.7,
        )
        ax.bar(
            x + 0.5 * width,
            ddit_costs[:, stage_index],
            width,
            bottom=ddit_bottom,
            color=color,
            edgecolor="black",
            linewidth=0.35,
            hatch="xxxx",
        )
        ax.bar(
            x + 1.5 * width,
            ddit_fcfs_costs[:, stage_index],
            width,
            bottom=ddit_fcfs_bottom,
            color=color,
            edgecolor="black",
            linewidth=0.35,
            hatch="\\\\\\\\",
            alpha=0.7,
        )
        fcfs_bottom += fcfs_costs[:, stage_index]
        edf_bottom += edf_costs[:, stage_index]
        ddit_bottom += ddit_costs[:, stage_index]
        ddit_fcfs_bottom += ddit_fcfs_costs[:, stage_index]

    x_labels = [
        "Single" if qpm == 0.5 else f"{qpm:g}" for qpm in df_qpm["QPM"]
    ]
    ax.set_yscale("log")
    ax.set_ylim(1, None)
    ax.set_yticks(Y_TICKS)
    ax.set_yticklabels(Y_TICK_LABELS)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Cost ($/hour)")
    ax.set_xlabel("Queries per minute (QPM)")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    tallest_totals = np.maximum(fcfs_totals, ddit_fcfs_totals)
    ax.set_ylim(1, tallest_totals.max() * 9)

    for bar_x, total, saved_percent in zip(
        x - 0.5 * width,
        fcfs_totals,
        savings_percent,
        strict=True,
    ):
        if saved_percent <= 0:
            continue
        ax.text(
            bar_x,
            total * 1.06,
            f"{saved_percent:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )
    for bar_x, total, saved_percent in zip(
        x + 1.5 * width,
        ddit_fcfs_totals,
        ddit_fcfs_vs_stream_pilot_percent,
        strict=True,
    ):
        ax.text(
            bar_x,
            total * 1.06,
            f"{saved_percent:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )
    stage_handles = {
        stage: Patch(
            facecolor=color,
            label=STAGE_LABELS.get(stage, stage),
        )
        for stage, color in zip(stage_columns, colors, strict=True)
    }
    stream_pilot_handle = Patch(
        facecolor="white",
        edgecolor="black",
        label="StreamPilot",
    )
    fcfs_handle = Patch(
        facecolor="white",
        edgecolor="black",
        hatch="////",
        alpha=0.7,
        label="StreamPilot (FCFS)",
    )
    ddit_handle = Patch(
        facecolor="white",
        edgecolor="black",
        hatch="xxxx",
        label="DDiT (EDF)",
    )
    ddit_fcfs_handle = Patch(
        facecolor="white",
        edgecolor="black",
        hatch="\\\\\\\\",
        alpha=0.7,
        label="DDiT (FCFS)",
    )
    legend_handles = [
        stage_handles[stage_columns[0]],
        stage_handles[stage_columns[5]],
        stage_handles[stage_columns[1]],
        stage_handles[stage_columns[6]],
        stage_handles[stage_columns[2]],
        stage_handles[stage_columns[3]],
        stage_handles[stage_columns[4]],
        stream_pilot_handle,
        fcfs_handle,
        ddit_handle,
        ddit_fcfs_handle,
    ]
    fig.legend(
        handles=legend_handles,
        ncols=4,
        loc="upper center",
        bbox_to_anchor=(0.06, 0.72, 0.88, 0.275),
        mode="expand",
        fontsize=ax.xaxis.label.get_fontsize()-1,
        borderaxespad=0,
        borderpad=0.2,
        columnspacing=0.4,
        handlelength=1.4,
        handletextpad=0.4,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.005, 0.005, 0.995, 0.72), pad=0)
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
