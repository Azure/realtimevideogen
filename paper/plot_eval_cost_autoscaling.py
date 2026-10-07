"""Cost of a warm-pool autoscaled deployment replaying the Azure LMM production trace.

Compares StreamPilot and DDiT resizing the warm GPU pool at the same SLO attainment (99%), for a static pool,
a reactive autoscaler, and a predictive (forecast-driven) warm pool. See ``simulator/autoscaling.py``.
"""
from argparse import ArgumentParser
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


SIMULATOR_DIR = Path(__file__).resolve().parent.parent / "simulator"
sys.path.insert(0, str(SIMULATOR_DIR))
try:
    from autoscaling import AutoscalingConfig
    from autoscaling import MINUTES_PER_DAY
    from autoscaling import Predictor
    from autoscaling import SimulationResult
    from autoscaling import forecast_error
    from autoscaling import load_stage_cost_curve
    from autoscaling import load_trace
    from autoscaling import run_experiment
    from autoscaling import stream_pilot_and_ddit
    from deadline_scheduling import get_latency_matched_reference
finally:
    sys.path.pop(0)


PAPER_FIG_SIZE = (4.0, 3.6)
DPI = 300
LEGEND_FONT_SIZE = 9
# Right of the timeline twin axis; shared by both panels so the legends align.
LEGEND_ANCHOR = (1.18, 0.5)
TIMELINE_BIN_MIN = 30
BAR_WIDTH = 0.36
TIMELINE_STRATEGY = "Predictive warm pool"
STRATEGY_LABELS = {
    "Static peak": "Static",
    "Reactive": "Reactive",
    "Predictive warm pool": "Predictive",
}
# (system, color, line style) in plotting order.
SYSTEM_STYLES = (
    ("StreamPilot", "#404040", "-"),
    ("DDiT", "#b0b0b0", "--"),
)
TIMELINE_COLORS = {"StreamPilot": "#202020", "DDiT": "#8c8c8c"}
WARM_HATCH = "////"
LOAD_COLOR = "#cfe2f3"


def get_ddit_cost_multiplier(qpm_path: Path, stream_pilot_single_path: Path, ddit_single_path: Path) -> float:
    """Latency-matched DDiT/StreamPilot cost ratio (same as the steady-state QPM figure)."""
    df_qpm = pd.read_csv(qpm_path)
    df_stream_pilot = pd.read_csv(stream_pilot_single_path, comment="#")
    df_ddit = pd.read_csv(ddit_single_path, comment="#")
    _, reference_cost, _, ddit_cost, _ = get_latency_matched_reference(
        target_cost=float(df_qpm["Total"].iloc[0]),
        reference_ttff=df_stream_pilot["ttff_s"].to_numpy(),
        reference_costs=df_stream_pilot["cost"].to_numpy(),
        candidate_ttff=df_ddit["ttff_s"].to_numpy(),
        candidate_costs=df_ddit["cost"].to_numpy(),
    )
    return float(ddit_cost / reference_cost)


def _bin(values: np.ndarray, size: int) -> np.ndarray:
    usable = len(values) // size * size
    return np.asarray(values[:usable].reshape(-1, size).mean(axis=1))


def plot_timeline(
    ax: plt.Axes,
    arrivals: np.ndarray,
    results: dict[str, SimulationResult],
) -> None:
    hours = _bin(np.arange(len(arrivals)) / 60.0, TIMELINE_BIN_MIN)
    load_ax = ax.twinx()
    load_ax.fill_between(hours, _bin(arrivals, TIMELINE_BIN_MIN), color=LOAD_COLOR, linewidth=0, label="Load")
    load_ax.set_ylim(0, None)
    load_ax.set_ylabel("Load (QPM)")
    ax.set_zorder(load_ax.get_zorder() + 1)
    ax.patch.set_visible(False)
    for system, color, style in SYSTEM_STYLES:
        result = results[system]
        ax.plot(
            hours,
            _bin(result.cost_rate, TIMELINE_BIN_MIN) / 1000,
            color=TIMELINE_COLORS.get(system, color),
            linestyle=style,
            linewidth=1.1,
            label=system,
        )
    num_days = len(arrivals) // MINUTES_PER_DAY
    ax.set_xticks(np.arange(num_days + 1) * 24, [str(day) for day in range(num_days + 1)])
    ax.set_xlim(0, hours[-1])
    ax.set_ylim(0, None)
    ax.set_xlabel("Time (days)", labelpad=1)
    ax.set_ylabel("Cost (k$/hour)")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    load_handles, load_labels = load_ax.get_legend_handles_labels()
    ax.legend(
        handles + load_handles,
        labels + load_labels,
        title=STRATEGY_LABELS[TIMELINE_STRATEGY],
        title_fontsize=LEGEND_FONT_SIZE,
        loc="center left",
        bbox_to_anchor=LEGEND_ANCHOR,
        fontsize=LEGEND_FONT_SIZE,
    )


def plot_strategy_bars(
    ax: plt.Axes,
    results: dict[str, dict[str, SimulationResult]],
) -> None:
    strategies = list(results)
    x = np.arange(len(strategies))
    offsets = (np.arange(len(SYSTEM_STYLES)) - (len(SYSTEM_STYLES) - 1) / 2) * BAR_WIDTH
    for offset, (system, color, _) in zip(offsets, SYSTEM_STYLES, strict=True):
        serving = np.array([results[strategy][system].avg_serving_cost for strategy in strategies]) / 1000
        warm = np.array([results[strategy][system].avg_warm_cost for strategy in strategies]) / 1000
        ax.bar(
            x + offset, serving, BAR_WIDTH, label=system, color=color, edgecolor="black", linewidth=0.35
        )
        ax.bar(
            x + offset,
            warm,
            BAR_WIDTH,
            bottom=serving,
            hatch=WARM_HATCH,
            alpha=0.55,
            color=color,
            edgecolor="black",
            linewidth=0.35,
        )
    # Annotate DDiT bars with the StreamPilot cost saving relative to them.
    for bar_x, strategy in zip(x + offsets[-1], strategies, strict=True):
        stream_pilot = results[strategy]["StreamPilot"].avg_cost
        ddit = results[strategy]["DDiT"].avg_cost
        ax.text(
            bar_x,
            ddit / 1000 * 1.03,
            f"-{(1 - stream_pilot / ddit) * 100:.0f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    max_cost = max(result.avg_cost for by_system in results.values() for result in by_system.values())
    ax.set_ylim(0, max_cost / 1000 * 1.18)
    ax.set_xticks(x, [STRATEGY_LABELS.get(strategy, strategy) for strategy in strategies])
    ax.set_xlim(-0.5, len(strategies) - 0.5)
    ax.set_ylabel("Avg. cost (k$/hour)")
    ax.set_xlabel("Pool provisioning (iso-SLO, 99%)", labelpad=1)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="white", edgecolor="black", linewidth=0.35, hatch=WARM_HATCH))
    labels.append("Warm pool")
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=LEGEND_ANCHOR, fontsize=LEGEND_FONT_SIZE)


def plot_autoscaling(
    trace_path: Path,
    qpm_path: Path,
    stream_pilot_single_path: Path,
    ddit_single_path: Path,
    output_path: Path,
    config: AutoscalingConfig = AutoscalingConfig(),
) -> dict[str, dict[str, SimulationResult]]:
    arrivals = load_trace(trace_path, config.rate_scale)
    curve = load_stage_cost_curve(qpm_path)
    ddit_multiplier = get_ddit_cost_multiplier(qpm_path, stream_pilot_single_path, ddit_single_path)
    results = run_experiment(arrivals, curve, stream_pilot_and_ddit(ddit_multiplier), config)

    fig, (timeline_ax, bar_ax) = plt.subplots(2, 1, figsize=PAPER_FIG_SIZE, gridspec_kw={"hspace": 0.45})
    plot_timeline(timeline_ax, arrivals, results[TIMELINE_STRATEGY])
    plot_strategy_bars(bar_ax, results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(
        f"Trace: {len(arrivals)} min, mean {arrivals.mean():.1f} QPM, peak {arrivals.max():.1f} QPM; "
        f"DDiT cost multiplier {ddit_multiplier:.3f}; forecast MAPE "
        f"predictive {forecast_error(arrivals, config, Predictor.ORACLE) * 100:.1f}%, "
        f"reactive {forecast_error(arrivals, config, Predictor.PERSISTENCE) * 100:.1f}%"
    )
    print(f"{'Strategy':>22} {'System':>12} {'Cost $/h':>10} {'Serving':>9} {'Warm':>9} {'Warm%':>7} {'SLO':>7}")
    for strategy, by_system in results.items():
        for system, result in by_system.items():
            print(
                f"{strategy:>22} {system:>12} {result.avg_cost:>10.0f} {result.avg_serving_cost:>9.0f} "
                f"{result.avg_warm_cost:>9.0f} {result.avg_warm_cost / result.avg_cost * 100:>6.1f}% "
                f"{result.slo_attainment * 100:>6.2f}%"
            )
        saving = 1 - by_system["StreamPilot"].avg_cost / by_system["DDiT"].avg_cost
        print(f"{'':>22} StreamPilot saves {saving * 100:.1f}% vs DDiT")
    return results


def parse_args() -> ArgumentParser:
    paper_dir = Path(__file__).resolve().parent
    parser = ArgumentParser(description="Warm-pool autoscaling cost: StreamPilot vs DDiT on a production trace.")
    parser.add_argument(
        "--trace",
        type=Path,
        default=paper_dir / "data" / "azure_lmm_trace_2024_per_minute.csv",
        help="Per-minute arrivals (see prepare_azure_lmm_trace.py).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=paper_dir / "data" / "provisioning_qpm.csv",
        help="StreamPilot provisioning QPM CSV.",
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
    parser.add_argument(
        "--output",
        type=Path,
        default=paper_dir / "figures" / "eval_cost_autoscaling.pdf",
        help="Output PDF path.",
    )
    return parser


if __name__ == "__main__":
    args = parse_args().parse_args()
    plot_autoscaling(
        trace_path=args.trace,
        qpm_path=args.data,
        stream_pilot_single_path=args.stream_pilot_single_data,
        ddit_single_path=args.ddit_single_data,
        output_path=args.output,
    )
