"""
Utilities for plotting.
"""
from __future__ import annotations

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.markers import MarkerStyle

from typing import Optional
from typing import cast

from utils import get_pareto_frontier

from sim_types import ProvisioningResult
from sim_types import GPUType
from sim_types import Model
from sim_types import QualityLevel


FIG_SIZE = (7, 5)


def get_color_map() -> list[tuple[float, float, float]]:
    return cast(ListedColormap, plt.get_cmap('tab10')).colors


def _get_time_ticklabels(
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    exclude_x: list[int] = [],
) -> tuple[list[int], list[str]]:
    ticks_seconds = [1, 2, 5, 10, 15, 30]

    # Define desired minute values (in seconds) to match the log scale
    minute_ticks_minutes = [1, 2, 5, 10, 20, 40]  # in minutes
    minute_ticks_seconds = [mins * 60 for mins in minute_ticks_minutes]  # convert to seconds

    hour_ticks_hours = [1, 3, 5, 8, 12]  # in hours
    hour_ticks_seconds = [hours * 60 * 60 for hours in hour_ticks_hours]  # convert to seconds

    day_ticks_days: list[int] = [1]  # in days
    day_ticks_seconds = [days * 24 * 60 * 60 for days in day_ticks_days]  # convert to seconds
    tick_labels = \
        [f"{s:g}s" for s in ticks_seconds] + \
        [f"{m:g}m" for m in minute_ticks_minutes] + \
        [f"{h:g}h" for h in hour_ticks_hours] + \
        [f"{d:g}d" for d in day_ticks_days]

    ticks: list[int] = ticks_seconds + minute_ticks_seconds + hour_ticks_seconds + day_ticks_seconds

    if xmin is not None or xmax is not None or exclude_x:
        filtered_ticks: list[int] = []
        filtered_labels: list[str] = []
        for tick, label in zip(ticks, tick_labels):
            if tick in exclude_x:
                continue
            if xmin is not None and tick < xmin:
                continue
            if xmax is not None and tick > xmax:
                continue
            filtered_ticks.append(tick)
            filtered_labels.append(label)
        ticks = filtered_ticks
        tick_labels = filtered_labels

    return ticks, tick_labels


def plot_x_vs_y(
    x_data: list[float],
    y_data: list[float],
    provisions: list[dict[GPUType, int]],
    verbose: bool = False,
    # Plot
    figsize: tuple[int, int] = FIG_SIZE,
    # X
    xlabel: str = "Latency (sec)",
    xmin: Optional[float] = 20,
    xmax: Optional[float] = None,
    # Y
    ylabel: str = "Cost ($)",
    ymin: Optional[float] = 0,
    ymax: Optional[float] = None,
    # Marker
    marker_size: int = 80,
) -> None:
    # Create the figure
    plt.figure(figsize=figsize)

    # Change the font size to 12
    plt.rcParams.update({'font.size': 12})

    # Define color map
    tab10 = cast(ListedColormap, plt.get_cmap("tab10")).colors
    colors = tab10
    markers = ["o", "^", "s", "D", "P", "X"]  # circle, triangle_up, square, diamond, plus, x

    # scatter plot for mixed provisioning
    idx_list_mixed: list[int] = []

    # Data for each GPU type
    idx_lists_single: dict[GPUType, list[int]] = {}
    for gpu_idx, gpu_type in enumerate(GPUType):
        idx_lists_single[gpu_type] = []
        for idx, provision in enumerate(provisions):
            if provision.get(gpu_type, 0) > 0 and len(provision) == 1:
                idx_lists_single[gpu_type].append(idx)
            else:
                idx_list_mixed.append(idx)

    # Scatter plot for mixed provisioning
    idx_list_mixed = list(set(idx_list_mixed))  # Remove duplicates
    idx_list_single = [
        idx
        for gpu_type in GPUType
        for idx in idx_lists_single[gpu_type]
    ]
    idx_list_mixed = [
        idx
        for idx in idx_list_mixed
        if idx not in idx_list_single
    ]
    x_data_mixed = [x_data[i] for i in idx_list_mixed]
    y_data_mixed = [y_data[i] for i in idx_list_mixed]
    if x_data_mixed and y_data_mixed:
        plt.scatter(
            x_data_mixed,
            y_data_mixed,
            s=marker_size,
            color=tab10[0],
            label="Mixed",
            marker=MarkerStyle('x'),
            alpha=0.5,
        )

    # Scatter plot for single GPU type provisioning
    for gpu_idx, gpu_type in enumerate(GPUType):
        x_data_gpu_type = [x_data[i] for i in idx_lists_single[gpu_type]]
        y_data_gpu_type = [y_data[i] for i in idx_lists_single[gpu_type]]
        if x_data_gpu_type and y_data_gpu_type:
            plt.scatter(
                x_data_gpu_type,
                y_data_gpu_type,
                s=marker_size,
                color=colors[gpu_idx + 1],
                marker=MarkerStyle(markers[gpu_idx]),
                label=gpu_type.name,
                alpha=0.7,
            )

    # Pareto frontier for all points
    pareto_front = get_pareto_frontier(x_data, y_data)

    # Find the provisioning options that correspond to the Pareto front points
    pareto_provision = []
    for point in pareto_front.tolist():
        x_val = point[0]
        y_val = point[1]
        try:
            idx = np.where((np.array(x_data) == x_val) & (np.array(y_data) == y_val))[0][0]
            pareto_provision.append(provisions[idx])
        except IndexError:
            pass  # Ignore artificial points

    if verbose:
        print("Pareto Front Provisioning Options:")
        for point in pareto_front.tolist():
            x_val = point[0]
            y_val = point[1]
            try:
                idx = np.where((np.array(x_data) == x_val) & (np.array(y_data) == y_val))[0][0]
                num_gpus = provisions[idx]
                print(
                    f"{num_gpus} -> "
                    f"TTFF: {point[0]:.2f} seconds, Cost: ${point[1]:.2f}")
            except IndexError:
                pass  # Ignore artificial points

    # plot the parento curve
    plt.plot(
        pareto_front[:, 0],
        pareto_front[:, 1],
        color=tab10[0],
        linewidth=3,
        # linestyle='--',
        label="Frontier",
        zorder=0
    )

    # add arrow point to the left bottom corner with 'Better' inside the arrow
    # plt.text(30, 20, "Better", ha="center", va="center",
    # rotation=45, size=12,
    # bbox=dict(boxstyle="larrow,pad=0.2", fc="white", ec="black", lw=1))

    if xmin is None or xmin > 0:
        plt.xscale("log")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    # add a vertical line at x=600 seconds
    # plt.axvline(x=600, color='red', linestyle='--', label='Realtime')

    ticks, tick_labels = _get_time_ticklabels(
        xmin=xmin,
        xmax=xmax,
    )
    plt.xticks(
        ticks,
        tick_labels
    )

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.7)

    plt.legend()

    # Improve formatting
    plt.tight_layout(pad=0)

    plt.show()
    # plt.savefig('ttff_vs_cost.pdf', dpi=300, bbox_inches='tight')


def plot_ttff_vs_cost(
    ttffs: list[float],
    costs: list[float],
    provisions: list[dict[GPUType, int]],
    verbose: bool = False,
    # Plot
    figsize: tuple[int, int] = FIG_SIZE,
    # X
    xlabel: str = "Time to First Frame (TTFF)",
    xmin: Optional[float] = 20,
    xmax: Optional[float] = None,
    # Y
    ylabel: str = "Cost ($)",
    ymin: Optional[float] = 0,
    ymax: Optional[float] = None,
) -> None:
    """
    Plots Time to First Frame (TTFF) against Cost for different provisioning options.
    Args:
        ttff_list (list): List of TTFF values corresponding to each provisioning option.
        costs (list): List of cost values corresponding to each provisioning option.
        provision (list): List of tuples representing the provisioning options (num_a100s, num_h100s, num_h200s).
    """
    plot_x_vs_y(
        x_data=ttffs,
        y_data=costs,
        provisions=provisions,
        xlabel=xlabel,
        ylabel=ylabel,
        verbose=verbose,
        figsize=figsize,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
    )


def plot_ttff_vs_energy(
    ttff_list: list[float],
    energy_list: list[float],
    actual_provision: list[dict[GPUType, int]],
    verbose: bool = False,
    # Plot
    figsize: tuple[int, int] = FIG_SIZE,
    # X
    xlabel: str = "Time to First Frame (TTFF)",
    xmin: Optional[float] = 20,
    xmax: Optional[float] = None,
    # Y
    ylabel: str = "Energy (kWh)",
    ymin: Optional[float] = 0,
    ymax: Optional[float] = None,
) -> None:
    # convert energy from Ws to kWh
    energy_list_copy = [
        energy / (60 * 60 * 1000)  # convert from Ws to kWh
        for energy in energy_list
    ]
    plot_x_vs_y(
        x_data=ttff_list,
        y_data=energy_list_copy,
        provisions=actual_provision,
        xlabel=xlabel,
        ylabel=ylabel,
        verbose=verbose,
        figsize=figsize,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
    )


def plot_adaptive_quality(
    provisioning_result_adaptive: ProvisioningResult,
    provisioning_qualities: dict[QualityLevel, ProvisioningResult],
    slide_seconds: float = 2.2,  # Value in seconds
    # Plot
    figsize: tuple[int, int] = FIG_SIZE,
    # X: 1 second to 10 minutes
    xmin: Optional[float] = 1,
    xmax: Optional[float] = 10 * 60,
    # Y: Cost ($)
    ymin: Optional[float] = 0,
    ymax: Optional[float] = 85,
) -> None:
    # Plot quality pareto frontiers
    pareto_front_adaptive = get_pareto_frontier(
        provisioning_result_adaptive.ttffs,
        provisioning_result_adaptive.costs,
        max_x=ymax,
        max_y=xmax,
    )

    pareto_fronts: dict[QualityLevel, np.ndarray] = {}
    for quality in [QualityLevel.HIGH, QualityLevel.MEDIUM, QualityLevel.LOW]:
        pareto_fronts[quality] = get_pareto_frontier(
            provisioning_qualities[quality].ttffs,
            provisioning_qualities[quality].costs,
            max_x=ymax,
            max_y=xmax,
        )

    _, ax = plt.subplots(figsize=figsize)
    for quality in [QualityLevel.HIGH, QualityLevel.MEDIUM, QualityLevel.LOW]:
        ax.plot(
            pareto_fronts[quality][:, 0],
            pareto_fronts[quality][:, 1],
            linewidth=3,
            label=quality.name.lower().capitalize(),
        )

    # Adaptive quality
    ax.plot(
        pareto_front_adaptive[:, 0],
        pareto_front_adaptive[:, 1],
        linewidth=3,
        linestyle='--',
        label="Adaptive",
    )

    # Adaptive + Slide
    plt.plot(
        pareto_front_adaptive[:, 0] - slide_seconds,
        pareto_front_adaptive[:, 1],
        linewidth=3,
        linestyle='-.',
        label="+Slide")

    plt.legend(loc="upper right")
    plt.grid()

    ax.grid(True, which='major', linestyle='--', alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', alpha=0.4)

    ax.set_xlabel("TTFF (s)", labelpad=-8)
    ax.set_ylabel("Cost ($)")

    if xmin is None or xmin > 0:
        plt.xscale("log")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    ticks, tick_labels = _get_time_ticklabels(
        xmin=xmin,
        xmax=xmax,
    )
    plt.xticks(
        ticks,
        tick_labels
    )

    ax2 = ax.twinx()
    if ymax is not None:
        ax2.set_ylim(0, ymax / 10)  # Scale by 1/10 cost/10 minutes
        ax2.set_ylabel("Cost ($/minute)", rotation=-90, labelpad=15)

    plt.tight_layout(pad=0)
    plt.show()


def plot_cost_vs_qpm(
    costs: dict[GPUType, dict[Model, list[float]]],
    qpms: list[float] = [1, 2],
    # Plot
    figsize: tuple[int, int] = FIG_SIZE,
    xlabel: str = "Requests per Minute (QPM)",
    ylabel: str = "Cost ($)",
) -> None:
    """Plot cost vs QPM for each component."""
    plt.figure(figsize=figsize)

    for gpu_type in costs.keys():
        for model in costs[gpu_type].keys():
            plt.plot(
                qpms,
                costs[gpu_type][model],
                marker='o',
                label=f"{model.name} ({gpu_type.name})")

    # plt.plot(QPM_LIST, total_costs, marker='o', label='Total Cost', color='black', linewidth=2)

    plt.xscale('log')

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title('Cost vs. Requests per Minute (QPM) for Each Component', fontsize=14)

    plt.xticks(qpms, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    if len(costs) > 0:
        plt.legend()
    plt.tight_layout(pad=0)
    plt.show()


def plot_policies_ttff_vs_cost(
    provision_results: dict[str, ProvisioningResult],
    points: bool = True,
    front: bool = True,
    # Plot
    figsize: tuple[int, int] = FIG_SIZE,
    # X
    xlabel: str = "Time to first frame (TTFF)",
    xmin: Optional[int] = 20,
    xmax: Optional[int] = 3 * 60 * 60,  # 3 hours
    # Y
    ylabel: str = "Cost ($)",
    ymin: Optional[int] = 0,
    ymax: Optional[int] = 1500,
) -> None:
    assert front or points, "At least one of 'front' or 'points' must be True"

    _, ax = plt.subplots(figsize=figsize)

    # Plot results
    for policy_name, provision_result in provision_results.items():
        # Plot points
        if points:
            ax.scatter(
                provision_result.ttffs,
                provision_result.costs,
                marker=MarkerStyle("o"),
                alpha=0.5,
                label=policy_name
            )
        # Plot Pareto frontier
        if front:
            pareto_frontier = get_pareto_frontier(
                provision_result.ttffs,
                provision_result.costs,
                max_x=ymax,
                max_y=xmax,
            )
            ax.plot(
                pareto_frontier[:, 0],
                pareto_frontier[:, 1],
                linewidth=3,
                label=f"{policy_name}" if not points else None
            )

    if xmin is None or xmin > 0:
        plt.xscale("log")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    ticks, tick_labels = _get_time_ticklabels(
        xmin=xmin,
        xmax=xmax,
    )
    plt.xticks(
        ticks,
        tick_labels
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.grid()
    if len(provision_results) > 0:
        plt.legend()
    plt.tight_layout(pad=0)
    plt.show()
