"""
Utilities for the simulator.
"""

from __future__ import annotations

from copy import deepcopy

import pandas as pd
import numpy as np

from scipy.interpolate import interp1d

from sim_types import ProvisioningResult
from sim_types import GPUType
from sim_types import Model
from sim_types import ModelAllocation

from typing import Optional


def to_models_df(
    models: dict[GPUType, dict[Model, list[ModelAllocation]]]
) -> pd.DataFrame:
    """
    Convert the models dictionary to a pandas DataFrame for easier analysis and visualization.
    """
    records = []
    for gpu_type, model_allocations in models.items():
        for model, allocations in model_allocations.items():
            for allocation in allocations:
                if allocation is None or allocation.get_num_gpus() == 0:
                    continue  # Ignoring empty allocations
                record = {
                    "GPU": gpu_type.value,
                    "Model": model.value,
                    "Devices": allocation.devices,
                    "Replicas": allocation.replicas,
                    "Work": allocation.work,
                    "#GPUs": allocation.get_num_gpus(),
                    "Time (s)": allocation.time,
                    "TTFF (s)": allocation.time_first,
                    "Energy (kWh)": allocation.energy / (60 * 60) / 1000.0,  # Convert to kWh
                    "Cost ($)": allocation.cost,
                }
                records.append(record)
    df = pd.DataFrame(records)
    df = df.set_index(["GPU", "Model"])
    df = df.round(2)

    total = df.sum(numeric_only=True)
    total["Time (s)"] = df["Time (s)"].groupby(level="Model").max().sum()
    total["TTFF (s)"] = df["TTFF (s)"].groupby(level="Model").min().sum()
    total.name = ("TOTAL", "")
    df = pd.concat([df, total.to_frame().T])

    df[["Devices", "Replicas", "#GPUs", "Work"]] = df[["Devices", "Replicas", "#GPUs", "Work"]].astype(int)

    return df


def coalesce_models(
    models: dict[GPUType, dict[Model, list[ModelAllocation]]]
) -> dict[GPUType, dict[Model, list[ModelAllocation]]]:
    """The models with the same parallelism and same work, should be accounted as replicas."""
    merged: dict[GPUType, dict[Model, list[ModelAllocation]]] = {}
    for gpu_type, model_dict in models.items():
        merged[gpu_type] = {}
        for model_name, allocations in model_dict.items():
            merged_allocations: list[ModelAllocation] = []
            for alloc in allocations:
                # Check if there's an existing allocation with the same devices and work
                match = next((
                    model_alloc
                    for model_alloc in merged_allocations
                    if model_alloc.devices == alloc.devices and model_alloc.work == alloc.work
                ), None)
                if match:
                    # If found, increment replicas and aggregate energy/cost
                    match.replicas += 1
                    match.energy += alloc.energy
                    match.cost += alloc.cost
                else:
                    # Otherwise, add as new allocation
                    merged_allocations.append(deepcopy(alloc))
            merged[gpu_type][model_name] = merged_allocations
    return merged


def simplify_model_allocations(
    models: dict[GPUType, dict[Model, list[ModelAllocation]]],
) -> dict[GPUType, dict[Model, list[ModelAllocation]]]:
    """
    Simplify model allocations by merging replicas with the same number of devices.
    This is to reduce the search space for the optimization loop.
    """
    new_models = deepcopy(models)
    for gpu_type in new_models.keys():
        for model in new_models[gpu_type].keys():
            model_instances = new_models[gpu_type][model]
            alloc_map: dict[int, ModelAllocation] = {}
            for model_instance in model_instances:
                if model_instance.get_num_gpus() == 0:
                    continue
                if model_instance.devices not in alloc_map:
                    alloc_map[model_instance.devices] = deepcopy(model_instance)
                else:
                    alloc_map[model_instance.devices].replicas += model_instance.replicas
            new_models[gpu_type][model] = list(alloc_map.values())
    return new_models


def find_fastest_provisioning(
    provisioning: ProvisioningResult,
) -> int:
    """Find the fastest provisioning option."""
    min_latency = min(provisioning.latencies)
    min_latency_index = provisioning.latencies.index(min_latency)
    return min_latency_index


def find_fastest_ttff_provisioning(
    provisioning: ProvisioningResult,
) -> int:
    """Find the fastest provisioning option."""
    min_ttff = min(provisioning.ttffs)
    min_ttff_index = provisioning.ttffs.index(min_ttff)
    return min_ttff_index


def find_cheapest_provisioning(
    provisioning: ProvisioningResult,
) -> int:
    """Find the cheapest provisioning option."""
    min_cost = min(provisioning.costs)
    min_cost_index = provisioning.costs.index(min_cost)
    return min_cost_index


def find_most_cost_effective_provisioning(
    provisioning: ProvisioningResult,
) -> int:
    """Find the most cost-effective provisioning option."""
    min_cost = min(provisioning.costs)
    min_latency = min(provisioning.latencies)
    min_cost_index = provisioning.costs.index(min_cost)
    min_latency_index = provisioning.latencies.index(min_latency)
    if min_cost_index == min_latency_index:
        return min_cost_index

    # if the indices are different, return the provisioning option with the minimum cost*latency
    cost_latency_list = [
        cost * latency
        for cost, latency in zip(provisioning.costs, provisioning.latencies)
    ]
    min_cost_latency = min(cost_latency_list)
    min_cost_latency_index = cost_latency_list.index(min_cost_latency)
    return min_cost_latency_index


def find_most_energy_efficient_provisioning(
    provisioning: ProvisioningResult,
) -> int:
    """Find the most energy-efficient provisioning option."""
    min_energy = min(provisioning.energies)
    min_latency = min(provisioning.latencies)
    min_energy_index = provisioning.energies.index(min_energy)
    min_latency_index = provisioning.latencies.index(min_latency)
    if min_energy_index == min_latency_index:
        return min_energy_index

    # if the indices are different, return the provisioning option with the minimum energy*latency
    energy_latency_list = [
        energy * latency
        for energy, latency in zip(provisioning.energies, provisioning.latencies)
    ]
    min_energy_latency = min(energy_latency_list)
    min_energy_latency_index = energy_latency_list.index(min_energy_latency)
    return min_energy_latency_index


def find_pareto_frontier(
    latency_list: list[float],
    energy_list: list[float],
    provision: list[float]
) -> tuple[list[float], list[float], list[float]]:
    pareto_provision = []
    pareto_latency = []
    pareto_energy = []
    for i in range(len(latency_list)):
        dominated = False
        for j in range(len(latency_list)):
            if i != j:
                if latency_list[j] <= latency_list[i] and energy_list[j] <= energy_list[i]:
                    if latency_list[j] < latency_list[i] or energy_list[j] < energy_list[i]:
                        dominated = True
                        break
        if not dominated:
            pareto_provision.append(provision[i])
            pareto_latency.append(latency_list[i])
            pareto_energy.append(energy_list[i])
    return pareto_provision, pareto_latency, pareto_energy


def get_pareto_frontier_paper(
    points: np.ndarray,
    max_y: Optional[float] = None,
    max_x: Optional[float] = None,
) -> np.ndarray:
    """
    Calculate the Pareto frontier from a set of data points
    """
    if points.size == 0:
        return points.copy()

    # points = points[np.argsort(points[:, 0])]
    points = points[np.lexsort((points[:, 1], points[:, 0]))]

    pareto_front = [points[0]]
    for point in points[1:]:
        if point[1] < pareto_front[-1][1]:
            pareto_front.append(point)

    # Add extreme points to the Pareto frontier
    extreme_point_0 = [pareto_front[0][0], max(points[:, 1])]
    extreme_point_1 = [max(points[:, 0]), pareto_front[-1][1]]
    pareto_front.append(extreme_point_0)
    pareto_front.append(extreme_point_1)

    if max_x is not None:
        candidate = np.array([max_x, min(points[:, 1])])
        if candidate[0] > pareto_front[-1][0] and candidate[1] <= pareto_front[-1][1]:
            pareto_front.append(candidate)
    if max_y is not None:
        candidate = np.array([min(points[:, 0]), max_y])
        if candidate[1] > pareto_front[0][1] and candidate[0] <= pareto_front[0][0]:
            pareto_front.append(candidate)

    pareto_front_np = np.array(pareto_front)
    pareto_front_np = pareto_front_np[np.lexsort((
        -pareto_front_np[:, 1],
        pareto_front_np[:, 0]))]

    # Avoid repeated points
    _, idx = np.unique(pareto_front_np, axis=0, return_index=True)
    pareto_front_np = pareto_front_np[np.sort(idx)]

    return pareto_front_np


def get_pareto_frontier(
    ttff_list: list[float],
    costs: list[float],
    max_y: Optional[float] = None,
    max_x: Optional[float] = None,
) -> np.ndarray:
    points = np.array(list(zip(ttff_list, costs)))
    return get_pareto_frontier_paper(
        points,
        max_x,
        max_y,
    )


def clean_frontier(
    frontier: np.ndarray
) -> np.ndarray:
    F = frontier[np.argsort(frontier[:, 0])]
    xs = []
    ys = []
    i = 0
    while i < len(F):
        x = F[i, 0]
        same_x = F[F[:, 0] == x]
        xs.append(x)
        ys.append(same_x[:, 1].min())
        i += len(same_x)
    return np.column_stack([xs, ys])


def area_between_frontiers(
    A: np.ndarray,
    B: np.ndarray,
    n: int = 5000
) -> np.ndarray:
    A = clean_frontier(A)
    B = clean_frontier(B)
    xmin = max(A[:, 0].min(), B[:, 0].min())
    xmax = min(A[:, 0].max(), B[:, 0].max())
    xs = np.linspace(xmin, xmax, n)
    fA = interp1d(A[:, 0], A[:, 1], kind="linear")
    fB = interp1d(B[:, 0], B[:, 1], kind="linear")
    yA = fA(xs)
    yB = fB(xs)
    # return np.trapezoid(yB - yA, xs)
    delta = yB - yA
    return 100.0 * delta / yB
