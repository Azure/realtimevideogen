from __future__ import annotations

import numpy as np


REAL_TIME_FRACTION = 1 / 3
RELAXED_FRACTION = 1 / 3
BATCH_FRACTION = 1 / 3
RELAXED_SLO_MULTIPLIER = 1.5
BATCH_CAPACITY_MULTIPLIER = 0.3
FCFS_TARGET_UTILIZATION = 0.9
DEADLINE_AWARE_TARGET_UTILIZATION = 0.95


def get_fcfs_incremental_capacity_factor() -> float:
    deadline_aware_load_fraction = (
        REAL_TIME_FRACTION
        + RELAXED_FRACTION / RELAXED_SLO_MULTIPLIER
        + BATCH_FRACTION * BATCH_CAPACITY_MULTIPLIER
    )
    fcfs_capacity_per_request = 1 / FCFS_TARGET_UTILIZATION
    deadline_aware_capacity_per_request = (
        deadline_aware_load_fraction
        / DEADLINE_AWARE_TARGET_UTILIZATION
    )
    return (
        fcfs_capacity_per_request
        / deadline_aware_capacity_per_request
    )


def get_load_aware_fcfs_costs(
    stream_pilot_costs: np.ndarray,
) -> np.ndarray:
    costs = np.asarray(stream_pilot_costs, dtype=float)
    if costs.ndim not in {1, 2}:
        raise ValueError("Costs must be a one- or two-dimensional array.")
    if costs.shape[0] == 0:
        raise ValueError("Costs must contain at least one load point.")
    if np.any(costs < 0):
        raise ValueError("Costs must be non-negative.")

    base_capacity = costs[0]
    incremental_capacity = np.maximum(costs - base_capacity, 0)
    return (
        base_capacity
        + incremental_capacity * get_fcfs_incremental_capacity_factor()
    )


def get_cost_savings(
    fcfs_costs: np.ndarray,
    stream_pilot_costs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fcfs = np.asarray(fcfs_costs, dtype=float)
    stream_pilot = np.asarray(stream_pilot_costs, dtype=float)
    if fcfs.shape != stream_pilot.shape:
        raise ValueError("FCFS and StreamPilot costs must have equal shape.")

    savings = fcfs - stream_pilot
    savings_percent = np.divide(
        savings,
        fcfs,
        out=np.zeros_like(savings),
        where=fcfs > 0,
    ) * 100
    return savings, savings_percent


def get_latency_matched_reference(
    target_cost: float,
    reference_ttff: np.ndarray,
    reference_costs: np.ndarray,
    candidate_ttff: np.ndarray,
    candidate_costs: np.ndarray,
) -> tuple[float, float, float, float, bool]:
    reference_ttff = np.asarray(reference_ttff, dtype=float)
    reference_costs = np.asarray(reference_costs, dtype=float)
    candidate_ttff = np.asarray(candidate_ttff, dtype=float)
    candidate_costs = np.asarray(candidate_costs, dtype=float)

    if reference_ttff.shape != reference_costs.shape:
        raise ValueError("Reference TTFF and costs must have equal shape.")
    if candidate_ttff.shape != candidate_costs.shape:
        raise ValueError("Candidate TTFF and costs must have equal shape.")
    if reference_ttff.size == 0 or candidate_ttff.size == 0:
        raise ValueError("Reference and candidate data must not be empty.")

    reference_index = int(
        np.argmin(np.abs(reference_costs - target_cost))
    )
    target_ttff = reference_ttff[reference_index]
    eligible = candidate_ttff <= target_ttff
    met_target = bool(np.any(eligible))

    if met_target:
        eligible_indices = np.flatnonzero(eligible)
        candidate_index = int(
            eligible_indices[
                np.argmin(candidate_costs[eligible_indices])
            ]
        )
    else:
        fastest_ttff = candidate_ttff.min()
        fastest_indices = np.flatnonzero(
            np.isclose(candidate_ttff, fastest_ttff)
        )
        candidate_index = int(
            fastest_indices[
                np.argmin(candidate_costs[fastest_indices])
            ]
        )

    return (
        target_ttff,
        reference_costs[reference_index],
        candidate_ttff[candidate_index],
        candidate_costs[candidate_index],
        met_target,
    )
