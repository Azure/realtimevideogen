"""Warm-pool autoscaling simulation for multi-request traffic replayed from a production trace.

A discrete-time (1-minute) fluid simulation of a GPU pool serving StreamCast requests:

* Arrivals come from the Azure LMM Inference Trace 2025 (per-minute counts), rescaled to a target load.
* A request occupies one pipeline "slot" for ``residency_min`` minutes (a 10-minute video is streamed in
  real time). A pool of ``C`` slots costs as much as serving ``C / residency_min`` QPM at steady state,
  taken from the per-stage StreamPilot provisioning curves (``provisioning_qpm.csv``).
* Slots ordered by the autoscaler become usable after ``cold_start_min`` minutes (GPU node provisioning +
  model weight loading). Provisioned slots that are not serving form the warm pool, which absorbs bursts.
* The autoscaler forecasts the load ``cold_start_min`` ahead and sizes the pool with a safety margin.
  The predictive pool assumes a traffic predictor with ~10% MAPE (true load with AR(1) relative error);
  the reactive pool uses recent load (persistence); the static pool is a fixed size.
  The margin (or static size) is tuned per system so every system meets the same SLO attainment target.
* DDiT capacity costs a latency-matched multiple of StreamPilot's (same as the steady-state QPM figure).
* StreamPilot schedules with EDF and sizes the pool deadline-aware (batch work is deferred into troughs);
  DDiT schedules FCFS and must provision for all arrivals.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from deadline_scheduling import BATCH_FRACTION
from deadline_scheduling import REAL_TIME_FRACTION
from deadline_scheduling import RELAXED_FRACTION


MINUTES_PER_DAY = 24 * 60
EPSILON = 1e-9


class Scheduler(Enum):
    FCFS = "fcfs"
    EDF = "edf"


class Predictor(Enum):
    ORACLE = "oracle"  # assumed traffic predictor: true future load with correlated relative error
    PERSISTENCE = "persistence"  # reactive: recent load persists


@dataclass(frozen=True)
class AutoscalingConfig:
    # Scale the production trace (mean ~99 req/min) down to a mean of ~10 QPM of 10-minute videos.
    rate_scale: float = 0.1
    # Minutes a request occupies a pipeline slot (10-minute video streamed in real time).
    residency_min: int = 10
    # GPU node provisioning (~6 min) + loading the largest model weights (~2 min).
    cold_start_min: int = 8
    # SLO classes: tight (real-time), relaxed, batch.
    class_fractions: tuple[float, float, float] = (REAL_TIME_FRACTION, RELAXED_FRACTION, BATCH_FRACTION)
    # Maximum queueing delay (minutes) before a request misses its SLO, per class.
    class_slack_min: tuple[int, int, int] = (0, 5, 60)
    target_slo_attainment: float = 0.99
    # Window to estimate the average batch rate that deadline-aware sizing spreads over time.
    batch_window_min: int = 60
    # Assumed predictor: AR(1) relative error (std, minute-to-minute correlation), ~10% MAPE.
    predictor_error_std: float = 0.125
    predictor_error_correlation: float = 0.9
    predictor_seed: int = 0
    persistence_window_min: int = 5
    max_margin: float = 4.0
    search_iterations: int = 16


@dataclass(frozen=True)
class SystemSpec:
    name: str
    scheduler: Scheduler
    deadline_aware_sizing: bool
    cost_multiplier: float = 1.0


@dataclass
class SimulationResult:
    capacity: np.ndarray
    busy: np.ndarray
    slo_attainment: float
    margin: float = 0.0
    cost_rate: np.ndarray = field(default_factory=lambda: np.zeros(0))
    serving_cost_rate: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def avg_cost(self) -> float:
        return float(self.cost_rate.mean())

    @property
    def avg_serving_cost(self) -> float:
        return float(self.serving_cost_rate.mean())

    @property
    def avg_warm_cost(self) -> float:
        return self.avg_cost - self.avg_serving_cost


@dataclass(frozen=True)
class StageCostCurve:
    qpm: np.ndarray
    stage_costs: np.ndarray  # shape (num_points, num_stages), $/hour

    def cost_rate(self, qpm: np.ndarray) -> np.ndarray:
        """Total $/hour to sustain ``qpm``; floor at the single-request config, linear extrapolation."""
        qpm = np.asarray(qpm, dtype=float)
        total = np.zeros_like(qpm)
        for stage in range(self.stage_costs.shape[1]):
            costs = self.stage_costs[:, stage]
            slope = (costs[-1] - costs[-2]) / (self.qpm[-1] - self.qpm[-2])
            stage_cost = np.interp(qpm, self.qpm, costs)
            beyond = qpm > self.qpm[-1]
            stage_cost[beyond] = costs[-1] + slope * (qpm[beyond] - self.qpm[-1])
            total += stage_cost
        return total


def load_trace(path: Path, rate_scale: float) -> np.ndarray:
    """Per-minute arrivals (fluid requests) rescaled by ``rate_scale``."""
    df = pd.read_csv(path, comment="#")
    return df["requests"].to_numpy(dtype=float) * rate_scale


def load_stage_cost_curve(path: Path) -> StageCostCurve:
    df = pd.read_csv(path)
    stage_columns = [column for column in df.columns if column not in {"QPM", "Total"}]
    return StageCostCurve(
        qpm=df["QPM"].to_numpy(dtype=float),
        stage_costs=df[stage_columns].to_numpy(dtype=float),
    )


def _window_sums(values: np.ndarray, window: int) -> np.ndarray:
    """``out[t] = sum(values[t - window + 1 : t + 1])`` (truncated at the start)."""
    cumsum = np.concatenate(([0.0], np.cumsum(values)))
    idx = np.arange(len(values))
    return cumsum[idx + 1] - cumsum[np.maximum(idx + 1 - window, 0)]


def predictor_error(num_minutes: int, config: AutoscalingConfig) -> np.ndarray:
    """Temporally correlated (AR(1)) relative forecast error with the configured marginal std."""
    rng = np.random.default_rng(config.predictor_seed)
    phi = config.predictor_error_correlation
    innovations = rng.normal(0.0, config.predictor_error_std * np.sqrt(1 - phi ** 2), num_minutes)
    error = np.zeros(num_minutes)
    error[0] = rng.normal(0.0, config.predictor_error_std)
    for t in range(1, num_minutes):
        error[t] = phi * error[t - 1] + innovations[t]
    return error


def forecast_concurrency(arrivals: np.ndarray, config: AutoscalingConfig, predictor: Predictor) -> np.ndarray:
    """Forecast slots in use at ``t + cold_start_min`` (arrivals over the residency window ending then)."""
    lead = config.cold_start_min
    window = config.residency_min
    if lead >= window:
        raise ValueError("Cold start must be shorter than the residency window.")
    if predictor is Predictor.ORACLE:
        actual = actual_concurrency(arrivals, config)
        last_valid = len(arrivals) - lead - 1
        actual[last_valid + 1:] = actual[last_valid]
        return np.maximum(actual * (1 + predictor_error(len(arrivals), config)), 0.0)

    recent = _window_sums(arrivals, config.persistence_window_min) / config.persistence_window_min
    rate = np.concatenate(([arrivals[0]], recent[:-1]))  # rate observed before minute t
    future = rate * (lead + 1)  # minutes t .. t + lead
    past_window = window - lead - 1  # minutes t - past_window .. t - 1
    past = np.concatenate(([0.0], _window_sums(arrivals, past_window)[:-1]))
    return future + past


def actual_concurrency(arrivals: np.ndarray, config: AutoscalingConfig) -> np.ndarray:
    """Slots needed at ``t + cold_start_min`` if every request started on arrival."""
    window_sums = _window_sums(arrivals, config.residency_min)
    shifted = np.full_like(window_sums, np.nan)
    shifted[:len(arrivals) - config.cold_start_min] = window_sums[config.cold_start_min:]
    return shifted


def forecast_error(arrivals: np.ndarray, config: AutoscalingConfig, predictor: Predictor) -> float:
    """Mean absolute percentage error of the concurrency forecast."""
    predicted = forecast_concurrency(arrivals, config, predictor)
    actual = actual_concurrency(arrivals, config)
    valid = ~np.isnan(actual) & (actual > 0)
    return float(np.mean(np.abs(predicted[valid] - actual[valid]) / actual[valid]))


def simulate(
    arrivals: np.ndarray,
    predicted_concurrency: np.ndarray,
    config: AutoscalingConfig,
    system: SystemSpec,
    margin: float,
    static_capacity: float | None = None,
) -> SimulationResult:
    """Simulate the pool, autoscaler, and scheduler; return capacity/busy slots and SLO attainment."""
    num_minutes = len(arrivals)
    lead = config.cold_start_min
    window = config.residency_min
    fractions = np.asarray(config.class_fractions, dtype=float)
    slacks = config.class_slack_min
    batch_rate = _window_sums(arrivals, config.batch_window_min) / np.minimum(
        np.arange(1, num_minutes + 1), config.batch_window_min
    ) * fractions[2]
    batch_rate = np.concatenate(([arrivals[0] * fractions[2]], batch_rate[:-1]))

    # Queues: EDF keeps one FIFO per class plus a low-priority FIFO of requests that already missed their
    # deadline (served only with spare capacity, avoiding EDF's overload domino effect). FCFS keeps one FIFO
    # of all classes (1/3 each). Items are [deadline, arrival, amount, class].
    num_queues = 3 if system.scheduler is Scheduler.EDF else 1
    queues: list[deque[list[float]]] = [deque() for _ in range(num_queues)]
    late: deque[list[float]] = deque()
    queued = np.zeros(num_queues)

    starts = np.zeros(num_minutes)
    capacity = np.zeros(num_minutes)
    busy = np.zeros(num_minutes)
    orders = np.zeros(num_minutes + lead + 1)
    targets: deque[float] = deque(maxlen=lead + 1)
    violated = 0.0
    counted = 0.0

    def target(t: int) -> float:
        if system.deadline_aware_sizing:
            urgent = (fractions[0] + fractions[1]) * predicted_concurrency[t]
            urgent_queue = queued[0] + queued[1] if num_queues == 3 else 0.0
            batch_queue = queued[2] if num_queues == 3 else queued[0]
            return (
                (1 + margin) * urgent
                + window * batch_rate[t]
                + urgent_queue
                + batch_queue * window / max(slacks[2], 1)
            )
        return float((1 + margin) * predicted_concurrency[t] + queued.sum())

    current = static_capacity if static_capacity is not None else target(0)
    for t in range(num_minutes):
        in_flight = starts[max(0, t - window + 1):t].sum()
        if static_capacity is None:
            current += orders[t]
            desired = target(t)
            targets.append(desired)
            pending = orders[t + 1:t + lead + 1].sum()
            if desired > current + pending:
                orders[t + lead] += desired - current - pending
            # Release idle slots not needed by any plan covering [t, t + lead].
            current = min(current, max(max(targets), in_flight))

        # Enqueue this minute's arrivals.
        if num_queues == 3:
            for cls in range(3):
                amount = arrivals[t] * fractions[cls]
                if amount > EPSILON:
                    queues[cls].append([t + slacks[cls], t, amount, cls])
                    queued[cls] += amount
            for queue in queues:
                while queue and queue[0][0] < t:
                    late.append(queue.popleft())
        elif arrivals[t] > EPSILON:
            queues[0].append([t, t, arrivals[t], 0])
            queued[0] += arrivals[t]

        free = max(current - in_flight, 0.0)
        while free > EPSILON:
            heads = [(queue[0][0], index) for index, queue in enumerate(queues) if queue]
            if heads:
                source = queues[min(heads)[1]]
            elif late:
                source = late
            else:
                break
            item = source[0]
            cls = int(item[3])
            served = min(free, item[2])
            delay = t - item[1]
            if num_queues == 3:
                violated += served * (delay > slacks[cls])
            else:
                violated += served * float(np.dot(fractions, [delay > slack for slack in slacks]))
            counted += served
            starts[t] += served
            queued[cls] -= served
            free -= served
            item[2] -= served
            if item[2] <= EPSILON:
                source.popleft()

        capacity[t] = current
        busy[t] = in_flight + starts[t]

    # Requests still queued at the end: count those already past their deadline.
    for queue in [*queues, late]:
        for _, arrival, amount, item_class in queue:
            cls = int(item_class)
            age = num_minutes - arrival
            if num_queues == 3:
                share = float(age > slacks[cls])
            else:
                share = float(np.dot(fractions, [age > slack for slack in slacks]))
            violated += amount * share
            counted += amount * share

    attainment = 1.0 - violated / counted if counted > 0 else 1.0
    return SimulationResult(capacity=capacity, busy=busy, slo_attainment=attainment, margin=margin)


def _bisect(
    attainment_of: Callable[[float], float],
    low: float,
    high: float,
    config: AutoscalingConfig,
) -> float:
    """Smallest value in [low, high] whose SLO attainment meets the target (attainment is monotone)."""
    if attainment_of(high) < config.target_slo_attainment:
        return high
    for _ in range(config.search_iterations):
        middle = (low + high) / 2
        if attainment_of(middle) >= config.target_slo_attainment:
            high = middle
        else:
            low = middle
    return high


def with_costs(result: SimulationResult, curve: StageCostCurve, config: AutoscalingConfig,
               cost_multiplier: float) -> SimulationResult:
    result.cost_rate = curve.cost_rate(result.capacity / config.residency_min) * cost_multiplier
    result.serving_cost_rate = np.minimum(
        curve.cost_rate(result.busy / config.residency_min) * cost_multiplier, result.cost_rate
    )
    return result


def provision_iso_slo(
    arrivals: np.ndarray,
    curve: StageCostCurve,
    config: AutoscalingConfig,
    system: SystemSpec,
    predictor: Predictor | None,
) -> SimulationResult:
    """Cheapest pool meeting the SLO target: static capacity if ``predictor`` is None, else autoscaled."""
    predicted = (
        np.zeros_like(arrivals) if predictor is None
        else forecast_concurrency(arrivals, config, predictor)
    )

    def run(value: float) -> SimulationResult:
        if predictor is None:
            return simulate(arrivals, predicted, config, system, 0.0, static_capacity=value)
        return simulate(arrivals, predicted, config, system, margin=value)

    if predictor is None:
        peak = float(_window_sums(arrivals, config.residency_min).max())
        value = _bisect(lambda capacity: run(capacity).slo_attainment, 0.0, peak, config)
    else:
        value = _bisect(lambda margin: run(margin).slo_attainment, -0.5, config.max_margin, config)
    return with_costs(run(value), curve, config, system.cost_multiplier)


STRATEGIES: dict[str, Predictor | None] = {
    "Static peak": None,
    "Reactive": Predictor.PERSISTENCE,
    "Predictive warm pool": Predictor.ORACLE,
}


def stream_pilot_and_ddit(ddit_cost_multiplier: float) -> tuple[SystemSpec, SystemSpec]:
    return (
        SystemSpec("StreamPilot", Scheduler.EDF, deadline_aware_sizing=True),
        SystemSpec("DDiT", Scheduler.FCFS, deadline_aware_sizing=False, cost_multiplier=ddit_cost_multiplier),
    )


def run_experiment(
    arrivals: np.ndarray,
    curve: StageCostCurve,
    systems: tuple[SystemSpec, ...],
    config: AutoscalingConfig = AutoscalingConfig(),
    strategies: dict[str, Predictor | None] = STRATEGIES,
) -> dict[str, dict[str, SimulationResult]]:
    """Results indexed by strategy name, then system name."""
    return {
        strategy: {
            system.name: provision_iso_slo(arrivals, curve, config, system, predictor)
            for system in systems
        }
        for strategy, predictor in strategies.items()
    }
