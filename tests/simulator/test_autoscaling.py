import sys
import os

import numpy as np
import pytest

sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path

with temp_sys_path("simulator"):
    from autoscaling import AutoscalingConfig
    from autoscaling import Predictor
    from autoscaling import Scheduler
    from autoscaling import StageCostCurve
    from autoscaling import SystemSpec
    from autoscaling import _window_sums
    from autoscaling import actual_concurrency
    from autoscaling import forecast_concurrency
    from autoscaling import forecast_error
    from autoscaling import predictor_error
    from autoscaling import provision_iso_slo
    from autoscaling import run_experiment
    from autoscaling import simulate
    from autoscaling import stream_pilot_and_baselines
    from autoscaling import stream_pilot_and_ddit


CURVE = StageCostCurve(
    qpm=np.array([0.5, 1.0, 10.0]),
    stage_costs=np.array([[1.0, 2.0], [1.0, 4.0], [10.0, 40.0]]),
)
CONFIG = AutoscalingConfig(search_iterations=12)
FCFS = SystemSpec("FCFS", Scheduler.FCFS, deadline_aware_sizing=False)
EDF = SystemSpec("EDF", Scheduler.EDF, deadline_aware_sizing=True)


def bursty_arrivals(num_minutes: int = 600) -> np.ndarray:
    minutes = np.arange(num_minutes)
    arrivals = 2.0 + np.sin(minutes / 60) + 0.0
    arrivals[(minutes % 120) < 10] *= 3
    return arrivals


def test_cost_curve_floor_interpolation_and_extrapolation() -> None:
    costs = CURVE.cost_rate(np.array([0.0, 0.5, 5.5, 10.0, 20.0]))
    assert costs[0] == pytest.approx(3.0)
    assert costs[1] == pytest.approx(3.0)
    assert costs[2] == pytest.approx(5.5 + 22.0)
    assert costs[3] == pytest.approx(50.0)
    assert costs[4] == pytest.approx(100.0)


def test_window_sums() -> None:
    assert np.allclose(_window_sums(np.array([1.0, 2.0, 3.0, 4.0]), 2), [1.0, 3.0, 5.0, 7.0])


def test_forecasts_on_constant_load() -> None:
    arrivals = np.full(100, 3.0)
    persistence = forecast_concurrency(arrivals, CONFIG, Predictor.PERSISTENCE)
    assert np.allclose(persistence[20:], 3.0 * CONFIG.residency_min)
    assert np.allclose(actual_concurrency(arrivals, CONFIG)[20:80], 3.0 * CONFIG.residency_min)
    exact = AutoscalingConfig(predictor_error_std=0.0)
    assert forecast_error(arrivals, exact, Predictor.ORACLE) == pytest.approx(0.0)


def test_forecast_requires_cold_start_shorter_than_residency() -> None:
    with pytest.raises(ValueError):
        forecast_concurrency(np.ones(50), AutoscalingConfig(cold_start_min=10), Predictor.ORACLE)


def test_predictor_error_is_reproducible_and_calibrated() -> None:
    error = predictor_error(20_000, CONFIG)
    assert np.array_equal(error, predictor_error(20_000, CONFIG))
    assert np.std(error) == pytest.approx(CONFIG.predictor_error_std, rel=0.15)
    assert np.corrcoef(error[:-1], error[1:])[0, 1] > 0.8


def test_simulate_static_capacity_meets_or_misses_slo() -> None:
    arrivals = np.full(200, 2.0)
    predicted = np.zeros_like(arrivals)
    enough = simulate(arrivals, predicted, CONFIG, FCFS, 0.0, static_capacity=20.0)
    assert enough.slo_attainment == pytest.approx(1.0)
    assert np.all(enough.busy <= enough.capacity + 1e-9)
    short = simulate(arrivals, predicted, CONFIG, FCFS, 0.0, static_capacity=10.0)
    assert short.slo_attainment < 0.9


def test_edf_prioritizes_tight_requests() -> None:
    arrivals = np.full(200, 3.0)
    predicted = np.zeros_like(arrivals)
    # Capacity for tight + relaxed requests only: EDF misses only batch, FCFS misses tight requests too.
    edf = simulate(arrivals, predicted, CONFIG, EDF, 0.0, static_capacity=20.0)
    fcfs = simulate(arrivals, predicted, CONFIG, FCFS, 0.0, static_capacity=20.0)
    assert edf.slo_attainment > fcfs.slo_attainment


def test_autoscaler_tracks_load_with_cold_start() -> None:
    arrivals = np.concatenate((np.full(100, 1.0), np.full(100, 4.0)))
    exact = AutoscalingConfig(predictor_error_std=0.0)
    predicted = forecast_concurrency(arrivals, exact, Predictor.ORACLE)
    result = simulate(arrivals, predicted, exact, FCFS, margin=0.0)
    assert result.capacity[50] == pytest.approx(10.0)
    assert result.capacity[-20] == pytest.approx(40.0)
    assert result.slo_attainment == pytest.approx(1.0)


def test_provision_iso_slo_meets_target_and_predictor_helps() -> None:
    arrivals = bursty_arrivals()
    results = {
        predictor: provision_iso_slo(arrivals, CURVE, CONFIG, FCFS, predictor)
        for predictor in (None, Predictor.PERSISTENCE, Predictor.ORACLE)
    }
    for result in results.values():
        assert result.slo_attainment >= CONFIG.target_slo_attainment
        assert 0 <= result.avg_serving_cost <= result.avg_cost
        assert result.avg_warm_cost >= 0
    assert results[Predictor.ORACLE].avg_cost < results[None].avg_cost


def test_stream_pilot_needs_smaller_warm_pool_than_ddit() -> None:
    arrivals = bursty_arrivals()
    stream_pilot, ddit = stream_pilot_and_ddit(2.0)
    results = run_experiment(arrivals, CURVE, (stream_pilot, ddit), CONFIG)
    assert list(results) == ["Static peak", "Reactive", "Predictive warm pool"]
    for by_system in results.values():
        assert by_system["StreamPilot"].avg_cost < by_system["DDiT"].avg_cost
        assert by_system["StreamPilot"].avg_warm_cost < by_system["DDiT"].avg_warm_cost


def test_naive_combo_sits_between_stream_pilot_and_ddit() -> None:
    systems = stream_pilot_and_baselines(1.5, 2.0)
    assert [system.name for system in systems] == ["StreamPilot", "Naive Combo", "DDiT"]
    naive_combo = systems[1]
    assert naive_combo.scheduler is Scheduler.FCFS
    assert not naive_combo.deadline_aware_sizing
    assert naive_combo.cost_multiplier == 1.5

    results = run_experiment(bursty_arrivals(), CURVE, systems, CONFIG)
    for by_system in results.values():
        assert by_system["StreamPilot"].avg_cost < by_system["Naive Combo"].avg_cost < by_system["DDiT"].avg_cost
