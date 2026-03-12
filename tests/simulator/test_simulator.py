"""
Test simulator module.
"""

import sys
import os
import pytest

from dataclasses import replace

# Add current path
sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path

with temp_sys_path("simulator"):
    from sim_types import WorkflowConfig
    from sim_types import Model
    from sim_types import Objective
    from sim_types import GPUType

    from constants import SECONDS_IN_HOUR
    from constants import DEFAULT_WORKFLOW_CONFIG

    from data_loading import load_latency_data
    from data_loading import load_power_data

    from auto_model_allocator import AutoModelAllocator
    from greedy import GreedyAllocator

    from policies import STREAMWISE_POLICY
    from policies import NAIVE_POLICY


def test_estimate_total_time() -> None:
    """8 A100 + 8 H100."""
    latency_data = load_latency_data("simulator/data/")

    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 8, GPUType.H100: 8},
        verbose=True,
    )
    assert result.gpus_used[GPUType.A100] == 8
    assert result.gpus_used[GPUType.H100] == 8
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s
    # assert 4000 <= result.total_time_s <= 4400
    assert 3000 <= result.total_time_s <= 4000
    # assert 3500 <= result.ttff_s <= 3800
    assert 2500 <= result.ttff_s <= 3000
    assert 0.2 <= result.tbf_s <= 0.4


def test_naive() -> None:
    """Naive policy with 8 A100 + 8 H100."""
    latency_data = load_latency_data("simulator/data/")
    allocator = AutoModelAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=NAIVE_POLICY,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 8, GPUType.H100: 8},
        verbose=True,
    )
    assert result.gpus_used[GPUType.A100] <= 8
    assert result.gpus_used[GPUType.H100] <= 8
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s

    video_seconds = DEFAULT_WORKFLOW_CONFIG.total_video_seconds
    expected_length_upper = 8 * SECONDS_IN_HOUR  # 8 hours
    expected_length_lower = 0.5 * SECONDS_IN_HOUR  # 30 minutes
    total_frames = DEFAULT_WORKFLOW_CONFIG.total_frames[Model.FT]
    assert expected_length_lower <= result.total_time_s <= expected_length_upper
    assert expected_length_lower - video_seconds <= result.ttff_s <= expected_length_upper - video_seconds
    assert expected_length_lower / total_frames <= result.tbf_s <= expected_length_upper / total_frames


def test_workflow_config() -> None:
    latency_data = load_latency_data("simulator/data/")

    workflow_config = WorkflowConfig(
        total_scenes=5,
        total_video_seconds=30,
        num_steps={
            Model.FLUX: 25,
            Model.HF: 20,
            Model.FT: 20,
        },
        hf_frames=[36, 72, 108, 144, 324],
        ft_frames=[9, 21, 41, 61, 77],
        frames_per_step_idx=2,
        total_frames={Model.HF: 20, Model.FT: 20},
        per_subscene_frames={Model.HF: 10, Model.FT: 10},
        total_subscenes=10,
    )
    allocator = GreedyAllocator(
        workflow=workflow_config,
        latency_data=latency_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 8, GPUType.H100: 8},
        verbose=True,
    )
    assert result.gpus_used[GPUType.A100] == 8
    assert result.gpus_used[GPUType.H100] == 8
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s
    assert 0 < result.tbf_s < 10

    # 8 A100 + 0 H100
    result = allocator.allocate(
        num_gpus={GPUType.A100: 8, GPUType.H100: 0},
        verbose=True,
    )
    assert result.gpus_used[GPUType.A100] == 8
    assert result.gpus_used.get(GPUType.H100, 0) == 0
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s

    # 0 A100 + 8 H100
    result = allocator.allocate(
        num_gpus={GPUType.H100: 8},
        verbose=True,
    )
    assert result.gpus_used.get(GPUType.A100, 0) == 0
    assert result.gpus_used[GPUType.H100] == 8
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s

    # 32 A100 + 128 H100
    result = allocator.allocate(
        num_gpus={GPUType.A100: 32, GPUType.H100: 128},
        verbose=True,
    )
    assert result.gpus_used[GPUType.A100] == 32
    # assert result.gpus_used[GPUType.H100] == 128
    assert 128 - 8 < result.gpus_used[GPUType.H100] <= 128  # TODO we should try to make it 128 no fragmenting
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s

    # 16 A100 + 8 H100 with verbose
    result = allocator.allocate(
        num_gpus={GPUType.A100: 16, GPUType.H100: 8},
        verbose=True,
    )
    assert result.gpus_used[GPUType.A100] == 16
    assert result.gpus_used[GPUType.H100] == 8
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s


def test_estimate_total_time_baseline() -> None:
    """Naive baseline with 8 A100 + 8 H100."""
    latency_data = load_latency_data("simulator/data/")
    allocator = AutoModelAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=NAIVE_POLICY,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 8, GPUType.H100: 8},
    )
    assert 0 < result.gpus_used[GPUType.A100] <= 8
    assert 0 < result.gpus_used[GPUType.H100] <= 8
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s


def test_estimate_total_energy() -> None:
    """Energy with 8 A100 + 8 H100."""
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")

    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=STREAMWISE_POLICY,
        power_data=power_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 8, GPUType.H100: 8},
    )
    assert 0 < result.gpus_used[GPUType.A100] <= 8
    assert 0 < result.gpus_used[GPUType.H100] == 8
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s
    assert 0 < result.total_energy
    assert 0 < result.cost


def test_AvH() -> None:
    """H100 should be better than A100."""
    latency_data = load_latency_data("simulator/data/")

    # 8 A100
    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    result_8a = allocator.allocate(
        num_gpus={GPUType.A100: 8, GPUType.H100: 0},
    )
    assert result_8a.gpus_used[GPUType.A100] == 8
    assert result_8a.gpus_used.get(GPUType.H100, 0) == 0

    # 8 H100
    result_8h = allocator.allocate(
        num_gpus={GPUType.H100: 8},
    )
    assert result_8h.gpus_used.get(GPUType.A100, 0) == 0
    assert result_8h.gpus_used[GPUType.H100] == 8

    # A100 should be worse than H100
    assert result_8a.total_time_s > result_8h.total_time_s
    assert result_8a.ttff_s > result_8h.ttff_s
    assert result_8a.tbf_s > result_8h.tbf_s


def test_estimate_total_time_A() -> None:
    """More A100s should lead to better performance."""
    latency_data = load_latency_data("simulator/data/")

    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )

    # 8 A100
    result_8a = allocator.allocate(
        num_gpus={GPUType.A100: 8, GPUType.H100: 0},
    )
    assert result_8a.gpus_used[GPUType.A100] == 8
    assert result_8a.gpus_used.get(GPUType.H100, 0) == 0
    assert 0 < result_8a.tbf_s < result_8a.ttff_s < result_8a.total_time_s

    # 16 A100
    result_16a = allocator.allocate(
        num_gpus={GPUType.A100: 16, GPUType.H100: 0},
    )
    assert result_16a.gpus_used[GPUType.A100] == 16
    assert result_16a.gpus_used.get(GPUType.H100, 0) == 0
    assert 0 < result_16a.tbf_s < result_16a.ttff_s < result_16a.total_time_s
    # 16 A100s should be faster than 8 A100s
    assert result_8a.total_time_s > result_16a.total_time_s
    assert result_8a.ttff_s > result_16a.ttff_s
    assert result_8a.tbf_s > result_16a.tbf_s

    # 24 A100
    result_24a = allocator.allocate(
        num_gpus={GPUType.A100: 24},
    )
    assert result_24a.gpus_used[GPUType.A100] == 24
    assert result_24a.gpus_used.get(GPUType.H100, 0) == 0
    assert result_24a.gpus_used.get(GPUType.H200, 0) == 0
    assert result_24a.gpus_used.get(GPUType.GB200, 0) == 0
    assert 0 < result_24a.tbf_s < result_24a.ttff_s < result_24a.total_time_s
    # 24 A100s should be faster than 16 A100s
    assert result_16a.total_time_s > result_24a.total_time_s
    assert result_16a.ttff_s > result_24a.ttff_s
    assert result_16a.tbf_s > result_24a.tbf_s

    # 4096 A100
    result_4096a = allocator.allocate(
        num_gpus={GPUType.A100: 4096},
    )
    assert result_4096a.gpus_used[GPUType.A100] <= 4096
    assert result_4096a.gpus_used.get(GPUType.H100, 0) == 0
    assert 0 < result_4096a.tbf_s < result_4096a.ttff_s < result_4096a.total_time_s
    # 4096 A100s should be faster than 24 A100s
    assert result_24a.total_time_s > result_4096a.total_time_s
    assert result_24a.ttff_s > result_4096a.ttff_s
    assert result_24a.tbf_s > result_4096a.tbf_s


def test_estimate_total_time_H() -> None:
    """More H100s should lead to better performance."""
    latency_data = load_latency_data("simulator/data/")

    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )

    # 8 H100
    result_8h = allocator.allocate(
        num_gpus={GPUType.H100: 8},
    )
    assert result_8h.gpus_used.get(GPUType.A100, 0) == 0
    assert result_8h.gpus_used[GPUType.H100] == 8
    assert 0 < result_8h.tbf_s < result_8h.ttff_s < result_8h.total_time_s

    # 16 H100
    result_16h = allocator.allocate(
        num_gpus={GPUType.A100: 0, GPUType.H100: 16},
    )
    assert result_16h.gpus_used.get(GPUType.A100, 0) == 0
    assert result_16h.gpus_used[GPUType.H100] == 16
    assert 0 < result_16h.tbf_s < result_16h.ttff_s < result_16h.total_time_s
    # 16 H100s should be faster than 8 H100s
    assert result_8h.total_time_s > result_16h.total_time_s
    assert result_8h.ttff_s > result_16h.ttff_s
    assert result_8h.tbf_s > result_16h.tbf_s

    # 24 H100
    result_24h = allocator.allocate(
        num_gpus={GPUType.H100: 24},
    )
    assert result_24h.gpus_used.get(GPUType.A100, 0) == 0
    assert result_24h.gpus_used[GPUType.H100] == 24
    assert 0 < result_24h.tbf_s < result_24h.ttff_s < result_24h.total_time_s
    # 24 H100s should be faster than 16 H100s
    assert result_16h.total_time_s > result_24h.total_time_s
    assert result_16h.ttff_s > result_24h.ttff_s
    assert result_16h.tbf_s > result_24h.tbf_s


def test_time_vs_power() -> None:
    """Compare time and power estimation for the same setup."""
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")

    # Time only
    allocator_time = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=STREAMWISE_POLICY,
    )
    result_time = allocator_time.allocate(
        num_gpus={GPUType.A100: 0, GPUType.H100: 8},
        verbose=True,
    )
    assert 0 < result_time.tbf_s < result_time.ttff_s < result_time.total_time_s

    # Time+Energy only
    allocator_energy = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
        policy=STREAMWISE_POLICY,
    )
    result_energy = allocator_energy.allocate(
        num_gpus={GPUType.H100: 8},
        verbose=True,
    )
    assert 0 < result_energy.tbf_s < result_energy.ttff_s < result_energy.total_time_s
    assert result_energy.total_time_s == result_time.total_time_s
    assert result_energy.ttff_s == result_time.ttff_s
    assert result_energy.tbf_s == result_time.tbf_s

    assert 0 == result_time.total_energy
    assert 0 < result_energy.total_energy


@pytest.mark.parametrize("objective", list(Objective))
def test_scheduler(
    objective: Objective,
) -> None:
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")

    policy = replace(STREAMWISE_POLICY)
    policy.objective = objective

    allocator = AutoModelAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
        policy=policy,
    )

    result = allocator.allocate(
        num_gpus={GPUType.A100: 8, GPUType.H100: 8},
        verbose=True,
    )
    assert result.cost > 0
    if objective == Objective.NONE:
        assert result.gpus_used[GPUType.A100] == 3
    else:
        assert result.gpus_used[GPUType.A100] == 8
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s
    assert 0 < result.total_energy

    # Single server type
    result = allocator.allocate(
        num_gpus={GPUType.H100: 8},
    )
    assert result.gpus_used.get(GPUType.A100, 0) == 0
    assert 0 < result.gpus_used[GPUType.H100] <= 8
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s
    assert 0 < result.total_energy


def test_scheduler_wrong() -> None:
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")

    policy = replace(STREAMWISE_POLICY)
    policy.objective = "wrong"  # type: ignore

    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
        policy=policy,
    )

    with pytest.raises(ValueError, match="Cannot recognize objective wrong"):
        allocator.allocate(
            num_gpus={
                GPUType.A100: 8,
                GPUType.H100: 8
            },
        )


@pytest.mark.parametrize("num_a100s", [num_a100s for num_a100s in range(8, 128 + 1, 8)])
def test_estimate_A100(
    num_a100s: int,
) -> None:
    latency_data = load_latency_data("simulator/data/")

    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=STREAMWISE_POLICY,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: num_a100s, GPUType.H100: 0},
        verbose=True,
    )
    assert result.gpus_used[GPUType.A100] == num_a100s
    assert result.gpus_used.get(GPUType.H100, 0) == 0
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s


@pytest.mark.parametrize("num_h100s", [num_h100s for num_h100s in range(8, 128 + 1, 8)])
def test_estimate_H100(
    num_h100s: int,
) -> None:
    latency_data = load_latency_data("simulator/data/")

    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.H100: num_h100s},
        verbose=True,
    )
    assert result.gpus_used.get(GPUType.A100, 0) == 0
    assert result.gpus_used[GPUType.H100] == num_h100s
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s


@pytest.mark.parametrize("num_h200s", [num_h200s for num_h200s in range(8, 128 + 1, 8)])
def test_estimate_H200(
    num_h200s: int,
) -> None:
    latency_data = load_latency_data("simulator/data/")

    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=STREAMWISE_POLICY,
    )
    result = allocator.allocate(
        num_gpus={GPUType.H200: num_h200s},
        verbose=True,
    )
    assert result.gpus_used.get(GPUType.A100, 0) == 0
    assert result.gpus_used.get(GPUType.H100, 0) == 0
    assert result.gpus_used[GPUType.H200] == num_h200s
    assert result.gpus_used.get(GPUType.GB200, 0) == 0
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s


@pytest.mark.parametrize("num_gb200s", [num_gb200s for num_gb200s in range(8, 128 + 1, 8)])
def test_estimate_GB200(
    num_gb200s: int,
) -> None:
    latency_data = load_latency_data("simulator/data/")

    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=STREAMWISE_POLICY,
    )
    result = allocator.allocate(
        num_gpus={GPUType.GB200: num_gb200s},
        verbose=True,
    )
    assert result.gpus_used.get(GPUType.A100, 0) == 0
    assert result.gpus_used.get(GPUType.H100, 0) == 0
    assert result.gpus_used.get(GPUType.H200, 0) == 0
    assert result.gpus_used[GPUType.GB200] == num_gb200s
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s
