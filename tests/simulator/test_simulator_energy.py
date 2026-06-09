import sys
import os
import pytest

from dataclasses import replace

# Add current path
sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path

with temp_sys_path("simulator", "streamwise"):
    from constants import DEFAULT_WORKFLOW_CONFIG

    from sim_types import GPUType
    from sim_types import Model
    from sim_types import Objective
    from sim_types import Solver

    from data_loading import load_latency_data
    from data_loading import load_power_data

    from auto_model_allocator import AutoModelAllocator
    from model_provisioner.greedy import GreedyAllocator
    from model_provisioner.naive_baseline import NaiveAllocator

    from model_provisioner.policies import NAIVE_POLICY
    from model_provisioner.policies import STREAMWISE_POLICY


def test_energy() -> None:
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")

    # 8 A100 + 8 H100
    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
        policy=STREAMWISE_POLICY,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 8, GPUType.H100: 8},
    )
    assert result.gpus_used[GPUType.A100] == 8
    assert result.gpus_used[GPUType.H100] == 8
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s
    assert 0 < result.total_energy
    assert 0 < result.cost


def test_energy_8A_0H() -> None:
    """8 A100."""
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")

    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 8},
    )
    assert result.gpus_used[GPUType.A100] == 8
    assert result.gpus_used.get(GPUType.H100, 0) == 0
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s
    assert 0 < result.total_energy


def test_energy_A() -> None:
    """A100 combinations."""
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")

    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
    )
    for num_a100s in range(8, 64 + 1, 8):
        result = allocator.allocate(
            num_gpus={GPUType.A100: num_a100s},
        )
        assert result.gpus_used[GPUType.A100] == num_a100s
        assert result.gpus_used.get(GPUType.H100, 0) == 0
        assert 0 < result.tbf_s < result.ttff_s < result.total_time_s


def test_energy_0A_8H() -> None:
    """8 H100."""
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")

    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.H100: 8},
        verbose=True,
    )
    assert result.gpus_used.get(GPUType.A100, 0) == 0
    assert result.gpus_used[GPUType.H100] == 8
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s

    video_seconds = DEFAULT_WORKFLOW_CONFIG.total_video_seconds
    expected_length_upper = 4 * 60 * 60  # 4 hours
    expected_length_lower = 2 * 60 * 60  # 2 hours
    total_frames = DEFAULT_WORKFLOW_CONFIG.total_frames[Model.FT]
    assert expected_length_lower <= result.total_time_s <= expected_length_upper
    assert expected_length_lower - video_seconds <= result.ttff_s <= expected_length_upper - video_seconds
    assert expected_length_lower / total_frames <= result.tbf_s <= expected_length_upper / total_frames

    idle_energy = 8 * power_data.gpus[GPUType.H100].idle * result.total_time_s
    active_energy = 8 * power_data.gpus[GPUType.H100].tdp * result.total_time_s
    assert idle_energy <= result.total_energy <= active_energy


def test_energy_H() -> None:
    """H100 combinations."""
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")

    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
    )
    for num_h100s in range(8, 64 + 1, 8):
        result = allocator.allocate(
            num_gpus={GPUType.H100: num_h100s},
        )
        assert result.gpus_used.get(GPUType.A100, 0) == 0
        assert result.gpus_used[GPUType.H100] == num_h100s
        assert 0 < result.tbf_s < result.ttff_s < result.total_time_s


def test_energy_0A_8H_naive() -> None:
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")

    allocator = NaiveAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
        policy=NAIVE_POLICY,
    )
    # 8 H100
    result = allocator.allocate(
        num_gpus={GPUType.H100: 8},
        verbose=True,
    )
    assert result.gpus_used.get(GPUType.A100, 0) == 0
    assert result.gpus_used.get(GPUType.H100, 0) == 8
    assert result.total_time_s > 0
    assert result.total_energy > 0


def test_energy_24A_0H() -> None:
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")

    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
    )

    # 24 A100 + 0 H100
    result = allocator.allocate(
        num_gpus={GPUType.A100: 24},
    )
    assert result.total_time_s > 0
    assert result.gpus_used[GPUType.A100] == 24
    assert result.gpus_used.get(GPUType.H100, 0) == 0
    assert result.total_energy > 0


def test_energy_0A_32H() -> None:
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")
    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
    )
    # 0 A100 + 32 H100
    result = allocator.allocate(
        num_gpus={GPUType.H100: 32},
    )
    assert result.total_time_s > 0
    assert result.gpus_used.get(GPUType.A100, 0) == 0
    assert result.gpus_used[GPUType.H100] == 32
    assert result.total_energy > 0


def test_energy_16A_32H() -> None:
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")
    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
    )
    # 16 A100 + 32 H100
    result = allocator.allocate(
        num_gpus={GPUType.A100: 16, GPUType.H100: 32},
        verbose=True,
    )
    assert result.total_time_s > 0
    assert result.gpus_used[GPUType.A100] == 16
    assert result.gpus_used[GPUType.H100] == 32
    assert result.total_energy > 0


def test_energy_8A_64H() -> None:
    # 8 A100 + 64 H100
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")
    allocator = AutoModelAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
        policy=NAIVE_POLICY,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 8, GPUType.H100: 64},
    )
    assert result.gpus_used[GPUType.A100] == 8
    assert result.gpus_used[GPUType.H100] == 64
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s
    assert 0 < result.total_energy


def test_energy_16H_naive() -> None:
    """16 H100 Naive."""
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")
    allocator = NaiveAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.H100: 16},
        verbose=True,
    )
    assert result.gpus_used.get(GPUType.A100, 0) == 0
    assert result.gpus_used[GPUType.H100] == 16
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s
    assert 0 < result.total_energy


def test_energy_0() -> None:
    """No GPUs."""
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")
    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
    )
    with pytest.raises(AssertionError, match="Total number of GPUs must be at least 8"):
        allocator.allocate(num_gpus={})


def test_energy_naive_parallelism() -> None:
    """Error from simulation."""
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")

    policy = replace(STREAMWISE_POLICY)
    policy.objective = Objective.TTFF
    policy.solver = Solver.NAIVE
    allocator = AutoModelAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
        policy=policy,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 1280, GPUType.H100: 576},
    )
    assert 1280 - 8 <= result.gpus_used[GPUType.A100] <= 1280
    assert 576 - 8 <= result.gpus_used[GPUType.H100] <= 576
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s
    assert 0 < result.total_energy
