import sys
import os
import pytest
from dataclasses import replace

# Add current path
sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path

with temp_sys_path("simulator"):
    from constants import DEFAULT_WORKFLOW_CONFIG
    from constants import SECONDS_IN_HOUR

    from workflows import WORKFLOWS

    from sim_types import GPUType
    from sim_types import QualityLevel

    from data_loading import load_latency_data
    from data_loading import load_power_data

    from greedy import GreedyAllocator

    from policies import STREAMWISE_POLICY


def test_allocate_8A_8H() -> None:
    """8 A100 + 8 H100."""
    latency_data = load_latency_data("simulator/data/")
    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 8, GPUType.H100: 8},
    )
    assert result.gpus_used[GPUType.A100] == 8
    assert result.gpus_used[GPUType.H100] == 8
    assert 0 < result.ttff_s < result.total_time_s
    # assert 4100 <= result.total_time_s <= 4400  # TODO old values
    assert 50 * 60 <= result.total_time_s <= 60 * 60  # 50-60 minutes
    # assert 3500 <= result.ttff_s <= 3800  # TODO old values
    assert 45 * 60 <= result.ttff_s <= 55 * 60  # 45-55 minutes
    assert 0.2 <= result.tbf_s <= 0.4


def test_allocate_32A_32H() -> None:
    """32 A100 + 32 H100"""
    latency_data = load_latency_data("simulator/data/")
    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 32, GPUType.H100: 32},
        verbose=True,
    )
    assert result.gpus_used[GPUType.A100] == 32
    assert result.gpus_used[GPUType.H100] == 32
    assert 0 < result.ttff_s < result.total_time_s
    assert 0 < result.tbf_s < 1


def test_allocate_16A() -> None:
    """16 A100 + 0 H100."""
    latency_data = load_latency_data("simulator/data/")
    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 16, GPUType.H100: 0},
        verbose=True,
    )
    assert result.gpus_used[GPUType.A100] == 16
    assert result.gpus_used.get(GPUType.H100, 0) == 0
    assert 0 < result.ttff_s < result.total_time_s < 10 * 60 * 60
    assert 0 < result.tbf_s < 1


def test_allocate_64A() -> None:
    """64 A100 + 0 H100."""
    latency_data = load_latency_data("simulator/data/")
    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 64, GPUType.H100: 0},
        verbose=True,
    )
    assert result.gpus_used[GPUType.A100] == 64
    assert result.gpus_used.get(GPUType.H100, 0) == 0
    assert 0 < result.ttff_s < result.total_time_s < 10 * 60 * 60
    assert 0 < result.tbf_s < 1


def test_allocate_64H() -> None:
    """0 A100 + 64 H100."""
    latency_data = load_latency_data("simulator/data/")
    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 0, GPUType.H100: 64},
    )
    assert result.gpus_used.get(GPUType.A100, 0) == 0
    assert result.gpus_used[GPUType.H100] == 64
    assert 0 < result.ttff_s < result.total_time_s < 10 * 60 * 60
    assert 0 < result.tbf_s < 1


def test_allocate_64GB() -> None:
    """64 GB200."""
    latency_data = load_latency_data("simulator/data/")
    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.GB200: 64},
        verbose=True,
    )
    assert result.gpus_used[GPUType.GB200] == 64
    assert result.gpus_used.get(GPUType.A100, 0) == 0
    assert result.gpus_used.get(GPUType.H100, 0) == 0
    assert result.gpus_used.get(GPUType.H200, 0) == 0
    assert 0 < result.ttff_s < result.total_time_s < 10 * 60 * 60
    assert 0 < result.tbf_s < 1


def test_allocate_32A_8H() -> None:
    """32 A100 + 8 H100."""
    latency_data = load_latency_data("simulator/data/")
    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 32, GPUType.H100: 8},
        verbose=True,
    )
    assert result.gpus_used[GPUType.A100] == 32
    assert result.gpus_used[GPUType.H100] == 8
    assert 0 < result.ttff_s < result.total_time_s < 10 * 60 * 60
    assert 0 < result.tbf_s < 1


def test_allocate_16A_80H() -> None:
    """16 A100 + 80 H100."""
    latency_data = load_latency_data("simulator/data/")
    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 16, GPUType.H100: 80},
        verbose=True,
    )
    assert result.gpus_used[GPUType.A100] == 16
    assert result.gpus_used[GPUType.H100] == 80
    assert 0 < result.ttff_s < result.total_time_s < 10 * 60 * 60
    assert 0 < result.tbf_s < 1


def test_allocate_0H() -> None:
    """No GPUs."""
    latency_data = load_latency_data("simulator/data/")
    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    with pytest.raises(AssertionError, match="Total number of GPUs must be at least 8"):
        allocator.allocate(
            num_gpus={GPUType.A100: 0, GPUType.H100: 0},
            verbose=True,
        )


def test_32A_32H_options() -> None:
    """32 A100 + 32 H100 with all options enabled."""
    latency_data = load_latency_data("simulator/data/")
    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    result = allocator.allocate(
        num_gpus={
            GPUType.A100: 32,
            GPUType.H100: 32,
        },
        verbose=True,
        allow_removal=True,
        allow_merging=True,
        look_ahead_replicas=4,
    )
    assert result.gpus_used[GPUType.A100] == 32
    assert result.gpus_used[GPUType.H100] == 32
    assert 0 < result.ttff_s < result.total_time_s < 10 * 60 * 60
    assert 0 < result.tbf_s < 1


@pytest.mark.parametrize("workflow", WORKFLOWS.values())
def test_workflows(workflow) -> None:
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")
    policy = STREAMWISE_POLICY
    if workflow.target_resolution != QualityLevel.HIGH:
        policy = replace(policy, use_upscaler=False)
    allocator = GreedyAllocator(
        workflow=workflow,
        latency_data=latency_data,
        power_data=power_data,
        policy=policy,
    )
    result = allocator.allocate(
        num_gpus={
            GPUType.A100: 16,
            GPUType.H100: 16
        },
        verbose=True,
    )
    assert result.gpus_used[GPUType.A100] == 16
    assert result.gpus_used[GPUType.H100] == 16
    assert 0 < result.ttff_s < result.total_time_s < 24 * SECONDS_IN_HOUR
    assert 0 < result.tbf_s < 1
