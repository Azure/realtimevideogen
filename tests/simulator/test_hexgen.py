import sys
import os
import pytest

# Add current path
sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path

with temp_sys_path("simulator", "streamwise"):
    from constants import DEFAULT_WORKFLOW_CONFIG
    from sim_types import GPUType
    from data_loading import load_latency_data
    from model_provisioner.hexgen import HexGenAllocator
    from model_provisioner.hexgen import _get_model_order
    from sim_types import MODEL_ORDER


def test_get_model_order() -> None:
    """Test that _get_model_order returns models sorted by MODEL_ORDER."""
    order = _get_model_order(DEFAULT_WORKFLOW_CONFIG)
    assert len(order) > 0
    # Check ordering is consistent with MODEL_ORDER
    for i in range(len(order) - 1):
        assert MODEL_ORDER[order[i]] < MODEL_ORDER[order[i + 1]]
    # All models in order should be in the workflow
    for m in order:
        assert m in DEFAULT_WORKFLOW_CONFIG.models


def test_8A() -> None:
    """8 x A100 (single server)."""
    latency_data = load_latency_data("simulator/data/")
    allocator = HexGenAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 8},
    )
    assert result.gpus_used[GPUType.A100] == 8
    assert 0 < result.ttff_s < result.total_time_s
    assert 0 < result.tbf_s


def test_8H() -> None:
    """8 x H100 (single server)."""
    latency_data = load_latency_data("simulator/data/")
    allocator = HexGenAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.H100: 8},
    )
    assert result.gpus_used[GPUType.H100] == 8
    assert 0 < result.ttff_s < result.total_time_s
    assert 0 < result.tbf_s


def test_16A() -> None:
    """16 x A100."""
    latency_data = load_latency_data("simulator/data/")
    allocator = HexGenAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 16},
        verbose=True,
    )
    assert result.gpus_used[GPUType.A100] == 16
    assert 0 < result.ttff_s < result.total_time_s < 24 * 60 * 60
    assert 0 < result.tbf_s < 5


def test_64A() -> None:
    """64 x A100."""
    latency_data = load_latency_data("simulator/data/")
    allocator = HexGenAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 64},
        verbose=True,
    )
    assert result.gpus_used[GPUType.A100] == 64
    assert 0 < result.ttff_s < result.total_time_s < 24 * 60 * 60
    assert 0 < result.tbf_s < 5


def test_64H() -> None:
    """64 x H100."""
    latency_data = load_latency_data("simulator/data/")
    allocator = HexGenAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.H100: 64},
    )
    assert result.gpus_used.get(GPUType.A100, 0) == 0
    assert result.gpus_used[GPUType.H100] == 64
    assert 0 < result.ttff_s < result.total_time_s < 24 * 60 * 60
    assert 0 < result.tbf_s < 3


def test_8A_8H() -> None:
    """8 x A100 + 8 x H100 (mixed GPU types)."""
    latency_data = load_latency_data("simulator/data/")
    allocator = HexGenAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 8, GPUType.H100: 8},
    )
    assert result.gpus_used[GPUType.A100] == 8
    assert result.gpus_used[GPUType.H100] == 8
    assert 0 < result.ttff_s < result.total_time_s
    assert 0 < result.tbf_s


def test_32A_32H() -> None:
    """32 x A100 + 32 x H100 (mixed, larger)."""
    latency_data = load_latency_data("simulator/data/")
    allocator = HexGenAllocator(
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
    assert 0 < result.tbf_s < 3


def test_no_gpus_error() -> None:
    """No GPUs should raise an error."""
    latency_data = load_latency_data("simulator/data/")
    allocator = HexGenAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    with pytest.raises(AssertionError, match="Total number of GPUs must be at least 8"):
        allocator.allocate(
            num_gpus={GPUType.A100: 0, GPUType.H100: 0},
        )


def test_is_subclass_of_greedy() -> None:
    """HexGenAllocator should extend GreedyAllocator."""
    from model_provisioner.greedy import GreedyAllocator
    latency_data = load_latency_data("simulator/data/")
    allocator = HexGenAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    assert isinstance(allocator, GreedyAllocator)


@pytest.mark.parametrize("gpu_type", [GPUType.A100, GPUType.H100, GPUType.H200])
def test_single_gpu_type_parametrized(gpu_type: GPUType) -> None:
    """Test HexGen with various single GPU types at 16 GPUs."""
    latency_data = load_latency_data("simulator/data/")
    allocator = HexGenAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    result = allocator.allocate(
        num_gpus={gpu_type: 16},
    )
    assert result.gpus_used[gpu_type] == 16
    assert 0 < result.ttff_s < result.total_time_s
    assert 0 < result.tbf_s


def test_produces_valid_result() -> None:
    """Verify the HexGen result has all required fields populated correctly."""
    latency_data = load_latency_data("simulator/data/")
    allocator = HexGenAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 32},
    )
    assert result.total_time_s > 0
    assert result.ttff_s > 0
    assert result.tbf_s >= 0
    assert result.cost >= 0
    assert result.total_energy >= 0
    assert result.models is not None
    assert len(result.models) > 0
