"""
Tests for the Helix allocator.

Helix optimizes models one-by-one following MODEL_ORDER using per-model MILP.
"""

import sys
import os

# Add current path
sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path

with temp_sys_path("simulator", "streamwise"):
    from constants import DEFAULT_WORKFLOW_CONFIG
    from sim_types import GPUType
    from sim_types import Model
    from sim_types import MODEL_ORDER
    from sim_types import Solver
    from data_loading import load_latency_data
    from data_loading import load_power_data
    from model_provisioner.helix import HelixAllocator
    from model_provisioner.policies import HELIX_POLICY


def test_get_model_order() -> None:
    """Test that get_model_order() returns models sorted by MODEL_ORDER."""
    order = DEFAULT_WORKFLOW_CONFIG.get_model_order()
    assert len(order) > 0
    # Check ordering is consistent with MODEL_ORDER
    for i in range(len(order) - 1):
        assert MODEL_ORDER[order[i]] < MODEL_ORDER[order[i + 1]]
    # All models in order should be in the workflow
    for m in order:
        assert m in DEFAULT_WORKFLOW_CONFIG.models


def test_helix_policy_solver() -> None:
    """Verify HELIX_POLICY uses Solver.HELIX."""
    assert HELIX_POLICY.solver == Solver.HELIX


def test_produces_valid_result() -> None:
    """Verify the Helix result has all required fields populated correctly."""
    latency_data = load_latency_data("simulator/data/")
    allocator = HelixAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 32},
        milp_solver=Solver.HIGHS,
    )
    assert result.total_time_s > 0
    assert result.ttff_s > 0
    assert result.tbf_s >= 0
    assert result.cost >= 0
    assert result.total_energy >= 0
    assert result.models is not None
    assert len(result.models) > 0


def test_produces_valid_result_verbose() -> None:
    """Verify verbose mode works without errors."""
    latency_data = load_latency_data("simulator/data/")
    allocator = HelixAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 32},
        verbose=True,
        milp_solver=Solver.HIGHS,
    )
    assert result.total_time_s > 0
    assert result.models is not None


def test_gpu_budget_respected() -> None:
    """Verify total GPUs used does not exceed the budget."""
    latency_data = load_latency_data("simulator/data/")
    allocator = HelixAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    budget = {GPUType.A100: 16}
    result = allocator.allocate(
        num_gpus=budget,
        milp_solver=Solver.HIGHS,
    )
    for gpu_type, used in result.gpus_used.items():
        assert used <= budget.get(gpu_type, 0), \
            f"{gpu_type.value}: used {used} > budget {budget.get(gpu_type, 0)}"


def test_all_workflow_models_allocated() -> None:
    """Verify models are allocated following MODEL_ORDER until GPUs are exhausted.

    Helix allocates sequentially, so later models may be skipped if earlier
    models consume the whole budget.  With a large GPU pool the first several
    models should still be present.
    """
    latency_data = load_latency_data("simulator/data/")
    allocator = HelixAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 64},
        milp_solver=Solver.HIGHS,
    )
    # Collect all models with non-empty allocations
    allocated_models: set[Model] = set()
    for gpu_type, model_dict in result.models.items():
        for model, allocs in model_dict.items():
            if allocs:
                allocated_models.add(model)

    # At least the first models in MODEL_ORDER (GEMMA, FLUX, OTHERS) should
    # always be allocated since they require few GPUs.
    model_order = DEFAULT_WORKFLOW_CONFIG.get_model_order()
    for model in model_order[:3]:
        if DEFAULT_WORKFLOW_CONFIG.model_work.get(model, 0) > 0:
            assert model in allocated_models, \
                f"Model {model.value} has work but was not allocated"

    # At least one model should be allocated
    assert len(allocated_models) >= 1


def test_with_power_data() -> None:
    """Verify allocation works with power data provided."""
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")
    allocator = HelixAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 32},
        milp_solver=Solver.HIGHS,
    )
    assert result.total_time_s > 0
    assert result.total_energy >= 0
    assert result.cost >= 0


def test_custom_per_model_time_limit() -> None:
    """Verify custom per-model time limit is accepted."""
    latency_data = load_latency_data("simulator/data/")
    allocator = HelixAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 16},
        per_model_time_limit=10,
        milp_solver=Solver.HIGHS,
    )
    assert result.total_time_s > 0
