"""
Test simulator module.
"""

import sys
import os
import pytest

from copy import deepcopy

# Add current path
sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path

with temp_sys_path("simulator", "streamwise"):
    from sim_types import LatencyData
    from sim_types import PowerData
    from sim_types import GPUType
    from sim_types import Objective
    from sim_types import Solver
    from sim_types import QualityLevel

    from data_loading import load_latency_data
    from data_loading import load_power_data

    from constants import DEFAULT_WORKFLOW_CONFIG
    from constants import SECONDS_IN_HOUR

    from model_provisioner.policies import STREAMWISE_MILP_POLICY

    from workflows import WORKFLOWS

    from model_provisioner.milp import MILPAllocator

    from evaluator import evaluate_model_allocation

    from utils import to_models_df


def test_base() -> None:
    workflow = DEFAULT_WORKFLOW_CONFIG

    data_dir = "simulator/data/"
    latency_data: LatencyData = load_latency_data(data_dir=data_dir)
    power_data: PowerData = load_power_data(data_dir=data_dir)

    policy = deepcopy(STREAMWISE_MILP_POLICY)
    policy.solver = Solver.HIGHS  # Options: "gurobi", "highs"
    policy.objective = Objective.TTFF

    num_gpus = {
        GPUType.H200: 16,
        GPUType.GB200: 16,
    }

    allocator = MILPAllocator(
        workflow=workflow,
        latency_data=latency_data,
        power_data=power_data,
        policy=policy,
    )
    result = allocator.allocate(
        num_gpus=num_gpus,
        running_cost=True,  # To make it work with "highs"
        verbose=True,
    )
    assert result is not None
    assert result.models is not None
    df_models = to_models_df(result.models)
    assert not df_models.empty
    assert 0 < result.ttff_s < result.total_time_s
    assert 0 < result.cost
    assert 0 < result.total_energy
    assert result.gpus_used[GPUType.H200] > 8
    assert result.gpus_used[GPUType.GB200] > 8

    # Validation that the model allocation is correct
    result_2 = evaluate_model_allocation(
        models=result.models,
        num_gpus=num_gpus,
        workflow=workflow,
        latency_data=latency_data,
        power_data=power_data,
        policy=policy,
    )
    assert result_2 is not None
    assert result_2.models is not None
    df_models = to_models_df(result_2.models)
    assert not df_models.empty
    assert 0 < result_2.ttff_s < result_2.total_time_s
    assert 0 < result_2.cost
    assert 0 < result_2.total_energy
    assert result_2.gpus_used[GPUType.H200] > 8
    assert result_2.gpus_used[GPUType.GB200] > 8


def test_objective_cost() -> None:
    """Test that the MILP allocator optimizes for cost."""
    data_dir = "simulator/data/"
    latency_data = load_latency_data(data_dir=data_dir)
    power_data = load_power_data(data_dir=data_dir)

    policy_cost = deepcopy(STREAMWISE_MILP_POLICY)
    policy_cost.solver = Solver.HIGHS
    policy_cost.objective = Objective.COST

    allocator = MILPAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
        policy=policy_cost,
    )
    result = allocator.allocate(
        num_gpus={
            GPUType.H200: 24,
            GPUType.GB200: 16,
        },
        running_cost=True,  # Avoid: Highs interface does not support expressions of degree None
        verbose=True,
        max_ttff=10 * SECONDS_IN_HOUR,  # 10 hours
        max_makespan=10 * SECONDS_IN_HOUR,  # 10 hours
        force_num_gpus=True,
    )
    assert 0 < result.ttff_s <= result.total_time_s <= 10 * SECONDS_IN_HOUR
    assert 0 < result.cost
    assert 0 < result.total_energy
    assert result.gpus_used[GPUType.H200] == 24
    assert result.gpus_used[GPUType.GB200] == 16


def test_objective_ttff() -> None:
    """Test that the MILP allocator optimizes for TTFF."""
    data_dir = "simulator/data/"
    latency_data = load_latency_data(data_dir=data_dir)
    power_data = load_power_data(data_dir=data_dir)

    policy_ttff = deepcopy(STREAMWISE_MILP_POLICY)
    policy_ttff.solver = Solver.HIGHS
    policy_ttff.objective = Objective.TTFF

    allocator = MILPAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
        policy=policy_ttff,
    )
    result = allocator.allocate(
        num_gpus={
            GPUType.A100: 16,
            GPUType.GB200: 16,
        },
        running_cost=True,
        verbose=True,
        max_cost=1000,
        force_num_gpus=True,
    )
    assert 0 < result.ttff_s < result.total_time_s
    assert 0 < result.cost <= 1000
    assert 0 < result.total_energy
    assert result.gpus_used[GPUType.A100] > 8
    assert result.gpus_used[GPUType.GB200] > 8


def test_unfeasible() -> None:
    """Test that the MILP allocator raises an exception for unfeasible constraints."""
    data_dir = "simulator/data/"
    latency_data = load_latency_data(data_dir=data_dir)
    power_data = load_power_data(data_dir=data_dir)

    policy_cost = deepcopy(STREAMWISE_MILP_POLICY)
    policy_cost.solver = Solver.HIGHS
    policy_cost.objective = Objective.COST

    allocator = MILPAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
        policy=policy_cost,
    )
    with pytest.raises(Exception, match="A feasible solution was not found"):
        allocator.allocate(
            num_gpus={GPUType.A100: 16},
            running_cost=True,
            force_num_gpus=True,
            max_cost=1,  # Unfeasible constraint
        )


def test_highs_exception() -> None:
    """Test that the MILP allocator with Highs raises an exception for unsupported expressions."""
    data_dir = "simulator/data/"
    latency_data = load_latency_data(data_dir=data_dir)
    power_data = load_power_data(data_dir=data_dir)

    policy_cost = deepcopy(STREAMWISE_MILP_POLICY)
    policy_cost.solver = Solver.HIGHS
    policy_cost.objective = Objective.COST

    allocator = MILPAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
        policy=policy_cost,
    )
    with pytest.raises(Exception, match="Highs interface does not support expressions of degree None"):
        allocator.allocate(
            num_gpus={GPUType.H200: 32},
            running_cost=False,  # Trigger: Highs interface does not support expressions of degree None
        )


def test_gurobi() -> None:
    """Test that the MILP allocator with Gurobi raises an exception ."""
    data_dir = "simulator/data/"
    latency_data = load_latency_data(data_dir=data_dir)
    power_data = load_power_data(data_dir=data_dir)

    policy_cost = deepcopy(STREAMWISE_MILP_POLICY)
    policy_cost.solver = Solver.GUROBI
    policy_cost.objective = Objective.COST

    allocator = MILPAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
        policy=policy_cost,
    )
    with pytest.raises(
        Exception,
        match=r"(Model too large for size-limited license|No executable found for solver 'gurobi')",
    ):
        allocator.allocate(
            num_gpus={GPUType.A100: 24},
        )


@pytest.mark.parametrize("workflow_name", WORKFLOWS.keys())
def test_workflows(workflow_name: str) -> None:
    """Test that the MILP allocator can allocate for all workflows."""
    data_dir = "simulator/data/"
    latency_data = load_latency_data(data_dir=data_dir)
    power_data = load_power_data(data_dir=data_dir)

    policy = deepcopy(STREAMWISE_MILP_POLICY)
    policy.solver = Solver.HIGHS
    policy.objective = Objective.TTFF

    if WORKFLOWS[workflow_name].target_resolution != QualityLevel.HIGH:
        policy.use_upscaler = False

    allocator = MILPAllocator(
        workflow=WORKFLOWS[workflow_name],
        latency_data=latency_data,
        power_data=power_data,
        policy=policy,
    )
    result = allocator.allocate(
        num_gpus={GPUType.H200: 16},
        running_cost=True,  # Avoid: Highs interface does not support expressions of degree None
        verbose=True,
        force_num_gpus=True,
    )
    assert result is not None
    assert result.models is not None
    assert 0 < result.ttff_s < result.total_time_s <= 10 * SECONDS_IN_HOUR
    assert 0 < result.cost
    assert 0 < result.total_energy
    # assert result.gpus_used[GPUType.H200] == 16
    assert 8 < result.gpus_used[GPUType.H200] <= 16
