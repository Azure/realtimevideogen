"""
Test baselines.
"""

import sys
import os
import pytest

# Add current path
sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path

with temp_sys_path("simulator", "streamwise"):
    from sim_types import GPUType
    from sim_types import Model

    from constants import DEFAULT_WORKFLOW_CONFIG
    from constants import SECONDS_IN_HOUR
    from constants import POWER_GPU_IDLE
    from constants import POWER_GPU_TDP

    from data_loading import load_latency_data
    from data_loading import load_power_data

    from auto_model_allocator import AutoModelAllocator
    from model_provisioner.naive_baseline import NaiveAllocator
    from model_provisioner.greedy import GreedyAllocator

    from model_provisioner.policies import NAIVE_POLICY
    from model_provisioner.policies import BASELINE_POLICIES
    from model_provisioner.policies import STREAMWISE_POLICY

    from workflows import SHORTS_WORKFLOW
    from workflows import WORKFLOWS


def test_baseline() -> None:
    """8 A100 + 8 H100."""
    latency_data = load_latency_data("simulator/data/")
    allocator = NaiveAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 8, GPUType.H100: 8},
    )
    assert 0 < result.gpus_used.get(GPUType.A100, 0) <= 8
    assert 0 < result.gpus_used.get(GPUType.H100, 0) <= 8
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s


@pytest.mark.parametrize("num_a100s", [num_a100s for num_a100s in range(8, 128 + 1, 8)])
def test_baseline_A_options(
    num_a100s: int,
) -> None:
    """A100 combinations."""
    latency_data = load_latency_data("simulator/data/")
    allocator = NaiveAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: num_a100s},
        verbose=True,
    )
    assert result.gpus_used.get(GPUType.A100, 0) == num_a100s
    assert result.gpus_used.get(GPUType.H100, 0) == 0
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s


def test_baseline_8A_upscaler() -> None:
    """8 A100 + 0 H100 with upscaler."""
    latency_data = load_latency_data("simulator/data/")
    allocator = NaiveAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=BASELINE_POLICIES["naive upscaler"],
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 8},
        verbose=True,
    )
    assert result.gpus_used.get(GPUType.A100, 0) == 8
    assert result.gpus_used.get(GPUType.H100, 0) == 0
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s


@pytest.mark.parametrize("num_h100s", [num_h100s for num_h100s in range(8, 128 + 1, 8)])
def test_baseline_H(
    num_h100s: int,
) -> None:
    """H100 combinations."""
    latency_data = load_latency_data("simulator/data/")
    allocator = NaiveAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.H100: num_h100s},
        verbose=True,
    )
    assert result.gpus_used.get(GPUType.A100, 0) == 0
    assert result.gpus_used.get(GPUType.H100, 0) == num_h100s
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s


@pytest.mark.parametrize("num_h100s", [num_h100s for num_h100s in range(8, 128 + 1, 8)])
def test_baseline_H_upscaler(
    num_h100s: int,
) -> None:
    """H100 with upscaler."""
    latency_data = load_latency_data("simulator/data/")
    allocator = NaiveAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=BASELINE_POLICIES["naive upscaler"],
    )
    result = allocator.allocate(
        num_gpus={GPUType.H100: num_h100s},
    )
    assert result.gpus_used.get(GPUType.A100, 0) == 0
    assert result.gpus_used.get(GPUType.H100, 0) <= num_h100s
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s


def test_baseline_24H() -> None:
    """0 A100 + 24 H100"""
    latency_data = load_latency_data("simulator/data/")

    allocator = NaiveAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 0, GPUType.H100: 24},
        verbose=True,
    )
    assert result.gpus_used.get(GPUType.A100, 0) == 0
    assert 24 - 8 < result.gpus_used.get(GPUType.H100, 0) <= 24
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s


def test_baseline_96A_96H() -> None:
    """96 A100 + 96 H100."""
    latency_data = load_latency_data("simulator/data/")

    allocator = NaiveAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=NAIVE_POLICY,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 96, GPUType.H100: 96},
        verbose=True,
    )
    assert result.gpus_used.get(GPUType.A100, 0) == 96
    assert result.gpus_used.get(GPUType.H100, 0) == 96
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s


def test_baseline_64A_16H() -> None:
    """64 A100 + 16 H100."""
    latency_data = load_latency_data("simulator/data/")

    allocator = NaiveAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=BASELINE_POLICIES["naive upscaler"],
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 64, GPUType.H100: 16},
        verbose=True,
    )
    assert result.gpus_used.get(GPUType.A100, 0) == 64
    assert result.gpus_used.get(GPUType.H100, 0) == 16
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s


def test_baseline_no_disaggregation() -> None:
    latency_data = load_latency_data("simulator/data/")

    # 8 A100 + 8 H100
    allocator = NaiveAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=NAIVE_POLICY,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 8, GPUType.H100: 8},
        verbose=True,
    )
    assert 0 < result.gpus_used.get(GPUType.A100, 0) <= 8
    assert 0 < result.gpus_used.get(GPUType.H100, 0) <= 8


def test_baseline_0() -> None:
    """No GPUs."""
    latency_data = load_latency_data("simulator/data/")
    with pytest.raises(AssertionError, match="Total number of GPUs must be at least 8"):
        allocator = NaiveAllocator(
            workflow=DEFAULT_WORKFLOW_CONFIG,
            latency_data=latency_data,
        )
        allocator.allocate(
            num_gpus={GPUType.A100: 0, GPUType.H100: 0},
        )


def test_baseline_timexcost_1024A() -> None:
    latency_data = load_latency_data("simulator/data/")
    policy = BASELINE_POLICIES["naive ttff*cost allocator"]
    allocator = AutoModelAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=policy,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 1024},
    )
    assert result.gpus_used[GPUType.A100] == 1024
    assert result.gpus_used.get(GPUType.H100, 0) == 0


@pytest.mark.parametrize("policy_name", BASELINE_POLICIES.keys())
def test_baseline_policies(
    policy_name: str
) -> None:
    latency_data = load_latency_data("simulator/data/")
    policy = BASELINE_POLICIES[policy_name]
    allocator = AutoModelAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=policy,
    )
    result = allocator.allocate(
        num_gpus={
            GPUType.A100: 64,
            GPUType.H100: 64
        },
    )
    assert 0 < result.gpus_used[GPUType.A100] <= 64
    assert 0 < result.gpus_used[GPUType.H100] <= 64


def test_baseline_streamwise() -> None:
    """StreamWise policy."""
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")

    policy_streamwise = STREAMWISE_POLICY
    assert policy_streamwise.objective.value == "ttff_cost"

    allocator = GreedyAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
        policy=policy_streamwise,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 8},
    )

    assert result.gpus_used[GPUType.A100] == 8
    assert GPUType.H100 not in result.gpus_used
    assert GPUType.H200 not in result.gpus_used
    assert GPUType.GB200 not in result.gpus_used

    assert len(result.models) == 1
    assert GPUType.A100 in result.models
    models = result.models[GPUType.A100]
    # 8 GPUs used
    assert models[Model.GEMMA][0].get_num_gpus() == 1
    assert models[Model.OTHERS][0].get_num_gpus() == 1
    assert models[Model.FLUX][0].get_num_gpus() == 1
    assert models[Model.HF][0].get_num_gpus() == 1
    assert models[Model.HF_VAE][0].get_num_gpus() == 1
    assert models[Model.FT][0].get_num_gpus() == 2
    assert models[Model.UPSCALER][0].get_num_gpus() == 1

    assert 4 * SECONDS_IN_HOUR < result.total_time_s < 5 * SECONDS_IN_HOUR  # 4-5 hours
    assert 4 * SECONDS_IN_HOUR < result.ttff_s < result.total_time_s  # 4-5 hours
    assert 1 < result.tbf_s < 2  # 1-2 seconds

    assert result.total_time_s * 8 * POWER_GPU_IDLE[GPUType.A100] < result.total_energy
    assert result.total_energy < result.total_time_s * 8 * POWER_GPU_TDP[GPUType.A100]

    assert 37 < result.cost < 38  # $37-38


def test_baseline_naive() -> None:
    """Naive policy."""
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")

    policy_naive = BASELINE_POLICIES["naive"]
    assert policy_naive.objective.value == "ttff"
    assert policy_naive.solver.value == "naive"

    allocator = NaiveAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
        policy=policy_naive,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 8},
    )

    assert result.gpus_used[GPUType.A100] == 8
    assert GPUType.H100 not in result.gpus_used
    assert GPUType.H200 not in result.gpus_used
    assert GPUType.GB200 not in result.gpus_used

    assert len(result.models) == 1
    assert GPUType.A100 in result.models
    models = result.models[GPUType.A100]
    # 8 GPUs used
    assert models[Model.GEMMA][0].get_num_gpus() == 1
    assert models[Model.OTHERS][0].get_num_gpus() == 1
    assert models[Model.FLUX][0].get_num_gpus() == 1
    assert models[Model.HF][0].get_num_gpus() == 1
    assert models[Model.HF_VAE][0].get_num_gpus() == 0  # No dissaggregation
    assert models[Model.FT][0].get_num_gpus() == 4
    assert models[Model.UPSCALER][0].get_num_gpus() == 0  # no upscaler

    # 8.3 hours from the paper
    assert 8 * SECONDS_IN_HOUR < result.total_time_s < 9 * SECONDS_IN_HOUR  # 8-9 hours
    assert 7 * SECONDS_IN_HOUR < result.ttff_s < result.total_time_s  # 7-8 hours
    assert 2 < result.tbf_s < 3  # 2-3 seconds

    assert result.total_time_s * 8 * POWER_GPU_IDLE[GPUType.A100] < result.total_energy
    assert result.total_energy < result.total_time_s * 8 * POWER_GPU_TDP[GPUType.A100]

    # With Spot is ~$71 and with Reserved is ~$225
    assert 225 < result.cost < 230  # $225-230


def test_baseline_upscaler() -> None:
    """Naive policy with upscaler."""
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")

    policy_naive_upscaler = BASELINE_POLICIES["naive upscaler"]
    assert policy_naive_upscaler.objective.value == "ttff"
    assert policy_naive_upscaler.solver.value == "naive"

    allocator = NaiveAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
        policy=policy_naive_upscaler,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 8},
    )

    assert result.gpus_used[GPUType.A100] == 8
    assert GPUType.H100 not in result.gpus_used
    assert GPUType.H200 not in result.gpus_used
    assert GPUType.GB200 not in result.gpus_used

    assert len(result.models) == 1
    assert GPUType.A100 in result.models
    models = result.models[GPUType.A100]
    # 8 GPUs used
    assert models[Model.GEMMA][0].get_num_gpus() == 1
    assert models[Model.OTHERS][0].get_num_gpus() == 1
    assert models[Model.FLUX][0].get_num_gpus() == 1
    assert models[Model.HF][0].get_num_gpus() == 1
    assert models[Model.HF_VAE][0].get_num_gpus() == 0  # No dissaggregation
    assert models[Model.FT][0].get_num_gpus() == 3
    assert models[Model.UPSCALER][0].get_num_gpus() == 1

    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s
    assert 3 * SECONDS_IN_HOUR < result.total_time_s < 4 * SECONDS_IN_HOUR  # 3-4 hours
    assert 3 * SECONDS_IN_HOUR < result.ttff_s < result.total_time_s  # 3-4 hours
    assert 0.5 < result.tbf_s < 1  # 0.5-1 seconds

    assert result.total_time_s * 8 * POWER_GPU_IDLE[GPUType.A100] < result.total_energy
    assert result.total_energy < result.total_time_s * 8 * POWER_GPU_TDP[GPUType.A100]

    assert 91 < result.cost < 92  # $91-92


def test_baseline_time_cost() -> None:
    """Naive policy with upscaler."""
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")

    policy_naive_sched = BASELINE_POLICIES["naive ttff*cost allocator"]
    assert policy_naive_sched.objective.value == "ttff_cost"
    assert policy_naive_sched.solver.value == "greedy"

    allocator = AutoModelAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
        policy=policy_naive_sched,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 8},
    )

    assert result.gpus_used[GPUType.A100] == 8
    assert GPUType.H100 not in result.gpus_used
    assert GPUType.H200 not in result.gpus_used
    assert GPUType.GB200 not in result.gpus_used

    assert len(result.models) == 1
    assert GPUType.A100 in result.models
    models = result.models[GPUType.A100]
    # 8 GPUs used
    assert models[Model.GEMMA][0].get_num_gpus() == 1
    assert models[Model.OTHERS][0].get_num_gpus() == 1
    assert models[Model.FLUX][0].get_num_gpus() == 1
    assert models[Model.HF][0].get_num_gpus() == 3
    assert models[Model.HF_VAE][0].get_num_gpus() == 0  # No dissaggregation
    assert models[Model.FT][0].get_num_gpus() == 2
    assert models[Model.UPSCALER][0].get_num_gpus() == 0

    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s
    # TODO figure if this is what is expected
    # assert 8 * SECONDS_IN_HOUR < result.total_time_s < 9 * SECONDS_IN_HOUR  # 8-9 hours
    assert 8 * SECONDS_IN_HOUR < result.total_time_s < 14 * SECONDS_IN_HOUR  # 8-14 hours
    assert 7 * SECONDS_IN_HOUR < result.ttff_s < result.total_time_s  # 7-8 hours
    assert 1 < result.tbf_s < 4  # 1-4 seconds
    assert result.total_time_s * 8 * POWER_GPU_IDLE[GPUType.A100] < result.total_energy
    assert result.total_energy < result.total_time_s * 8 * POWER_GPU_TDP[GPUType.A100]


def test_baseline_hardware_error() -> None:
    """Naive policy without naive parallelism."""
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")

    policy_naive_hardware = BASELINE_POLICIES["naive hardware"]
    assert policy_naive_hardware.objective.value == "ttff"
    assert policy_naive_hardware.solver.value == "naive"

    allocator = NaiveAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
        policy=policy_naive_hardware,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 24, GPUType.H200: 1280},
    )

    assert result.gpus_used[GPUType.A100] == 24
    assert result.gpus_used[GPUType.H200] == 1264
    assert GPUType.H100 not in result.gpus_used
    assert GPUType.GB200 not in result.gpus_used

    assert len(result.models) == 2
    assert GPUType.A100 in result.models
    assert GPUType.H200 in result.models
    models = result.models[GPUType.A100]
    assert models[Model.GEMMA][0].get_num_gpus() == 2
    assert models[Model.OTHERS][0].get_num_gpus() == 1
    assert models[Model.FLUX][0].get_num_gpus() == 2
    assert models[Model.HF][0].get_num_gpus() == 8
    assert models[Model.HF_VAE][0].get_num_gpus() == 0
    assert models[Model.FT][0].get_num_gpus() == 11
    assert models[Model.FT_VAE][0].get_num_gpus() == 0
    assert models[Model.UPSCALER][0].get_num_gpus() == 0

    assert 500 < result.total_time_s < 810
    assert 130 < result.ttff_s < result.total_time_s
    assert 0 < result.tbf_s < 1
    assert result.total_time_s * (1265 + 24) * POWER_GPU_IDLE[GPUType.A100] < result.total_energy
    assert result.total_energy < result.total_time_s * (1265 + 24) * POWER_GPU_TDP[GPUType.H200]


def test_workflow_short() -> None:
    """Test the shorts workflow with the naive policy."""
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")

    allocator = NaiveAllocator(
        workflow=SHORTS_WORKFLOW,
        latency_data=latency_data,
        power_data=power_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 56},
    )

    assert result.gpus_used[GPUType.A100] == 49
    assert GPUType.H100 not in result.gpus_used
    assert GPUType.H200 not in result.gpus_used
    assert GPUType.GB200 not in result.gpus_used

    assert len(result.models) == 1
    assert GPUType.A100 in result.models
    models = result.models[GPUType.A100]
    assert models[Model.GEMMA][0].get_num_gpus() == 48
    assert models[Model.OTHERS][0].get_num_gpus() == 1
    assert models[Model.FLUX][0].get_num_gpus() == 0
    assert models[Model.HF][0].get_num_gpus() == 0
    assert models[Model.HF_VAE][0].get_num_gpus() == 0
    assert models[Model.FT][0].get_num_gpus() == 0
    assert models[Model.UPSCALER][0].get_num_gpus() == 0

    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s
    assert result.total_time_s * 8 * POWER_GPU_IDLE[GPUType.A100] < result.total_energy
    # assert result.total_energy < result.total_time_s * 8 * POWER_GPU_TDP[GPUType.A100]


@pytest.mark.parametrize("workflow_name", WORKFLOWS.keys())
def test_workflows(workflow_name: str) -> None:
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")

    allocator = NaiveAllocator(
        workflow=WORKFLOWS[workflow_name],
        latency_data=latency_data,
        power_data=power_data,
    )
    result = allocator.allocate(
        num_gpus={GPUType.A100: 32},
    )
    assert GPUType.A100 in result.models
    assert 0 < result.tbf_s < result.ttff_s < result.total_time_s
