import sys
import os
import pytest

# Add current path
sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path

with temp_sys_path("simulator", "streamwise"):
    from constants import DEFAULT_WORKFLOW_CONFIG

    from provisioning import get_provisioning_results
    from provisioning import get_provisioning_adaptive_results
    from provisioning import Provision
    from provisioning import get_provisions
    from provisioning import GPU_PROVISIONS
    from provisioning import GPU_PROVISIONS_SHORT

    from sim_types import GPUType
    from sim_types import QualityLevel
    from sim_types import Solver

    from data_loading import load_latency_data

    from model_provisioner.policies import NAIVE_POLICY
    from model_provisioner.policies import STREAMWISE_POLICY
    from model_provisioner.policies import HEXGEN_POLICY


@pytest.mark.parametrize("gpu_type", [gpu_type for gpu_type in GPUType])
@pytest.mark.parametrize("num_gpus", [8, 16, 32, 64])
def test_provisioning_streamwise_single_gpu_type(
    gpu_type: GPUType,
    num_gpus: int,
) -> None:
    latency_data = load_latency_data("simulator/data/")
    provisions = [
        Provision({gpu_type: num_gpus}),
    ]
    result = get_provisioning_results(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=STREAMWISE_POLICY,
        provisions=provisions,
    )
    assert len(result.latencies) == 1
    assert len(result.costs) == 1
    assert len(result.ttffs) == 1
    assert len(result.tbfs) == 1
    assert len(result.actual_provision) == 1
    assert len(result.config_provision) == 1

    assert result.config_provision == [
        {gpu_type: num_gpus},
    ]
    assert result.actual_provision == [
        {gpu_type: num_gpus},
    ]


def test_provisioning() -> None:
    latency_data = load_latency_data("simulator/data/")
    result = get_provisioning_results(
        provisions=[
            # A100, H100, H200
            # This now gets ordered differently
            Provision({GPUType.H100: 16}),
            Provision({GPUType.A100: 32}),
            Provision({GPUType.A100: 64, GPUType.H200: 128}),
            Provision({GPUType.A100: 64, GPUType.H100: 64}),
        ],
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=STREAMWISE_POLICY,
        verbose=False,
    )
    assert len(result.latencies) == 4
    assert len(result.costs) == 4
    assert len(result.ttffs) == 4
    assert len(result.tbfs) == 4
    assert len(result.actual_provision) == 4
    assert len(result.config_provision) == 4

    assert result.config_provision == [
        {GPUType.H100: 16},
        {GPUType.A100: 32},
        {GPUType.A100: 64, GPUType.H200: 128},
        {GPUType.A100: 64, GPUType.H100: 64},
    ]
    assert result.actual_provision[0] == {GPUType.H100: 16}
    assert result.actual_provision[1] == {GPUType.A100: 32}
    assert result.actual_provision[2] == {GPUType.A100: 64, GPUType.H200: 128}
    assert result.actual_provision[3] == {GPUType.A100: 64, GPUType.H100: 64}


def test_provisioning_not_gpus_error() -> None:
    latency_data = load_latency_data("simulator/data/")
    with pytest.raises(AssertionError, match="No GPUs provisioned"):
        get_provisioning_results(
            workflow=DEFAULT_WORKFLOW_CONFIG,
            latency_data=latency_data,
            policy=NAIVE_POLICY,
            provisions=[
                Provision({  # No GPU provision
                    GPUType.A100: 0,
                    GPUType.H100: 0,
                    GPUType.H200: 0
                }),
            ],
        )


def test_provisioning_3_gpus_error() -> None:
    latency_data = load_latency_data("simulator/data/")
    with pytest.raises(AssertionError, match="Only support up to 2 GPU types in a provision"):
        get_provisioning_results(
            workflow=DEFAULT_WORKFLOW_CONFIG,
            latency_data=latency_data,
            policy=NAIVE_POLICY,
            provisions=[
                Provision({  # All not supported
                    GPUType.A100: 8,
                    GPUType.H100: 8,
                    GPUType.H200: 8,
                }),
            ],
        )


def test_provisioning_adaptive_too_small() -> None:
    latency_data = load_latency_data("simulator/data/")
    provisions = [
        Provision({GPUType.A100: 8}),
    ]
    provisioning_result_high = get_provisioning_results(
        provisions=provisions,
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=STREAMWISE_POLICY,
    )
    provisioning_result_medium = get_provisioning_results(
        provisions=provisions,
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=STREAMWISE_POLICY,
    )
    provisioning_result_low = get_provisioning_results(
        provisions=provisions,
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=STREAMWISE_POLICY,
    )

    with pytest.raises(ValueError, match="Low quality TTFF .+ exceeds video length .+"):
        get_provisioning_adaptive_results(
            workflow_config=DEFAULT_WORKFLOW_CONFIG,
            provisioning_qualities={
                QualityLevel.HIGH: provisioning_result_high,
                QualityLevel.MEDIUM: provisioning_result_medium,
                QualityLevel.LOW: provisioning_result_low,
            },
        )


def test_provisioning_adaptive() -> None:
    latency_data = load_latency_data("simulator/data/")
    provisions = [
        Provision({GPUType.A100: 128}),
        Provision({GPUType.H100: 128}),
        Provision({GPUType.A100: 24, GPUType.H100: 32}),
        Provision({GPUType.A100: 128, GPUType.H100: 32}),  # TODO This should work with fewer H100
    ]
    provisioning_result_high = get_provisioning_results(
        provisions=provisions,
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=STREAMWISE_POLICY,
    )
    provisioning_result_medium = get_provisioning_results(
        provisions=provisions,
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=STREAMWISE_POLICY,
    )
    provisioning_result_low = get_provisioning_results(
        provisions=provisions,
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=STREAMWISE_POLICY,
    )
    adaptive_results = get_provisioning_adaptive_results(
        workflow_config=DEFAULT_WORKFLOW_CONFIG,
        provisioning_qualities={
            QualityLevel.HIGH: provisioning_result_high,
            QualityLevel.MEDIUM: provisioning_result_medium,
            QualityLevel.LOW: provisioning_result_low,
        },
    )
    assert adaptive_results is not None
    assert len(adaptive_results.costs) == len(provisions)
    assert len(adaptive_results.latencies) == len(provisions)
    assert len(adaptive_results.ttffs) == len(provisions)
    assert len(adaptive_results.tbfs) == len(provisions)
    assert len(adaptive_results.actual_provision) == len(provisions)
    assert len(adaptive_results.config_provision) == len(provisions)


def test_get_provisions() -> None:
    """Test get_provisions() function for various GPU type combinations."""
    provisions_a100 = get_provisions([GPUType.A100])
    assert 10 < len(provisions_a100) == len(GPU_PROVISIONS) - 1
    for provision in provisions_a100:
        assert provision.num_gpus.get(GPUType.A100, 0) > 0
        assert provision.num_gpus.get(GPUType.H100, 0) == 0
        assert provision.num_gpus.get(GPUType.H200, 0) == 0

    provisions_h100 = get_provisions([GPUType.H100])
    assert 10 < len(provisions_h100) == len(GPU_PROVISIONS) - 1
    for provision in provisions_h100:
        assert provision.num_gpus.get(GPUType.A100, 0) == 0
        assert provision.num_gpus.get(GPUType.H100, 0) > 0
        assert provision.num_gpus.get(GPUType.H200, 0) == 0

    provisions_h200 = get_provisions([GPUType.H200])
    assert 10 < len(provisions_h200) == len(GPU_PROVISIONS) - 1
    for provision in provisions_h200:
        assert provision.num_gpus.get(GPUType.A100, 0) == 0
        assert provision.num_gpus.get(GPUType.H100, 0) == 0
        assert provision.num_gpus.get(GPUType.H200, 0) > 0

    # Mixed mode (A100 + H100)
    provisions_mixed = get_provisions([GPUType.A100, GPUType.H100])
    assert len(provisions_mixed) == len(GPU_PROVISIONS) ** 2 - 1  # num_gpus ^ num_types - 1 (empty)
    for provision in provisions_mixed:
        assert (
            provision.num_gpus.get(GPUType.A100, 0) > 0
            or provision.num_gpus.get(GPUType.H100, 0) > 0
        )

    # Mixed mode (all types)
    provisions_mixed = get_provisions([
        GPUType.A100,
        GPUType.H100,
        GPUType.H200,
    ], limits_pairs=False)
    assert len(provisions_mixed) == len(GPU_PROVISIONS) ** 3 - 1  # num_gpus ^ num_types - 1 (empty)
    for provision in provisions_mixed:
        assert (
            provision.num_gpus.get(GPUType.A100, 0) > 0
            or provision.num_gpus.get(GPUType.H100, 0) > 0
            or provision.num_gpus.get(GPUType.H200, 0) > 0
        )
    # Make sure specific cases exist
    assert any(
        provision.num_gpus.get(GPUType.A100, 0) == 8
        and provision.num_gpus.get(GPUType.H100, 0) == 8
        and provision.num_gpus.get(GPUType.H200, 0) == 0
        for provision in provisions_mixed
    )
    assert any(
        provision.num_gpus.get(GPUType.A100, 0) == 8
        and provision.num_gpus.get(GPUType.H100, 0) == 0
        and provision.num_gpus.get(GPUType.H200, 0) == 0
        for provision in provisions_mixed
    )
    assert any(
        provision.num_gpus.get(GPUType.A100, 0) == 8
        and provision.num_gpus.get(GPUType.H100, 0) == 8
        and provision.num_gpus.get(GPUType.H200, 0) == 8
        for provision in provisions_mixed
    )


def test_get_provisions_short() -> None:
    """Test get_provisions() function for various GPU type combinations."""
    provisions_a100 = get_provisions([GPUType.A100], short_list=True)
    assert 10 < len(provisions_a100) == len(GPU_PROVISIONS_SHORT) - 1
    for provision in provisions_a100:
        assert provision.num_gpus.get(GPUType.A100, 0) > 0
        assert provision.num_gpus.get(GPUType.H100, 0) == 0
        assert provision.num_gpus.get(GPUType.H200, 0) == 0

    provisions_mixed = get_provisions([
        GPUType.A100,
        GPUType.H100,
        GPUType.H200,
    ], limits_pairs=False, short_list=True)
    assert len(provisions_mixed) == len(GPU_PROVISIONS_SHORT) ** 3 - 1  # num_gpus ^ num_types - 1 (empty)
    for provision in provisions_mixed:
        assert (
            provision.num_gpus.get(GPUType.A100, 0) > 0
            or provision.num_gpus.get(GPUType.H100, 0) > 0
            or provision.num_gpus.get(GPUType.H200, 0) > 0
        )


# --- HexGenAllocator provisioning tests ---
@pytest.mark.parametrize("gpu_type", [GPUType.A100, GPUType.H100])
@pytest.mark.parametrize("num_gpus", [8, 16, 32])
def test_provisioning_hexgen_single_gpu_type(
    gpu_type: GPUType,
    num_gpus: int,
) -> None:
    """Test provisioning with HexGenAllocator for single GPU types."""
    latency_data = load_latency_data("simulator/data/")
    provisions = [
        Provision({gpu_type: num_gpus}),
    ]
    result = get_provisioning_results(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=HEXGEN_POLICY,
        provisions=provisions,
    )
    assert len(result.latencies) == 1
    assert len(result.costs) == 1
    assert len(result.ttffs) == 1
    assert len(result.tbfs) == 1
    assert len(result.actual_provision) == 1
    assert len(result.config_provision) == 1

    assert result.config_provision == [
        {gpu_type: num_gpus},
    ]
    assert result.actual_provision == [
        {gpu_type: num_gpus},
    ]


def test_hexgen_mixed() -> None:
    """Test provisioning with HexGenAllocator for mixed GPU types."""
    latency_data = load_latency_data("simulator/data/")
    result = get_provisioning_results(
        provisions=[
            Provision({GPUType.A100: 8, GPUType.H100: 8}),
            Provision({GPUType.A100: 32}),
        ],
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=HEXGEN_POLICY,
        verbose=False,
    )
    assert len(result.latencies) == 2
    assert len(result.costs) == 2
    assert len(result.ttffs) == 2
    assert len(result.tbfs) == 2
    assert len(result.actual_provision) == 2
    assert len(result.config_provision) == 2


def test_default_allocator_is_greedy() -> None:
    """Verify default policy allocator is greedy (backward compat)."""
    latency_data = load_latency_data("simulator/data/")
    provisions = [
        Provision({GPUType.A100: 8}),
    ]
    # STREAMWISE_POLICY uses default allocator="greedy"
    result = get_provisioning_results(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=STREAMWISE_POLICY,
        provisions=provisions,
    )
    assert STREAMWISE_POLICY.solver == Solver.GREEDY
    assert len(result.latencies) == 1


def test_hexgen_vs_greedy() -> None:
    """Both allocators should produce valid results for the same provision."""
    latency_data = load_latency_data("simulator/data/")
    provisions = [
        Provision({GPUType.A100: 32}),
    ]
    result_greedy = get_provisioning_results(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=STREAMWISE_POLICY,
        provisions=provisions,
    )
    result_hexgen = get_provisioning_results(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=HEXGEN_POLICY,
        provisions=provisions,
    )
    # Both should produce 1 result
    assert len(result_greedy.latencies) == 1
    assert len(result_hexgen.latencies) == 1
    # Both should have positive metrics
    assert result_greedy.latencies[0] > 0
    assert result_hexgen.latencies[0] > 0
    assert result_greedy.costs[0] > 0
    assert result_hexgen.costs[0] > 0
    assert result_greedy.ttffs[0] > 0
    assert result_hexgen.ttffs[0] > 0
