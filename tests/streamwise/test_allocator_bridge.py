"""
Tests for streamwise/allocator_bridge.py.

Covers:
- Model-to-container name mapping.
- Result to deployment specs conversion.
- run_allocator end-to-end (with real latency data).
- Error handling for invalid inputs.
"""

from __future__ import annotations

import sys
import os

import pytest

# Add current path and simulator/ permanently so lazy imports
# (e.g. GreedyAllocator via auto_model_allocator) resolve at test time.
sys.path.append(os.getcwd())
sys.path[:0] = [os.path.join(os.getcwd(), "simulator")]

from tests.test_utils import temp_sys_path

with temp_sys_path("streamwise", "simulator"):
    from allocator_bridge import (
        MODEL_TO_CONTAINERS,
        CONTAINER_RESOURCES,
        GPU_TYPE_TO_POD_STR,
        APP_TO_WORKFLOW,
        DeploymentSpec,
        DeploymentPlan,
        get_available_workflows,
        get_available_gpu_types,
        result_to_deployment_specs,
        deployment_plan_to_json,
        run_allocator,
    )
    from sim_types import GPUType, Model, Result
    from models import (
        GemmaModelAllocation,
        FluxModelAllocation,
        HFModelAllocation,
        HFVAEModelAllocation,
        FTModelAllocation,
        OthersModelAllocation,
        UpscalerModelAllocation,
    )


# ---------------------------------------------------------------------------
# Mapping correctness
# ---------------------------------------------------------------------------

def test_model_to_containers_covers_all_models() -> None:
    """Every Model enum value must have a mapping entry."""
    for model in Model:
        assert model in MODEL_TO_CONTAINERS, f"Missing mapping for {model}"


def test_container_resources_covers_all_mapped_containers() -> None:
    """Every container referenced in MODEL_TO_CONTAINERS must have resource defaults."""
    for model, containers in MODEL_TO_CONTAINERS.items():
        for container in containers:
            assert container in CONTAINER_RESOURCES, (
                f"Missing CONTAINER_RESOURCES for '{container}' (from {model})")


def test_gpu_type_to_pod_str_covers_all_gpu_types() -> None:
    """Every GPUType enum value must have a pod string mapping."""
    for gpu_type in GPUType:
        assert gpu_type in GPU_TYPE_TO_POD_STR


def test_app_to_workflow_has_expected_entries() -> None:
    """Key StreamWise apps should map to workflows."""
    assert "streamcast" in APP_TO_WORKFLOW
    assert "streampersona" in APP_TO_WORKFLOW
    assert "streamchat" in APP_TO_WORKFLOW


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def test_get_available_workflows() -> None:
    workflows = get_available_workflows()
    assert isinstance(workflows, list)
    assert "streamcast" in workflows
    assert len(workflows) >= 5


def test_get_available_gpu_types() -> None:
    gpu_types = get_available_gpu_types()
    assert isinstance(gpu_types, list)
    assert "A100" in gpu_types
    assert "H100" in gpu_types


# ---------------------------------------------------------------------------
# result_to_deployment_specs
# ---------------------------------------------------------------------------

def test_result_to_deployment_specs_basic() -> None:
    """A simple result with one active allocation maps to the right container."""
    models = {
        GPUType.A100: {
            Model.GEMMA: [GemmaModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=1)],
            Model.FLUX: [FluxModelAllocation(gpu_type=GPUType.A100, devices=2, replicas=1)],
            Model.HF: [HFModelAllocation(gpu_type=GPUType.A100, devices=2, replicas=2)],
            Model.HF_VAE: [HFVAEModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=1)],
            Model.FT: [FTModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=0)],
            Model.FT_VAE: [],
            Model.UPSCALER: [UpscalerModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=0)],
            Model.OTHERS: [OthersModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=1)],
        }
    }
    result = Result(
        total_time_s=100.0,
        ttff_s=10.0,
        cost=1.0,
        gpus_used={GPUType.A100: 8},
        gpus_total={GPUType.A100: 8},
        models=models,
    )

    specs = result_to_deployment_specs(result)
    assert isinstance(specs, list)
    assert len(specs) > 0

    container_names = [s.container_name for s in specs]
    assert "gemma" in container_names
    assert "flux" in container_names
    assert "hunyuanframepackf1" in container_names  # HF model
    assert "hunyuanframepackvae" in container_names  # HF_VAE model

    # OTHERS maps to kokoro + yolo
    assert "kokoro" in container_names
    assert "yolo" in container_names

    # Check GPU type mapping
    gemma_spec = next(s for s in specs if s.container_name == "gemma")
    assert gemma_spec.gpu_type == "a100"
    assert gemma_spec.gpu == 1

    # MIG containers get mig_profile set
    kokoro_spec = next(s for s in specs if s.container_name == "kokoro")
    assert kokoro_spec.mig_profile == "1g.10gb"


def test_result_to_deployment_specs_skips_zero_replicas() -> None:
    """Allocations with zero replicas should not produce deployment specs."""
    models = {
        GPUType.A100: {
            Model.GEMMA: [GemmaModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=0)],
            Model.FLUX: [FluxModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=0)],
            Model.HF: [HFModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=0)],
            Model.HF_VAE: [],
            Model.FT: [],
            Model.FT_VAE: [],
            Model.UPSCALER: [],
            Model.OTHERS: [],
        }
    }
    result = Result(
        total_time_s=0.0,
        ttff_s=0.0,
        cost=0.0,
        gpus_used={GPUType.A100: 0},
        gpus_total={GPUType.A100: 8},
        models=models,
    )
    specs = result_to_deployment_specs(result)
    assert specs == []


def test_result_to_deployment_specs_multiple_replicas() -> None:
    """Multiple replicas should produce multiple deployment specs for same container."""
    models = {
        GPUType.H100: {
            Model.GEMMA: [GemmaModelAllocation(gpu_type=GPUType.H100, devices=1, replicas=1)],
            Model.FLUX: [FluxModelAllocation(gpu_type=GPUType.H100, devices=1, replicas=1)],
            Model.HF: [HFModelAllocation(gpu_type=GPUType.H100, devices=2, replicas=3)],
            Model.HF_VAE: [],
            Model.FT: [],
            Model.FT_VAE: [],
            Model.UPSCALER: [],
            Model.OTHERS: [],
        }
    }
    result = Result(
        total_time_s=50.0,
        ttff_s=5.0,
        cost=0.5,
        gpus_used={GPUType.H100: 8},
        gpus_total={GPUType.H100: 16},
        models=models,
    )
    specs = result_to_deployment_specs(result)
    hf_specs = [s for s in specs if s.container_name == "hunyuanframepackf1"]
    assert len(hf_specs) == 3  # 3 replicas
    for spec in hf_specs:
        assert spec.gpu == 2
        assert spec.gpu_type == "h100"


# ---------------------------------------------------------------------------
# deployment_plan_to_json
# ---------------------------------------------------------------------------

def test_deployment_plan_to_json() -> None:
    """Serialization should produce all expected keys."""
    result = Result(
        total_time_s=100.0,
        ttff_s=10.0,
        cost=1.5,
        gpus_used={GPUType.A100: 8},
        gpus_total={GPUType.A100: 8},
        models={},
    )
    plan = DeploymentPlan(
        specs=[
            DeploymentSpec(
                container_name="gemma", cpu=16, memory_gib=192,
                ephemeral_storage_gib=64, gpu=2, gpu_type="a100", mig_profile=None)
        ],
        result=result,
        workflow_name="streamcast",
        gpu_budget={"A100": 8},
    )
    data = deployment_plan_to_json(plan)
    assert data["workflow_name"] == "streamcast"
    assert data["gpu_budget"] == {"A100": 8}
    assert data["metrics"]["total_time_s"] == 100.0
    assert data["metrics"]["ttff_s"] == 10.0
    assert len(data["specs"]) == 1
    assert data["specs"][0]["container_name"] == "gemma"


# ---------------------------------------------------------------------------
# run_allocator (integration with real data)
# ---------------------------------------------------------------------------

def test_run_allocator_streamcast_8_a100() -> None:
    """Run allocator for StreamCast with 8 A100s — should produce a valid plan."""
    plan = run_allocator(
        gpu_budget={"A100": 8},
        workflow_name="streamcast",
    )
    assert isinstance(plan, DeploymentPlan)
    assert len(plan.specs) > 0
    assert plan.result.total_time_s > 0
    assert plan.result.ttff_s > 0
    assert plan.workflow_name == "streamcast"


def test_run_allocator_streamchat_8_h100() -> None:
    """Run allocator for StreamChat with 8 H100s."""
    plan = run_allocator(
        gpu_budget={"H100": 8},
        workflow_name="streamchat",
    )
    assert isinstance(plan, DeploymentPlan)
    assert len(plan.specs) > 0


def test_run_allocator_invalid_workflow() -> None:
    """Unknown workflow name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown workflow"):
        run_allocator(gpu_budget={"A100": 8}, workflow_name="nonexistent")


def test_run_allocator_invalid_gpu_type() -> None:
    """Unknown GPU type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown GPU type"):
        run_allocator(gpu_budget={"RTX4090": 8}, workflow_name="streamcast")


def test_run_allocator_insufficient_gpus() -> None:
    """Too few GPUs raises ValueError."""
    with pytest.raises(ValueError, match="at least 8"):
        run_allocator(gpu_budget={"A100": 4}, workflow_name="streamcast")
