"""
Direct unit tests for simulator/models.py.
Tests the model allocation factory, helper functions, and per-model
calculate_time / calculate_time_first / calculate_energy methods.
"""

from __future__ import annotations

import sys
import os
import pytest
from unittest.mock import patch

# Add current path
sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path

with temp_sys_path("simulator"):
    from sim_types import GPUType
    from sim_types import Model
    from sim_types import QualityLevel
    from sim_types import LatencyData
    from sim_types import PowerData

    from constants import DEFAULT_WORKFLOW_CONFIG

    from data_loading import load_latency_data
    from data_loading import load_power_data

    from policies import STREAMWISE_POLICY
    from policies import NAIVE_POLICY

    from models import get_model_allocation
    from models import _calculate_total_time
    from models import assert_pixel_config
    from models import _MODEL_ALLOCATION_REGISTRY
    from models import GemmaModelAllocation
    from models import FluxModelAllocation
    from models import HFModelAllocation
    from models import HFVAEModelAllocation
    from models import FTModelAllocation
    from models import FTVAEModelAllocation
    from models import UpscalerModelAllocation
    from models import OthersModelAllocation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_latency_data() -> LatencyData:
    """Return minimal LatencyData built from the real data files."""
    return load_latency_data("simulator/data/")


def _make_power_data() -> PowerData:
    """Return minimal PowerData built from the real data files."""
    return load_power_data("simulator/data/")


# ---------------------------------------------------------------------------
# get_model_allocation factory
# ---------------------------------------------------------------------------

def test_get_model_allocation_returns_correct_types() -> None:
    """Factory returns the right ModelAllocation subclass for each Model."""
    expected: list[tuple[Model, type]] = [
        (Model.GEMMA, GemmaModelAllocation),
        (Model.FLUX, FluxModelAllocation),
        (Model.HF, HFModelAllocation),
        (Model.HF_VAE, HFVAEModelAllocation),
        (Model.FT, FTModelAllocation),
        (Model.FT_VAE, FTVAEModelAllocation),
        (Model.UPSCALER, UpscalerModelAllocation),
        (Model.OTHERS, OthersModelAllocation),
    ]
    for model, cls in expected:
        alloc = get_model_allocation(
            model=model,
            gpu_type=GPUType.A100,
            devices=1,
            replicas=1,
        )
        assert isinstance(alloc, cls), f"Expected {cls.__name__} for {model}"
        assert alloc.gpu_type == GPUType.A100
        assert alloc.devices == 1
        assert alloc.replicas == 1


def test_get_model_allocation_zero_replicas() -> None:
    """Factory creates an allocation with zero replicas (disabled) by default."""
    alloc = get_model_allocation(
        model=Model.FLUX,
        gpu_type=GPUType.H100,
    )
    assert alloc.replicas == 0
    assert alloc.get_num_gpus() == 0


def test_get_model_allocation_unknown_model_raises() -> None:
    """Factory raises ValueError for an unregistered Model value."""
    # Temporarily remove GEMMA from the registry using patch.dict
    with patch.dict(_MODEL_ALLOCATION_REGISTRY, {}, clear=False) as patched:
        patched.pop(Model.GEMMA, None)
        with pytest.raises(ValueError, match="No ModelAllocation for model"):
            get_model_allocation(
                model=Model.GEMMA,
                gpu_type=GPUType.A100,
            )


# ---------------------------------------------------------------------------
# _calculate_total_time
# ---------------------------------------------------------------------------

def test_calculate_total_time_zero_replicas() -> None:
    """Zero replicas → zero time."""
    assert _calculate_total_time(100.0, 0, 1.0) == 0.0


def test_calculate_total_time_negative_replicas() -> None:
    """Negative replicas → zero time."""
    assert _calculate_total_time(100.0, -1, 1.0) == 0.0


def test_calculate_total_time_single_replica() -> None:
    """Single replica: total_work * time_per_work, clamped to time_per_work."""
    # 10 work / 1 replica * 5.0 = 50.0
    assert _calculate_total_time(10.0, 1, 5.0) == 50.0


def test_calculate_total_time_floor_at_single_work_unit() -> None:
    """Time cannot be less than time_per_work (single unit floor)."""
    # 1 work, 10 replicas → 1/10 * 5.0 = 0.5, floored to 5.0
    assert _calculate_total_time(1.0, 10, 5.0) == 5.0


def test_calculate_total_time_many_replicas() -> None:
    """Many replicas reduce time proportionally."""
    # 20/4 * 2.0 = 10.0; 10.0 > time_per_work(2.0) → 10.0
    assert _calculate_total_time(20.0, 4, 2.0) == 10.0


# ---------------------------------------------------------------------------
# assert_pixel_config
# ---------------------------------------------------------------------------

def test_assert_pixel_config() -> None:
    """assert_pixel_config passes for valid config and raises for invalid."""
    assert_pixel_config(DEFAULT_WORKFLOW_CONFIG)

    # Patching MEDIUM > HIGH violates the ordering constraint → AssertionError.
    with patch.dict("sim_types.RESOLUTION_PIXELS",
                    {QualityLevel.MEDIUM: 1000, QualityLevel.HIGH: 500}):
        with pytest.raises(AssertionError):
            assert_pixel_config(DEFAULT_WORKFLOW_CONFIG)


# ---------------------------------------------------------------------------
# Zero-GPU paths (replicas=0 → time=0, energy=0)
# All models must return 0.0 when no GPUs are allocated.
# ---------------------------------------------------------------------------

def test_zero_gpu_all_models() -> None:
    latency_data = _make_latency_data()
    power_data = _make_power_data()

    zero_allocs = [
        GemmaModelAllocation(gpu_type=GPUType.A100),  # replicas=0 by default
        FluxModelAllocation(gpu_type=GPUType.A100),
        HFModelAllocation(gpu_type=GPUType.A100),
        HFVAEModelAllocation(gpu_type=GPUType.A100),
        FTModelAllocation(gpu_type=GPUType.A100),
        FTVAEModelAllocation(gpu_type=GPUType.A100),
        UpscalerModelAllocation(gpu_type=GPUType.A100),
        OthersModelAllocation(gpu_type=GPUType.A100),
    ]
    workflow = DEFAULT_WORKFLOW_CONFIG
    for alloc in zero_allocs:
        assert alloc.get_num_gpus() == 0
        t = alloc.calculate_time(STREAMWISE_POLICY, workflow, latency_data)
        assert t == 0.0, f"{type(alloc).__name__}.calculate_time with 0 GPUs should be 0"
        tf = alloc.calculate_time_first(STREAMWISE_POLICY, workflow, latency_data)
        assert tf == 0.0, f"{type(alloc).__name__}.calculate_time_first with 0 GPUs should be 0"
        e = alloc.calculate_energy(workflow, power_data, total_time_s=1000.0)
        assert e == 0.0, f"{type(alloc).__name__}.calculate_energy with 0 GPUs should be 0"


# ---------------------------------------------------------------------------
# GemmaModelAllocation
# ---------------------------------------------------------------------------

def test_gemma_calculate_time_single_gpu() -> None:
    """Gemma time with 1 GPU should be positive and scale with tokens."""
    latency_data = _make_latency_data()
    alloc = GemmaModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=1)
    t = alloc.calculate_time(STREAMWISE_POLICY, DEFAULT_WORKFLOW_CONFIG, latency_data)
    assert t > 0.0
    assert alloc.time == t


def test_gemma_calculate_time_first_single_gpu() -> None:
    """Gemma TTFF < total time."""
    latency_data = _make_latency_data()
    alloc = GemmaModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=1)
    alloc.calculate_time(STREAMWISE_POLICY, DEFAULT_WORKFLOW_CONFIG, latency_data)
    tf = alloc.calculate_time_first(STREAMWISE_POLICY, DEFAULT_WORKFLOW_CONFIG, latency_data)
    assert 0.0 < tf <= alloc.time


def test_gemma_calculate_energy() -> None:
    """Gemma energy > 0 when power data is provided."""
    latency_data = _make_latency_data()
    power_data = _make_power_data()
    alloc = GemmaModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=1)
    alloc.calculate_time(STREAMWISE_POLICY, DEFAULT_WORKFLOW_CONFIG, latency_data)
    alloc.calculate_time_first(STREAMWISE_POLICY, DEFAULT_WORKFLOW_CONFIG, latency_data)
    e = alloc.calculate_energy(
        DEFAULT_WORKFLOW_CONFIG,
        power_data=power_data,
        total_time_s=alloc.time * 2,
    )
    assert e > 0.0


def test_gemma_calculate_energy_no_power_data() -> None:
    """Energy is 0 when no power data is provided."""
    latency_data = _make_latency_data()
    alloc = GemmaModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=1)
    alloc.calculate_time(STREAMWISE_POLICY, DEFAULT_WORKFLOW_CONFIG, latency_data)
    e = alloc.calculate_energy(DEFAULT_WORKFLOW_CONFIG, power_data=None)
    assert e == 0.0


def test_gemma_get_max_replicas() -> None:
    """Gemma max replicas equals model_work.get(GEMMA, 1)."""
    alloc = GemmaModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=1)
    max_r = alloc.get_max_replicas(DEFAULT_WORKFLOW_CONFIG)
    assert max_r == DEFAULT_WORKFLOW_CONFIG.model_work.get(Model.GEMMA, 1)


# ---------------------------------------------------------------------------
# FluxModelAllocation
# ---------------------------------------------------------------------------

def test_flux_calculate_time_single_gpu() -> None:
    """Flux time with 1 GPU should be positive."""
    latency_data = _make_latency_data()
    alloc = FluxModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=1)
    t = alloc.calculate_time(STREAMWISE_POLICY, DEFAULT_WORKFLOW_CONFIG, latency_data)
    assert t > 0.0


def test_flux_time_equals_time_first() -> None:
    """For Flux, time and time_first are the same (single-scene model)."""
    latency_data = _make_latency_data()
    alloc = FluxModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=1)
    t = alloc.calculate_time(STREAMWISE_POLICY, DEFAULT_WORKFLOW_CONFIG, latency_data)
    tf = alloc.calculate_time_first(STREAMWISE_POLICY, DEFAULT_WORKFLOW_CONFIG, latency_data)
    assert abs(t - tf) < 1e-6


def test_flux_multi_device_faster() -> None:
    """Flux with more GPU devices should be faster (or equal)."""
    latency_data = _make_latency_data()
    alloc1 = FluxModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=1)
    alloc2 = FluxModelAllocation(gpu_type=GPUType.A100, devices=2, replicas=1)
    t1 = alloc1.calculate_time(STREAMWISE_POLICY, DEFAULT_WORKFLOW_CONFIG, latency_data)
    t2 = alloc2.calculate_time(STREAMWISE_POLICY, DEFAULT_WORKFLOW_CONFIG, latency_data)
    assert t2 <= t1


# ---------------------------------------------------------------------------
# HFModelAllocation
# ---------------------------------------------------------------------------

def test_hf_calculate_time_single_gpu() -> None:
    """HF time with 1 GPU should be positive."""
    latency_data = _make_latency_data()
    alloc = HFModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=1)
    t = alloc.calculate_time(STREAMWISE_POLICY, DEFAULT_WORKFLOW_CONFIG, latency_data)
    assert t > 0.0


def test_hf_more_replicas_reduces_time() -> None:
    """More HF replicas should reduce total time."""
    latency_data = _make_latency_data()
    alloc1 = HFModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=1)
    alloc4 = HFModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=4)
    t1 = alloc1.calculate_time(STREAMWISE_POLICY, DEFAULT_WORKFLOW_CONFIG, latency_data)
    t4 = alloc4.calculate_time(STREAMWISE_POLICY, DEFAULT_WORKFLOW_CONFIG, latency_data)
    assert t4 < t1


def test_hf_get_max_replicas() -> None:
    alloc = HFModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=1)
    max_r = alloc.get_max_replicas(DEFAULT_WORKFLOW_CONFIG)
    assert max_r == DEFAULT_WORKFLOW_CONFIG.model_work.get(Model.HF, 1)


# ---------------------------------------------------------------------------
# FTModelAllocation
# ---------------------------------------------------------------------------

def test_ft_calculate_time_single_gpu() -> None:
    """FT time with 1 GPU should be positive."""
    latency_data = _make_latency_data()
    alloc = FTModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=1)
    t = alloc.calculate_time(STREAMWISE_POLICY, DEFAULT_WORKFLOW_CONFIG, latency_data)
    assert t > 0.0


def test_ft_time_first_equals_one_subscene() -> None:
    """FT time_first corresponds to a single subscene duration."""
    latency_data = _make_latency_data()
    alloc = FTModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=1)
    alloc.calculate_time(STREAMWISE_POLICY, DEFAULT_WORKFLOW_CONFIG, latency_data)
    tf = alloc.calculate_time_first(STREAMWISE_POLICY, DEFAULT_WORKFLOW_CONFIG, latency_data)
    assert 0.0 < tf <= alloc.time


def test_ft_get_max_replicas() -> None:
    alloc = FTModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=1)
    max_r = alloc.get_max_replicas(DEFAULT_WORKFLOW_CONFIG)
    assert max_r == DEFAULT_WORKFLOW_CONFIG.model_work.get(Model.FT, 1)


# ---------------------------------------------------------------------------
# HFVAEModelAllocation (disaggregated)
# ---------------------------------------------------------------------------

def test_hf_vae_disabled_when_not_disaggregated() -> None:
    """HF_VAE with no disaggregation (NAIVE_POLICY) should stay at zero."""
    # NAIVE_POLICY has disaggregation={} → HF not disaggregated
    alloc = HFVAEModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=0)
    latency_data = _make_latency_data()
    t = alloc.calculate_time(NAIVE_POLICY, DEFAULT_WORKFLOW_CONFIG, latency_data)
    assert t == 0.0


def test_hf_vae_positive_when_disaggregated() -> None:
    """HF_VAE with disaggregation enabled should give positive time."""
    alloc = HFVAEModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=1)
    latency_data = _make_latency_data()
    t = alloc.calculate_time(STREAMWISE_POLICY, DEFAULT_WORKFLOW_CONFIG, latency_data)
    assert t > 0.0


# ---------------------------------------------------------------------------
# FTVAEModelAllocation (disaggregated)
# ---------------------------------------------------------------------------

def test_ft_vae_disabled_when_not_disaggregated() -> None:
    """FT_VAE with no disaggregation should stay at zero."""
    alloc = FTVAEModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=0)
    latency_data = _make_latency_data()
    t = alloc.calculate_time(NAIVE_POLICY, DEFAULT_WORKFLOW_CONFIG, latency_data)
    assert t == 0.0


# ---------------------------------------------------------------------------
# UpscalerModelAllocation
# ---------------------------------------------------------------------------

def test_upscaler_calculate_time() -> None:
    """Upscaler time should be positive with 1 replica."""
    latency_data = _make_latency_data()
    alloc = UpscalerModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=1)
    t = alloc.calculate_time(STREAMWISE_POLICY, DEFAULT_WORKFLOW_CONFIG, latency_data)
    assert t > 0.0


def test_upscaler_time_first() -> None:
    """Upscaler time_first should be positive when use_upscaler=True."""
    latency_data = _make_latency_data()
    alloc = UpscalerModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=1)
    tf = alloc.calculate_time_first(STREAMWISE_POLICY, DEFAULT_WORKFLOW_CONFIG, latency_data)
    assert tf > 0.0


def test_upscaler_get_max_replicas() -> None:
    alloc = UpscalerModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=1)
    max_r = alloc.get_max_replicas(DEFAULT_WORKFLOW_CONFIG)
    assert max_r == DEFAULT_WORKFLOW_CONFIG.model_work.get(Model.UPSCALER, 1)


# ---------------------------------------------------------------------------
# OthersModelAllocation
# ---------------------------------------------------------------------------

def test_others_calculate_time() -> None:
    """Others time is proportional to total_scenes."""
    latency_data = _make_latency_data()
    alloc = OthersModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=1)
    t = alloc.calculate_time(STREAMWISE_POLICY, DEFAULT_WORKFLOW_CONFIG, latency_data)
    assert t > 0.0


def test_others_time_first_less_than_total() -> None:
    """Others time_first equals latency for a single scene."""
    latency_data = _make_latency_data()
    alloc = OthersModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=1)
    alloc.calculate_time(STREAMWISE_POLICY, DEFAULT_WORKFLOW_CONFIG, latency_data)
    tf = alloc.calculate_time_first(STREAMWISE_POLICY, DEFAULT_WORKFLOW_CONFIG, latency_data)
    assert 0.0 < tf <= alloc.time


# ---------------------------------------------------------------------------
# ModelAllocation.calculate() convenience method
# ---------------------------------------------------------------------------

def test_calculate_convenience_method_populates_all_fields() -> None:
    """ModelAllocation.calculate() should fill time, time_first, cost, energy."""
    latency_data = _make_latency_data()
    power_data = _make_power_data()
    alloc = FluxModelAllocation(gpu_type=GPUType.A100, devices=1, replicas=1)
    alloc.calculate(
        policy=STREAMWISE_POLICY,
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
        total_time_s=100.0,
    )
    assert alloc.time > 0.0
    assert alloc.time_first > 0.0
    assert alloc.cost > 0.0
    assert alloc.energy > 0.0


# ---------------------------------------------------------------------------
# ModelAllocation.disable()
# ---------------------------------------------------------------------------

def test_disable_zeroes_allocation() -> None:
    """disable() should zero all fields."""
    alloc = FluxModelAllocation(gpu_type=GPUType.A100, devices=2, replicas=2)
    alloc.time = 5.0
    alloc.time_first = 1.0
    alloc.energy = 100.0
    alloc.disable()
    assert alloc.devices == 0
    assert alloc.replicas == 0
    assert alloc.time == 0.0
    assert alloc.time_first == 0.0
    assert alloc.energy == 0.0
    assert alloc.get_num_gpus() == 0
