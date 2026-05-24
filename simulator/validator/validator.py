"""
Validator for external simulators that mimic the StreamWise system.

This module provides:
1. A set of representative GPU allocation scenarios.
2. A function to generate ground truth results from the current simulator.
3. A function to validate external simulator outputs against the ground truth.

Usage:
    # Generate ground truth (run from repository root):
    python -m simulator.validator --generate

    # Programmatic validation:
    from simulator.validator import validate, load_ground_truth
    ground_truth = load_ground_truth()
    results = validate(my_simulator_outputs, ground_truth)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from sim_types import GPUType
from sim_types import Model
from sim_types import Policy
from sim_types import Objective

from constants import DEFAULT_WORKFLOW_CONFIG
from constants import GPU_SPOT_COST

from data_loading import load_latency_data
from data_loading import load_power_data

from evaluator import evaluate_model_allocation

from models import FluxModelAllocation
from models import GemmaModelAllocation
from models import HFModelAllocation
from models import HFVAEModelAllocation
from models import FTModelAllocation
from models import FTVAEModelAllocation
from models import UpscalerModelAllocation
from models import OthersModelAllocation
from models import ModelAllocation

logger = logging.getLogger(__name__)

# Fixed policy for ground truth generation.
# The validator uses a fixed policy so that cost/time calculations are deterministic.
VALIDATOR_POLICY = Policy(
    name="validator",
    gpu_cost=GPU_SPOT_COST,
    objective=Objective.TTFF_COST,
    disaggregation={
        Model.HF: True,
        Model.FT: False,
    },
    use_upscaler=True,
    hardware=list(GPUType),
)

RELATIVE_TOLERANCE = 0.01  # 1% relative error

_DEFAULT_GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "validator_ground_truth.json"


@dataclass
class ScenarioAllocation:
    """Describes a model's allocation in a scenario."""
    gpu_type: str  # GPUType value e.g. "A100"
    model: str  # Model value e.g. "gemma"
    devices: int
    replicas: int


@dataclass
class Scenario:
    """A named GPU allocation scenario for validation."""
    name: str
    allocations: list[ScenarioAllocation]
    num_gpus: dict[str, int]  # GPUType value -> total GPUs available


@dataclass
class ValidationResult:
    """Result of validating one scenario."""
    scenario_name: str
    passed: bool
    errors: list[str]


def _get_model_allocation_instance(
    model: Model,
    gpu_type: GPUType,
    devices: int,
    replicas: int,
) -> ModelAllocation:
    """Create a ModelAllocation instance for the given model."""
    cls_map = {
        Model.GEMMA: GemmaModelAllocation,
        Model.FLUX: FluxModelAllocation,
        Model.HF: HFModelAllocation,
        Model.HF_VAE: HFVAEModelAllocation,
        Model.FT: FTModelAllocation,
        Model.FT_VAE: FTVAEModelAllocation,
        Model.UPSCALER: UpscalerModelAllocation,
        Model.OTHERS: OthersModelAllocation,
    }
    cls = cls_map[model]
    return cls(gpu_type=gpu_type, devices=devices, replicas=replicas)


def _build_models_dict(
    allocations: list[ScenarioAllocation],
) -> dict[GPUType, dict[Model, list[ModelAllocation]]]:
    """Build the models dict structure from a flat list of allocations."""
    models: dict[GPUType, dict[Model, list[ModelAllocation]]] = {}
    for alloc in allocations:
        gpu_type = GPUType(alloc.gpu_type)
        model = Model(alloc.model)
        if gpu_type not in models:
            models[gpu_type] = {}
        if model not in models[gpu_type]:
            models[gpu_type][model] = []
        models[gpu_type][model].append(
            _get_model_allocation_instance(model, gpu_type, alloc.devices, alloc.replicas)
        )
    return models


def get_scenarios() -> list[Scenario]:
    """
    Return a comprehensive set of GPU allocation scenarios for validation.

    Scenarios vary across:
    - GPU types: A100, H100, H200
    - Total GPU counts: 8, 16, 24, 32, 40, 48, 64
    - Tensor parallelism (devices per model): 1, 2, 4, 8, 16
    - Replica counts: 1 to 10+
    - Mixed vs single GPU type configurations
    - Bottleneck-focused allocations (FT-heavy, HF-heavy, balanced)
    """
    scenarios = [
        # ============================================================
        # GROUP 1: Single server (8 GPUs) — minimal configurations
        # ============================================================
        Scenario(
            name="8xA100_minimal",
            allocations=[
                ScenarioAllocation("A100", "gemma", devices=1, replicas=1),
                ScenarioAllocation("A100", "flux", devices=1, replicas=1),
                ScenarioAllocation("A100", "hf", devices=1, replicas=1),
                ScenarioAllocation("A100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("A100", "ft", devices=1, replicas=2),
                ScenarioAllocation("A100", "upscaler", devices=1, replicas=1),
                ScenarioAllocation("A100", "others", devices=1, replicas=1),
            ],
            num_gpus={"A100": 8},
        ),
        Scenario(
            name="8xH100_minimal",
            allocations=[
                ScenarioAllocation("H100", "gemma", devices=1, replicas=1),
                ScenarioAllocation("H100", "flux", devices=1, replicas=1),
                ScenarioAllocation("H100", "hf", devices=1, replicas=1),
                ScenarioAllocation("H100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H100", "ft", devices=1, replicas=2),
                ScenarioAllocation("H100", "upscaler", devices=1, replicas=1),
                ScenarioAllocation("H100", "others", devices=1, replicas=1),
            ],
            num_gpus={"H100": 8},
        ),
        Scenario(
            name="8xH200_minimal",
            allocations=[
                ScenarioAllocation("H200", "gemma", devices=1, replicas=1),
                ScenarioAllocation("H200", "flux", devices=1, replicas=1),
                ScenarioAllocation("H200", "hf", devices=1, replicas=1),
                ScenarioAllocation("H200", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H200", "ft", devices=1, replicas=2),
                ScenarioAllocation("H200", "upscaler", devices=1, replicas=1),
                ScenarioAllocation("H200", "others", devices=1, replicas=1),
            ],
            num_gpus={"H200": 8},
        ),
        # ============================================================
        # GROUP 2: Two servers (16 GPUs) — moderate parallelism
        # ============================================================
        Scenario(
            name="16xA100_balanced",
            allocations=[
                ScenarioAllocation("A100", "gemma", devices=1, replicas=1),
                ScenarioAllocation("A100", "flux", devices=2, replicas=1),
                ScenarioAllocation("A100", "hf", devices=2, replicas=2),
                ScenarioAllocation("A100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("A100", "ft", devices=1, replicas=4),
                ScenarioAllocation("A100", "upscaler", devices=1, replicas=2),
                ScenarioAllocation("A100", "others", devices=1, replicas=1),
            ],
            num_gpus={"A100": 16},
        ),
        Scenario(
            name="16xH100_balanced",
            allocations=[
                ScenarioAllocation("H100", "gemma", devices=2, replicas=1),
                ScenarioAllocation("H100", "flux", devices=2, replicas=1),
                ScenarioAllocation("H100", "hf", devices=2, replicas=2),
                ScenarioAllocation("H100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H100", "ft", devices=1, replicas=4),
                ScenarioAllocation("H100", "upscaler", devices=1, replicas=1),
                ScenarioAllocation("H100", "others", devices=1, replicas=1),
            ],
            num_gpus={"H100": 16},
        ),
        Scenario(
            name="16xH200_balanced",
            allocations=[
                ScenarioAllocation("H200", "gemma", devices=1, replicas=1),
                ScenarioAllocation("H200", "flux", devices=1, replicas=1),
                ScenarioAllocation("H200", "hf", devices=2, replicas=2),
                ScenarioAllocation("H200", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H200", "ft", devices=2, replicas=2),
                ScenarioAllocation("H200", "upscaler", devices=1, replicas=2),
                ScenarioAllocation("H200", "others", devices=1, replicas=1),
            ],
            num_gpus={"H200": 16},
        ),
        Scenario(
            name="16xA100_ft_heavy",
            allocations=[
                ScenarioAllocation("A100", "gemma", devices=1, replicas=1),
                ScenarioAllocation("A100", "flux", devices=1, replicas=1),
                ScenarioAllocation("A100", "hf", devices=1, replicas=1),
                ScenarioAllocation("A100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("A100", "ft", devices=1, replicas=8),
                ScenarioAllocation("A100", "upscaler", devices=1, replicas=2),
                ScenarioAllocation("A100", "others", devices=1, replicas=1),
            ],
            num_gpus={"A100": 16},
        ),
        Scenario(
            name="16xH200_hf_heavy",
            allocations=[
                ScenarioAllocation("H200", "gemma", devices=1, replicas=1),
                ScenarioAllocation("H200", "flux", devices=1, replicas=1),
                ScenarioAllocation("H200", "hf", devices=8, replicas=1),
                ScenarioAllocation("H200", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H200", "ft", devices=1, replicas=2),
                ScenarioAllocation("H200", "upscaler", devices=1, replicas=1),
                ScenarioAllocation("H200", "others", devices=1, replicas=1),
            ],
            num_gpus={"H200": 16},
        ),
        Scenario(
            name="16xH100_high_tp",
            allocations=[
                ScenarioAllocation("H100", "gemma", devices=4, replicas=1),
                ScenarioAllocation("H100", "flux", devices=4, replicas=1),
                ScenarioAllocation("H100", "hf", devices=4, replicas=1),
                ScenarioAllocation("H100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H100", "ft", devices=1, replicas=1),
                ScenarioAllocation("H100", "upscaler", devices=1, replicas=1),
                ScenarioAllocation("H100", "others", devices=1, replicas=1),
            ],
            num_gpus={"H100": 16},
        ),
        # ============================================================
        # GROUP 3: Three servers (24 GPUs) — scaled configurations
        # ============================================================
        Scenario(
            name="24xH100_scaled",
            allocations=[
                ScenarioAllocation("H100", "gemma", devices=2, replicas=1),
                ScenarioAllocation("H100", "flux", devices=2, replicas=1),
                ScenarioAllocation("H100", "hf", devices=4, replicas=2),
                ScenarioAllocation("H100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H100", "ft", devices=2, replicas=4),
                ScenarioAllocation("H100", "upscaler", devices=1, replicas=2),
                ScenarioAllocation("H100", "others", devices=1, replicas=1),
            ],
            num_gpus={"H100": 24},
        ),
        Scenario(
            name="24xA100_ft_scaled",
            allocations=[
                ScenarioAllocation("A100", "gemma", devices=1, replicas=1),
                ScenarioAllocation("A100", "flux", devices=1, replicas=1),
                ScenarioAllocation("A100", "hf", devices=2, replicas=1),
                ScenarioAllocation("A100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("A100", "ft", devices=2, replicas=6),
                ScenarioAllocation("A100", "upscaler", devices=1, replicas=4),
                ScenarioAllocation("A100", "others", devices=1, replicas=1),
            ],
            num_gpus={"A100": 24},
        ),
        Scenario(
            name="24xH200_balanced",
            allocations=[
                ScenarioAllocation("H200", "gemma", devices=2, replicas=1),
                ScenarioAllocation("H200", "flux", devices=2, replicas=1),
                ScenarioAllocation("H200", "hf", devices=4, replicas=2),
                ScenarioAllocation("H200", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H200", "ft", devices=2, replicas=3),
                ScenarioAllocation("H200", "upscaler", devices=1, replicas=2),
                ScenarioAllocation("H200", "others", devices=1, replicas=1),
            ],
            num_gpus={"H200": 24},
        ),
        # ============================================================
        # GROUP 4: Four servers (32 GPUs) — large scale
        # ============================================================
        Scenario(
            name="32xH200_large",
            allocations=[
                ScenarioAllocation("H200", "gemma", devices=2, replicas=1),
                ScenarioAllocation("H200", "flux", devices=2, replicas=1),
                ScenarioAllocation("H200", "hf", devices=4, replicas=3),
                ScenarioAllocation("H200", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H200", "ft", devices=2, replicas=4),
                ScenarioAllocation("H200", "upscaler", devices=1, replicas=4),
                ScenarioAllocation("H200", "others", devices=1, replicas=1),
            ],
            num_gpus={"H200": 32},
        ),
        Scenario(
            name="32xA100_high_replicas",
            allocations=[
                ScenarioAllocation("A100", "gemma", devices=1, replicas=1),
                ScenarioAllocation("A100", "flux", devices=1, replicas=1),
                ScenarioAllocation("A100", "hf", devices=1, replicas=4),
                ScenarioAllocation("A100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("A100", "ft", devices=1, replicas=10),
                ScenarioAllocation("A100", "upscaler", devices=1, replicas=4),
                ScenarioAllocation("A100", "others", devices=1, replicas=1),
            ],
            num_gpus={"A100": 32},
        ),
        Scenario(
            name="32xH100_mixed_tp",
            allocations=[
                ScenarioAllocation("H100", "gemma", devices=4, replicas=1),
                ScenarioAllocation("H100", "flux", devices=4, replicas=1),
                ScenarioAllocation("H100", "hf", devices=8, replicas=1),
                ScenarioAllocation("H100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H100", "ft", devices=4, replicas=2),
                ScenarioAllocation("H100", "upscaler", devices=2, replicas=1),
                ScenarioAllocation("H100", "others", devices=1, replicas=1),
            ],
            num_gpus={"H100": 32},
        ),
        # ============================================================
        # GROUP 5: Five servers (40 GPUs)
        # ============================================================
        Scenario(
            name="40xH200_high_parallelism",
            allocations=[
                ScenarioAllocation("H200", "gemma", devices=2, replicas=1),
                ScenarioAllocation("H200", "flux", devices=4, replicas=1),
                ScenarioAllocation("H200", "hf", devices=4, replicas=4),
                ScenarioAllocation("H200", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H200", "ft", devices=2, replicas=5),
                ScenarioAllocation("H200", "upscaler", devices=1, replicas=4),
                ScenarioAllocation("H200", "others", devices=1, replicas=1),
            ],
            num_gpus={"H200": 40},
        ),
        Scenario(
            name="40xA100_ft_dominated",
            allocations=[
                ScenarioAllocation("A100", "gemma", devices=1, replicas=1),
                ScenarioAllocation("A100", "flux", devices=2, replicas=1),
                ScenarioAllocation("A100", "hf", devices=2, replicas=2),
                ScenarioAllocation("A100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("A100", "ft", devices=2, replicas=10),
                ScenarioAllocation("A100", "upscaler", devices=2, replicas=4),
                ScenarioAllocation("A100", "others", devices=1, replicas=1),
            ],
            num_gpus={"A100": 40},
        ),
        # ============================================================
        # GROUP 6: Six servers (48 GPUs)
        # ============================================================
        Scenario(
            name="48xH100_large_scale",
            allocations=[
                ScenarioAllocation("H100", "gemma", devices=4, replicas=1),
                ScenarioAllocation("H100", "flux", devices=4, replicas=1),
                ScenarioAllocation("H100", "hf", devices=8, replicas=2),
                ScenarioAllocation("H100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H100", "ft", devices=4, replicas=4),
                ScenarioAllocation("H100", "upscaler", devices=2, replicas=2),
                ScenarioAllocation("H100", "others", devices=1, replicas=1),
            ],
            num_gpus={"H100": 48},
        ),
        Scenario(
            name="48xH200_max_ft",
            allocations=[
                ScenarioAllocation("H200", "gemma", devices=2, replicas=1),
                ScenarioAllocation("H200", "flux", devices=2, replicas=1),
                ScenarioAllocation("H200", "hf", devices=4, replicas=2),
                ScenarioAllocation("H200", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H200", "ft", devices=4, replicas=6),
                ScenarioAllocation("H200", "upscaler", devices=1, replicas=4),
                ScenarioAllocation("H200", "others", devices=1, replicas=1),
            ],
            num_gpus={"H200": 48},
        ),
        # ============================================================
        # GROUP 7: Eight servers (64 GPUs) — very large scale
        # ============================================================
        Scenario(
            name="64xH200_massive",
            allocations=[
                ScenarioAllocation("H200", "gemma", devices=4, replicas=1),
                ScenarioAllocation("H200", "flux", devices=4, replicas=1),
                ScenarioAllocation("H200", "hf", devices=8, replicas=3),
                ScenarioAllocation("H200", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H200", "ft", devices=4, replicas=5),
                ScenarioAllocation("H200", "upscaler", devices=2, replicas=4),
                ScenarioAllocation("H200", "others", devices=1, replicas=1),
            ],
            num_gpus={"H200": 64},
        ),
        Scenario(
            name="64xA100_distributed",
            allocations=[
                ScenarioAllocation("A100", "gemma", devices=2, replicas=1),
                ScenarioAllocation("A100", "flux", devices=4, replicas=1),
                ScenarioAllocation("A100", "hf", devices=4, replicas=4),
                ScenarioAllocation("A100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("A100", "ft", devices=2, replicas=10),
                ScenarioAllocation("A100", "upscaler", devices=2, replicas=6),
                ScenarioAllocation("A100", "others", devices=1, replicas=1),
            ],
            num_gpus={"A100": 64},
        ),
        # ============================================================
        # GROUP 8: Mixed GPU type configurations
        # ============================================================
        Scenario(
            name="mixed_8A100_8H200",
            allocations=[
                ScenarioAllocation("A100", "gemma", devices=1, replicas=1),
                ScenarioAllocation("A100", "flux", devices=2, replicas=1),
                ScenarioAllocation("A100", "others", devices=1, replicas=1),
                ScenarioAllocation("H200", "hf", devices=4, replicas=1),
                ScenarioAllocation("H200", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H200", "ft", devices=1, replicas=2),
                ScenarioAllocation("H200", "upscaler", devices=1, replicas=1),
            ],
            num_gpus={"A100": 8, "H200": 8},
        ),
        Scenario(
            name="mixed_8A100_16H200",
            allocations=[
                ScenarioAllocation("A100", "gemma", devices=1, replicas=1),
                ScenarioAllocation("A100", "flux", devices=2, replicas=1),
                ScenarioAllocation("A100", "others", devices=1, replicas=1),
                ScenarioAllocation("H200", "hf", devices=4, replicas=2),
                ScenarioAllocation("H200", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H200", "ft", devices=2, replicas=2),
                ScenarioAllocation("H200", "upscaler", devices=1, replicas=1),
            ],
            num_gpus={"A100": 8, "H200": 16},
        ),
        Scenario(
            name="mixed_16A100_16H100",
            allocations=[
                ScenarioAllocation("A100", "gemma", devices=1, replicas=1),
                ScenarioAllocation("A100", "flux", devices=1, replicas=1),
                ScenarioAllocation("A100", "hf", devices=2, replicas=2),
                ScenarioAllocation("A100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("A100", "others", devices=1, replicas=1),
                ScenarioAllocation("H100", "ft", devices=2, replicas=4),
                ScenarioAllocation("H100", "upscaler", devices=1, replicas=4),
            ],
            num_gpus={"A100": 16, "H100": 16},
        ),
        Scenario(
            name="mixed_8H100_24H200",
            allocations=[
                ScenarioAllocation("H100", "gemma", devices=2, replicas=1),
                ScenarioAllocation("H100", "flux", devices=4, replicas=1),
                ScenarioAllocation("H100", "others", devices=1, replicas=1),
                ScenarioAllocation("H200", "hf", devices=4, replicas=3),
                ScenarioAllocation("H200", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H200", "ft", devices=2, replicas=4),
                ScenarioAllocation("H200", "upscaler", devices=1, replicas=2),
            ],
            num_gpus={"H100": 8, "H200": 24},
        ),
        Scenario(
            name="mixed_16A100_32H200",
            allocations=[
                ScenarioAllocation("A100", "gemma", devices=2, replicas=1),
                ScenarioAllocation("A100", "flux", devices=4, replicas=1),
                ScenarioAllocation("A100", "hf", devices=2, replicas=2),
                ScenarioAllocation("A100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("A100", "others", devices=1, replicas=1),
                ScenarioAllocation("H200", "ft", devices=4, replicas=6),
                ScenarioAllocation("H200", "upscaler", devices=2, replicas=4),
            ],
            num_gpus={"A100": 16, "H200": 32},
        ),
        Scenario(
            name="mixed_triple_8A_8H1_8H2",
            allocations=[
                ScenarioAllocation("A100", "gemma", devices=1, replicas=1),
                ScenarioAllocation("A100", "flux", devices=1, replicas=1),
                ScenarioAllocation("A100", "others", devices=1, replicas=1),
                ScenarioAllocation("H100", "hf", devices=2, replicas=2),
                ScenarioAllocation("H100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H200", "ft", devices=2, replicas=2),
                ScenarioAllocation("H200", "upscaler", devices=1, replicas=2),
            ],
            num_gpus={"A100": 8, "H100": 8, "H200": 8},
        ),
        # ============================================================
        # GROUP 9: High tensor parallelism configurations
        # ============================================================
        Scenario(
            name="16xH200_tp8_hf",
            allocations=[
                ScenarioAllocation("H200", "gemma", devices=1, replicas=1),
                ScenarioAllocation("H200", "flux", devices=1, replicas=1),
                ScenarioAllocation("H200", "hf", devices=8, replicas=1),
                ScenarioAllocation("H200", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H200", "ft", devices=2, replicas=1),
                ScenarioAllocation("H200", "upscaler", devices=1, replicas=1),
                ScenarioAllocation("H200", "others", devices=1, replicas=1),
            ],
            num_gpus={"H200": 16},
        ),
        Scenario(
            name="32xH100_tp16_hf",
            allocations=[
                ScenarioAllocation("H100", "gemma", devices=2, replicas=1),
                ScenarioAllocation("H100", "flux", devices=4, replicas=1),
                ScenarioAllocation("H100", "hf", devices=16, replicas=1),
                ScenarioAllocation("H100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H100", "ft", devices=4, replicas=1),
                ScenarioAllocation("H100", "upscaler", devices=1, replicas=1),
                ScenarioAllocation("H100", "others", devices=1, replicas=1),
            ],
            num_gpus={"H100": 32},
        ),
        Scenario(
            name="24xA100_tp4_all",
            allocations=[
                ScenarioAllocation("A100", "gemma", devices=4, replicas=1),
                ScenarioAllocation("A100", "flux", devices=4, replicas=1),
                ScenarioAllocation("A100", "hf", devices=4, replicas=2),
                ScenarioAllocation("A100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("A100", "ft", devices=4, replicas=1),
                ScenarioAllocation("A100", "upscaler", devices=1, replicas=1),
                ScenarioAllocation("A100", "others", devices=1, replicas=1),
            ],
            num_gpus={"A100": 24},
        ),
        # ============================================================
        # GROUP 10: High replica, low tensor parallelism
        # ============================================================
        Scenario(
            name="24xH200_many_replicas",
            allocations=[
                ScenarioAllocation("H200", "gemma", devices=1, replicas=1),
                ScenarioAllocation("H200", "flux", devices=1, replicas=1),
                ScenarioAllocation("H200", "hf", devices=1, replicas=4),
                ScenarioAllocation("H200", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H200", "ft", devices=1, replicas=8),
                ScenarioAllocation("H200", "upscaler", devices=1, replicas=4),
                ScenarioAllocation("H200", "others", devices=1, replicas=1),
            ],
            num_gpus={"H200": 24},
        ),
        Scenario(
            name="32xA100_many_replicas",
            allocations=[
                ScenarioAllocation("A100", "gemma", devices=1, replicas=1),
                ScenarioAllocation("A100", "flux", devices=1, replicas=1),
                ScenarioAllocation("A100", "hf", devices=1, replicas=6),
                ScenarioAllocation("A100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("A100", "ft", devices=1, replicas=10),
                ScenarioAllocation("A100", "upscaler", devices=1, replicas=8),
                ScenarioAllocation("A100", "others", devices=1, replicas=1),
            ],
            num_gpus={"A100": 32},
        ),
        Scenario(
            name="40xH100_many_replicas",
            allocations=[
                ScenarioAllocation("H100", "gemma", devices=1, replicas=1),
                ScenarioAllocation("H100", "flux", devices=1, replicas=1),
                ScenarioAllocation("H100", "hf", devices=1, replicas=6),
                ScenarioAllocation("H100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H100", "ft", devices=1, replicas=16),
                ScenarioAllocation("H100", "upscaler", devices=1, replicas=8),
                ScenarioAllocation("H100", "others", devices=1, replicas=1),
            ],
            num_gpus={"H100": 40},
        ),
        # ============================================================
        # GROUP 11: Upscaler-heavy configurations
        # ============================================================
        Scenario(
            name="16xA100_upscaler_heavy",
            allocations=[
                ScenarioAllocation("A100", "gemma", devices=1, replicas=1),
                ScenarioAllocation("A100", "flux", devices=1, replicas=1),
                ScenarioAllocation("A100", "hf", devices=1, replicas=1),
                ScenarioAllocation("A100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("A100", "ft", devices=1, replicas=2),
                ScenarioAllocation("A100", "upscaler", devices=1, replicas=8),
                ScenarioAllocation("A100", "others", devices=1, replicas=1),
            ],
            num_gpus={"A100": 16},
        ),
        Scenario(
            name="24xH200_upscaler_heavy",
            allocations=[
                ScenarioAllocation("H200", "gemma", devices=1, replicas=1),
                ScenarioAllocation("H200", "flux", devices=1, replicas=1),
                ScenarioAllocation("H200", "hf", devices=2, replicas=2),
                ScenarioAllocation("H200", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H200", "ft", devices=2, replicas=2),
                ScenarioAllocation("H200", "upscaler", devices=2, replicas=4),
                ScenarioAllocation("H200", "others", devices=1, replicas=1),
            ],
            num_gpus={"H200": 24},
        ),
        # ============================================================
        # GROUP 12: Gemma/Flux with higher TP (LLM-focused)
        # ============================================================
        Scenario(
            name="16xH100_gemma_tp8",
            allocations=[
                ScenarioAllocation("H100", "gemma", devices=8, replicas=1),
                ScenarioAllocation("H100", "flux", devices=1, replicas=1),
                ScenarioAllocation("H100", "hf", devices=2, replicas=1),
                ScenarioAllocation("H100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H100", "ft", devices=1, replicas=1),
                ScenarioAllocation("H100", "upscaler", devices=1, replicas=1),
                ScenarioAllocation("H100", "others", devices=1, replicas=1),
            ],
            num_gpus={"H100": 16},
        ),
        Scenario(
            name="24xH200_flux_tp8",
            allocations=[
                ScenarioAllocation("H200", "gemma", devices=2, replicas=1),
                ScenarioAllocation("H200", "flux", devices=8, replicas=1),
                ScenarioAllocation("H200", "hf", devices=4, replicas=1),
                ScenarioAllocation("H200", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H200", "ft", devices=2, replicas=2),
                ScenarioAllocation("H200", "upscaler", devices=1, replicas=1),
                ScenarioAllocation("H200", "others", devices=1, replicas=1),
            ],
            num_gpus={"H200": 24},
        ),
        # ============================================================
        # GROUP 13: Heterogeneous multi-allocation configurations
        # Same model allocated across multiple GPU types simultaneously
        # ============================================================
        Scenario(
            name="hetero_24H200_40A100_hf_split",
            allocations=[
                ScenarioAllocation("A100", "gemma", devices=1, replicas=1),
                ScenarioAllocation("A100", "flux", devices=2, replicas=1),
                ScenarioAllocation("A100", "others", devices=1, replicas=1),
                ScenarioAllocation("A100", "hf", devices=4, replicas=4),
                ScenarioAllocation("A100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("A100", "ft", devices=2, replicas=6),
                ScenarioAllocation("A100", "upscaler", devices=1, replicas=4),
                ScenarioAllocation("H200", "hf", devices=4, replicas=3),
                ScenarioAllocation("H200", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H200", "ft", devices=2, replicas=2),
                ScenarioAllocation("H200", "upscaler", devices=1, replicas=2),
            ],
            num_gpus={"A100": 40, "H200": 24},
        ),
        Scenario(
            name="hetero_8H100_16A100_ft_split",
            allocations=[
                ScenarioAllocation("A100", "gemma", devices=1, replicas=1),
                ScenarioAllocation("A100", "flux", devices=1, replicas=1),
                ScenarioAllocation("A100", "others", devices=1, replicas=1),
                ScenarioAllocation("A100", "hf", devices=2, replicas=2),
                ScenarioAllocation("A100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("A100", "ft", devices=1, replicas=4),
                ScenarioAllocation("A100", "upscaler", devices=1, replicas=2),
                ScenarioAllocation("H100", "ft", devices=2, replicas=4),
            ],
            num_gpus={"A100": 16, "H100": 8},
        ),
        Scenario(
            name="hetero_16H200_16H100_balanced",
            allocations=[
                ScenarioAllocation("H100", "gemma", devices=2, replicas=1),
                ScenarioAllocation("H100", "flux", devices=2, replicas=1),
                ScenarioAllocation("H100", "others", devices=1, replicas=1),
                ScenarioAllocation("H100", "hf", devices=4, replicas=1),
                ScenarioAllocation("H100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H100", "ft", devices=2, replicas=2),
                ScenarioAllocation("H200", "hf", devices=4, replicas=2),
                ScenarioAllocation("H200", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H200", "ft", devices=2, replicas=2),
                ScenarioAllocation("H200", "upscaler", devices=1, replicas=2),
            ],
            num_gpus={"H100": 16, "H200": 16},
        ),
        Scenario(
            name="hetero_8A100_8H100_8H200_triple",
            allocations=[
                ScenarioAllocation("A100", "gemma", devices=1, replicas=1),
                ScenarioAllocation("A100", "flux", devices=1, replicas=1),
                ScenarioAllocation("A100", "others", devices=1, replicas=1),
                ScenarioAllocation("A100", "hf", devices=1, replicas=2),
                ScenarioAllocation("A100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H100", "hf", devices=2, replicas=1),
                ScenarioAllocation("H100", "ft", devices=1, replicas=4),
                ScenarioAllocation("H100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H200", "ft", devices=2, replicas=2),
                ScenarioAllocation("H200", "upscaler", devices=1, replicas=2),
                ScenarioAllocation("H200", "hf_vae", devices=1, replicas=1),
            ],
            num_gpus={"A100": 8, "H100": 8, "H200": 8},
        ),
        Scenario(
            name="hetero_24H200_8A100_hf_heavy",
            allocations=[
                ScenarioAllocation("A100", "gemma", devices=1, replicas=1),
                ScenarioAllocation("A100", "flux", devices=2, replicas=1),
                ScenarioAllocation("A100", "others", devices=1, replicas=1),
                ScenarioAllocation("A100", "hf", devices=1, replicas=2),
                ScenarioAllocation("A100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H200", "hf", devices=4, replicas=4),
                ScenarioAllocation("H200", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H200", "ft", devices=2, replicas=2),
                ScenarioAllocation("H200", "upscaler", devices=1, replicas=1),
            ],
            num_gpus={"A100": 8, "H200": 24},
        ),
        Scenario(
            name="hetero_32A100_16H100_ft_heavy",
            allocations=[
                ScenarioAllocation("A100", "gemma", devices=1, replicas=1),
                ScenarioAllocation("A100", "flux", devices=2, replicas=1),
                ScenarioAllocation("A100", "others", devices=1, replicas=1),
                ScenarioAllocation("A100", "hf", devices=2, replicas=4),
                ScenarioAllocation("A100", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("A100", "ft", devices=2, replicas=7),
                ScenarioAllocation("A100", "upscaler", devices=1, replicas=4),
                ScenarioAllocation("H100", "ft", devices=4, replicas=4),
            ],
            num_gpus={"A100": 32, "H100": 16},
        ),
        Scenario(
            name="hetero_16H200_8H100_upscaler_split",
            allocations=[
                ScenarioAllocation("H100", "gemma", devices=2, replicas=1),
                ScenarioAllocation("H100", "flux", devices=2, replicas=1),
                ScenarioAllocation("H100", "others", devices=1, replicas=1),
                ScenarioAllocation("H100", "upscaler", devices=1, replicas=2),
                ScenarioAllocation("H200", "hf", devices=4, replicas=2),
                ScenarioAllocation("H200", "hf_vae", devices=1, replicas=1),
                ScenarioAllocation("H200", "ft", devices=2, replicas=2),
                ScenarioAllocation("H200", "upscaler", devices=1, replicas=2),
            ],
            num_gpus={"H100": 8, "H200": 16},
        ),
    ]
    return scenarios


def generate_ground_truth(
    data_dir: Optional[str | Path] = None,
    output_path: Optional[Path] = None,
) -> dict[str, dict]:
    """
    Run all scenarios through the simulator and produce ground truth results.

    Args:
        data_dir: Path to simulator data directory. Defaults to simulator/data/.
        output_path: Path to save JSON output. Defaults to validator_ground_truth.json.

    Returns:
        Dictionary mapping scenario name to ground truth metrics.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "data"
    if output_path is None:
        output_path = _DEFAULT_GROUND_TRUTH_PATH

    latency_data = load_latency_data(data_dir)
    power_data = load_power_data(data_dir)
    workflow = DEFAULT_WORKFLOW_CONFIG

    scenarios = get_scenarios()
    ground_truth: dict[str, dict] = {}

    for scenario in scenarios:
        models_dict = _build_models_dict(scenario.allocations)
        num_gpus = {GPUType(k): v for k, v in scenario.num_gpus.items()}

        result = evaluate_model_allocation(
            models=models_dict,
            num_gpus=num_gpus,
            workflow=workflow,
            latency_data=latency_data,
            power_data=power_data,
            policy=VALIDATOR_POLICY,
        )

        ground_truth[scenario.name] = {
            "allocations": [
                {
                    "gpu_type": a.gpu_type,
                    "model": a.model,
                    "devices": a.devices,
                    "replicas": a.replicas,
                }
                for a in scenario.allocations
            ],
            "num_gpus": scenario.num_gpus,
            "expected": {
                "cost": round(result.cost, 4),
                "ttff_s": round(result.ttff_s, 4),
                "tbf_s": round(result.tbf_s, 4),
                "total_time_s": round(result.total_time_s, 4),
            },
        }
        logger.info(
            "Scenario %s: cost=$%.2f, ttff=%.2fs, tbf=%.4fs, total=%.2fs",
            scenario.name, result.cost, result.ttff_s, result.tbf_s, result.total_time_s,
        )

    with open(output_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    logger.info("Ground truth saved to %s", output_path)
    return ground_truth


def load_ground_truth(
    path: Optional[Path] = None,
) -> dict[str, dict]:
    """Load previously generated ground truth from JSON."""
    if path is None:
        path = _DEFAULT_GROUND_TRUTH_PATH
    with open(path, "r") as f:
        return json.load(f)


def _check_relative_error(
    actual: float,
    expected: float,
    metric_name: str,
    tolerance: float = RELATIVE_TOLERANCE,
) -> Optional[str]:
    """Check if actual is within tolerance of expected. Returns error message or None."""
    if expected == 0.0:
        if actual == 0.0:
            return None
        return (
            f"{metric_name}: expected 0.0, got {actual:.6f}"
        )
    relative_error = abs(actual - expected) / abs(expected)
    if relative_error > tolerance:
        return (
            f"{metric_name}: expected {expected:.6f}, got {actual:.6f} "
            f"(relative error: {relative_error:.4%}, tolerance: {tolerance:.2%})"
        )
    return None


def validate(
    simulator_outputs: dict[str, dict[str, float]],
    ground_truth: Optional[dict[str, dict]] = None,
) -> list[ValidationResult]:
    """
    Validate external simulator outputs against ground truth.

    Args:
        simulator_outputs: Dict mapping scenario name to metrics dict.
            Each metrics dict must have keys: "cost", "ttff_s", "tbf_s", "total_time_s".
        ground_truth: Ground truth dict (loaded from JSON). If None, loads from default path.

    Returns:
        List of ValidationResult for each scenario in the ground truth.
    """
    if ground_truth is None:
        ground_truth = load_ground_truth()

    results: list[ValidationResult] = []

    for scenario_name, gt_data in ground_truth.items():
        if scenario_name not in simulator_outputs:
            results.append(ValidationResult(
                scenario_name=scenario_name,
                passed=False,
                errors=[f"Scenario '{scenario_name}' not found in simulator outputs"],
            ))
            continue

        output = simulator_outputs[scenario_name]
        expected = gt_data["expected"]
        errors: list[str] = []

        for metric in ("cost", "ttff_s", "tbf_s", "total_time_s"):
            if metric not in output:
                errors.append(f"Missing metric: {metric}")
                continue
            error = _check_relative_error(
                actual=output[metric],
                expected=expected[metric],
                metric_name=metric,
            )
            if error is not None:
                errors.append(error)

        results.append(ValidationResult(
            scenario_name=scenario_name,
            passed=len(errors) == 0,
            errors=errors,
        ))

    return results


def validate_all_passed(
    simulator_outputs: dict[str, dict[str, float]],
    ground_truth: Optional[dict[str, dict]] = None,
) -> bool:
    """Convenience function: returns True if all scenarios pass validation."""
    results = validate(simulator_outputs, ground_truth)
    return all(r.passed for r in results)


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Simulator validator")
    parser.add_argument(
        "--generate", action="store_true",
        help="Generate ground truth JSON from the current simulator",
    )
    parser.add_argument(
        "--validate", type=str, default=None,
        help="Path to external simulator results JSON to validate",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to simulator data directory",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for ground truth JSON",
    )
    args = parser.parse_args()

    if args.generate:
        output_path = Path(args.output) if args.output else None
        data_dir = Path(args.data_dir) if args.data_dir else None
        generate_ground_truth(data_dir=data_dir, output_path=output_path)
        print("Ground truth generated successfully.")

    elif args.validate:
        with open(args.validate, "r") as f:
            external_results = json.load(f)
        results = validate(external_results)
        all_passed = True
        for r in results:
            if r.passed:
                print(f"  PASS: {r.scenario_name}")
            else:
                all_passed = False
                print(f"  FAIL: {r.scenario_name}")
                for err in r.errors:
                    print(f"        {err}")
        sys.exit(0 if all_passed else 1)

    else:
        parser.print_help()
