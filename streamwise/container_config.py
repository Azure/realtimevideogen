"""
Shared container deployment configuration for StreamWise.

Central source of truth for container resource defaults, MIG profiles,
GPU type mappings, and related deployment constants. Both allocator_bridge
and streamwise.py import from here to avoid duplication.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from typing import Union

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# model_provisioner import adds simulator/ to sys.path
import model_provisioner  # noqa: E402, F401

from sim_types import GPUType  # noqa: E402
from sim_types import Model  # noqa: E402


@dataclass(frozen=True)
class ContainerResourceSpec:
    """Default resource settings for a container deployment."""
    cpu: int
    memory_gib: int
    ephemeral_storage_gib: int
    gpu: Union[int, str] = 0


# Default CPU/memory/storage and baseline GPU settings for key services.
# Keep in sync with deployment defaults in streamwise/templates/add_pod.html.
MODEL_TO_CONTAINER_NAME: dict[Model, str] = {
    Model.GEMMA: "gemma",
    Model.FLUX: "flux",
    Model.HF: "hunyuanframepackf1",
    Model.HF_VAE: "hunyuanframepackvae",
    Model.FT: "fantasytalking",
}

MODEL_CONTAINER_RESOURCES: dict[Model, ContainerResourceSpec] = {
    Model.GEMMA: ContainerResourceSpec(cpu=16, memory_gib=192, ephemeral_storage_gib=64, gpu=2),
    Model.FLUX: ContainerResourceSpec(cpu=12, memory_gib=128, ephemeral_storage_gib=64, gpu=2),
    Model.HF: ContainerResourceSpec(cpu=24, memory_gib=128, ephemeral_storage_gib=64, gpu=2),
    Model.HF_VAE: ContainerResourceSpec(cpu=4, memory_gib=32, ephemeral_storage_gib=16, gpu=1),
    Model.FT: ContainerResourceSpec(cpu=12, memory_gib=192, ephemeral_storage_gib=64, gpu=2),
}

CONTAINER_RESOURCES: dict[str, ContainerResourceSpec] = {
    MODEL_TO_CONTAINER_NAME[model]: spec
    for model, spec in MODEL_CONTAINER_RESOURCES.items()
}
CONTAINER_RESOURCES.update({
    "realesrgan": ContainerResourceSpec(cpu=4, memory_gib=32, ephemeral_storage_gib=16, gpu="1g.10gb"),
    "kokoro": ContainerResourceSpec(cpu=2, memory_gib=8, ephemeral_storage_gib=16, gpu="1g.10gb"),
    "yolo": ContainerResourceSpec(cpu=4, memory_gib=8, ephemeral_storage_gib=16, gpu="1g.10gb"),
})

# GPU type string used by pod_manager (lowercase).
GPU_TYPE_TO_POD_STR: dict[GPUType, str] = {
    GPUType.A100: "a100",
    GPUType.H100: "h100",
    GPUType.H200: "h200",
    GPUType.GB200: "gb200",
}

# MIG is only supported by pod_manager on these GPU types.
MIG_CAPABLE_GPU_TYPES: frozenset[GPUType] = frozenset({GPUType.A100, GPUType.H100})

# Containers that prefer a MIG slice when the selected GPU type supports MIG.
# When MIG is available on the cluster, these services use a MIG slice (shared GPU).
# When MIG is NOT available, they fall back to 1 full GPU each and the extra GPUs
# are counted against the budget (with a warning if exceeded).
MIG_CONTAINERS: dict[str, str] = {
    "kokoro": "1g.10gb",
    "yolo": "1g.10gb",
    "realesrgan": "1g.10gb",
}

# Containers that are co-located with their parent model (sharing GPUs on the same server).
# The allocator counts their GPUs as part of the parent model's allocation, so they should
# deploy with gpu=0 to avoid double-counting.
COLOCATED_CONTAINERS: frozenset[str] = frozenset({"hunyuanframepackvae"})

# Whether MIG is actually configured on the cluster.
# When False, MIG_CONTAINERS entries fall back to full GPUs.
MIG_AVAILABLE: bool = False


def get_minimum_service_container_specs(max_gpus: int) -> dict[str, ContainerResourceSpec]:
    """Build container defaults for /api/service minimal deployment.

    Negative `max_gpus` values are clamped to 0 to avoid invalid GPU assignments.
    """
    validated_max_gpus = max(0, max_gpus)
    container_specs: dict[str, ContainerResourceSpec] = {
        "podcasttranscript": ContainerResourceSpec(cpu=1, memory_gib=4, ephemeral_storage_gib=16, gpu=0),
        "slidetranscript": ContainerResourceSpec(cpu=1, memory_gib=4, ephemeral_storage_gib=16, gpu=0),
        "fluxkontext": ContainerResourceSpec(cpu=12, memory_gib=128, ephemeral_storage_gib=64, gpu=1),
        "whisper": ContainerResourceSpec(cpu=2, memory_gib=8, ephemeral_storage_gib=16, gpu=1),
    }

    for name, spec in CONTAINER_RESOURCES.items():
        if name in MIG_CONTAINERS:
            assigned_gpu: Union[int, str] = MIG_CONTAINERS[name]
        elif name == "hunyuanframepackvae":
            assigned_gpu = 1
        else:
            assigned_gpu = min(2, validated_max_gpus)

        container_specs[name] = ContainerResourceSpec(
            cpu=spec.cpu,
            memory_gib=spec.memory_gib,
            ephemeral_storage_gib=spec.ephemeral_storage_gib,
            gpu=assigned_gpu,
        )

    return container_specs
