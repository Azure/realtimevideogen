"""
Shared container deployment configuration for StreamWise.

Central source of truth for container resource defaults, MIG profiles,
GPU type mappings, and related deployment constants. Both allocator_bridge
and streamwise.py import from here to avoid duplication.
"""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# model_provisioner import adds simulator/ to sys.path
import model_provisioner  # noqa: E402, F401

from sim_types import GPUType  # noqa: E402


# Default CPU/memory/storage for each container when deployed via auto-deploy.
# Format: (cpu_cores, memory_gib, ephemeral_storage_gib)
# Keep in sync with the Helm values in deployment/helm/values.yaml.
CONTAINER_RESOURCES: dict[str, tuple[int, int, int]] = {
    "gemma": (16, 192, 64),
    "flux": (12, 128, 64),
    "hunyuanframepackf1": (24, 128, 64),
    "hunyuanframepackvae": (4, 32, 16),
    "fantasytalking": (12, 192, 64),
    "realesrgan": (4, 32, 16),
    "kokoro": (2, 8, 16),
    "yolo": (4, 8, 16),
}

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
