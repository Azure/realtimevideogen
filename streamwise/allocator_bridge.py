"""
Bridge between the model provisioner's allocator output and StreamWise pod deployment.

Translates ModelAllocation results (abstract Model enum + GPU counts) into concrete
container deployment parameters compatible with pod_manager.add_pod().
"""

from __future__ import annotations

import sys
import os

# Ensure the directory containing this file is on sys.path so model_provisioner is importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import model_provisioner  # noqa: E402, F401 — adds simulator/ to sys.path

from dataclasses import dataclass
from typing import Optional

from sim_types import GPUType
from sim_types import Model
from sim_types import Result

from auto_model_allocator import AutoModelAllocator
from data_loading import load_latency_data
from model_provisioner.policies import STREAMWISE_POLICY
from workflows import WORKFLOWS


# Mapping from simulator Model enum to concrete container names used by pod_manager.
# Some Model entries map to multiple containers (e.g., OTHERS -> kokoro + yolo).
MODEL_TO_CONTAINERS: dict[Model, list[str]] = {
    Model.GEMMA: ["gemma"],
    Model.FLUX: ["flux"],
    Model.HF: ["hunyuanframepackf1"],
    Model.HF_VAE: ["hunyuanframepackvae"],
    Model.FT: ["fantasytalking"],
    Model.FT_VAE: [],  # FT_VAE is handled within fantasytalking container
    Model.UPSCALER: ["realesrgan"],
    Model.OTHERS: ["kokoro", "yolo"],
}

# Default CPU/memory/storage for each container when deployed via auto-deploy.
# Format: (cpu_cores, memory_gib, ephemeral_storage_gib)
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

# GPU type string used by pod_manager (lowercase)
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


def get_mig_profile(container_name: str, gpu_type: GPUType) -> Optional[str]:
    """Return a MIG profile only when MIG is available and the GPU type supports it."""
    if not MIG_AVAILABLE:
        return None
    if gpu_type not in MIG_CAPABLE_GPU_TYPES:
        return None
    return MIG_CONTAINERS.get(container_name)


# Mapping from StreamWise app name to simulator workflow key
APP_TO_WORKFLOW: dict[str, str] = {
    "streamcast": "podcast",
    "streampersona": "slide",
    "streamchat": "chat",
    "streamshort": "short",
    "streammovie": "movie",
    "streamanimate": "story",
    "streamlecture": "lecture",
    "streamdub": "dubbing",
    "streamedit": "editing",
}


@dataclass
class DeploymentSpec:
    """A single container deployment specification."""
    container_name: str
    cpu: int
    memory_gib: int
    ephemeral_storage_gib: int
    gpu: int
    gpu_type: Optional[str]
    mig_profile: Optional[str]


@dataclass
class DeploymentPlan:
    """Complete deployment plan produced by the auto-allocator."""
    specs: list[DeploymentSpec]
    result: Result
    workflow_name: str
    gpu_budget: dict[str, int]


def _get_data_dir() -> str:
    """Get the path to the simulator data directory."""
    default_path = os.path.join(os.path.dirname(__file__), "simulator", "data")
    return os.getenv("SIMULATOR_DATA_DIR", default_path)


# Reverse mapping from pod gpu_type string to GPUType enum
_POD_STR_TO_GPU_TYPE: dict[str, GPUType] = {v: k for k, v in GPU_TYPE_TO_POD_STR.items()}


def _calc_actual_gpus_per_type(specs: list['DeploymentSpec']) -> dict[GPUType, int]:
    """Calculate actual GPUs needed per GPUType from deployment specs."""
    result: dict[GPUType, int] = {}
    for spec in specs:
        if spec.mig_profile:
            continue
        gpu_type = _POD_STR_TO_GPU_TYPE.get(spec.gpu_type or "")
        if gpu_type is not None:
            result[gpu_type] = result.get(gpu_type, 0) + spec.gpu
    return result


def _trim_specs_for_type(
    specs: list['DeploymentSpec'], gpu_type_str: str, excess: int
) -> list['DeploymentSpec']:
    """
    Remove replicas from specs to reduce GPU usage on a specific type by `excess` GPUs.

    Prefers removing replicas of the most-replicated scalable container (typically
    realesrgan/upscaler) to minimize impact on pipeline throughput.
    """
    # Count replicas per container on this GPU type (only scalable ones)
    from collections import Counter
    type_counts: Counter[str] = Counter()
    for spec in specs:
        if spec.gpu_type == gpu_type_str and spec.gpu > 0 and spec.container_name not in COLOCATED_CONTAINERS:
            type_counts[spec.container_name] += 1

    # Prefer trimming containers with most replicas (least impact per removal)
    trimmed = 0
    result_specs = list(specs)
    for container_name, _count in type_counts.most_common():
        if trimmed >= excess:
            break
        # Remove replicas from the end of the list
        for i in range(len(result_specs) - 1, -1, -1):
            if trimmed >= excess:
                break
            spec = result_specs[i]
            if (spec.container_name == container_name
                    and spec.gpu_type == gpu_type_str
                    and spec.gpu > 0):
                trimmed += spec.gpu
                result_specs.pop(i)
    return result_specs


def get_available_workflows() -> list[str]:
    """Return list of available workflow names for the UI."""
    return list(APP_TO_WORKFLOW.keys())


def get_available_gpu_types() -> list[str]:
    """Return list of available GPU type strings for the UI."""
    return [gpu_type.value for gpu_type in GPUType]


def run_allocator(
    gpu_budget: dict[str, int],
    workflow_name: str,
) -> DeploymentPlan:
    """
    Run the greedy model allocator and return a deployment plan.

    Args:
        gpu_budget: GPU counts keyed by GPU type string (e.g., {"A100": 8, "H100": 0}).
        workflow_name: StreamWise app name (e.g., "streamcast").

    Returns:
        DeploymentPlan with concrete container deployment specs.

    Raises:
        ValueError: If workflow_name or GPU types are invalid.
    """
    # Validate workflow
    workflow_key = APP_TO_WORKFLOW.get(workflow_name)
    if workflow_key is None:
        raise ValueError(
            f"Unknown workflow '{workflow_name}'. "
            f"Available: {list(APP_TO_WORKFLOW.keys())}")

    workflow = WORKFLOWS[workflow_key]

    # Parse GPU budget into GPUType enum
    num_gpus: dict[GPUType, int] = {}
    for gpu_str, count in gpu_budget.items():
        try:
            gpu_type = GPUType(gpu_str)
        except ValueError:
            raise ValueError(
                f"Unknown GPU type '{gpu_str}'. "
                f"Available: {[g.value for g in GPUType]}")
        if count > 0:
            num_gpus[gpu_type] = count

    if not num_gpus or sum(num_gpus.values()) < 8:
        raise ValueError("Total GPU budget must be at least 8 GPUs.")

    # Load latency data and run allocator
    data_dir = _get_data_dir()
    latency_data = load_latency_data(data_dir=data_dir)

    allocator = AutoModelAllocator(
        workflow=workflow,
        latency_data=latency_data,
        policy=STREAMWISE_POLICY,
    )

    result = allocator.allocate(num_gpus=num_gpus, verbose=False)

    # Convert result to deployment specs
    specs = result_to_deployment_specs(result)

    # When MIG is unavailable, deployment specs may use more GPUs per type than the
    # allocator budgeted (e.g., OTHERS allocates 1 GPU but kokoro+yolo each need a
    # full GPU = 2). Detect per-type overflow and trim excess replicas.
    if not MIG_AVAILABLE:
        actual_per_type = _calc_actual_gpus_per_type(specs)
        for gpu_type, budget_count in num_gpus.items():
            actual = actual_per_type.get(gpu_type, 0)
            if actual <= budget_count:
                continue
            # Need to trim (actual - budget_count) GPUs from this type.
            # Remove replicas of the most-replicated scalable container on this type.
            excess = actual - budget_count
            gpu_type_str = GPU_TYPE_TO_POD_STR[gpu_type]
            specs = _trim_specs_for_type(specs, gpu_type_str, excess)

    return DeploymentPlan(
        specs=specs,
        result=result,
        workflow_name=workflow_name,
        gpu_budget=gpu_budget,
    )


def result_to_deployment_specs(result: Result) -> list[DeploymentSpec]:
    """
    Convert an allocator Result into a list of DeploymentSpec objects.

    Each ModelAllocation with replicas > 0 is mapped to one or more container deployments.
    When MIG is unavailable, containers that would normally use MIG slices get 1 full GPU instead.
    """
    specs: list[DeploymentSpec] = []

    for gpu_type, model_dict in result.models.items():
        gpu_type_str = GPU_TYPE_TO_POD_STR[gpu_type]

        for model, allocations in model_dict.items():
            containers = MODEL_TO_CONTAINERS.get(model, [])
            if not containers:
                continue

            for allocation in allocations:
                if allocation.replicas <= 0:
                    continue

                for container_name in containers:
                    resources = CONTAINER_RESOURCES.get(container_name, (4, 16, 16))
                    cpu, memory_gib, ephemeral_storage_gib = resources

                    mig_profile: Optional[str] = None
                    if container_name in COLOCATED_CONTAINERS:
                        # Co-located with parent model; shares GPU on the same server
                        gpu_count = 0
                    elif MIG_AVAILABLE and container_name in MIG_CONTAINERS:
                        mig_profile = MIG_CONTAINERS[container_name]
                        gpu_count = 1
                    elif container_name in MIG_CONTAINERS:
                        # MIG not available: use 1 full GPU instead of a MIG slice
                        gpu_count = 1
                    else:
                        gpu_count = allocation.devices

                    for _ in range(allocation.replicas):
                        specs.append(DeploymentSpec(
                            container_name=container_name,
                            cpu=cpu,
                            memory_gib=memory_gib,
                            ephemeral_storage_gib=ephemeral_storage_gib,
                            gpu=gpu_count,
                            gpu_type=gpu_type_str,
                            mig_profile=mig_profile,
                        ))

    return specs


def deployment_plan_to_json(plan: DeploymentPlan) -> dict:
    """Serialize a DeploymentPlan to a JSON-friendly dict."""
    # Calculate actual GPUs used by the deployment specs (may differ from allocator
    # when MIG is unavailable and services fall back to full GPUs).
    actual_gpus: dict[str, int] = {}
    for spec in plan.specs:
        if spec.mig_profile:
            continue  # MIG slices don't count against full GPU budget
        gpu_type_key = spec.gpu_type or "unknown"
        actual_gpus[gpu_type_key] = actual_gpus.get(gpu_type_key, 0) + spec.gpu

    total_budget = sum(plan.gpu_budget.values())
    total_actual = sum(actual_gpus.values())
    budget_exceeded = total_actual > total_budget

    warnings: list[str] = []
    if budget_exceeded:
        mig_hint = (
            "Enable MIG to fit lightweight services (kokoro, yolo, realesrgan) "
            "on shared GPU slices."
        ) if not MIG_AVAILABLE else ""
        warnings.append(
            f"Deployment requires {total_actual} full GPUs but budget is "
            f"{total_budget}. {mig_hint}"
        )

    return {
        "workflow_name": plan.workflow_name,
        "gpu_budget": plan.gpu_budget,
        "metrics": {
            "total_time_s": round(plan.result.total_time_s, 2),
            "ttff_s": round(plan.result.ttff_s, 2),
            "cost": round(plan.result.cost, 4),
            "gpus_used": {
                gpu_type.value: count
                for gpu_type, count in plan.result.gpus_used.items()
            },
            "actual_gpus_needed": actual_gpus,
            "budget_exceeded": budget_exceeded,
        },
        "warnings": warnings,
        "mig_available": MIG_AVAILABLE,
        "specs": [
            {
                "container_name": spec.container_name,
                "cpu": spec.cpu,
                "memory_gib": spec.memory_gib,
                "ephemeral_storage_gib": spec.ephemeral_storage_gib,
                "gpu": spec.gpu,
                "gpu_type": spec.gpu_type,
                "mig_profile": spec.mig_profile,
            }
            for spec in plan.specs
        ],
    }
