from __future__ import annotations

import math

from sim_types import GPUType
from sim_types import Model
from sim_types import QualityLevel
from sim_types import RESOLUTION_PIXELS
from sim_types import WorkflowConfig
from sim_types import LatencyData


# Queries per minute
QPM_LIST = [0.1, 1, 2, 5, 10, 20, 30, 50, 100]


# ---------------------------------------------------------------------------
# Hardware budget used to derive the constants below.
# To regenerate, run:  python multirequests_derive.py
# ---------------------------------------------------------------------------
HARDWARE_BUDGET: dict[GPUType, int] = {
    GPUType.A100: 256,
    GPUType.H100: 64,
}

# Derived by running `python multirequests_derive.py`

# ---------------------------------------------------------------------------
# TIME_PER_REQ — wall-clock seconds for each model to process one full request
# (10-min podcast video) at the hardware budget above.
#
# Derived by running the StreamWise greedy allocator (STREAMWISE_POLICY) on
# PODCAST_WORKFLOW at 256 A100 + 64 H100 GPUs.  Each value is the bottleneck
# time across all instances of that model on that GPU type.
# ---------------------------------------------------------------------------
TIME_PER_REQ: dict[GPUType, dict[Model, float]] = {
    GPUType.A100: {
        Model.GEMMA: 8.57,
        Model.FLUX: 1.65,
        Model.HF_VAE: 21.80,
        Model.FT: 246.97,
        Model.UPSCALER: 49.40,
        Model.OTHERS: 25.80,
    },
    GPUType.H100: {
        Model.HF: 56.96,
        Model.HF_VAE: 21.80,
        Model.FT: 250.70,
        Model.UPSCALER: 49.40,
    },
}


# ---------------------------------------------------------------------------
# INIT_REPLICAS — total GPU count allocated to each model on each GPU type at
# the Pareto-optimal operating point (256 A100 + 64 H100).
#
# These are NOT literal replica counts; each entry represents
# ``devices_per_replica × num_replicas`` summed across all instances.
# For multi-request scaling, each unit is treated as one GPU for cost purposes.
# ---------------------------------------------------------------------------
INIT_REPLICAS: dict[GPUType, dict[Model, int]] = {
    GPUType.A100: {
        Model.GEMMA: 8,
        Model.FLUX: 8,
        Model.HF_VAE: 7,
        Model.FT: 192,
        Model.UPSCALER: 40,
        Model.OTHERS: 1,
    },
    GPUType.H100: {
        Model.HF: 14,
        Model.HF_VAE: 4,
        Model.FT: 38,
        Model.UPSCALER: 8,
    },
}


# ---------------------------------------------------------------------------
# TIME_PER_REQ_ADAPTIVE — per-quality-level time per request (seconds).
#
# Derived by running the StreamWise allocator at each quality level (HIGH,
# MEDIUM, LOW) on the same hardware budget.  Quality scaling affects latency
# through resolution-dependent models (HF, FT, UPSCALER, FLUX).
# ---------------------------------------------------------------------------
TIME_PER_REQ_ADAPTIVE: dict[GPUType, dict[Model, dict[QualityLevel, float]]] = {
    GPUType.A100: {
        Model.GEMMA: {
            QualityLevel.HIGH: 8.57,
            QualityLevel.MEDIUM: 8.57,
            QualityLevel.LOW: 8.57,
        },
        Model.FLUX: {
            QualityLevel.HIGH: 1.65,
            QualityLevel.MEDIUM: 0.41,
            QualityLevel.LOW: 0.10,
        },
        Model.HF_VAE: {
            QualityLevel.HIGH: 21.80,
            QualityLevel.MEDIUM: 2.79,
            QualityLevel.LOW: 1.25,
        },
        Model.HF: {
            QualityLevel.MEDIUM: 10.01,
            QualityLevel.LOW: 4.24,
        },
        Model.FT: {
            QualityLevel.HIGH: 246.97,
            QualityLevel.MEDIUM: 57.57,
            QualityLevel.LOW: 22.99,
        },
        Model.UPSCALER: {
            QualityLevel.HIGH: 49.40,
            QualityLevel.MEDIUM: 8.50,
            QualityLevel.LOW: 3.50,
        },
        Model.OTHERS: {
            QualityLevel.HIGH: 25.80,
            QualityLevel.MEDIUM: 25.80,
            QualityLevel.LOW: 25.80,
        },
    },
    GPUType.H100: {
        Model.HF: {
            QualityLevel.HIGH: 56.96,
            QualityLevel.MEDIUM: 9.96,
            QualityLevel.LOW: 4.26,
        },
        Model.HF_VAE: {
            QualityLevel.HIGH: 21.80,
            QualityLevel.MEDIUM: 2.79,
            QualityLevel.LOW: 1.25,
        },
        Model.FT: {
            QualityLevel.HIGH: 250.70,
            QualityLevel.MEDIUM: 57.41,
            QualityLevel.LOW: 23.14,
        },
        Model.UPSCALER: {
            QualityLevel.HIGH: 49.40,
            QualityLevel.MEDIUM: 8.52,
            QualityLevel.LOW: 3.50,
        },
    },
}


# ---------------------------------------------------------------------------
# INIT_REPLICAS_ADAPTIVE — GPU allocation for the adaptive-quality scenario.
#
# Uses the HIGH-quality allocation as the base (worst-case demand).  Same
# hardware budget as INIT_REPLICAS since the adaptive policy dynamically
# adjusts quality rather than hardware.
# ---------------------------------------------------------------------------
INIT_REPLICAS_ADAPTIVE: dict[GPUType, dict[Model, int]] = {
    GPUType.A100: {
        Model.GEMMA: 8,
        Model.FLUX: 8,
        Model.HF_VAE: 7,
        Model.FT: 192,
        Model.UPSCALER: 40,
        Model.OTHERS: 1,
    },
    GPUType.H100: {
        Model.HF: 14,
        Model.HF_VAE: 4,
        Model.FT: 38,
        Model.UPSCALER: 8,
    },
}

# Quality distribution portions for adaptive quality cost aggregation.
# These represent relative weights for each quality level in the adaptive mix.
QUALITY_PORTIONS = {
    QualityLevel.LOW: 112,
    QualityLevel.MEDIUM: 305,
    QualityLevel.HIGH: 13383,
}


# Initial setup based on minimal 8 A100 configuration
# 1 for Kokoro, 1 for Gemma, 1 for Flux, 1 for HF+VAE (co-located), 4 for FT
INIT_REPLICAS_BASELINE: dict[GPUType, dict[Model, int]] = {
    GPUType.A100: {
        Model.OTHERS: 1,
        Model.GEMMA: 1,
        Model.FLUX: 1,
        Model.HF: 1,  # HF and VAE co-located on same GPU
        Model.FT: 4,
    },
    GPUType.H100: {
        # Empty for baseline
    }
}


def get_time_per_request_baseline(
    workflow_config: WorkflowConfig,
    latency_data: LatencyData,
    init_replicas: dict[GPUType, dict[Model, int]] = INIT_REPLICAS_BASELINE,
) -> dict[GPUType, dict[Model, float]]:
    """Get time per request for baseline (single quality)."""
    # Calculate time per request for each component (using baseline latencies)
    # Using A100 latencies from the Naive Baseline section
    # NOTE: In naive baseline, HF and VAE are co-located and run sequentially (not concurrently)

    total_scenes = workflow_config.total_scenes
    num_steps_flux = workflow_config.num_steps[Model.FLUX]

    total_frames_hf = workflow_config.total_frames[Model.HF]
    num_steps_hf = workflow_config.num_steps[Model.HF]
    hf_frames = workflow_config.hf_frames
    frames_per_step_idx = workflow_config.frames_per_step_idx
    total_frames_ft = workflow_config.total_frames[Model.FT]
    num_steps_ft = workflow_config.num_steps[Model.FT]
    ft_frames = workflow_config.ft_frames

    num_pixels_high = RESOLUTION_PIXELS[QualityLevel.HIGH]
    num_pixels_medium = RESOLUTION_PIXELS[QualityLevel.MEDIUM]

    # Latencies
    latency_hf_mapping_a100 = {
        k: v * num_pixels_high / num_pixels_medium
        for k, v in latency_data.gpus[GPUType.A100].hf.items()
    }
    latency_hf_vae_a100 = latency_data.gpus[GPUType.A100][Model.HF_VAE, 1] * num_pixels_high / num_pixels_medium
    latency_ft_mapping_a100 = {
        k: v * num_pixels_high / num_pixels_medium
        for k, v in latency_data.gpus[GPUType.A100].ft.items()
    }
    latency_ft_vae_a100 = latency_data.gpus[GPUType.A100][Model.FT_VAE, 1] * num_pixels_high / num_pixels_medium

    num_gemma_gpus = 1
    num_flux_gpus = 1
    num_hf_gpus = 1
    num_ft_gpus = 1
    num_ft_replicas = init_replicas[GPUType.A100][Model.FT]

    latency_gemma_first = latency_data.gpus[GPUType.A100].gemma_first_scene[num_gemma_gpus]
    latency_gemma_per = latency_data.gpus[GPUType.A100].gemma_per_scene[num_gemma_gpus]
    latency_flux = latency_data.gpus[GPUType.A100][Model.FLUX, num_flux_gpus]
    latency_hf = latency_hf_mapping_a100[num_hf_gpus]
    time_hf = (
        (total_frames_hf / hf_frames[frames_per_step_idx] * latency_hf * num_steps_hf)
        + (total_frames_hf / hf_frames[frames_per_step_idx] * latency_hf_vae_a100)
    )
    latency_ft = latency_ft_mapping_a100[num_ft_gpus] / num_ft_replicas
    time_ft = (
        (total_frames_ft / ft_frames[frames_per_step_idx] * latency_ft * num_steps_ft)
        + (total_frames_ft / ft_frames[frames_per_step_idx] * latency_ft_vae_a100)
    )

    return {
        GPUType.A100: {
            Model.OTHERS: total_scenes * 0.6,
            Model.GEMMA: latency_gemma_first + latency_gemma_per * (total_scenes - 1),
            Model.FLUX: latency_flux * num_steps_flux,
            Model.HF: time_hf,
            Model.FT: time_ft,
        },
        GPUType.H100: {
            # None in baseline
        },
    }


def aggregate_time_per_request_by_quality(
    time_per_req: dict[GPUType, dict[Model, float | dict[QualityLevel, float]]],
    quality_portions: dict[QualityLevel, int],
) -> dict[GPUType, dict[Model, float]]:
    """Aggregate time per request metrics."""
    ret: dict[GPUType, dict[Model, float]] = {}

    total_portions = sum(quality_portions.values())

    for gpu_type in time_per_req.keys():
        ret[gpu_type] = {}
        for model in time_per_req[gpu_type].keys():
            val = time_per_req[gpu_type][model]
            if isinstance(val, float):
                time_val: float = val
                ret[gpu_type][model] = time_val
            elif isinstance(val, dict):
                dict_quality: dict[QualityLevel, float] = val
                agg_val = 0.0
                for quality_level in dict_quality.keys():
                    fraction = quality_portions[quality_level] / total_portions
                    agg_val += dict_quality[quality_level] * fraction
                ret[gpu_type][model] = agg_val
            else:
                raise ValueError("Invalid time_per_req format")
    return ret


def required_replicas(
    name: str,
    video_seconds: float,
    ttff: float,
    per_sec: float,
    partition: str,
    req_per_min: float,
) -> float:
    """Calculate required replicas for a model."""
    ttff_total = 0.0
    per_sec_total = 0.0

    if partition == "scenes":
        ttff_total = ttff
        per_sec_total = (video_seconds * per_sec)
    elif partition == "frames":
        if name == "hf_vae":
            ttff_total = ttff
            per_sec_total = (video_seconds * per_sec)
        if name == "upscaler":
            ttff_total = ttff
            per_sec_total = (video_seconds * per_sec)
    elif partition == "subscenes":
        ttff_total = ttff
        per_sec_total = (video_seconds * per_sec)
    else:
        ttff_total = ttff
        per_sec_total = video_seconds * per_sec

    total_time_per_request = ttff_total + per_sec_total
    total_time_per_minute = total_time_per_request * req_per_min
    return total_time_per_minute / 60.0


def get_replicas(
    video_seconds: float = 10 * 60,  # 10 minutes video
    requests_per_minute: float = 0.5,
    time_per_req: dict[GPUType, dict[Model, float]] = TIME_PER_REQ,
    init_replicas: dict[GPUType, dict[Model, int]] = INIT_REPLICAS,
    qpms: list[float] = QPM_LIST,
) -> dict[GPUType, dict[Model, list[int]]]:
    """Get required replicas for different QPM levels."""
    assert video_seconds > 0
    assert requests_per_minute > 0
    assert 0 < len(time_per_req) == len(init_replicas)
    assert len(qpms) > 0

    video_minutes = video_seconds / 60

    capacity: dict[GPUType, dict[Model, float]] = {}
    for gpu_type in time_per_req.keys():
        capacity[gpu_type] = {}
        for model in time_per_req[gpu_type].keys():
            capacity[gpu_type][model] = video_seconds / time_per_req[gpu_type][model]

    replicas: dict[GPUType, dict[Model, list[int]]] = {}
    for qpm in qpms:
        arrival = qpm * video_minutes
        for gpu_type in init_replicas.keys():
            if gpu_type not in replicas:
                replicas[gpu_type] = {}
            for model in init_replicas[gpu_type].keys():
                if model not in replicas[gpu_type]:
                    replicas[gpu_type][model] = []
                # num_replicas = arrival * time_per_req[gpu_type][model]
                scale_factor = max(1, arrival / capacity[gpu_type][model])
                num_replicas = math.ceil(init_replicas[gpu_type][model] * scale_factor)
                replicas[gpu_type][model].append(num_replicas)

    return replicas


def get_costs(
    replicas: dict[GPUType, dict[Model, list[int]]],
    gpu_costs: dict[GPUType, float],
) -> dict[GPUType, dict[Model, list[float]]]:
    costs: dict[GPUType, dict[Model, list[float]]] = {}
    for gpu_type in replicas.keys():
        costs[gpu_type] = {}
        for model in replicas[gpu_type].keys():
            costs[gpu_type][model] = [
                replica * gpu_costs[gpu_type]
                for replica in replicas[gpu_type][model]
            ]
    return costs


def get_total_costs(
    costs: dict[GPUType, dict[Model, list[float]]],
    qpms: list[float] = QPM_LIST,
) -> list[float]:
    total_costs = [
        sum(
            costs[gpu_type][model][i]
            for gpu_type in costs.keys()
            for model in costs[gpu_type].keys()
        )
        for i in range(len(qpms))
    ]
    return total_costs
