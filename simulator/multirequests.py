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


# Single quality
TIME_PER_REQ: dict[GPUType, dict[Model, float]] = {
    GPUType.A100: {
        Model.FLUX: 9.75,
        Model.HF: 123.08,
        Model.HF_VAE: 114.32,
        Model.FT: 130.82,
        Model.FT_VAE: 52.4,  # TODO proper value
        Model.UPSCALER: 126.83,
        Model.GEMMA: 6.5 + 42 * 0.6,  # First scene + per scene
        Model.OTHERS: 43 * 0.6,  # Kokoro: 43 scenes at 0.6 seconds each
    },
    GPUType.H100: {
        Model.FT: 130.82 / 2.2,  # hunyuanframepackf1_time_per_req
        Model.FT_VAE: 52.4 / 2.2,  # TODO proper value
    }
}


# Optimal point in Pareto Frontier for StreamWise
INIT_REPLICAS: dict[GPUType, dict[Model, int]] = {
    GPUType.A100: {
        Model.OTHERS: 1,
        Model.GEMMA: 1,
        Model.FLUX: 1,
        Model.HF: 12,
        Model.HF_VAE: 3,
        Model.FT: 172,
        Model.FT_VAE: 10,  # TODO proper value
        Model.UPSCALER: 21,
    },
    GPUType.H100: {
        Model.FT: 78,
        Model.FT_VAE: 1,  # TODO proper value
    }
}

# Adaptive quality
# Time per request in seconds
TIME_PER_REQ_ADAPTIVE: dict[GPUType, dict[Model, dict[QualityLevel, float]]] = {
    GPUType.A100: {
        Model.GEMMA: {
            # Same quality for all levels: First + per scene
            QualityLevel.LOW: 2.3 + 42 * 0.176,
            QualityLevel.MEDIUM: 2.3 + 42 * 0.176,
            QualityLevel.HIGH: 2.3 + 42 * 0.176,
        },
        Model.OTHERS: {
            # Kokoro: 42 scenes at 0.6 seconds each
            QualityLevel.LOW: 43 * 0.6,
            QualityLevel.MEDIUM: 43 * 0.6,
            QualityLevel.HIGH: 43 * 0.6,
        },
        Model.FLUX: {
            QualityLevel.LOW: 0.10,
            QualityLevel.MEDIUM: 0.81,
            QualityLevel.HIGH: 0.95,
        },
        Model.HF: {
            QualityLevel.LOW: 3.41,
            QualityLevel.MEDIUM: 8.06,
            QualityLevel.HIGH: 27.1,
        },
        Model.HF_VAE: {
            QualityLevel.LOW: 0.75,
            QualityLevel.MEDIUM: 3.18,
            QualityLevel.HIGH: 52.4,
        },
        Model.UPSCALER: {
            QualityLevel.LOW: 2.01,
            QualityLevel.MEDIUM: 8.30,
            QualityLevel.HIGH: 34.4,
        },
    },
    GPUType.H100: {
        Model.HF_VAE: {
            QualityLevel.LOW: 0.75,
            QualityLevel.MEDIUM: 3.18,
            QualityLevel.HIGH: 52.4,
        },
        Model.FT: {
            QualityLevel.LOW: 8.74,
            QualityLevel.MEDIUM: 39.62,
            QualityLevel.HIGH: 131.14,
        },
        Model.FT_VAE: {  # TODO proper values
            QualityLevel.LOW: 0.75,
            QualityLevel.MEDIUM: 3.18,
            QualityLevel.HIGH: 52.4,
        },
        Model.UPSCALER: {
            QualityLevel.LOW: 2.01,
            QualityLevel.MEDIUM: 8.30,
            QualityLevel.HIGH: 34.4,
        },
    }
}


# This is a point in the Pareto Frontier found via simulation
INIT_REPLICAS_ADAPTIVE: dict[GPUType, dict[Model, int]] = {
    GPUType.A100: {
        Model.OTHERS: 1,  # Kokoro
        Model.GEMMA: 8,
        Model.FLUX: 16,
        Model.HF: 25,
        Model.HF_VAE: 10,
        Model.UPSCALER: 5,
    },
    GPUType.H100: {
        Model.HF_VAE: 1,
        Model.FT: 96,
        Model.FT_VAE: 1,  # Proper values
        Model.UPSCALER: 38,
    }
}

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
    time_ft = total_frames_ft / ft_frames[frames_per_step_idx] * latency_ft * num_steps_ft

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
