from __future__ import annotations

import math
import os
from dataclasses import replace

from sim_types import GPUType
from sim_types import Model
from sim_types import QualityLevel
from sim_types import RESOLUTION_PIXELS
from sim_types import Result
from sim_types import WorkflowConfig
from sim_types import LatencyData

from data_loading import load_latency_data
from data_loading import load_power_data
from data_loading import load_adaptive_quality_data

from workflows import PODCAST_WORKFLOW

from policies import STREAMWISE_POLICY

from auto_model_allocator import AutoModelAllocator


# Queries per minute
QPM_LIST = [0.1, 1, 2, 5, 10, 20, 30, 50, 100]

# Resolve the data directory relative to this file so imports work from any cwd.
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ---------------------------------------------------------------------------
# Hardware budget — the Pareto-optimal operating point used in the paper.
# ---------------------------------------------------------------------------
HARDWARE_BUDGET: dict[GPUType, int] = {
    GPUType.A100: 256,
    GPUType.H100: 64,
}


# ---------------------------------------------------------------------------
# Derivation helpers
# ---------------------------------------------------------------------------

def _extract_from_result(
    result: Result,
) -> tuple[dict[GPUType, dict[Model, int]], dict[GPUType, dict[Model, float]]]:
    """Extract init_replicas (GPU counts) and time_per_req from a simulation result.

    Returns
    -------
    init_replicas:
        ``{gpu_type: {model: total_gpus}}`` — total GPU count allocated to each
        model on each GPU type (i.e. ``devices × replicas`` summed across instances).
    time_per_req:
        ``{gpu_type: {model: seconds}}`` — wall-clock time for the model to process
        one full request (10-min video) given the allocated resources.  When a model
        has multiple instances on the same GPU type, we take the *maximum* time
        (the bottleneck).
    """
    init_replicas: dict[GPUType, dict[Model, int]] = {}
    time_per_req: dict[GPUType, dict[Model, float]] = {}

    for gpu_type, model_allocs in result.models.items():
        init_replicas[gpu_type] = {}
        time_per_req[gpu_type] = {}
        for model, allocs in model_allocs.items():
            total_gpus = sum(a.get_num_gpus() for a in allocs)
            times = [a.time for a in allocs if a.get_num_gpus() > 0]
            if total_gpus > 0:
                init_replicas[gpu_type][model] = total_gpus
                time_per_req[gpu_type][model] = max(times) if times else 0.0

    return init_replicas, time_per_req


def derive_multirequest_params(
    budget: dict[GPUType, int] | None = None,
    data_dir: str = _DATA_DIR,
) -> tuple[dict[GPUType, dict[Model, int]], dict[GPUType, dict[Model, float]]]:
    """Run the StreamWise simulator and derive multi-request parameters.

    Runs the greedy allocator with ``STREAMWISE_POLICY`` on ``PODCAST_WORKFLOW``
    at the given hardware *budget* and extracts:

    * **init_replicas** — total GPU count per model per GPU type
    * **time_per_req** — total time (seconds) per request per model per GPU type

    Parameters
    ----------
    budget:
        ``{GPUType: num_gpus}`` hardware budget to allocate.
        Defaults to ``HARDWARE_BUDGET`` when ``None``.
    data_dir:
        Path to the latency/power CSV data directory.
    """
    if budget is None:
        budget = dict(HARDWARE_BUDGET)
    latency_data = load_latency_data(data_dir=data_dir)
    power_data = load_power_data(data_dir=data_dir)

    allocator = AutoModelAllocator(
        workflow=PODCAST_WORKFLOW,
        latency_data=latency_data,
        power_data=power_data,
        policy=STREAMWISE_POLICY,
    )
    result = allocator.allocate(
        num_gpus=budget,
        verbose=False,
    )

    return _extract_from_result(result)


def derive_adaptive_params(
    budget: dict[GPUType, int] | None = None,
    data_dir: str = _DATA_DIR,
) -> tuple[
    dict[GPUType, dict[Model, int]],
    dict[GPUType, dict[Model, dict[QualityLevel, float]]],
]:
    """Run the simulator at each quality level and derive adaptive parameters.

    Returns
    -------
    init_replicas_adaptive:
        ``{gpu_type: {model: total_gpus}}`` from the HIGH-quality simulation run
        (the worst-case / most-demanding quality level sets the base allocation).
    time_per_req_adaptive:
        ``{gpu_type: {model: {quality: seconds}}}`` — per-quality time per request,
        normalized against the HIGH-quality allocation so every
        ``(gpu_type, model)`` present in ``init_replicas_adaptive`` has a timing
        entry for every quality level.
    """
    if budget is None:
        budget = dict(HARDWARE_BUDGET)

    power_data = load_power_data(data_dir=data_dir)

    qualities = [QualityLevel.HIGH, QualityLevel.MEDIUM, QualityLevel.LOW]
    results_by_quality: dict[QualityLevel, Result] = {}
    for quality in qualities:
        policy = replace(STREAMWISE_POLICY)
        policy.name = f"{STREAMWISE_POLICY.name} {quality.value}"

        latency_data = load_adaptive_quality_data(
            data_dir=data_dir,
            level=quality,
        )

        allocator = AutoModelAllocator(
            workflow=PODCAST_WORKFLOW,
            latency_data=latency_data,
            power_data=power_data,
            policy=policy,
        )
        result = allocator.allocate(
            num_gpus=budget,
            verbose=False,
        )
        results_by_quality[quality] = result

    init_replicas_adaptive, time_per_req_high = _extract_from_result(
        results_by_quality[QualityLevel.HIGH],
    )

    time_per_req_by_quality: dict[QualityLevel, dict[GPUType, dict[Model, float]]] = {}
    for quality, result in results_by_quality.items():
        _, time_per_req_q = _extract_from_result(result)
        time_per_req_by_quality[quality] = time_per_req_q

    time_per_req_adaptive: dict[GPUType, dict[Model, dict[QualityLevel, float]]] = {}
    for gpu_type, models in init_replicas_adaptive.items():
        time_per_req_adaptive[gpu_type] = {}
        for model in models:
            high_time = time_per_req_high[gpu_type][model]
            quality_times: dict[QualityLevel, float] = {}
            for quality in qualities:
                quality_times[quality] = (
                    time_per_req_by_quality
                    .get(quality, {})
                    .get(gpu_type, {})
                    .get(model, high_time)
                )
            time_per_req_adaptive[gpu_type][model] = quality_times

    return init_replicas_adaptive, time_per_req_adaptive


# ---------------------------------------------------------------------------
# Derived constants — computed by running the simulator at HARDWARE_BUDGET.
#
# TIME_PER_REQ / INIT_REPLICAS: single (HIGH) quality operating point.
# TIME_PER_REQ_ADAPTIVE / INIT_REPLICAS_ADAPTIVE: per-quality-level values.
# ---------------------------------------------------------------------------
INIT_REPLICAS, TIME_PER_REQ = derive_multirequest_params(budget=dict(HARDWARE_BUDGET))
INIT_REPLICAS_ADAPTIVE, TIME_PER_REQ_ADAPTIVE = derive_adaptive_params(budget=dict(HARDWARE_BUDGET))

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
