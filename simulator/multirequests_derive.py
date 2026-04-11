"""
Derive multi-request parameters from simulation runs.

This module runs the StreamWise simulator at a specified hardware budget and extracts
the model allocation parameters (replicas/GPUs and time-per-request) needed for
multi-request cost estimation.

Usage
-----
Run as a script to regenerate and print the derived constants::

    cd simulator/
    python multirequests_derive.py

Or import and call ``derive_multirequest_params()`` programmatically.

The derived values should be copied into ``multirequests.py`` whenever the latency
data, workflow configuration, or simulator logic changes.
"""
from __future__ import annotations

from dataclasses import replace

from sim_types import GPUType
from sim_types import Model
from sim_types import QualityLevel
from sim_types import Result

from data_loading import load_latency_data
from data_loading import load_power_data
from data_loading import load_adaptive_quality_data

from workflows import PODCAST_WORKFLOW

from policies import STREAMWISE_POLICY

from auto_model_allocator import AutoModelAllocator


# ---------------------------------------------------------------------------
# Default hardware budget for deriving multi-request parameters.
# This corresponds to the Pareto-optimal operating point used in the paper.
# ---------------------------------------------------------------------------
DEFAULT_BUDGET: dict[GPUType, int] = {
    GPUType.A100: 256,
    GPUType.H100: 64,
}

# Hardware budget for the adaptive-quality scenario.
ADAPTIVE_BUDGET: dict[GPUType, int] = {
    GPUType.A100: 256,
    GPUType.H100: 64,
}


def _extract_from_result(
    result: Result,
) -> tuple[dict[GPUType, dict[Model, int]], dict[GPUType, dict[Model, float]]]:
    """Extract init_replicas (GPU counts) and time_per_req from a simulation result.

    Parameters
    ----------
    result:
        The ``Result`` returned by the allocator.

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
    budget: dict[GPUType, int] = DEFAULT_BUDGET,
    data_dir: str = "data/",
) -> tuple[dict[GPUType, dict[Model, int]], dict[GPUType, dict[Model, float]]]:
    """Run the StreamWise simulator and derive multi-request parameters.

    Runs the greedy allocator with ``STREAMWISE_POLICY`` on ``PODCAST_WORKFLOW``
    at the given hardware *budget* and extracts:

    * **init_replicas** — total GPU count per model per GPU type
    * **time_per_req** — total time (seconds) per request per model per GPU type

    These values are the single-request operating point from which multi-request
    scaling is computed in ``multirequests.py``.

    Parameters
    ----------
    budget:
        ``{GPUType: num_gpus}`` hardware budget to allocate.
    data_dir:
        Path to the latency/power CSV data directory.
    """
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
    budget: dict[GPUType, int] = ADAPTIVE_BUDGET,
    data_dir: str = "data/",
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
        ``{gpu_type: {model: {quality: seconds}}}`` — per-quality time per request.
    """
    power_data = load_power_data(data_dir=data_dir)

    # Run simulation at each quality level
    results_by_quality: dict[QualityLevel, Result] = {}
    for quality in [QualityLevel.HIGH, QualityLevel.MEDIUM, QualityLevel.LOW]:
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

    # Use HIGH quality result for init_replicas (worst-case allocation)
    init_replicas_adaptive, _ = _extract_from_result(results_by_quality[QualityLevel.HIGH])

    # Build per-quality time_per_req
    time_per_req_adaptive: dict[GPUType, dict[Model, dict[QualityLevel, float]]] = {}
    for quality, result in results_by_quality.items():
        _, time_per_req_q = _extract_from_result(result)
        for gpu_type in time_per_req_q:
            if gpu_type not in time_per_req_adaptive:
                time_per_req_adaptive[gpu_type] = {}
            for model, t in time_per_req_q[gpu_type].items():
                if model not in time_per_req_adaptive[gpu_type]:
                    time_per_req_adaptive[gpu_type][model] = {}
                time_per_req_adaptive[gpu_type][model][quality] = t

    return init_replicas_adaptive, time_per_req_adaptive


def _format_dict(
    d: dict[GPUType, dict[Model, int | float | dict[QualityLevel, float]]],
    name: str,
    type_hint: str,
) -> str:
    """Format a nested dict as valid Python source code."""
    lines = [f"{name}: {type_hint} = {{"]
    for gpu_type in d:
        lines.append(f"    GPUType.{gpu_type.name}: {{")
        for model, val in d[gpu_type].items():
            if isinstance(val, dict):
                lines.append(f"        Model.{model.name}: {{")
                for k, v in val.items():
                    if isinstance(v, float):
                        lines.append(f"            QualityLevel.{k.name}: {v:.2f},")
                    else:
                        lines.append(f"            QualityLevel.{k.name}: {v},")
                lines.append("        },")
            elif isinstance(val, float):
                lines.append(f"        Model.{model.name}: {val:.2f},")
            else:
                lines.append(f"        Model.{model.name}: {val},")
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 70)
    print(f"Deriving multi-request params at budget: {DEFAULT_BUDGET}")
    print("=" * 70)

    init_replicas, time_per_req = derive_multirequest_params()

    print()
    print("# Single quality (high)")
    print(_format_dict(
        time_per_req, "TIME_PER_REQ",
        "dict[GPUType, dict[Model, float]]"))
    print()
    print(_format_dict(
        init_replicas, "INIT_REPLICAS",
        "dict[GPUType, dict[Model, int]]"))

    print()
    print("=" * 70)
    print(f"Deriving adaptive params at budget: {ADAPTIVE_BUDGET}")
    print("=" * 70)

    init_replicas_a, time_per_req_a = derive_adaptive_params()

    print()
    print("# Adaptive quality")
    print(_format_dict(
        time_per_req_a, "TIME_PER_REQ_ADAPTIVE",
        "dict[GPUType, dict[Model, dict[QualityLevel, float]]]"))
    print()
    print(_format_dict(
        init_replicas_a, "INIT_REPLICAS_ADAPTIVE",
        "dict[GPUType, dict[Model, int]]"))

    # Verify GPU totals
    for label, replicas, budget in [
        ("Single quality", init_replicas, DEFAULT_BUDGET),
        ("Adaptive quality", init_replicas_a, ADAPTIVE_BUDGET),
    ]:
        for gpu_type, expected in budget.items():
            actual = sum(replicas.get(gpu_type, {}).values())
            status = "✓" if actual == expected else "✗"
            print(f"  {status} {label} {gpu_type.value}: {actual}/{expected} GPUs")
