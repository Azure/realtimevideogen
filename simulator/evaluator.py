"""
Evaluate the performance of a given model allocation in terms of time, energy, and cost.
It includes some assertions (e.g., only one instance of Gemma and Flux).
"""
from __future__ import annotations

import math
import logging

from typing import Optional

from constants import NUM_GPUS_PER_SERVER
from constants import TOTAL_INPUT_TOKENS

from sim_types import Result
from sim_types import GPUType
from sim_types import WorkflowConfig
from sim_types import PowerData
from sim_types import LatencyData
from sim_types import Model
from sim_types import ModelAllocation
from sim_types import Policy

from sim_types_json import models_to_json
from sim_types_json import workflow_to_json
from sim_types_json import policy_to_json

SECONDS_IN_HOUR = 60 * 60


def _count_instances(
    models: dict[GPUType, dict[Model, list[ModelAllocation]]],
    model: Model,
) -> int:
    num_instances = 0
    for model_gpus in models.values():
        if model in model_gpus:
            for model_allocation in model_gpus[model]:
                if model_allocation.get_num_gpus() > 0:
                    num_instances += 1
    return num_instances


def _assert_single_instance(
    models: dict[GPUType, dict[Model, list[ModelAllocation]]],
    model: Model,
) -> None:
    num_instances = _count_instances(models, model)
    assert num_instances == 1, f"Expected exactly one instance of {model}, but found {num_instances}"


def _assert_at_least_one_instance(
    models: dict[GPUType, dict[Model, list[ModelAllocation]]],
    model: Model,
) -> None:
    num_instances = _count_instances(models, model)
    assert num_instances > 0, f"Expected at least one instance of {model}, but found {num_instances}"


def _assert_no_instances(
    models: dict[GPUType, dict[Model, list[ModelAllocation]]],
    model: Model,
) -> None:
    num_instances = _count_instances(models, model)
    assert num_instances == 0, f"Expected no instances of {model}, but found {num_instances}"


def evaluate_times(
    models: dict[GPUType, dict[Model, list[ModelAllocation]]],
    latency_data: LatencyData,
    workflow: WorkflowConfig,
    policy: Policy,
    include_models: Optional[list[Model]] = None,
) -> None:
    """
    Compute the total time for the given model allocation and workflow, using the latency data.
    It only evaluates the models specified in "include_models" if provided.
    """
    gpu_types = list(models.keys())

    upscaler_gpus = sum(
        model_alloc.get_num_gpus()
        for gpu_type in gpu_types
        for model_alloc in models.get(gpu_type, {}).get(Model.UPSCALER, [])
    )
    if not policy.use_upscaler:
        assert upscaler_gpus == 0

    for model_name in workflow.models:
        if include_models is not None and model_name not in include_models:
            continue

        # Special conditions: models that require a policy flag
        if model_name == Model.HF_VAE and not policy.is_disaggregated(Model.HF):
            _assert_no_instances(models, Model.HF_VAE)
            continue
        if model_name == Model.FT_VAE and not policy.is_disaggregated(Model.FT):
            _assert_no_instances(models, Model.FT_VAE)
            continue
        if model_name == Model.UPSCALER and not policy.use_upscaler:
            _assert_no_instances(models, Model.UPSCALER)
            continue

        if model_name not in (Model.HF_VAE, Model.FT_VAE):
            # VAE models are optional and can be 0
            _assert_at_least_one_instance(models, model_name)

        if not workflow.is_parallelizable(model_name):
            # Single-instance: no work splitting
            for gpu_type in gpu_types:
                if model_name in models[gpu_type]:
                    for model_alloc in models[gpu_type][model_name]:
                        model_alloc.calculate_time(
                            policy, workflow, latency_data)
                        model_alloc.calculate_time_first(
                            policy, workflow, latency_data)
            continue

        # Parallel: capacity-based work splitting (throughput-weighted)
        capacities: dict[GPUType, list[float]] = {}
        for gpu_type in gpu_types:
            capacities[gpu_type] = []
            if model_name not in models[gpu_type]:
                continue
            for model_alloc in models[gpu_type][model_name]:
                if model_alloc.get_num_gpus() > 0:
                    latency = latency_data[gpu_type][model_name, model_alloc.devices]
                    if model_name in (Model.HF, Model.HF_VAE, Model.FT, Model.FT_VAE):
                        latency *= workflow.get_resolution_scale(policy.use_upscaler)
                    if model_name == Model.GEMMA:
                        latency *= workflow.total_input_tokens / TOTAL_INPUT_TOKENS
                    if latency == 0:
                        capacities[gpu_type].append(0.0)
                    else:
                        capacities[gpu_type].append(model_alloc.replicas / latency)

        total_capacity = sum(sum(c) for c in capacities.values())
        for gpu_type in gpu_types:
            if model_name not in models[gpu_type]:
                continue
            cap_idx = 0
            for model_alloc in models[gpu_type][model_name]:
                if model_alloc.get_num_gpus() > 0:
                    work_pct = capacities[gpu_type][cap_idx] / total_capacity if total_capacity > 0 else 0.0
                    model_alloc.calculate_time(
                        policy, workflow, latency_data,
                        work_pct=work_pct)
                    model_alloc.calculate_time_first(
                        policy, workflow, latency_data)
                    cap_idx += 1


def evaluate_energy(
    models: dict[GPUType, dict[Model, list[ModelAllocation]]],
    power_data: PowerData,
    workflow: WorkflowConfig,
    total_time_s: float = 0.0,
) -> None:
    """
    Calculate total energy (power * time * replicas for each model).
    Need to run after evaluate_times since energy calculation depends on time.
    """
    for gpu_type_allocs in models.values():
        for model_allocation_list in gpu_type_allocs.values():
            for model_allocation in model_allocation_list:
                model_allocation.calculate_energy(
                    workflow,
                    power_data,
                    total_time_s)


def evaluate_cost(
    models: dict[GPUType, dict[Model, list[ModelAllocation]]],
    total_time_s: float,
    policy: Policy,
) -> None:
    """
    Calculate total cost based on GPU hours used.
    Need to run after evaluate_times since cost calculation depends on time.
    """
    for gpu_type_allocs in models.values():
        for model_allocation_list in gpu_type_allocs.values():
            for model in model_allocation_list:
                model.calculate_cost(policy, total_time_s)


_EVALUATOR_CACHE: dict[str, Result] = {}


def evaluate_model_allocation(
    models: dict[GPUType, dict[Model, list[ModelAllocation]]],
    num_gpus: dict[GPUType, int],
    workflow: WorkflowConfig,
    latency_data: LatencyData,
    power_data: Optional[PowerData],
    policy: Policy,
    include_models: Optional[list[Model]] = None,
    cache_results: bool = False,
    round_up_cost_to_server: bool = False,
) -> Result:
    """
    Evaluate the metrics for a given allocation of models to GPUs.
    It only evaluates the models in "include_models" if specified.
    """
    cache_key = None
    if cache_results:
        cache_key = models_to_json(models) + \
            workflow_to_json(workflow) + \
            str(latency_data) + \
            str(power_data) + \
            policy_to_json(policy) + \
            str(include_models)
        if cache_key in _EVALUATOR_CACHE:
            return _EVALUATOR_CACHE[cache_key]

    # Check if setup is possible
    gpus_used = {}
    for gpu_type, model_gpu in models.items():
        gpus_used[gpu_type] = calc_used_gpus({gpu_type: model_gpu})
        assert num_gpus[gpu_type] % NUM_GPUS_PER_SERVER[gpu_type] == 0, \
            f"{gpu_type.value}: {num_gpus[gpu_type]} % {NUM_GPUS_PER_SERVER[gpu_type]}"
        assert gpus_used[gpu_type] <= num_gpus[gpu_type], \
            f"{gpu_type.value}: {gpus_used[gpu_type]} > {num_gpus[gpu_type]}"

    # Assert input models are built correctly
    for gpu_type in models.keys():
        for model_name in models[gpu_type].keys():
            for instance_id in range(len(models[gpu_type][model_name])):
                assert models[gpu_type][model_name][instance_id].model == model_name
                assert models[gpu_type][model_name][instance_id].gpu_type == gpu_type

    # Actual evaluation
    evaluate_times(
        models, latency_data, workflow, policy,
        include_models=include_models,
    )
    time_s = calc_total_time(models)

    first_chunk_time = calc_ttff(models)
    ttff_s = max(
        first_chunk_time,
        time_s - workflow.total_video_seconds
    )

    num_frames = (workflow.total_frames[Model.FT] - workflow.per_subscene_frames[Model.FT])
    tbf_s = (time_s - first_chunk_time) / num_frames
    if tbf_s < 0:
        logging.debug(
            f"Negative TBF: "
            F"{tbf_s:.2f} = ({time_s:.2f} - {first_chunk_time:.2f}) / {num_frames}")
        tbf_s = 0.0

    # Calculate total energy (power * time * replicas for each model)
    energy = 0.0
    if power_data is not None:
        evaluate_energy(models, power_data, workflow, time_s)
        energy = calc_energy(models=models)

    evaluate_cost(models, time_s, policy)
    cost = calc_cost(
        models, time_s, policy,
        round_up_to_server=round_up_cost_to_server)

    ret = Result(
        models=models,
        gpus_used=gpus_used,
        gpus_total=num_gpus,
        total_time_s=time_s,
        first_chunk_time=first_chunk_time,
        ttff_s=ttff_s,
        tbf_s=tbf_s,
        total_energy=energy if power_data else 0.0,
        cost=cost,
    )

    if cache_key is not None:
        _EVALUATOR_CACHE[cache_key] = ret

    return ret


def calc_energy(
    models: dict[GPUType, dict[Model, list[ModelAllocation]]],
) -> float:
    """
    Calculate total energy (power * time * replicas for each model).
    Energy in Watt x seconds (Joules).
    This assumes that evaluate_energy() has been called already.
    """
    energy = 0.0  # Total energy in Watt-seconds (Joules = Watt x second)
    for model_dict in models.values():
        for model_allocations in model_dict.values():
            for model_allocation in model_allocations:
                energy += model_allocation.energy
    return energy


def calc_model_cost(
    models: dict[GPUType, dict[Model, list[ModelAllocation]]],
) -> float:
    """
    Calculate total cost based on GPU hours used.
    This assumes that evaluate_cost() has been called already.
    """
    costs = {}
    for gpu_type, model_dict in models.items():
        costs[gpu_type] = 0.0
        for model_allocations in model_dict.values():
            for model_allocation in model_allocations:
                costs[gpu_type] += model_allocation.cost
    return sum(costs.values())


def calc_cost(
    models: dict[GPUType, dict[Model, list[ModelAllocation]]],
    time_s: float,
    policy: Policy,
    round_up_to_server: bool = True,
) -> float:
    """
    Calculate total cost based on GPU hours used.
    """
    used_gpus = calc_used_gpus_per_type(models)

    # Round up to the nearest server (pack of GPUs) since we pay for whole servers
    if round_up_to_server:
        for gpu_type, used in used_gpus.items():
            used_pack = math.ceil(used / NUM_GPUS_PER_SERVER[gpu_type]) * NUM_GPUS_PER_SERVER[gpu_type]
            used_gpus[gpu_type] = used_pack

    return calc_cost_total(used_gpus, time_s, policy)


def calc_cost_total(
    num_gpus: dict[GPUType, int],
    time_s: float,
    policy: Policy,
) -> float:
    """
    Calculate total cost based on GPU hours used.
    It includes the idle GPUs not assigned to a model.
    """
    cost = 0.0
    for gpu_type, num in num_gpus.items():
        cost += num * (time_s / SECONDS_IN_HOUR) * policy.gpu_cost[gpu_type]
    return cost


def calc_used_gpus_per_type(
    models: dict[GPUType, dict[Model, list[ModelAllocation]]],
) -> dict[GPUType, int]:
    """
    Calculate number of GPUs used per GPU type across all models.
    """
    gpus_used = {}
    for gpu_type, model_gpu in models.items():
        gpus_used[gpu_type] = 0
        for model_allocations in model_gpu.values():
            for model_allocation in model_allocations:
                gpus_used[gpu_type] += model_allocation.get_num_gpus()
    return gpus_used


def calc_used_gpus(
    models: dict[GPUType, dict[Model, list[ModelAllocation]]],
) -> int:
    """
    Calculate total number of GPUs used across all models and GPU types.
    """
    gpus_used = calc_used_gpus_per_type(models)
    return sum(gpus_used.values())


def calc_total_time(
    models: dict[GPUType, dict[Model, list[ModelAllocation]]],
) -> float:
    """
    Calculate total time considering all stages and dependencies.
    This assumes that evaluate_time() has been called already.
    """
    total_time_secs = 0.0
    for model_name in Model:
        model_alloc_times = [
            model_alloc.time
            for gpu_type in GPUType
            if gpu_type in models and model_name in models[gpu_type]
            for model_alloc in models[gpu_type][model_name]
        ]
        model_time = max(model_alloc_times) if model_alloc_times else 0.0
        total_time_secs += model_time
    return total_time_secs


def calc_ttff(
    models: dict[GPUType, dict[Model, list[ModelAllocation]]],
) -> float:
    """
    Calculate time to first frame (chunk).
    It takes the time to first frame (TTFF) for each model.
    This assumes that evaluate_time() has been called already.
    """
    models_time_first: dict[Model, float] = {}
    for model_name in Model:
        times_first = []
        for gpu_type in models.keys():
            if model_name in models[gpu_type]:
                for model_alloc in models[gpu_type][model_name]:
                    if model_alloc.get_num_gpus() > 0:
                        times_first.append(model_alloc.time_first)
        if len(times_first) > 0:
            models_time_first[model_name] = min(times_first)  # The fastest model determines TTFF
    return sum(models_time_first.values())
