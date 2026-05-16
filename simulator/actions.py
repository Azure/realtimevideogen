"""
Actions for scaling models for the greedy allocator.
"""

from __future__ import annotations

import random

from collections import Counter

from copy import deepcopy

from typing import Optional

from constants import DEVICE_OPTIONS
from constants import SINGLE_INSTANCE_MODELS
from constants import SINGLE_DEVICE_MODELS

from sim_types import Action
from sim_types import ActionName
from sim_types import Model
from sim_types import ModelAllocation
from sim_types import GPUType
from sim_types import WorkflowConfig
from sim_types import LatencyData
from sim_types import PowerData
from sim_types import Objective
from sim_types import Policy

from model_provisioner.policies import STREAMWISE_POLICY

from models import get_model_allocation

from evaluator import evaluate_model_allocation
from evaluator import calc_used_gpus


def _is_single_instance(
    model_name: Model,
    workflow: Optional[WorkflowConfig] = None,
) -> bool:
    """Check if a model is single-instance, considering workflow parallelism settings."""
    if model_name not in SINGLE_INSTANCE_MODELS:
        return False
    if workflow is not None and workflow.is_parallelizable(model_name):
        return False
    return True


def find_next_devices(
    device_options: list[int],
    num_devices: int,
    num_replicas: int,
    remaining_devices: int,
    max_num_devices: Optional[int] = None,
) -> Optional[int]:
    """
    Find the next device combination.
    For example, with device options [2, 4, 8, 16, 40], current devices 8, 1 replica, we get 16.
    """
    if num_replicas == 0:
        # means we haven't allocated any replicas yet so start from smallest device option
        return device_options[0] if device_options[0] <= remaining_devices else None

    for device_option in device_options:
        # if device_option > num_devices and device_option <= remaining_devices + num_devices:
        if (
            device_option > num_devices
            and (device_option - num_devices) * num_replicas <= remaining_devices
            and (max_num_devices is None or device_option <= max_num_devices)
        ):
            return device_option
    return None


def choose_action(
    actions: list[Action],
    objective: Objective,
    switch_objective: bool = False,
) -> Optional[Action]:
    """Schedule requests."""
    if not actions:
        return None

    if objective == Objective.TIME_COST:
        # return min(actions, key=lambda a: a.time)
        return min(
            actions,
            key=lambda a: (
                a.time_cost(),
                a.time,
            ),
        )
    if objective == Objective.TIME_COST:
        return min(
            actions,
            key=lambda a: (
                a.time_cost(),
                a.time,
            ),
        )
    if objective == Objective.TTFF_COST:
        return min(
            actions,
            key=lambda a: (
                a.ttff_cost(),
                a.ttff,
            ),
        )
    if objective == Objective.FIFO:
        # return min(actions, key=lambda a: a.arrival_time_s)
        return min(actions, key=lambda a: a.get_order())
    if objective == Objective.TIME:
        return min(actions, key=lambda a: a.time)
    if objective == Objective.TTFF:
        return min(actions, key=lambda a: a.ttff)
    if objective == Objective.COST:
        return min(actions, key=lambda a: a.cost)
    if objective == Objective.ENERGY:
        return min(actions, key=lambda a: a.energy)
    if objective == Objective.TIME_ENERGY:
        return min(actions, key=lambda a: a.time_energy())
    if objective == Objective.ENERGY_COST:
        return min(actions, key=lambda a: a.energy_cost())
    if objective == Objective.RANDOM:
        # randomly pick an improvement to simulate naive allocation
        return random.choice(actions)
    if objective == Objective.TTFF_THEN_TIME:
        if switch_objective:
            return min(actions, key=lambda a: a.time)
        else:
            return min(actions, key=lambda a: a.ttff)
    if objective == Objective.NONE:
        return None
    raise ValueError(f"Cannot recognize objective {objective}")


def apply_action(
    action: Action,
    models: dict[GPUType, dict[Model, list[ModelAllocation]]],
) -> dict[GPUType, dict[Model, list[ModelAllocation]]]:
    """Apply the chosen action to the models and update remaining devices."""

    for gpu_type in action.models.keys():
        if gpu_type not in models:
            raise ValueError(f"Cannot find gpu type {gpu_type} in {models.keys()}")
        for model in action.models[gpu_type].keys():
            if model not in models[gpu_type]:
                raise ValueError(f"Cannot find model {model} in {models[gpu_type].keys()}")
            allocs_to_remove = []
            for alloc_id in range(len(action.models[gpu_type][model])):
                # check if devices and replicas are non-negative
                num_devices = action.models[gpu_type][model][alloc_id].devices
                if num_devices < 0:
                    raise ValueError(f"Action devices {num_devices} must be >= 0")
                if action.models[gpu_type][model][alloc_id].replicas <= 0:
                    # remove that instance if replicas is 0 or negative
                    allocs_to_remove.append(alloc_id)
            for alloc_id in reversed(allocs_to_remove):
                del action.models[gpu_type][model][alloc_id]

    return action.models


def gen_actions(
    workflow: WorkflowConfig,
    num_gpus: dict[GPUType, int],
    latency_data: LatencyData,
    power_data: Optional[PowerData] = None,
    models: dict[GPUType, dict[Model, list[ModelAllocation]]] = {},
    policy: Policy = STREAMWISE_POLICY,
    allow_removal: bool = False,
    allow_merging: bool = False,
    look_ahead_replicas: int = 3,
) -> list[Action]:
    actions: list[Action] = []

    # Extract GPU types from models
    gpu_types = list(models.keys())
    assert len(gpu_types) == len(num_gpus), \
        f"Number of GPU types in models {len(gpu_types)} must match num_gpus {len(num_gpus)}"

    remaining_gpus = {}
    for gpu_type in num_gpus.keys():
        remaining_gpus[gpu_type] = num_gpus[gpu_type] - calc_used_gpus({gpu_type: models[gpu_type]})

    # Option 1: Provision more by increasing <devices, replicas> for each model allocation
    for model in Model:
        if model not in workflow.models:
            continue
        for gpu_type in gpu_types:
            for alloc_id in range(len(models[gpu_type][model])):
                actions.extend(_gen_add_device_replica_actions(
                    models=models,
                    num_gpus=num_gpus,
                    remaining_gpus=remaining_gpus[gpu_type],
                    gpu_type=gpu_type,
                    model_name=model,
                    allocation_id=alloc_id,
                    workflow=workflow,
                    policy=policy,
                    latency_data=latency_data,
                    power_data=power_data,
                    look_ahead_replicas=look_ahead_replicas,
                ))

    # Option 2: Add a model instance of <devices, replicas>
    for model in Model:
        if model not in workflow.models:
            continue
        for gpu_type in gpu_types:
            actions.extend(_gen_add_instance(
                models=models,
                num_gpus=num_gpus,
                remaining_gpus=remaining_gpus[gpu_type],
                gpu_type=gpu_type,
                model_name=model,
                workflow=workflow,
                policy=policy,
                latency_data=latency_data,
                power_data=power_data,
                look_ahead_replicas=look_ahead_replicas,
            ))

    if allow_removal:
        # Option 3: Remove replicas for each model allocation
        for model in Model:
            if model not in workflow.models:
                continue
            for gpu_type in gpu_types:
                model_instances = models[gpu_type][model]
                for alloc_id in range(len(model_instances)):
                    action = _gen_remove_replica_action(
                        models=models,
                        num_gpus=num_gpus,
                        gpu_type=gpu_type,
                        model_name=model,
                        allocation_id=alloc_id,
                        workflow=workflow,
                        policy=policy,
                        latency_data=latency_data,
                        power_data=power_data,
                    )
                    if action:
                        actions.append(action)

    if allow_merging:
        # Option 4: Merge across model allocations
        for model in Model:
            if model not in workflow.models:
                continue
            for gpu_type in gpu_types:
                actions.extend(_gen_merge_replicas_actions(
                    models=models,
                    num_gpus=num_gpus,
                    gpu_type=gpu_type,
                    model_name=model,
                    workflow=workflow,
                    policy=policy,
                    latency_data=latency_data,
                    power_data=power_data,
                ))

    return actions


def _get_min_device_combinations(
    num_gpus: int,
    model: Model,
) -> list[tuple[int, int]]:
    """
    Get the minimum device combinations for a given number of GPUs and model.
    [(device_count, num_replicas), ...]
    For example, for 64, it would return [(40, 1), (16, 1)].
    """
    remaining = num_gpus
    result: list[int] = []
    for size in sorted(DEVICE_OPTIONS[model], reverse=True):
        while remaining >= size:
            result.append(size)
            remaining -= size
    if remaining > 0:
        raise ValueError(f"Cannot exactly decompose {num_gpus} with DEVICE_OPTIONS")
    counts = Counter(result)
    return sorted(counts.items(), reverse=True)  # Sort by device count descending


def _get_large_instance_many_small_combinations(
    num_gpus: int,
    model: Model,
) -> list[tuple[int, int]]:
    """
    Get the largest instance possible and then split the rest into 1 GPU instances.
    For example, for 64, it would return [(40, 1), (1, 16)].
    """
    assert num_gpus > 0
    assert model in DEVICE_OPTIONS
    assert DEVICE_OPTIONS[model][0] == 1  # must have 1 GPU option to use this function

    remaining_gpus = num_gpus
    result: list[tuple[int, int]] = []
    for size in sorted(DEVICE_OPTIONS[model], reverse=True):
        if remaining_gpus >= size:
            result = [(size, 1)]
            remaining_gpus -= size
            break
    if remaining_gpus > 0:
        result.append((1, remaining_gpus))
    return result


def _gen_add_device_replica_actions(
    models: dict[GPUType, dict[Model, list[ModelAllocation]]],
    num_gpus: dict[GPUType, int],
    remaining_gpus: int,
    gpu_type: GPUType,
    model_name: Model,
    allocation_id: int,
    workflow: WorkflowConfig,
    policy: Policy,
    latency_data: LatencyData,
    power_data: Optional[PowerData] = None,
    look_ahead_replicas: int = 3,
) -> list[Action]:
    """
    Generate actions that explore all valid (replicas, devices) provisioning
    options for a given model allocation, using the remaining GPUs.

    From the current replicas * devices, find the next options by distributing the remaining devices.
    For example, if currently 2 replicas at parallelism 4 with 4 remaining devices, options include:
      - 3 replicas, 4 devices  (uses 12 total, 4 more than current 8)
      - 1 replica, 10 devices  (uses 10 total, 2 more than current 8)
      - etc.
    """
    actions: list[Action] = []

    if model_name in SINGLE_DEVICE_MODELS and _is_single_instance(model_name, workflow):
        return actions  # No scaling possible

    alloc = models[gpu_type][model_name][allocation_id]
    current_total = alloc.devices * max(alloc.replicas, 0)
    current_replicas = alloc.replicas
    total_available = current_total + remaining_gpus

    max_num_devices = latency_data[gpu_type].get_max_parallelism(model_name)
    max_replicas = alloc.get_max_replicas(workflow)
    is_single_instance = _is_single_instance(model_name, workflow)
    is_single_device = model_name in SINGLE_DEVICE_MODELS

    seen: set[tuple[int, int]] = set()
    seen.add((max(alloc.replicas, 0), alloc.devices))  # skip current config

    for new_devices in DEVICE_OPTIONS[model_name]:
        if new_devices > max_num_devices:
            continue  # Exceeds max parallelism from latency data
        if is_single_device and new_devices > 1:
            continue  # Model only supports single device
        if (model_name, new_devices) not in latency_data[gpu_type]:
            continue  # No latency data for this device count

        # Determine the range of replicas possible with this device count
        if is_single_instance:
            replica_candidates = [1]
        else:
            max_r = min(max_replicas, total_available // new_devices) if new_devices > 0 else 0
            # limit max replicas to original replicas + X to avoid too many combinations
            max_r = min(max_r, current_replicas + look_ahead_replicas)
            replica_candidates = list(range(1, max_r + 1))

        for new_replicas in replica_candidates:
            new_total = new_replicas * new_devices
            if new_total <= current_total:
                continue  # Must be an increase
            if new_total > total_available:
                continue  # Not enough GPUs
            if (new_replicas, new_devices) in seen:
                continue
            seen.add((new_replicas, new_devices))

            try:
                new_models = deepcopy(models)
                new_models[gpu_type][model_name][allocation_id] = get_model_allocation(
                    model=model_name,
                    gpu_type=gpu_type,
                    devices=new_devices,
                    replicas=new_replicas,
                )
                action_result = evaluate_model_allocation(
                    models=new_models,
                    num_gpus=num_gpus,
                    workflow=workflow,
                    latency_data=latency_data,
                    power_data=power_data,
                    policy=policy,
                    include_models=[model_name],
                )
                actions.append(Action(
                    name=ActionName.ADD_DEVICE_REPLICA,
                    model=model_name,
                    gpu_type=gpu_type,
                    models=new_models,
                    action_result=action_result,
                    arrival_time_s=alloc.time,
                ))
            except Exception:
                pass  # Invalid configuration, skip

    return actions


def _gen_add_device_action(
    models: dict[GPUType, dict[Model, list[ModelAllocation]]],
    num_gpus: dict[GPUType, int],
    remaining_gpus: int,
    gpu_type: GPUType,
    model_name: Model,
    allocation_id: int,
    workflow: WorkflowConfig,
    policy: Policy,
    latency_data: LatencyData,
    power_data: Optional[PowerData] = None,
) -> Optional[Action]:
    """
    Action to add devices (increase parallelism) for a specific model allocation.
    """
    action: Optional[Action] = None

    if model_name in SINGLE_DEVICE_MODELS:
        return action  # These models only run on a single GPU, so we don't add more devices

    alloc = models[gpu_type][model_name][allocation_id]

    max_num_devices = latency_data[gpu_type].get_max_parallelism(model_name)
    next_num_devices = find_next_devices(
        DEVICE_OPTIONS[model_name],
        num_devices=alloc.devices,
        num_replicas=alloc.replicas,
        remaining_devices=remaining_gpus,
        max_num_devices=max_num_devices)

    if not next_num_devices:
        return action  # No valid next device option, skip
    if (model_name, next_num_devices) not in latency_data[gpu_type]:
        return action  # No latency data for this device option, skip

    new_models = deepcopy(models)
    new_models[gpu_type][model_name][allocation_id] = get_model_allocation(
        model=model_name,
        gpu_type=gpu_type,
        devices=next_num_devices,
        replicas=max(1, alloc.replicas),
    )
    try:
        action_result = evaluate_model_allocation(
            models=new_models,
            num_gpus=num_gpus,
            workflow=workflow,
            latency_data=latency_data,
            power_data=power_data,
            policy=policy,
            include_models=[model_name],
        )
        action = Action(
            name=ActionName.ADD_DEVICE,
            model=model_name,
            gpu_type=gpu_type,
            models=new_models,
            action_result=action_result,
            arrival_time_s=alloc.time,
        )
    except Exception:
        pass  # Invalid action

    return action


def _gen_merge_replicas_actions(
    models: dict[GPUType, dict[Model, list[ModelAllocation]]],
    gpu_type: GPUType,
    model_name: Model,
    num_gpus: dict[GPUType, int],
    workflow: WorkflowConfig,
    policy: Policy,
    latency_data: LatencyData,
    power_data: Optional[PowerData] = None,
) -> list[Action]:
    actions: list[Action] = []

    if _is_single_instance(model_name, workflow):
        return actions  # These models only support a single instance, so no need to merge

    model_instances = models[gpu_type][model_name]
    model_num_gpus = 0
    for model_instance in model_instances:
        model_num_gpus += model_instance.get_num_gpus()
    if model_num_gpus <= 1:
        return actions  # No replicas to merge for this model and GPU type

    for device_combos in [
        _get_min_device_combinations(model_num_gpus, model_name),
        _get_large_instance_many_small_combinations(model_num_gpus, model_name)
    ]:
        new_models = deepcopy(models)
        new_models[gpu_type][model_name] = []

        for new_num_devices, new_num_replicas in device_combos:
            new_models[gpu_type][model_name].append(get_model_allocation(
                model=model_name,
                gpu_type=gpu_type,
                devices=new_num_devices,
                replicas=new_num_replicas,
            ))

        try:
            action_result = evaluate_model_allocation(
                models=new_models,
                num_gpus=num_gpus,
                workflow=workflow,
                latency_data=latency_data,
                power_data=power_data,
                policy=policy,
                include_models=[model_name],
            )

            instance_id = 0
            actions.append(Action(
                name=ActionName.MERGE,
                model=model_name,
                gpu_type=gpu_type,
                models=new_models,
                action_result=action_result,
                arrival_time_s=new_models[gpu_type][model_name][instance_id].time,
            ))
        except Exception:
            pass  # Invalid action

    return actions


def _gen_add_instance(
    models: dict[GPUType, dict[Model, list[ModelAllocation]]],
    num_gpus: dict[GPUType, int],
    remaining_gpus: int,
    gpu_type: GPUType,
    model_name: Model,
    workflow: WorkflowConfig,
    policy: Policy,
    latency_data: LatencyData,
    power_data: Optional[PowerData] = None,
    look_ahead_replicas: int = 3,
) -> list[Action]:
    actions: list[Action] = []

    if _is_single_instance(model_name, workflow):
        return actions  # These models only support a single instance, so we don't add more

    for new_num_devices in DEVICE_OPTIONS[model_name]:
        for new_num_replicas in list(range(1, look_ahead_replicas + 1)):
            new_instance = get_model_allocation(
                model=model_name,
                gpu_type=gpu_type,
                devices=new_num_devices,
                replicas=new_num_replicas,
            )
            if new_instance.get_num_gpus() > remaining_gpus:
                continue  # Not enough remaining GPUs for this new instance

            new_models = deepcopy(models)
            new_models[gpu_type][model_name].append(new_instance)

            try:
                action_result = evaluate_model_allocation(
                    models=new_models,
                    num_gpus=num_gpus,
                    workflow=workflow,
                    latency_data=latency_data,
                    power_data=power_data,
                    policy=policy,
                    include_models=[model_name],
                )
                action = Action(
                    name=ActionName.ADD_INSTANCE,
                    model=model_name,
                    gpu_type=gpu_type,
                    models=new_models,
                    action_result=action_result,
                    arrival_time_s=new_instance.time,
                )
                actions.append(action)
            except Exception:
                pass  # Invalid action

    return actions


def _gen_remove_replica_action(
    models: dict[GPUType, dict[Model, list[ModelAllocation]]],
    num_gpus: dict[GPUType, int],
    gpu_type: GPUType,
    model_name: Model,
    allocation_id: int,
    workflow: WorkflowConfig,
    policy: Policy,
    latency_data: LatencyData,
    power_data: Optional[PowerData] = None,
) -> Optional[Action]:
    action: Optional[Action] = None

    model = models[gpu_type][model_name][allocation_id]

    if model.replicas == 0:
        return action  # No replicas to remove for this model and GPU type

    new_models = deepcopy(models)
    new_models[gpu_type][model_name][allocation_id] = get_model_allocation(
        model=model_name,
        gpu_type=gpu_type,
        devices=model.devices,
        replicas=model.replicas - 1,
    )

    if len(num_gpus) == 2:
        # For dual GPU setting, initialize removed replica on the other GPU type to see if it improves performance
        gpu_types = list(num_gpus.keys())
        other_gpu_type = gpu_types[0] if gpu_type == gpu_types[1] else gpu_types[1]
        if _is_single_instance(model_name, workflow):
            if new_models[gpu_type][model_name][allocation_id].replicas == 0:
                # If this is a single instance model and we're removing the only replica, add it to the other GPU type
                new_models[other_gpu_type][model_name].append(get_model_allocation(
                    model=model_name,
                    gpu_type=other_gpu_type,
                    devices=model.devices,
                    replicas=1,
                ))

    try:
        action_result = evaluate_model_allocation(
            models=new_models,
            num_gpus=num_gpus,
            workflow=workflow,
            latency_data=latency_data,
            power_data=power_data,
            policy=policy,
            include_models=[model_name],
        )
        action = Action(
            name=ActionName.REMOVE_REPLICA,
            model=model_name,
            gpu_type=gpu_type,
            models=new_models,
            action_result=action_result,
            arrival_time_s=new_models[gpu_type][model_name][allocation_id].time,
        )
    except Exception:
        pass  # Ignore not possible action
    return action


def _gen_add_replica_action(
    models: dict[GPUType, dict[Model, list[ModelAllocation]]],
    num_gpus: dict[GPUType, int],
    remaining_gpus: int,
    gpu_type: GPUType,
    model_name: Model,
    allocation_id: int,
    workflow: WorkflowConfig,
    policy: Policy,
    latency_data: LatencyData,
    power_data: Optional[PowerData] = None,
) -> Optional[Action]:
    """
    Action to add replicas for a specific model allocation.
    """
    action: Optional[Action] = None

    if _is_single_instance(model_name, workflow):
        return action  # These models don't support replication, so we skip

    model = models[gpu_type][model_name][allocation_id]

    if remaining_gpus < model.devices:
        return action  # Not enough remaining GPUs to add another replica

    max_replicas = model.get_max_replicas(workflow)
    if model.replicas >= max_replicas:
        return action  # Already at max replicas, skip

    new_num_replicas = min(
        model.replicas + 1,
        max_replicas,  # - models[other_gpu_type][Model.HF].replicas
        model.replicas + remaining_gpus // model.devices
    )
    if new_num_replicas == model.replicas:
        return action  # No changes, skip

    new_models = deepcopy(models)
    new_models[gpu_type][model_name][allocation_id] = get_model_allocation(
        model=model_name,
        gpu_type=gpu_type,
        devices=model.devices,
        replicas=new_num_replicas,
    )

    try:
        action_result = evaluate_model_allocation(
            models=new_models,
            num_gpus=num_gpus,
            workflow=workflow,
            latency_data=latency_data,
            power_data=power_data,
            policy=policy,
            include_models=[model_name],
        )
        action = Action(
            name=ActionName.ADD_REPLICA,
            model=model_name,
            gpu_type=gpu_type,
            models=new_models,
            action_result=action_result,
            arrival_time_s=model.time,
        )
    except Exception:
        pass  # Invalid action

    return action


def max_time(
    models: dict[GPUType, dict[Model, list[ModelAllocation]]],
    model_name: Model,
) -> float:
    values = []
    for models_gpu in models.values():
        if model_name in models_gpu:
            for alloc in models_gpu[model_name]:
                values.append(alloc.time)
    return max(values)
