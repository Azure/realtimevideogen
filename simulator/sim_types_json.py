from __future__ import annotations

import json

from dataclasses import asdict

from sim_types import Model
from sim_types import Policy
from sim_types import GPUType
from sim_types import ModelAllocation
from sim_types import WorkflowConfig


def models_to_json(
    models: dict[GPUType, dict[Model, list[ModelAllocation]]]
) -> str:
    result = {}
    for gpu_type, model_dict in models.items():
        inner_result = {}
        for model, allocation_list in model_dict.items():
            for allocation in allocation_list:
                alloc_dict = {
                    'devices': allocation.devices,
                    'replicas': allocation.replicas,
                }
                inner_result[model.value] = alloc_dict
        result[gpu_type.name] = inner_result
    return str(result).replace("}}, '", "}},'")


def workflow_to_json(workflow: WorkflowConfig) -> str:
    d = asdict(workflow)
    # Convert Model enum keys in dict fields to string values
    for dict_field in ('total_frames', 'per_subscene_frames', 'num_steps', 'model_work'):
        if dict_field in d:
            d[dict_field] = {
                (k.value if hasattr(k, 'value') else k): v
                for k, v in d[dict_field].items()
            }
    # Convert QualityLevel enum to string value
    if 'target_resolution' in d and hasattr(d['target_resolution'], 'value'):
        d['target_resolution'] = d['target_resolution'].value
    return json.dumps(d)


def policy_to_json(policy: Policy) -> str:
    result = {
        'name': policy.name,
        'objective': str(policy.objective),
        'disaggregation': {model.value: enabled for model, enabled in policy.disaggregation.items()},
        'use_upscaler': policy.use_upscaler,
        'hardware': [gpu.name for gpu in policy.hardware],
    }
    return json.dumps(result)


def model_list_to_json(models: list[Model]) -> str:
    return json.dumps(models, default=lambda o: o.value)
