from __future__ import annotations

import sys
import os

# Add current path
sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path

with temp_sys_path("simulator", "streamwise"):
    from sim_types import Model
    from sim_types import GPUType

    from sim_types_json import models_to_json
    from sim_types_json import workflow_to_json
    from sim_types_json import policy_to_json
    from sim_types_json import model_list_to_json

    from models import GemmaModelAllocation
    from models import FluxModelAllocation

    from model_provisioner.policies import STREAMWISE_POLICY

    from workflows import PODCAST_WORKFLOW


def test_serialize_models() -> None:
    models = {
        GPUType.A100: {
            Model.GEMMA: [GemmaModelAllocation(
                gpu_type=GPUType.A100,
                devices=1, replicas=1)]
        },
        GPUType.H200: {
            Model.FLUX: [FluxModelAllocation(
                gpu_type=GPUType.H200,
                devices=2, replicas=1)]
        },
    }

    models_json = models_to_json(models)

    assert models_json == (
        "{"
        "'A100': {'gemma': {'devices': 1, 'replicas': 1}},"
        "'H200': {'flux': {'devices': 2, 'replicas': 1}}"
        "}"
    )


def test_serialize_workflow() -> None:
    workflow = PODCAST_WORKFLOW
    workflow_json = workflow_to_json(workflow)
    assert workflow_json == (
        '{'
        '"total_video_seconds": 600, '
        '"total_scenes": 43, '
        '"total_frames": {"hf": 18000, "ft": 13800}, '
        '"total_subscenes": 171, '
        '"per_subscene_frames": {"hf": 106, "ft": 81}, '
        '"num_steps": {"flux": 25, "hf": 10, "ft": 10}, '
        '"hf_frames": [36, 72, 108, 144, 324], '
        '"ft_frames": [9, 21, 41, 61, 77], '
        '"frames_per_step_idx": 4, '
        '"target_resolution": "high", '
        '"total_input_tokens": 20480, '
        '"model_work": {'
        '"gemma": 1, '
        '"flux": 1, '
        '"hf": 171, '
        '"hf_vae": 18000, '
        '"ft": 171, '
        '"ft_vae": 13800, '
        '"upscaler": 13800, '
        '"others": 1}'
        '}'
    )


def test_serialize_policy() -> None:
    policy = STREAMWISE_POLICY
    policy_json = policy_to_json(policy)
    assert policy_json == (
        '{'
        '"name": "streamwise", '
        '"objective": "Objective.TTFF_COST", '
        '"disaggregation": {"hf": true, "ft": false}, '
        '"use_upscaler": true, '
        '"hardware": ["A100", "H100", "H200", "GB200"]'
        '}'
    )


def test_serialize_model_list() -> None:
    models = [Model.GEMMA, Model.FLUX]
    models_json = model_list_to_json(models)
    assert models_json == '["gemma", "flux"]'
