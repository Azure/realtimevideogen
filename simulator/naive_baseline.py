"""
Naive baseline for the StreamWise workflow allocation problem.
"""

from __future__ import annotations

from typing import Optional

from constants import NUM_GPUS_PER_SERVER
from constants import DEVICE_OPTIONS

from sim_types import Result
from sim_types import GPUType
from sim_types import WorkflowConfig
from sim_types import LatencyData
from sim_types import PowerData
from sim_types import Policy
from sim_types import Solver
from sim_types import Model
from sim_types import ModelAllocation
from sim_types import Objective

from models import FluxModelAllocation
from models import GemmaModelAllocation
from models import HFModelAllocation
from models import HFVAEModelAllocation
from models import FTModelAllocation
from models import FTVAEModelAllocation
from models import UpscalerModelAllocation
from models import OthersModelAllocation

from evaluator import evaluate_model_allocation

from policies import NAIVE_POLICY
from policies import MAX_DEVICES

from model_allocator import ModelAllocator


class NaiveAllocator(ModelAllocator):
    """
    Naive allocator that implements a simple heuristic.
    """
    def __init__(
        self,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
        power_data: Optional[PowerData] = None,
        policy: Policy = NAIVE_POLICY,
    ) -> None:
        super().__init__(
            workflow,
            latency_data,
            power_data,
            policy,
        )
        assert self.policy.solver == Solver.NAIVE
        assert self.policy.objective == Objective.TTFF

    def allocate(
        self,
        num_gpus: dict[GPUType, int],
        verbose: bool = False,
    ) -> Result:
        total_gpus = sum(num_gpus.values())
        assert total_gpus >= 8, f"Total number of GPUs must be at least 8 ({num_gpus})"

        gpu_types = [
            gpu_type
            for gpu_type, count in num_gpus.items()
            if count > 0
        ]
        assert 1 <= len(gpu_types) <= 2, f"Only up to two GPU types are supported ({len(gpu_types)})"
        gpu_type1 = gpu_types[0]

        if len(gpu_types) == 1:
            models = self._naive_single(
                num_gpus.get(gpu_type1, 0),
                gpu_type=gpu_type1,
            )
        else:
            # Mixed setup of GPU types (e.g., A100 and H100)
            models = self._naive_two(num_gpus)

        result = evaluate_model_allocation(
            models=models,
            num_gpus=num_gpus,
            workflow=self.workflow,
            latency_data=self.latency_data,
            power_data=self.power_data,
            policy=self.policy,
            round_up_cost_to_server=True,
        )
        return result

    def _naive_single(
        self,
        num_gpus: int,
        gpu_type: GPUType,
    ) -> dict[GPUType, dict[Model, list[ModelAllocation]]]:
        """Naive allocation for single GPU type."""
        return self._naive_parallelism_allocation(gpu_type, num_gpus)

    def _naive_two(
        self,
        num_gpus: dict[GPUType, int],
    ) -> dict[GPUType, dict[Model, list[ModelAllocation]]]:
        """Naive allocation for two GPU types."""
        gpu_types = list(num_gpus.keys())
        assert len(gpu_types) == 2
        assert len(num_gpus) == 2
        gpu_type1 = gpu_types[0]
        gpu_type2 = gpu_types[1]
        assert num_gpus[gpu_type1] >= NUM_GPUS_PER_SERVER[gpu_type1]
        assert num_gpus[gpu_type2] >= NUM_GPUS_PER_SERVER[gpu_type2]

        # Initialize allocations with minimal setup
        models: dict[GPUType, dict[Model, list[ModelAllocation]]] = {
            gpu_type1: {  # 3 x A100s (type1)
                Model.GEMMA: [GemmaModelAllocation(
                    gpu_type=gpu_type1,
                    devices=1, replicas=1)],
                Model.FLUX: [FluxModelAllocation(
                    gpu_type=gpu_type1,
                    devices=1, replicas=1)],
                Model.HF: [],
                Model.HF_VAE: [],
                Model.FT: [],
                Model.FT_VAE: [],
                Model.UPSCALER: [],
                Model.OTHERS: [OthersModelAllocation(
                    gpu_type=gpu_type1,
                    devices=1, replicas=1)],  # + 1 for Kokoro/YOLO
            },
            gpu_type2: {  # 4 (+1) X H100 GPUs (type2)
                Model.GEMMA: [],
                Model.FLUX: [],
                Model.HF: [HFModelAllocation(
                    gpu_type=gpu_type2,
                    devices=1, replicas=1)],
                Model.HF_VAE: [HFVAEModelAllocation(
                    gpu_type=gpu_type2,
                    devices=1, replicas=1)],
                Model.FT: [FTModelAllocation(
                    gpu_type=gpu_type2,
                    devices=2, replicas=1)],
                Model.FT_VAE: [FTVAEModelAllocation(
                    gpu_type=gpu_type2,
                    devices=1, replicas=1)],
                Model.UPSCALER: [UpscalerModelAllocation(
                    gpu_type=gpu_type2)],
                Model.OTHERS: [],
            },
        }

        # Calculate remaining: starting - assigned
        if not self.policy.is_disaggregated(Model.HF):
            models[gpu_type2][Model.HF][0].replicas = 2
            models[gpu_type2][Model.HF_VAE][0].replicas = 0
        if not self.policy.is_disaggregated(Model.FT):
            models[gpu_type2][Model.FT_VAE][0].replicas = 0

        if self.policy.use_upscaler:
            models[gpu_type2][Model.UPSCALER][0].replicas = 1

        models_gpu_type1 = self._naive_parallelism_allocation(
            gpu_type1,
            num_gpus.get(gpu_type1, 0),
        )
        models_gpu_type2 = self._naive_parallelism_allocation(
            gpu_type2,
            num_gpus.get(gpu_type2, 0),
            # Already allocated in first GPU type
            skip_non_paralelizable_models=True,
        )
        models[gpu_type1] = models_gpu_type1[gpu_type1]
        models[gpu_type2] = models_gpu_type2[gpu_type2]

        # Apply per-GPU-type overrides after allocation
        if self.policy.use_upscaler:
            models[gpu_type2][Model.UPSCALER][0].replicas = 1

        return models

    def _naive_parallelism_allocation(
        self,
        gpu_type: GPUType,
        num_devices: int,
        skip_non_paralelizable_models: bool = False,
    ) -> dict[GPUType, dict[Model, list[ModelAllocation]]]:
        """
        Device allocation for naive parallelism.
        Max devices for each model.
        Allocate devices to each model proportional to their max devices.
        """
        models: dict[GPUType, dict[Model, list[ModelAllocation]]] = {
            gpu_type: {
                Model.GEMMA: [GemmaModelAllocation(
                    gpu_type=gpu_type,
                    replicas=1)],
                Model.FLUX: [FluxModelAllocation(
                    gpu_type=gpu_type,
                    replicas=1)],
                Model.HF: [HFModelAllocation(
                    gpu_type=gpu_type,
                    replicas=1)],
                Model.HF_VAE: [HFVAEModelAllocation(
                    gpu_type=gpu_type,
                    replicas=1 if self.policy.is_disaggregated(Model.HF) else 0)],
                Model.FT: [FTModelAllocation(
                    gpu_type=gpu_type,
                    replicas=4)],
                Model.FT_VAE: [FTVAEModelAllocation(
                    gpu_type=gpu_type,
                    replicas=1 if self.policy.is_disaggregated(Model.FT) else 0)],
                Model.OTHERS: [OthersModelAllocation(
                    gpu_type=gpu_type,
                    replicas=1)],  # + 1 for Kokoro/YOLO
                Model.UPSCALER: [UpscalerModelAllocation(
                    gpu_type=gpu_type,
                    replicas=1 if self.policy.use_upscaler else 0)],
            },
        }

        # Zero out replicas for models not in workflow
        for model in Model:
            if model not in self.workflow.models:
                for alloc in models[gpu_type][model]:
                    alloc.replicas = 0

        # Zero out replicas for models that are not parallelizable when skip_non_paralelizable_models is True
        if skip_non_paralelizable_models:
            for model in Model:
                if not self.workflow.is_parallelizable(model):
                    for alloc in models[gpu_type][model]:
                        alloc.replicas = 0

        # Assert only 1 allocation instance per model for naive parallelism
        for model in Model:
            assert len(models[gpu_type][model]) == 1, \
                f"Expected only 1 allocation instance for {model}, got {len(models[gpu_type][model])}"

        alloc_id = 0
        model_gemma = models[gpu_type][Model.GEMMA][alloc_id]
        model_flux = models[gpu_type][Model.FLUX][alloc_id]
        model_hf = models[gpu_type][Model.HF][alloc_id]
        model_vae = models[gpu_type][Model.HF_VAE][alloc_id]
        model_ft = models[gpu_type][Model.FT][alloc_id]
        model_ft_vae = models[gpu_type][Model.FT_VAE][alloc_id]
        model_upscaler = models[gpu_type][Model.UPSCALER][alloc_id]

        # TODO do we need to do something for Model.OTHERS

        if num_devices == 8:
            # single server case, use fixed allocation
            if Model.FT in self.workflow.models:
                model_ft.replicas = 4
            if self.policy.use_upscaler and Model.UPSCALER in self.workflow.models:
                model_upscaler.replicas = 1
                if Model.FT in self.workflow.models:
                    model_ft.replicas -= 1
            if self.policy.is_disaggregated(Model.HF) and Model.HF_VAE in self.workflow.models:
                model_vae.replicas = 1
                if Model.FT in self.workflow.models:
                    model_ft.replicas -= 1
            if self.policy.is_disaggregated(Model.FT) and Model.FT_VAE in self.workflow.models:
                model_ft_vae.replicas = 1
                if Model.FT in self.workflow.models:
                    model_ft.replicas -= 1
            return models

        init_num_devices = sum([
            model[0].devices * model[0].replicas
            for model in models[gpu_type].values()
        ])

        # Allocate devices proportional to each model's max devices
        max_devices = MAX_DEVICES
        models_in_workflow = [
            model
            for model in max_devices.keys()
            if model in self.workflow.models
        ]
        if skip_non_paralelizable_models:
            for model in max_devices.keys():
                if not self.workflow.is_parallelizable(model):
                    models_in_workflow.remove(model)

        total_max_devices = sum([
            max_devices[model]
            for model in models_in_workflow
        ])
        for model in models_in_workflow:
            # Calculate the number of devices to allocate for the model, proportional to its max devices among others
            alloc_devices = int((num_devices - init_num_devices) * max_devices[model] / total_max_devices)
            if model == Model.GEMMA:
                max_devices_gemma = max_devices[Model.GEMMA]
                if self.latency_data:
                    max_devices_gemma = min(max_devices_gemma, self.latency_data[gpu_type].get_max_parallelism(model))
                model_gemma.devices += min(alloc_devices, max_devices_gemma)
                # Round down nearest in DEVICE_OPTIONS_GEMMA
                num_gemma_devices = max([
                    d
                    for d in DEVICE_OPTIONS[Model.GEMMA]
                    if d <= model_gemma.devices
                ])
                model_gemma.devices = num_gemma_devices
            elif model == Model.FLUX:
                max_devices_flux = max_devices[Model.FLUX]
                if self.latency_data:
                    max_devices_flux = min(max_devices_flux, self.latency_data[gpu_type].get_max_parallelism(model))
                model_flux.devices += min(alloc_devices, max_devices_flux)
                # Round down nearest in DEVICE_OPTIONS_FLUX
                model_flux.devices = max([
                    d
                    for d in DEVICE_OPTIONS[Model.FLUX]
                    if d <= model_flux.devices
                ])
            elif model == Model.HF:
                max_devices_hf = max_devices[Model.HF]
                if self.latency_data:
                    max_devices_hf = min(max_devices_hf, self.latency_data[gpu_type].get_max_parallelism(model))
                model_hf.replicas += min(alloc_devices, max_devices_hf)
            elif model == Model.HF_VAE:
                if self.policy.is_disaggregated(Model.HF):
                    max_devices_vae = max_devices[Model.HF_VAE]
                    if self.latency_data:
                        max_devices_vae = min(max_devices_vae, self.latency_data[gpu_type].get_max_parallelism(model))
                    model_vae.replicas += min(alloc_devices, max_devices_vae)
            elif model == Model.FT:
                max_devices_ft = max_devices[Model.FT]
                if self.latency_data:
                    max_devices_ft = min(max_devices_ft, self.latency_data[gpu_type].get_max_parallelism(model))
                model_ft.replicas += min(alloc_devices, max_devices_ft)
            elif model == Model.FT_VAE:
                if self.policy.is_disaggregated(Model.FT):
                    max_devices_ft_vae = max_devices[Model.FT_VAE]
                    if self.latency_data:
                        max_devices_ft_vae = min(
                            max_devices_ft_vae, self.latency_data[gpu_type].get_max_parallelism(model)
                        )
                    model_ft_vae.replicas += min(alloc_devices, max_devices_ft_vae)
            else:
                raise ValueError(f"Unrecognized model {model}")

        remaining_devices = num_devices
        for model_name in models[gpu_type].keys():
            for model_alloc in models[gpu_type][model_name]:
                remaining_devices -= model_alloc.get_num_gpus()

        # Distribute remaining devices to parallelizable models
        distribute_models = self.workflow.filter_parallelizable_models(
            models_in_workflow,
            disaggregation=self.policy.disaggregation,
        )
        # Prioritise models that already hold more GPUs
        distribute_models.sort(
            key=lambda m: models[gpu_type][m][alloc_id].get_num_gpus(),
            reverse=True,
        )
        num_distribute = len(distribute_models)
        if num_distribute > 0 and remaining_devices > 0:
            made_progress = True
            while remaining_devices > 0 and made_progress:
                made_progress = False
                for model_name in distribute_models:
                    gpus_per_replica = models[gpu_type][model_name][alloc_id].devices
                    if gpus_per_replica <= 0 or remaining_devices < gpus_per_replica:
                        continue
                    models[gpu_type][model_name][alloc_id].replicas += 1
                    remaining_devices -= gpus_per_replica
                    made_progress = True
                    if remaining_devices <= 0:
                        break

        remaining_devices = num_devices
        for model_name in models[gpu_type].keys():
            for model_alloc in models[gpu_type][model_name]:
                remaining_devices -= model_alloc.get_num_gpus()

        # TODO we should try to assign all resources
        # assert remaining_devices == 0, \
        assert remaining_devices >= 0, \
            f"remaining={remaining_devices} != 0: " \
            f"gpu={gpu_type.value} total={num_devices} remaining={remaining_devices}"

        # Update replicas based on total devices
        # Gemma (when parallelizable)
        if self.workflow.is_parallelizable(Model.GEMMA) and Model.GEMMA in models_in_workflow:
            model_gemma.devices, model_gemma.replicas, remaining_devices = _calculate_naive_num_devices(
                model_gemma.devices,
                model_gemma.replicas,
                remaining_devices,
                device_options=DEVICE_OPTIONS[Model.GEMMA],
                replica_upper_bound=self.workflow.total_scenes)

        # Flux (when parallelizable)
        if self.workflow.is_parallelizable(Model.FLUX) and Model.FLUX in models_in_workflow:
            model_flux.devices, model_flux.replicas, remaining_devices = _calculate_naive_num_devices(
                model_flux.devices,
                model_flux.replicas,
                remaining_devices,
                device_options=DEVICE_OPTIONS[Model.FLUX],
                replica_upper_bound=self.workflow.total_scenes)

        # Hunyuan FramePack
        if Model.HF in self.workflow.models:
            model_hf.devices, model_hf.replicas, remaining_devices = _calculate_naive_num_devices(
                model_hf.devices,
                model_hf.replicas,
                remaining_devices,
                device_options=DEVICE_OPTIONS[Model.HF],
                replica_upper_bound=self.workflow.total_scenes)

        # Hunyuan FramePack VAE
        if self.policy.is_disaggregated(Model.HF) and Model.HF_VAE in self.workflow.models:
            model_vae.devices, model_vae.replicas, remaining_devices = _calculate_naive_num_devices(
                model_vae.devices,
                model_vae.replicas,
                remaining_devices,
                device_options=None,
                replica_upper_bound=self.workflow.total_frames[Model.HF],
            )

        # Fantasy Talking
        if Model.FT in self.workflow.models:
            model_ft.devices, model_ft.replicas, remaining_devices = _calculate_naive_num_devices(
                model_ft.devices,
                model_ft.replicas,
                remaining_devices,
                device_options=DEVICE_OPTIONS[Model.FT],
                replica_upper_bound=self.workflow.total_subscenes,
            )

        # Fantasy Talking VAE
        if self.policy.is_disaggregated(Model.FT) and Model.FT_VAE in self.workflow.models:
            model_ft_vae.devices, model_ft_vae.replicas, remaining_devices = _calculate_naive_num_devices(
                model_ft_vae.devices,
                model_ft_vae.replicas,
                remaining_devices,
                device_options=None,
                replica_upper_bound=self.workflow.total_frames[Model.FT],
            )

        return models


def _calculate_naive_num_devices(
    num_devices: int,
    num_replicas: int,
    remaining_devices: int,
    device_options: Optional[list[int]] = [1],
    replica_upper_bound: Optional[int] = None,
) -> tuple[int, int, int]:
    """Find the parallelism that maximizes the device usage."""
    assert remaining_devices >= 0

    model_quota = num_devices * num_replicas

    if device_options:
        best_product = 0
        best_devices_per_replica = 1
        best_replicas = 1
        for devices_per_replica in device_options:
            if devices_per_replica > model_quota:
                continue
            max_replicas = model_quota // devices_per_replica
            if replica_upper_bound and max_replicas > replica_upper_bound:
                max_replicas = replica_upper_bound
            product = devices_per_replica * max_replicas
            if product > best_product:
                best_product = product
                best_devices_per_replica = devices_per_replica
                best_replicas = max_replicas
    else:
        # start with parallelism=1 instead
        best_devices_per_replica = 1
        best_replicas = model_quota

    num_devices = best_devices_per_replica
    num_replicas = best_replicas
    remaining_devices += model_quota - num_replicas * num_devices

    return num_devices, num_replicas, remaining_devices
