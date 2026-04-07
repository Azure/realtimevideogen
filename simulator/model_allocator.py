"""
Defines the ModelAllocator abstract base class and its interface for model allocation strategies.
"""

from __future__ import annotations

from typing import Optional

from abc import ABC
from abc import abstractmethod

from sim_types import GPUType
from sim_types import Model
from sim_types import ModelAllocation
from sim_types import Policy
from sim_types import WorkflowConfig
from sim_types import LatencyData
from sim_types import PowerData
from sim_types import Result

from models import FluxModelAllocation
from models import GemmaModelAllocation
from models import HFModelAllocation
from models import HFVAEModelAllocation
from models import FTModelAllocation
from models import FTVAEModelAllocation
from models import UpscalerModelAllocation
from models import OthersModelAllocation

from policies import NAIVE_POLICY


class ModelAllocator(ABC):
    """
    Abstract base class for model allocators.
    """

    def __init__(
        self,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
        power_data: Optional[PowerData] = None,
        policy: Policy = NAIVE_POLICY,
    ) -> None:
        self.workflow = workflow
        self.latency_data = latency_data
        self.power_data = power_data
        self.policy = policy

    @abstractmethod
    def allocate(
        self,
        num_gpus: dict[GPUType, int],
        verbose: bool = False,
    ) -> Result:
        """Allocate models to GPUs and return the provisioning result."""
        ...

    def _init_single_server_models(
        self,
        gpu_type: GPUType,
    ) -> dict[GPUType, dict[Model, list[ModelAllocation]]]:
        """
        Initialize model allocations for a single server (8 GPUs or fewer).
        Each model gets a single allocation entry.
        """
        models: dict[GPUType, dict[Model, list[ModelAllocation]]] = {
            gpu_type: {
                Model.GEMMA: [
                    GemmaModelAllocation(
                        gpu_type=gpu_type,
                        devices=1, replicas=1)
                ],
                Model.FLUX: [
                    FluxModelAllocation(
                        gpu_type=gpu_type,
                        devices=1, replicas=1)
                ],
                Model.HF: [
                    HFModelAllocation(
                        gpu_type=gpu_type,
                        devices=1, replicas=2)
                ],
                Model.HF_VAE: [
                    HFVAEModelAllocation(
                        gpu_type=gpu_type,
                        devices=1, replicas=1)
                ],
                Model.FT: [
                    FTModelAllocation(
                        gpu_type=gpu_type,
                        devices=1, replicas=1)
                ],
                Model.FT_VAE: [
                    FTVAEModelAllocation(
                        gpu_type=gpu_type,
                        devices=1, replicas=1)
                ],
                Model.UPSCALER: [
                    UpscalerModelAllocation(
                        gpu_type=gpu_type)
                ],
                Model.OTHERS: [
                    OthersModelAllocation(
                        gpu_type=gpu_type,
                        devices=1, replicas=1)  # + 1 for Kokoro/YOLO
                ],
            },
        }

        if self.policy.use_upscaler:
            # HF -> UPSCALER
            models[gpu_type][Model.HF][0].replicas -= 1
            models[gpu_type][Model.UPSCALER][0].replicas += 1

        if not self.policy.is_disaggregated(Model.HF):
            # HF_VAE -> HF
            models[gpu_type][Model.HF_VAE][0].replicas -= 1
            models[gpu_type][Model.HF][0].replicas += 1
        if not self.policy.is_disaggregated(Model.FT):
            # FT_VAE -> FT
            models[gpu_type][Model.FT_VAE][0].replicas -= 1
            models[gpu_type][Model.FT][0].replicas += 1

        self._zero_out_unused_models(models)
        return models

    def _init_single_device_models(
        self,
        gpu_type: GPUType,
    ) -> dict[GPUType, dict[Model, list[ModelAllocation]]]:
        """
        Initialize model allocations for a single GPU type with >8 GPUs.
        Each model gets two allocation entries (active and inactive).
        """
        models: dict[GPUType, dict[Model, list[ModelAllocation]]] = {
            gpu_type: {
                Model.GEMMA: [
                    GemmaModelAllocation(
                        gpu_type=gpu_type,
                        devices=1, replicas=1),
                    GemmaModelAllocation(
                        gpu_type=gpu_type),
                ],
                Model.FLUX: [
                    FluxModelAllocation(
                        gpu_type=gpu_type,
                        devices=1, replicas=1),
                    FluxModelAllocation(
                        gpu_type=gpu_type),
                ],
                Model.HF: [
                    HFModelAllocation(
                        gpu_type=gpu_type,
                        devices=1, replicas=1),
                    HFModelAllocation(
                        gpu_type=gpu_type),
                ],
                Model.HF_VAE: [
                    HFVAEModelAllocation(
                        gpu_type=gpu_type,
                        devices=1, replicas=1),
                    HFVAEModelAllocation(
                        gpu_type=gpu_type),
                ],
                Model.FT: [
                    FTModelAllocation(
                        gpu_type=gpu_type,
                        devices=2, replicas=1),
                    FTModelAllocation(
                        gpu_type=gpu_type),
                ],
                Model.FT_VAE: [
                    FTVAEModelAllocation(
                        gpu_type=gpu_type,
                        devices=1, replicas=1),
                    FTVAEModelAllocation(
                        gpu_type=gpu_type),
                ],
                Model.UPSCALER: [
                    UpscalerModelAllocation(
                        gpu_type=gpu_type),
                    UpscalerModelAllocation(
                        gpu_type=gpu_type),
                ],
                Model.OTHERS: [
                    OthersModelAllocation(
                        gpu_type=gpu_type,
                        devices=1, replicas=1),
                    OthersModelAllocation(
                        gpu_type=gpu_type),
                ],
            },
        }

        if self.policy.use_upscaler:
            models[gpu_type][Model.UPSCALER][0].replicas = 1

        if not self.policy.is_disaggregated(Model.HF):
            # HF_VAE -> HF
            models[gpu_type][Model.HF_VAE][0].replicas -= 1
            models[gpu_type][Model.HF][0].replicas += 1
        if not self.policy.is_disaggregated(Model.FT):
            # FT_VAE -> FT
            models[gpu_type][Model.FT_VAE][0].replicas -= 1
            models[gpu_type][Model.FT][0].replicas += 1

        self._zero_out_unused_models(models)
        return models

    def _init_both_devices_models(
        self,
        gpu_type1: GPUType,
        gpu_type2: GPUType,
    ) -> dict[GPUType, dict[Model, list[ModelAllocation]]]:
        """
        Initialize model allocations for two GPU types.
        gpu_type1 gets GEMMA, FLUX, OTHERS; gpu_type2 gets HF, VAE, FT, UPSCALER.
        """
        models: dict[GPUType, dict[Model, list[ModelAllocation]]] = {
            gpu_type1: {
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
            gpu_type2: {
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

        if not self.policy.is_disaggregated(Model.HF):
            # HF_VAE -> HF
            models[gpu_type2][Model.HF_VAE][0].replicas -= 1
            models[gpu_type2][Model.HF][0].replicas += 1
        if not self.policy.is_disaggregated(Model.FT):
            # FT_VAE -> FT
            models[gpu_type2][Model.FT_VAE][0].replicas -= 1
            models[gpu_type2][Model.FT][0].replicas += 1

        if self.policy.use_upscaler:
            models[gpu_type2][Model.UPSCALER][0].replicas = 1

        self._zero_out_unused_models(models)
        return models

    def _zero_out_unused_models(
        self,
        models: dict[GPUType, dict[Model, list[ModelAllocation]]],
    ) -> None:
        """Zero out replicas for models not in the workflow."""
        for gpu_type in models:
            for model in Model:
                if model not in self.workflow.models:
                    for alloc in models[gpu_type][model]:
                        alloc.replicas = 0
