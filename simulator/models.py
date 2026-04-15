"""
Contains the definition for each model.
It includes the calculations for time, energy, and cost.
"""
from __future__ import annotations

import math

from typing import override
from typing import Callable
from typing import Optional
from typing import Type
from typing import ClassVar

from sim_types import LatencyData
from sim_types import PowerData
from sim_types import ModelAllocation
from sim_types import Model
from sim_types import Policy
from sim_types import QualityLevel
from sim_types import WorkflowConfig
from sim_types import GPUType

from constants import TOTAL_INPUT_TOKENS


# ModelAllocation Factory
ModelAllocationCls = Type[ModelAllocation]

_MODEL_ALLOCATION_REGISTRY: dict[Model, ModelAllocationCls] = {}


def register_model(
    model: Model
) -> Callable[[ModelAllocationCls], ModelAllocationCls]:
    """Register a ModelAllocation class for the factory."""
    def decorator(cls: ModelAllocationCls) -> ModelAllocationCls:
        _MODEL_ALLOCATION_REGISTRY[model] = cls
        return cls
    return decorator


def get_model_allocation(
    *,
    model: Model,
    gpu_type: GPUType,
    devices: int = 1,
    replicas: int = 0,
) -> ModelAllocation:
    """Factory to get the ModelAllocation instance for a specific model."""
    if model not in _MODEL_ALLOCATION_REGISTRY:
        raise ValueError(f"No ModelAllocation for model {model}")
    cls = _MODEL_ALLOCATION_REGISTRY[model]
    return cls(
        gpu_type=gpu_type,
        devices=devices,
        replicas=replicas,
    )


def _calculate_total_time(
    total_work: float,
    num_replicas: int,
    time_per_work: float,
) -> float:
    """Calculate total time given work, replicas, and time per work unit."""
    if num_replicas <= 0:
        return 0.0
    total_time = (total_work / num_replicas) * time_per_work
    if total_time < time_per_work:  # We cannot go faster than single work unit time
        total_time = time_per_work
    return total_time


def assert_pixel_config(
    workflow: WorkflowConfig
) -> None:
    """Verify that the workflow's pixel configuration is valid for upscaling."""
    from sim_types import RESOLUTION_PIXELS
    assert 0 < RESOLUTION_PIXELS[QualityLevel.MEDIUM] < RESOLUTION_PIXELS[QualityLevel.HIGH]


@register_model(Model.GEMMA)
class GemmaModelAllocation(ModelAllocation):
    """Gemma model allocation."""
    model: ClassVar[Model] = Model.GEMMA

    @override
    def get_max_replicas(
        self,
        workflow: WorkflowConfig,
    ) -> int:
        return workflow.model_work.get(Model.GEMMA, 1)

    @override
    def calculate_time(
        self,
        policy: Policy,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
        work_pct: float = 1.0,
    ) -> float:
        if self.get_num_gpus() == 0:
            self.time = 0.0
            return self.time
        latency_first = latency_data[self.gpu_type].gemma_first_scene[self.devices]
        latency_per_scene = latency_data[self.gpu_type].gemma_per_scene[self.devices]
        latency_first *= workflow.total_input_tokens / TOTAL_INPUT_TOKENS
        latency_per_scene *= workflow.total_input_tokens / TOTAL_INPUT_TOKENS
        total_work = workflow.model_work.get(Model.GEMMA, 1)
        if total_work > 1:
            num_scenes = math.ceil(work_pct * total_work)
            total_time_per_scene = latency_first + latency_per_scene * (num_scenes - 1)
            self.time = _calculate_total_time(
                num_scenes,
                self.replicas,
                total_time_per_scene / num_scenes)
        else:
            self.time = latency_first + latency_per_scene * (workflow.total_scenes - 1)
        return self.time

    @override
    def calculate_time_first(
        self,
        policy: Policy,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
    ) -> float:
        if self.get_num_gpus() == 0:
            self.time_first = 0.0
            return self.time_first
        latency_first = latency_data[self.gpu_type].gemma_first_scene[self.devices]
        latency_first *= workflow.total_input_tokens / TOTAL_INPUT_TOKENS
        self.time_first = latency_first
        return self.time_first

    @override
    def calculate_energy(
        self,
        workflow: WorkflowConfig,
        power_data: Optional[PowerData] = None,
        total_time_s: float = 0.0,
    ) -> float:
        if self.get_num_gpus() == 0 or power_data is None:
            self.energy = 0.0
            return self.energy
        # Gemma energy
        latency_first = self.time_first
        latency_per_scene = max(0.0, self.time - latency_first)
        power_first = power_data[self.gpu_type].gemma_first_scene[self.devices]
        power_per_scene = power_data[self.gpu_type].gemma_per_scene[self.devices]
        self.energy = \
            power_first * latency_first + \
            power_per_scene * latency_per_scene * (workflow.total_scenes - 1)
        # Idle energy
        power_idle = power_data[self.gpu_type]["idle"] * self.get_num_gpus()
        time_idle = total_time_s - self.time
        if time_idle > 0:
            self.energy += power_idle * time_idle
        return self.energy


@register_model(Model.FLUX)
class FluxModelAllocation(ModelAllocation):
    """Flux model allocation."""
    model: ClassVar[Model] = Model.FLUX

    def _calc_time_per_scene(
        self,
        policy: Policy,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
    ) -> float:
        return (
            latency_data[self.gpu_type][self.model, self.devices]
            * workflow.num_steps[Model.FLUX]
        )

    @override
    def get_max_replicas(
        self,
        workflow: WorkflowConfig,
    ) -> int:
        return workflow.model_work.get(Model.FLUX, 1)

    @override
    def calculate_time(
        self,
        policy: Policy,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
        work_pct: float = 1.0,
    ) -> float:
        if self.get_num_gpus() == 0:
            self.time = 0.0
            return self.time
        time_per_scene = self._calc_time_per_scene(
            policy,
            workflow,
            latency_data,
        )
        total_work = workflow.model_work.get(Model.FLUX, 1)
        if total_work > 1:
            num_scenes = math.ceil(work_pct * total_work)
            self.time = _calculate_total_time(
                num_scenes,
                self.replicas,
                time_per_scene)
        else:
            self.time = time_per_scene
        return self.time

    @override
    def calculate_time_first(
        self,
        policy: Policy,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
    ) -> float:
        if self.get_num_gpus() == 0:
            self.time_first = 0.0
            return self.time_first
        time_per_scene = self._calc_time_per_scene(
            policy,
            workflow,
            latency_data,
        )
        self.time_first = time_per_scene
        return self.time_first

    @override
    def calculate_energy(
        self,
        workflow: WorkflowConfig,
        power_data: Optional[PowerData] = None,
        total_time_s: float = 0.0,
    ) -> float:
        if self.get_num_gpus() == 0 or power_data is None:
            self.energy = 0.0
            return self.energy
        power_flux = power_data[self.gpu_type][Model.FLUX, self.devices]
        self.energy = power_flux * self.time * self.replicas
        # Idle energy
        power_idle = power_data[self.gpu_type]["idle"] * self.get_num_gpus()
        time_idle = total_time_s - self.time
        if time_idle > 0:
            self.energy += power_idle * time_idle
        return self.energy


@register_model(Model.HF)
class HFModelAllocation(ModelAllocation):
    """HunyuanFramePack model allocation."""
    model: ClassVar[Model] = Model.HF

    def _calc_time_per_frame(
        self,
        policy: Policy,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
    ) -> float:
        return (
            latency_data[self.gpu_type][self.model, self.devices]
            * workflow.get_resolution_scale(policy.use_upscaler)
            * workflow.num_steps[Model.HF]
        )

    def _calc_time_per_subscene(
        self,
        policy: Policy,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
    ) -> float:
        return (
            workflow.per_subscene_frames[Model.HF]
            / workflow.hf_frames[workflow.frames_per_step_idx]
            * latency_data[self.gpu_type][self.model, self.devices]
            * workflow.get_resolution_scale(policy.use_upscaler)  # latency_ratio
            * workflow.num_steps[Model.HF]
        )

    @override
    def calculate_time(
        self,
        policy: Policy,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
        work_pct: float = 1.0,
    ) -> float:
        if self.get_num_gpus() == 0:
            self.time = 0.0
            return self.time

        hf_time_per_subscene = self._calc_time_per_subscene(
            policy,
            workflow,
            latency_data,
        )
        self.time = _calculate_total_time(
            math.ceil(work_pct * workflow.total_subscenes),
            self.replicas,
            hf_time_per_subscene)

        if not policy.is_disaggregated(Model.HF):
            # Include VAE time in the same GPU when disaggregation is disabled
            hf_vae_time_per_frame = (
                latency_data[self.gpu_type][Model.HF_VAE, 1]  # VAE is single-device only in current policy
                * workflow.get_resolution_scale(policy.use_upscaler)
                / workflow.hf_frames[workflow.frames_per_step_idx]
            )
            self.time += _calculate_total_time(
                math.ceil(work_pct * workflow.total_frames[Model.HF]),
                self.replicas,
                hf_vae_time_per_frame)

        return self.time

    @override
    def calculate_time_first(
        self,
        policy: Policy,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
    ) -> float:
        if self.get_num_gpus() == 0:
            self.time_first = 0.0
            return self.time_first

        if policy.is_disaggregated(Model.HF):
            # HF for the first chunk
            self.time_first = min(
                # Option 1: the first few frames until the first chunk is done
                workflow.hf_frames[0]
                / workflow.hf_frames[workflow.frames_per_step_idx]
                * self._calc_time_per_frame(
                    policy,
                    workflow,
                    latency_data
                ),
                # Option 2: the full subscene
                self._calc_time_per_subscene(
                    policy,
                    workflow,
                    latency_data
                ),
            )
        else:
            # HF + VAE for the full subscene
            hf_time_per_subscene = self._calc_time_per_subscene(
                policy,
                workflow,
                latency_data)
            hf_vae_time_per_subscene = (
                workflow.per_subscene_frames[Model.HF]
                / workflow.hf_frames[workflow.frames_per_step_idx]
                * latency_data[self.gpu_type][Model.HF_VAE, 1]  # VAE is single-device only in current policy
                * workflow.get_resolution_scale(policy.use_upscaler)
            )
            self.time_first = hf_time_per_subscene + hf_vae_time_per_subscene

        return self.time_first

    @override
    def calculate_energy(
        self,
        workflow: WorkflowConfig,
        power_data: Optional[PowerData] = None,
        total_time_s: float = 0.0,
    ) -> float:
        if self.get_num_gpus() == 0 or power_data is None:
            self.energy = 0.0
            return self.energy
        power_hf = power_data[self.gpu_type][Model.HF, self.devices]
        self.energy = power_hf * self.time * self.replicas
        # Idle energy
        power_idle = power_data[self.gpu_type]["idle"] * self.get_num_gpus()
        time_idle = total_time_s - self.time
        if time_idle > 0:
            self.energy += power_idle * time_idle
        return self.energy

    @override
    def get_max_replicas(
        self,
        workflow: WorkflowConfig,
    ) -> int:
        return workflow.model_work.get(Model.HF, 1)


@register_model(Model.HF_VAE)
class HFVAEModelAllocation(ModelAllocation):
    """HunyuanFramePack VAE model allocation."""
    model: ClassVar[Model] = Model.HF_VAE

    def _calc_time_per_frame(
        self,
        policy: Policy,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
    ) -> float:
        return (
            latency_data[self.gpu_type][Model.HF_VAE, self.devices]
            * workflow.get_resolution_scale(policy.use_upscaler)
            / workflow.hf_frames[workflow.frames_per_step_idx]
        )

    @override
    def calculate_time(
        self,
        policy: Policy,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
        work_pct: float = 1.0,
    ) -> float:
        if not policy.is_disaggregated(Model.HF):
            assert self.get_num_gpus() == 0
            self.time = 0.0
            return self.time
        if self.get_num_gpus() == 0:
            self.time = 0.0
            return self.time

        vae_time_per_frame = self._calc_time_per_frame(
            policy,
            workflow,
            latency_data
        )
        self.time = _calculate_total_time(
            math.ceil(workflow.total_frames[Model.HF] * work_pct),
            self.replicas,
            vae_time_per_frame)
        return self.time

    @override
    def calculate_time_first(
        self,
        policy: Policy,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
    ) -> float:
        if not policy.is_disaggregated(Model.HF):
            assert self.get_num_gpus() == 0
            self.time_first = 0.0
            return self.time_first
        if self.get_num_gpus() == 0:
            self.time_first = 0.0
            return self.time_first

        vae_time_per_frame = self._calc_time_per_frame(
            policy,
            workflow,
            latency_data,
        )
        num_frames = workflow.per_subscene_frames[Model.HF]
        self.time_first = num_frames * vae_time_per_frame
        return self.time_first

    @override
    def calculate_energy(
        self,
        workflow: WorkflowConfig,
        power_data: Optional[PowerData] = None,
        total_time_s: float = 0.0,
    ) -> float:
        if self.get_num_gpus() == 0 or power_data is None:
            self.energy = 0.0
            return self.energy
        self.energy = power_data[self.gpu_type][Model.HF_VAE, self.devices] * self.time * self.replicas
        # Idle energy
        power_idle = power_data[self.gpu_type]["idle"] * self.get_num_gpus()
        time_idle = total_time_s - self.time
        if time_idle > 0:
            self.energy += power_idle * time_idle
        return self.energy

    @override
    def get_max_replicas(
        self,
        workflow: WorkflowConfig,
    ) -> int:
        return workflow.model_work.get(Model.HF_VAE, 1)


@register_model(Model.FT)
class FTModelAllocation(ModelAllocation):
    """FantasyTalking model allocation."""
    model: ClassVar[Model] = Model.FT

    def _calc_time_per_subscene(
        self,
        policy: Policy,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
    ) -> float:
        return (
            workflow.per_subscene_frames[Model.FT]
            / workflow.ft_frames[workflow.frames_per_step_idx]
            * latency_data[self.gpu_type][Model.FT, self.devices]
            * workflow.get_resolution_scale(policy.use_upscaler)
            * workflow.num_steps[Model.FT]
        )

    @override
    def calculate_time(
        self,
        policy: Policy,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
        work_pct: float = 1.0,
    ) -> float:
        if self.get_num_gpus() == 0:
            self.time = 0.0
            return self.time

        ft_time_per_subscene = self._calc_time_per_subscene(
            policy,
            workflow,
            latency_data,
        )
        self.time = _calculate_total_time(
            math.ceil(work_pct * workflow.total_subscenes),
            self.replicas,
            ft_time_per_subscene)

        if not policy.is_disaggregated(Model.FT):
            # Include VAE time in the same GPU when disaggregation is disabled
            # Note: VAE latency uses devices=1 as VAE processing is not parallelized
            # across multiple devices in the same way as the main FT diffusion
            ft_vae_time_per_frame = (
                latency_data[self.gpu_type][Model.FT_VAE, 1]
                * workflow.get_resolution_scale(policy.use_upscaler)
                / workflow.ft_frames[workflow.frames_per_step_idx]
            )
            self.time += _calculate_total_time(
                math.ceil(work_pct * workflow.total_frames[Model.FT]),
                self.replicas,
                ft_vae_time_per_frame)

        return self.time

    @override
    def calculate_time_first(
        self,
        policy: Policy,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
    ) -> float:
        if self.get_num_gpus() == 0:
            self.time_first = 0.0
            return self.time_first

        ft_time_per_subscene = self._calc_time_per_subscene(
            policy,
            workflow,
            latency_data,
        )
        self.time_first = ft_time_per_subscene

        if not policy.is_disaggregated(Model.FT):
            # Include VAE time_first when FT-VAE is not disaggregated
            # Note: VAE latency uses devices=1 (see note in calculate_time)
            ft_vae_time_per_subscene = (
                workflow.per_subscene_frames[Model.FT]
                / workflow.ft_frames[workflow.frames_per_step_idx]
                * latency_data[self.gpu_type][Model.FT_VAE, 1]
                * workflow.get_resolution_scale(policy.use_upscaler)
            )
            self.time_first += ft_vae_time_per_subscene

        return self.time_first

    @override
    def calculate_energy(
        self,
        workflow: WorkflowConfig,
        power_data: Optional[PowerData] = None,
        total_time_s: float = 0.0,
    ) -> float:
        if self.get_num_gpus() == 0 or power_data is None:
            self.energy = 0.0
            return self.energy
        power_ft = power_data[self.gpu_type][Model.FT, self.devices]
        self.energy = power_ft * self.time * self.replicas
        # Idle energy
        power_idle = power_data[self.gpu_type]["idle"] * self.get_num_gpus()
        time_idle = total_time_s - self.time
        if time_idle > 0:
            self.energy += power_idle * time_idle
        return self.energy

    @override
    def get_max_replicas(
        self,
        workflow: WorkflowConfig,
    ) -> int:
        return workflow.model_work.get(Model.FT, 1)


@register_model(Model.FT_VAE)
class FTVAEModelAllocation(ModelAllocation):
    """FantasyTalking VAE model allocation."""
    model: ClassVar[Model] = Model.FT_VAE

    def _calc_time_per_frame(
        self,
        policy: Policy,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
    ) -> float:
        return (
            latency_data[self.gpu_type][Model.FT_VAE, self.devices]
            * workflow.get_resolution_scale(policy.use_upscaler)
            / workflow.ft_frames[workflow.frames_per_step_idx]
        )

    @override
    def calculate_time(
        self,
        policy: Policy,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
        work_pct: float = 1.0,
    ) -> float:
        if not policy.is_disaggregated(Model.FT):
            assert self.get_num_gpus() == 0
            self.time = 0.0
            return self.time
        if self.get_num_gpus() == 0:
            self.time = 0.0
            return self.time

        vae_time_per_frame = self._calc_time_per_frame(
            policy,
            workflow,
            latency_data,
        )
        self.time = _calculate_total_time(
            math.ceil(workflow.total_frames[Model.FT] * work_pct),
            self.replicas,
            vae_time_per_frame)
        return self.time

    @override
    def calculate_time_first(
        self,
        policy: Policy,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
    ) -> float:
        if not policy.is_disaggregated(Model.FT):
            assert self.get_num_gpus() == 0
            self.time_first = 0.0
            return self.time_first
        if self.get_num_gpus() == 0:
            self.time_first = 0.0
            return self.time_first

        vae_time_per_frame = self._calc_time_per_frame(
            policy,
            workflow,
            latency_data,
        )
        num_frames = workflow.per_subscene_frames[Model.FT]
        self.time_first = num_frames * vae_time_per_frame
        return self.time_first

    @override
    def calculate_energy(
        self,
        workflow: WorkflowConfig,
        power_data: Optional[PowerData] = None,
        total_time_s: float = 0.0,
    ) -> float:
        if self.get_num_gpus() == 0 or power_data is None:
            self.energy = 0.0
            return self.energy
        self.energy = power_data[self.gpu_type][Model.FT_VAE, self.devices] * self.time * self.replicas
        # Idle energy
        power_idle = power_data[self.gpu_type]["idle"] * self.get_num_gpus()
        time_idle = total_time_s - self.time
        if time_idle > 0:
            self.energy += power_idle * time_idle
        return self.energy

    @override
    def get_max_replicas(
        self,
        workflow: WorkflowConfig,
    ) -> int:
        return workflow.model_work.get(Model.FT_VAE, 1)


@register_model(Model.UPSCALER)
class UpscalerModelAllocation(ModelAllocation):
    """Upscaler model allocation."""
    model: ClassVar[Model] = Model.UPSCALER

    @override
    def calculate_time(
        self,
        policy: Policy,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
        work_pct: float = 1.0,
    ) -> float:
        if self.get_num_gpus() == 0:
            self.time = 0.0
            return self.time
        self.time = _calculate_total_time(
            math.ceil(work_pct * workflow.total_frames[Model.FT]),
            self.replicas,
            latency_data[self.gpu_type][self.model, self.devices])
        return self.time

    @override
    def calculate_time_first(
        self,
        policy: Policy,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
    ) -> float:
        if not policy.use_upscaler:
            assert self.get_num_gpus() == 0
        if self.get_num_gpus() == 0:
            self.time_first = 0.0
            return self.time_first

        self.time_first = (
            workflow.per_subscene_frames[Model.FT]
            * latency_data[self.gpu_type][self.model, self.devices]
        )
        return self.time_first

    @override
    def calculate_energy(
        self,
        workflow: WorkflowConfig,
        power_data: Optional[PowerData] = None,
        total_time_s: float = 0.0,
    ) -> float:
        if self.get_num_gpus() == 0 or power_data is None:
            self.energy = 0.0
            return self.energy
        # Assumes a single device and multiple replicas
        self.energy = power_data[self.gpu_type][self.model, self.devices] * self.time * self.replicas
        # Idle energy
        power_idle = power_data[self.gpu_type]["idle"] * self.get_num_gpus()
        time_idle = total_time_s - self.time
        if time_idle > 0:
            self.energy += power_idle * time_idle
        return self.energy

    @override
    def get_max_replicas(
        self,
        workflow: WorkflowConfig,
    ) -> int:
        return workflow.model_work.get(Model.UPSCALER, 1)


@register_model(Model.OTHERS)
class OthersModelAllocation(ModelAllocation):
    """Others: Kokoro + YOLO."""
    model: ClassVar[Model] = Model.OTHERS

    @override
    def calculate_time(
        self,
        policy: Policy,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
        work_pct: float = 1.0,
    ) -> float:
        if self.get_num_gpus() == 0:
            self.time = 0.0
            return self.time

        self.time = (
            workflow.total_scenes
            * latency_data[self.gpu_type][self.model, self.devices]
        )
        return self.time

    @override
    def calculate_time_first(
        self,
        policy: Policy,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
    ) -> float:
        if self.get_num_gpus() == 0:
            self.time_first = 0.0
            return self.time_first

        self.time_first = latency_data[self.gpu_type][self.model, self.devices]
        return self.time_first

    @override
    def calculate_energy(
        self,
        workflow: WorkflowConfig,
        power_data: Optional[PowerData] = None,
        total_time_s: float = 0.0,
    ) -> float:
        if self.get_num_gpus() == 0 or power_data is None:
            self.energy = 0.0
            return self.energy
        # Idle energy; not much GPU usage
        power_idle = power_data[self.gpu_type]["idle"] * self.get_num_gpus()
        self.energy = power_idle * self.time
        return self.energy
