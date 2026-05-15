from __future__ import annotations

import pandas as pd
import numpy as np

from typing import Optional
from typing import ClassVar

from abc import ABC
from abc import abstractmethod

from dataclasses import dataclass
from dataclasses import field

from enum import Enum


class GPUType(Enum):
    A100 = "A100"
    H100 = "H100"
    H200 = "H200"
    GB200 = "GB200"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, GPUType):
            return NotImplemented
        order = [GPUType.A100, GPUType.H100, GPUType.H200, GPUType.GB200]
        return order.index(self) < order.index(other)


class QualityLevel(Enum):
    ORIGINAL = "original"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Pixel counts per quality level (16:10 aspect ratio).
# Latency data is profiled at MEDIUM resolution.
RESOLUTION_PIXELS: dict[QualityLevel, int] = {
    QualityLevel.HIGH: 1280 * 800,
    QualityLevel.MEDIUM: 640 * 400,
    QualityLevel.LOW: 320 * 200,
}


class Model(Enum):
    GEMMA = "gemma"
    FLUX = "flux"
    HF = "hf"  # HunyuanFramePack
    HF_VAE = "hf_vae"  # HunyuanFramePack VAE
    FT = "ft"  # FantasyTalking
    FT_VAE = "ft_vae"  # FantasyTalking VAE
    UPSCALER = "upscaler"
    OTHERS = "others"  # YOLO + Kokoro


# Used for FIFO
MODEL_ORDER: dict[Model, int] = {
    Model.GEMMA: 0,
    Model.FLUX: 1,
    Model.OTHERS: 2,
    Model.HF: 3,
    Model.HF_VAE: 4,
    Model.FT: 5,
    Model.FT_VAE: 6,
    Model.UPSCALER: 7,
}


@dataclass
class ModelAllocation(ABC):
    model: ClassVar[Model]

    # policy TODO
    # workflow TODO
    gpu_type: GPUType
    devices: int = 1
    replicas: int = 0  # No replicas by default
    work: int = 0
    time: float = 0.0
    time_first: float = 0.0
    energy: float = 0.0
    cost: float = 0.0

    def __str__(self) -> str:
        if self.replicas <= 0:
            assert self.time == 0.0, f"time must be 0 when no replicas, got {self.time:.2f}"
            assert self.energy == 0.0, f"energy must be 0 when no replicas, got {self.energy:.2f}"
            return "--"
        return \
            f"devices={self.devices:2d}, " \
            f"replicas={self.replicas}, " \
            f"work={self.work}, " \
            f"time={self.time:.2f} secs, " \
            f"time_first={self.time_first:.2f} secs, " \
            f"energy={self.energy / 60.0 / 60.0:.2f} Wh, " \
            f"cost=${self.cost:.2f}"

    def __repr__(self) -> str:
        return self.__str__()

    def __post_init__(self) -> None:
        if self.replicas > 0:
            return
        if self.time != 0.0 or self.energy != 0.0:
            raise ValueError(
                f"time and energy must be 0.0 when no replicas, got time={self.time:.2f}, energy={self.energy:.2f}")

    def get_num_gpus(self) -> int:
        if self.replicas <= 0:
            return 0
        return self.devices * self.replicas

    def disable(self) -> None:
        self.devices = 0
        self.replicas = 0
        self.time = 0.0
        self.time_first = 0.0
        self.energy = 0.0

    @abstractmethod
    def calculate_time(
        self,
        policy: Policy,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
        work_pct: float = 1.0,
    ) -> float:
        ...

    @abstractmethod
    def calculate_time_first(
        self,
        policy: Policy,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
    ) -> float:
        ...

    @abstractmethod
    def calculate_energy(
        self,
        workflow: WorkflowConfig,
        power_data: Optional[PowerData] = None,
        total_time_s: float = 0.0,
    ) -> float:
        ...

    def calculate_cost(
        self,
        policy: Policy,
        total_time_s: float = 0.0,
    ) -> float:
        """Calculate the cost for this model allocation."""
        SECONDS_IN_HOUR = 60 * 60
        gpu_cost = policy.gpu_cost[self.gpu_type]
        self.cost = total_time_s * (self.get_num_gpus() * gpu_cost) / SECONDS_IN_HOUR
        return self.cost

    def calculate(
        self,
        policy: Policy,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
        power_data: Optional[PowerData] = None,
        total_time_s: float = 0.0,
        work_pct: float = 1.0,
    ) -> None:
        """Calculate all the values for this model allocation."""
        self.calculate_time(policy, workflow, latency_data, work_pct)
        self.calculate_time_first(policy, workflow, latency_data)
        self.calculate_cost(policy, total_time_s)
        self.calculate_energy(workflow, power_data, total_time_s)

    def get_max_replicas(
        self,
        workflow: WorkflowConfig,
    ) -> int:
        """Get the maximum number of replicas that can leverage parallelism."""
        return 1


class Objective(Enum):
    FIFO = "fifo"
    TIME = "time"
    TTFF = "ttff"
    COST = "cost"
    ENERGY = "energy"
    TIME_COST = "time_cost"
    TTFF_COST = "ttff_cost"
    ENERGY_COST = "energy_cost"
    TIME_ENERGY = "time_energy"
    RANDOM = "random"
    NONE = "none"

    TTFF_THEN_TIME = "ttff_then_time"  # first minimize ttff, then minimize time

    def is_monotonic(self) -> bool:
        return self not in {Objective.RANDOM, Objective.FIFO}


@dataclass
class WorkflowConfig:
    total_video_seconds: int
    total_scenes: int
    total_frames: dict[Model, int]
    total_subscenes: int
    per_subscene_frames: dict[Model, int]
    # default per-frame number of denoising steps
    num_steps: dict[Model, int]
    # supported number of generation frames
    hf_frames: list[int]
    ft_frames: list[int]
    frames_per_step_idx: int
    # target output resolution (default: HIGH)
    target_resolution: QualityLevel = QualityLevel.HIGH

    # total input tokens
    total_input_tokens: int = 0

    # work per model (determines parallelism; work > 1 means parallelizable across replicas)
    # models included in the workflow are derived from the keys of this dict
    model_work: dict[Model, int] = field(default_factory=dict)

    @property
    def models(self) -> list[Model]:
        """Models included in the workflow (derived from model_work keys)."""
        return list(self.model_work.keys())

    @property
    def work(self) -> dict[Model, int]:
        """Units of work per model (0 for models not in the workflow)."""
        return {
            model_name: self.model_work.get(model_name, 0)
            for model_name in Model
        }

    def get_model_order(self) -> list[Model]:
        """Get ordered list of models in the workflow, sorted by MODEL_ORDER."""
        return sorted(
            [m for m in self.models if m in MODEL_ORDER],
            key=lambda m: MODEL_ORDER[m],
        )

    def get_resolution_scale(self, use_upscaler: bool) -> float:
        """Compute latency scaling factor based on target resolution.

        Latency data is profiled at MEDIUM resolution.  The scale factor
        adjusts for the actual generation resolution:

        1. Upscaler used, HIGH   → 1.0 (models generate at MEDIUM)
        2. Upscaler used, MEDIUM → LOW / MEDIUM (models generate at LOW)
        3. No upscaler, HIGH     → HIGH / MEDIUM  (scale up)
        4. No upscaler, MEDIUM   → 1.0
        5. No upscaler, LOW      → LOW / MEDIUM   (scale down)
        """
        if use_upscaler:
            assert self.target_resolution in (QualityLevel.HIGH, QualityLevel.MEDIUM), \
                "Upscaler can only be used when target resolution is HIGH or MEDIUM"
            if self.target_resolution == QualityLevel.HIGH:
                return 1.0
            # MEDIUM target with upscaler: generate at LOW, upscale to MEDIUM
            return RESOLUTION_PIXELS[QualityLevel.LOW] / RESOLUTION_PIXELS[QualityLevel.MEDIUM]
        if self.target_resolution == QualityLevel.MEDIUM:
            return 1.0
        return RESOLUTION_PIXELS[self.target_resolution] / RESOLUTION_PIXELS[QualityLevel.MEDIUM]

    def is_parallelizable(self, model: Model) -> bool:
        """Whether the given model can be parallelized across multiple replicas."""
        return self.model_work.get(model, 0) > 1

    def filter_parallelizable_models(
        self,
        models: list[Model],
        disaggregation: dict[Model, bool],
    ) -> list[Model]:
        filtered_models = [
            model
            for model in models
            if self.is_parallelizable(model)
        ]
        # Remove VAE models when their parent model disaggregation is disabled
        if not disaggregation.get(Model.HF, False):
            filtered_models = [m for m in filtered_models if m != Model.HF_VAE]
        if not disaggregation.get(Model.FT, False):
            filtered_models = [m for m in filtered_models if m != Model.FT_VAE]
        return filtered_models

    def __post_init__(self) -> None:
        assert self.total_frames[Model.HF] > self.per_subscene_frames[Model.HF]
        assert self.total_frames[Model.FT] > self.per_subscene_frames[Model.FT]

        # If no models specified, populate defaults for all models
        if not self.model_work:
            defaults: dict[Model, int] = {
                Model.GEMMA: 1,
                Model.FLUX: 1,
                Model.HF: self.total_subscenes,
                Model.HF_VAE: self.total_frames[Model.HF],
                Model.FT: self.total_subscenes,
                Model.FT_VAE: self.total_frames[Model.FT],
                Model.UPSCALER: self.total_frames[Model.FT],
                Model.OTHERS: 1,
            }
            for model, work in defaults.items():
                self.model_work[model] = work
        if self.target_resolution != QualityLevel.HIGH:
            if Model.UPSCALER in self.model_work:
                del self.model_work[Model.UPSCALER]

    @property
    def num_frames(self) -> int:
        """Number of frames generated by the workflow."""
        if Model.FT in self.total_frames:
            return self.total_frames[Model.FT]
        return 0


class ActionName(Enum):
    MERGE = "merge"
    ADD_DEVICE = "add device"
    ADD_REPLICA = "add replica"
    ADD_DEVICE_REPLICA = "add device replica"
    ADD_INSTANCE = "add instance"
    REMOVE_DEVICE = "remove device"
    REMOVE_REPLICA = "remove replica"


@dataclass
class Action:
    """
    Optimization action to take.
    """
    name: ActionName
    model: Model
    gpu_type: GPUType
    models: dict[GPUType, dict[Model, list[ModelAllocation]]]

    action_result: Result = field(repr=False)

    arrival_time_s: float = 0.0  # For FIFO scheduling

    # Derived fields from action_result (not passed by caller)
    time: float = field(init=False)  # Total execution time
    ttff: float = field(init=False)  # Time to first frame
    cost: float = field(init=False)  # Cost in $
    energy: float = field(init=False)  # Energy in W*s

    def __post_init__(self) -> None:
        # ---- type checks ----
        if not isinstance(self.model, Model):
            raise ValueError(f"Model {self.model} [{type(self.model)}] not supported")
        if not isinstance(self.name, ActionName):
            raise ValueError(f"Action name {self.name} [{type(self.name)}] not supported")
        if not isinstance(self.models, dict):
            raise ValueError(f"models must be a dict, got {type(self.models)}")
        if not isinstance(self.gpu_type, GPUType):
            raise ValueError(f"Device type {self.gpu_type} [{type(self.gpu_type)}] not supported")
        """
        if not isinstance(self.allocation_id, int) or self.allocation_id < 0:
            raise ValueError(f"Allocation ID {self.allocation_id} must be a non-negative integer")
        if self.num_replicas <= 0:
            raise ValueError(f"num_replicas {self.num_replicas} must be > 0")
        if self.num_devices <= 0:
            raise ValueError(f"num_devices {self.num_devices} must be > 0")
        """
        # ---- derive values ----
        self.time = self.action_result.total_time_s
        self.ttff = self.action_result.ttff_s
        self.cost = self.action_result.cost
        self.energy = self.action_result.total_energy
        if self.cost < 0.0:
            raise ValueError("cost must be >= 0")

    def __str__(self) -> str:
        return (
            f"Action("
            f"{self.name.value}, "
            f"model={self.model.value}, "
            f"gpu={self.gpu_type.value}, "
            f"time={self.time:.2f} s, "
            f"ttff={self.ttff:.2f} s, "
            f"cost=${self.cost:.2f}, "
            f"time*cost={self.time_cost():.2f}, "
            f"ttff*cost={self.ttff_cost():.2f}, "
            f"energy*cost={self.energy_cost():.2f}, "
            f"time*energy={self.time_energy():.2f}, "
            f"energy={self.energy:.2f} Ws, "
            f"models={self.models}"
            f")"
        )

    def time_cost(self) -> float:
        """We use improvement in time * $."""
        if self.time <= 0:
            return self.cost
        if self.cost <= 0:
            return self.time
        return self.time * self.cost

    def ttff_cost(self) -> float:
        """We use improvement in TTFF * $."""
        if self.ttff <= 0:
            return self.cost
        if self.cost <= 0:
            return self.ttff
        return self.ttff * self.cost

    def energy_cost(self) -> float:
        """We use improvement in Wh * $."""
        if self.cost <= 0:
            return self.energy
        if self.energy <= 0:
            return self.cost
        return self.energy * self.cost

    def time_energy(self) -> float:
        """We use improvement in TTFF * Wh."""
        if self.energy <= 0:
            return self.time
        if self.time <= 0:
            return self.energy
        return self.time * self.energy

    def get_order(self) -> int:
        " ""For FIFO scheduling."" "
        return MODEL_ORDER[self.model]

    def get_metric(
        self,
        obj: Objective,
        switch_objective: bool = False,
    ) -> float:
        if obj == Objective.RANDOM:
            return 0.0
        if obj == Objective.TIME:
            return self.time
        if obj == Objective.TTFF:
            return self.ttff
        if obj == Objective.COST:
            return self.cost
        if obj == Objective.ENERGY:
            return self.energy
        if obj == Objective.TIME_COST:
            return self.time_cost()
        if obj == Objective.TTFF_COST:
            return self.ttff_cost()
        if obj == Objective.ENERGY_COST:
            return self.energy_cost()
        if obj == Objective.TIME_ENERGY:
            return self.time_energy()
        if obj == Objective.FIFO:
            # return self.get_order()
            return 0  # TODO
        if obj == Objective.TTFF_THEN_TIME:
            if switch_objective:
                return self.time
            else:
                return self.ttff
        raise ValueError(f"Unknown objective {obj}")


@dataclass
class Result:
    total_time_s: float = 0.0
    first_chunk_time: float = 0.0  # Time to first chunk
    ttff_s: float = 0.0  # Time to first frame (accounts for total time and workflow length)
    tbf_s: float = 0.0  # Time between frames
    total_energy: float = 0.0  # Watts x second
    cost: float = 0.0  # Total $ cost
    gpus_used: dict[GPUType, int] = field(default_factory=dict)
    gpus_total: dict[GPUType, int] = field(default_factory=dict)
    models: dict[GPUType, dict[Model, list[ModelAllocation]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.total_time_s >= 0.0, f"total_time_s={self.total_time_s} must be >= 0.0"
        assert self.first_chunk_time >= 0.0, f"first_chunk_time={self.first_chunk_time} must be >= 0.0"
        assert self.ttff_s >= 0.0, f"ttff_s={self.ttff_s} must be >= 0.0"
        assert self.tbf_s >= 0.0, f"tbf_s={self.tbf_s} must be >= 0.0"
        assert self.total_energy >= 0.0, f"total_energy={self.total_energy} must be >= 0.0"
        assert self.cost >= 0.0, f"cost={self.cost} must be >= 0.0"
        assert len(self.gpus_used) >= 0, f"gpus_used cannot be empty: {self.gpus_used}"
        for gpu_used in self.gpus_used.values():
            assert gpu_used >= 0, f"all gpus_used value {self.gpus_used} must be >= 0"

    def to_csv(self) -> str:
        num_a100 = self.gpus_used.get(GPUType.A100, 0)
        num_h100 = self.gpus_used.get(GPUType.H100, 0)
        num_h200 = self.gpus_used.get(GPUType.H200, 0)
        num_gb200 = self.gpus_used.get(GPUType.GB200, 0)
        return (
            f"{num_a100},{num_h100},{num_h200},{num_gb200},"
            f"{self.ttff_s:.2f},{self.tbf_s:.2f},{self.cost:.2f},"
            f"{self.total_time_s:.2f},{self.total_energy:.2f}"
        )

    def __str__(self) -> str:
        SECONDS_IN_HOUR = 60 * 60
        return (
            f"Time:{self.total_time_s:.2f} s TTFF:{self.ttff_s:.2f} s "
            f"Cost:${self.cost:.2f} TTFF*Cost:{self.ttff_s * self.cost:.2f} "
            f"Energy:{self.total_energy / SECONDS_IN_HOUR / 1000:.2f} kWh "
            f"GPUS: {num_gpus_to_str(self.gpus_used)}"
        )

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class LatencyGPUTypeData:
    gpu_type: GPUType
    # TP -> latency mappings
    flux: dict[int, float] = field(default_factory=dict)
    hf: dict[int, float] = field(default_factory=dict)
    hf_high: dict[int, float] = field(default_factory=dict)
    hf_vae: dict[int, float] = field(default_factory=dict)
    hf_vae_high: dict[int, float] = field(default_factory=dict)
    ft: dict[int, float] = field(default_factory=dict)
    ft_high: dict[int, float] = field(default_factory=dict)
    ft_vae: dict[int, float] = field(default_factory=dict)
    ft_vae_high: dict[int, float] = field(default_factory=dict)
    upscaler: dict[int, float] = field(default_factory=dict)
    gemma_first_scene: dict[int, float] = field(default_factory=dict)
    gemma_per_scene: dict[int, float] = field(default_factory=dict)
    others: dict[int, float] = field(default_factory=dict)

    def __getitem__(
        self,
        key: Model | tuple[Model, int]
    ) -> float:
        if isinstance(key, tuple):
            assert isinstance(key[0], Model)
            assert isinstance(key[1], int)
            model, num_devices = key
            if model == Model.FLUX:
                return self.flux[num_devices]
            if model == Model.HF:
                return self.hf[num_devices]
            if model == Model.HF_VAE:
                return self.hf_vae[num_devices]
            if model == Model.FT:
                return self.ft[num_devices]
            if model == Model.FT_VAE:
                return self.ft_vae[num_devices]
            if model == Model.GEMMA:
                return self.gemma_first_scene[num_devices]
            if model == Model.UPSCALER:
                return self.upscaler[num_devices]
            if model == Model.OTHERS:
                return self.others[num_devices]
        raise KeyError(f"Latency for model {key} not found")

    def __contains__(self, key: Model | tuple[Model, int]) -> bool:
        if isinstance(key, tuple):
            assert isinstance(key[0], Model)
            assert isinstance(key[1], int)
            model, num_devices = key
            if model == Model.GEMMA:
                return num_devices in self.gemma_first_scene
            if model == Model.FLUX:
                return num_devices in self.flux
            if model == Model.HF:
                return num_devices in self.hf
            if model == Model.HF_VAE:
                return num_devices in self.hf_vae
            if model == Model.FT:
                return num_devices in self.ft
            if model == Model.FT_VAE:
                return num_devices in self.ft_vae
            if model == Model.UPSCALER:
                return num_devices in self.upscaler
            if model == Model.HF_VAE:
                return num_devices in self.hf_vae
            if model == Model.OTHERS:
                return num_devices in self.others
        return False

    def get_max_parallelism(self, model: Model) -> int:
        """Max number of devices supported for the given model."""
        if model == Model.FLUX:
            return max(self.flux.keys())
        if model == Model.HF:
            return max(self.hf.keys())
        if model == Model.FT:
            return max(self.ft.keys())
        if model == Model.FT_VAE:
            return max(self.ft_vae.keys())
        if model == Model.GEMMA:
            return max(self.gemma_first_scene.keys())
        if model == Model.UPSCALER:
            return max(self.upscaler.keys())
        if model == Model.HF_VAE:
            return max(self.hf_vae.keys())
        if model == Model.OTHERS:
            return max(self.others.keys())
        raise KeyError(f"Model {model} not found in latency data")


@dataclass
class PowerGPUTypeData:
    gpu_type: GPUType
    # TP -> power mappings
    flux: dict[int, float] = field(default_factory=dict)
    hf: dict[int, float] = field(default_factory=dict)
    hf_high: dict[int, float] = field(default_factory=dict)
    hf_vae: dict[int, float] = field(default_factory=dict)
    hf_vae_high: dict[int, float] = field(default_factory=dict)
    ft: dict[int, float] = field(default_factory=dict)
    ft_high: dict[int, float] = field(default_factory=dict)
    ft_vae: dict[int, float] = field(default_factory=dict)
    ft_vae_high: dict[int, float] = field(default_factory=dict)
    upscaler: dict[int, float] = field(default_factory=dict)
    gemma_first_scene: dict[int, float] = field(default_factory=dict)
    gemma_per_scene: dict[int, float] = field(default_factory=dict)
    # Other values
    idle: float = 0.0  # Idle power in Watts
    tdp: float = 0.0  # TDP power in Watts

    def __getitem__(
        self,
        key: Model | tuple[Model, int] | str
    ) -> float:
        if isinstance(key, tuple):
            assert isinstance(key[0], Model)
            assert isinstance(key[1], int)
            model, devices = key
            if model == Model.FLUX:
                return self.flux[devices]
            if model == Model.HF:
                return self.hf[devices]
            if model == Model.HF_VAE:
                return self.hf_vae[devices]
            if model == Model.FT:
                return self.ft[devices]
            if model == Model.FT_VAE:
                return self.ft_vae[devices]
            if model == Model.UPSCALER:
                return self.upscaler[devices]
        if isinstance(key, str):
            if key == "idle":
                return self.idle
            if key == "tdp":
                return self.tdp
        raise KeyError(f"Power for {key} not found")


@dataclass
class LatencyData:
    gpus: dict[GPUType, LatencyGPUTypeData]

    def __getitem__(self, gpu_type: GPUType) -> LatencyGPUTypeData:
        return self.gpus[gpu_type]

    def __setitem__(
        self,
        gpu_type: GPUType,
        latency_data: LatencyGPUTypeData
    ) -> None:
        self.gpus[gpu_type] = latency_data


@dataclass
class PowerData:
    gpus: dict[GPUType, PowerGPUTypeData]

    def __getitem__(self, gpu_type: GPUType) -> PowerGPUTypeData:
        return self.gpus[gpu_type]

    def __setitem__(
        self,
        gpu_type: GPUType,
        power_data: PowerGPUTypeData
    ) -> None:
        self.gpus[gpu_type] = power_data


def num_gpus_to_str(
    provision: dict[GPUType, int]
) -> str:
    return "+".join([
        f"{num_gpus}x{gpu_type.name}"
        for gpu_type, num_gpus in provision.items()
        if num_gpus > 0
    ])


@dataclass
class Provision:
    num_gpus: dict[GPUType, int] = field(default_factory=dict)

    def __getitem__(self, gpu_type: GPUType) -> int:
        return self.num_gpus[gpu_type]

    def __str__(self) -> str:
        return num_gpus_to_str(self.num_gpus)


@dataclass
class ProvisioningResult:
    latencies: list[float]
    costs: list[float]
    ttffs: list[float]
    tbfs: list[float]
    actual_provision: list[dict[GPUType, int]]
    config_provision: list[dict[GPUType, int]]
    model_provision: list[dict[GPUType, dict[Model, list[ModelAllocation]]]]
    qualities: list[float] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)

    def save(
        self,
        policy_name: str,
        results_dir: str,
    ) -> None:
        """Save the provisioning results to a CSV file."""
        num_a100: list[int] = []
        num_h100: list[int] = []
        num_h200: list[int] = []
        num_gb200: list[int] = []
        for provision in self.actual_provision:
            num_a100.append(provision.get(GPUType.A100, 0))
            num_h100.append(provision.get(GPUType.H100, 0))
            num_h200.append(provision.get(GPUType.H200, 0))
            num_gb200.append(provision.get(GPUType.GB200, 0))
        df_latency = pd.DataFrame({
            'num_a100': num_a100,
            'num_h100': num_h100,
            'num_h200': num_h200,
            'num_gb200': num_gb200,
            'ttff_s': self.ttffs,
            'tbf_s': self.tbfs,
            'cost': self.costs,
            'total_time': self.latencies,
            'energy': self.energies,
        })
        df_latency[['ttff_s', 'tbf_s', 'cost', 'total_time', 'energy']] = (
            df_latency[['ttff_s', 'tbf_s', 'cost', 'total_time', 'energy']].round(2)
        )
        policy_name_clean = policy_name.replace(" ", "_").replace("*", "x").replace("/", "_").lower()
        file_name = results_dir + f"provisioning_{policy_name_clean}.csv"
        df_latency.to_csv(file_name, index=False)

    def get_pareto_frontier(
        self,
        max_x: Optional[float] = None,
        max_y: Optional[float] = None,
    ) -> np.ndarray:
        from utils import get_pareto_frontier  # TODO this is a lazy fix, we need to reset
        # points = np.array(list(zip(self.ttffs, self.costs)))
        return get_pareto_frontier(
            self.ttffs,
            self.costs,
            max_x=max_x,
            max_y=max_y,
        )


class Solver(Enum):
    GUROBI = "gurobi"
    HIGHS = "highs"
    GREEDY = "greedy"
    NAIVE = "naive"
    HEXGEN = "hexgen"
    HELIX = "helix"


@dataclass
class Policy:
    name: str
    gpu_cost: dict[GPUType, float]
    objective: Objective
    disaggregation: dict[Model, bool]
    use_upscaler: bool
    hardware: list[GPUType] = field(default_factory=lambda: [GPUType.A100, GPUType.H100, GPUType.H200, GPUType.GB200])
    solver: Solver = Solver.GREEDY

    def is_disaggregated(self, model: Model) -> bool:
        """Check if a model has disaggregation enabled."""
        return self.disaggregation.get(model, False)

    def __str__(self) -> str:
        disag_str = {
            model.value: disaggregated
            for model, disaggregated in self.disaggregation.items()
            if disaggregated
        }
        return (
            f"Policy({self.name}, "
            f"objective={self.objective}, "
            f"disag={disag_str}, "
            f"upscaler={self.use_upscaler}, "
            f"cost={self.gpu_cost}, "
            f"solver={self.solver})"
        )
