"""
MILP formulation for the StreamWise workflow allocation problem.
"""

from __future__ import annotations

import json
import logging

from typing import Callable
from typing import Optional

from pyomo.environ import ConcreteModel
from pyomo.environ import Var
from pyomo.environ import Set
from pyomo.environ import Objective as OptObjective
from pyomo.environ import Binary
from pyomo.environ import NonNegativeIntegers
from pyomo.environ import NonNegativeReals
from pyomo.environ import minimize
from pyomo.environ import SolverFactory
from pyomo.environ import ConstraintList

from sim_types import GPUType
from sim_types import Model
from sim_types import WorkflowConfig
from sim_types import LatencyData
from sim_types import PowerData
from sim_types import Result
from sim_types import Policy
from sim_types import ModelAllocation
from sim_types import Objective
from sim_types import Solver

from models import get_model_allocation

from model_allocator import ModelAllocator

from constants import DEVICE_OPTIONS
from constants import NUM_GPUS_PER_SERVER
from constants import SECONDS_IN_HOUR

from policies import STREAMWISE_MILP_POLICY


MAX_INSTANCES = 16

# Maximum time it can take: 24 hours in seconds
# Used for big-M constraints to link TTFF and makespan to instance variables
MAX_TIME = 24 * SECONDS_IN_HOUR


# Allocators that require quadratic (bilinear) objectives - need Gurobi
QUADRATIC_OBJECTIVES = [
    Objective.TTFF_COST,
    Objective.TIME_ENERGY,
    Objective.ENERGY_COST,
]


def idx(
    gpu_type: GPUType,
    model_name: Model,
    instance_id: int
) -> tuple[str, str, int]:
    """Helper to convert enum to index key for instance variables."""
    return (gpu_type.value, model_name.value, instance_id)


def dev_idx(
    gpu_type: GPUType,
    model_name: Model,
    instance_id: int,
    num_devices: int
) -> tuple[str, str, int, int]:
    """Helper to convert enum to index key for device variables."""
    return (gpu_type.value, model_name.value, instance_id, num_devices)


class MILPAllocator(ModelAllocator):
    """
    MILP-based allocator that computes the optimal model allocation.
    """
    def __init__(
        self,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
        power_data: Optional[PowerData] = None,
        policy: Policy = STREAMWISE_MILP_POLICY,
    ) -> None:
        super().__init__(
            workflow,
            latency_data,
            power_data,
            policy,
        )
        assert self.policy.solver in [Solver.GUROBI, Solver.HIGHS]

    def allocate(
        self,
        num_gpus: dict[GPUType, int],
        verbose: bool = False,
        running_cost: bool = False,  # If True, cost = active time only; False = makespan x GPUs
        max_cost: Optional[float] = None,  # If set, adds a constraint to limit cost
        max_ttff: Optional[float] = None,  # If set, adds a constraint to limit TTFF
        max_makespan: Optional[float] = None,  # If set, adds a constraint to limit makespan
        time_limit: Optional[int] = None,  # Time limit for the solver in seconds
        save_solution_path: Optional[str] = None,  # If set, saves the solution to a JSON file
        warm_start_path: Optional[str] = None,  # If set, loads a warm start solution from a JSON file
        force_num_gpus: bool = False,  # If True, adds constraints to force the use of all available GPUs
        skip_server_constraint: bool = False,  # If True, skips the GPU-per-server constraint
    ) -> Result:
        """
        Calculate the optimal model allocation and resulting metrics using MILP formulation.
        """
        m = ConcreteModel()

        # Options: "gurobi", "highs"
        solver_name = self.policy.solver.value

        # Define index sets
        gpu_types = list(num_gpus.keys())

        model_names = [
            Model.GEMMA,
            Model.FLUX,
            Model.HF,
            # Model.HF_VAE,
            Model.FT,
            # Model.FT_VAE,
            # Model.UPSCALER,
            Model.OTHERS,
        ]
        if self.policy.use_upscaler:
            model_names.append(Model.UPSCALER)
        if self.policy.is_disaggregated(Model.HF):
            model_names.append(Model.HF_VAE)
        if self.policy.is_disaggregated(Model.FT):
            model_names.append(Model.FT_VAE)

        # Remove models not in the workflow
        model_names = [
            model_name
            for model_name in model_names
            if model_name in self.workflow.models
        ]

        instance_ids = list(range(MAX_INSTANCES))

        # The units of work that each model has to do
        work: dict[Model, int] = self.workflow.work

        # Create Pyomo Sets
        m.GPU_TYPES = Set(initialize=[g.value for g in gpu_types])
        m.MODEL_NAMES = Set(initialize=[mn.value for mn in model_names])
        m.INSTANCES = Set(initialize=instance_ids)

        # Create index set for device choices: (gpu_type, model_name, instance_id, device_count)
        device_index_set = [
            (gpu_type.value, model_name.value, instance_id, num_devices)
            for gpu_type in gpu_types
            for model_name in model_names
            for instance_id in instance_ids
            for num_devices in [0] + DEVICE_OPTIONS[model_name]
        ]
        m.DEVICE_INDEX = Set(initialize=device_index_set)

        # Create index set for instance variables: (gpu_type, model_name, instance_id)
        instance_index_set = [
            (gpu_type.value, model_name.value, instance_id)
            for gpu_type in gpu_types
            for model_name in model_names
            for instance_id in instance_ids
        ]
        m.INSTANCE_INDEX = Set(initialize=instance_index_set)

        # Define indexed variables
        m.device_choice = Var(m.DEVICE_INDEX, domain=Binary)
        m.work_device = Var(m.DEVICE_INDEX, domain=NonNegativeIntegers)  # Linearization: work per device choice
        m.gpus = Var(m.INSTANCE_INDEX, domain=NonNegativeIntegers)
        m.is_active = Var(m.INSTANCE_INDEX, domain=Binary)
        m.is_min = Var(m.INSTANCE_INDEX, domain=Binary)
        m.work = Var(m.INSTANCE_INDEX, domain=NonNegativeIntegers)
        m.time = Var(m.INSTANCE_INDEX, domain=NonNegativeReals)
        m.ttff = Var(m.INSTANCE_INDEX, domain=NonNegativeReals)

        # Objective variables
        m.makespan = Var(domain=NonNegativeReals)
        m.ttff_user = Var(domain=NonNegativeReals)
        m.ttff_min = Var(m.MODEL_NAMES, domain=NonNegativeReals)  # Per-model minimum TTFF
        m.time_max = Var(m.MODEL_NAMES, domain=NonNegativeReals)  # Per-model maximum time
        m.cost = Var(domain=NonNegativeReals)
        m.energy = Var(domain=NonNegativeReals)

        # Constraint list for dynamic constraints
        m.constraints = ConstraintList()

        for gpu_type in gpu_types:
            for model_name in model_names:
                for instance_id in instance_ids:
                    key = idx(gpu_type, model_name, instance_id)

                    # GPUs used = sum of num_devices * device_choice[num_devices]
                    m.constraints.add(
                        m.gpus[key] == sum(
                            num_devices * m.device_choice[dev_idx(gpu_type, model_name, instance_id, num_devices)]
                            for num_devices in [0] + DEVICE_OPTIONS[model_name]
                        )
                    )

                    # Cannot select inactive instance as min
                    m.constraints.add(m.is_min[key] <= m.is_active[key])
                    # If active = 0 -> GPUs = 0
                    m.constraints.add(m.gpus[key] <= num_gpus[gpu_type] * m.is_active[key])
                    # If active = 1 -> GPUs ≥ 1
                    m.constraints.add(m.gpus[key] >= m.is_active[key])
                    # If work = 0 -> active = 0 -> GPUs = 0
                    m.constraints.add(m.is_active[key] <= m.work[key])

                    # If device = 0 -> work = 0
                    dev_idx_0 = dev_idx(gpu_type, model_name, instance_id, 0)
                    m.constraints.add(
                        m.work[key]
                        <= work[model_name] * (1 - m.device_choice[dev_idx_0])
                    )

                    # Linearization: work_device links device_choice and work
                    # work = sum(work_device[d] for d in devices) - excludes 0 GPUs since they can't do work
                    m.constraints.add(
                        m.work[key] == sum(
                            m.work_device[dev_idx(gpu_type, model_name, instance_id, num_devices)]
                            for num_devices in DEVICE_OPTIONS[model_name]
                        )
                    )
                    # If any non-zero device is selected, work must be >= 1
                    m.constraints.add(
                        m.work[key] >= sum(
                            m.device_choice[dev_idx(gpu_type, model_name, instance_id, num_devices)]
                            for num_devices in DEVICE_OPTIONS[model_name]
                        )
                    )
                    # work_device[d] <= TOTAL_WORK * device_choice[d]
                    for num_devices in [0] + DEVICE_OPTIONS[model_name]:
                        didx = dev_idx(gpu_type, model_name, instance_id, num_devices)
                        m.constraints.add(
                            m.work_device[didx] <= work[model_name] * m.device_choice[didx]
                        )

                    # Link instance time to per-model max time
                    m.constraints.add(m.time[key] <= m.time_max[model_name.value])

                    # Link TTFF to per-model TTFF min
                    # If selected → ttff_min[model] == ttff_var
                    m.constraints.add(m.ttff_min[model_name.value] >= m.ttff[key] - MAX_TIME * (1 - m.is_min[key]))
                    m.constraints.add(m.ttff_min[model_name.value] <= m.ttff[key] + MAX_TIME * (1 - m.is_active[key]))

                # One device per instance
                for instance_id in instance_ids:
                    m.constraints.add(
                        sum(
                            m.device_choice[dev_idx(gpu_type, model_name, instance_id, num_devices)]
                            for num_devices in [0] + DEVICE_OPTIONS[model_name]
                        ) == 1
                    )

                # Symmetry breaking (fill earlier instances first)
                for instance_id in range(MAX_INSTANCES - 1):
                    m.constraints.add(
                        m.gpus[idx(gpu_type, model_name, instance_id)]
                        >= m.gpus[idx(gpu_type, model_name, instance_id + 1)]
                    )

        # Makespan is the sum of max times per model (models run sequentially)
        m.constraints.add(m.makespan == sum(m.time_max[model_name.value] for model_name in model_names))

        # User TTFF definition: sum of min TTFF per model
        m.constraints.add(m.ttff_user >= sum(m.ttff_min[model_name.value] for model_name in model_names))
        m.constraints.add(m.ttff_user >= m.makespan - self.workflow.total_video_seconds)

        # Select exactly 1 instance as the min TTFF instance per model
        for model_name in model_names:
            m.constraints.add(
                sum(
                    m.is_min[idx(gpu_type, model_name, instance_id)]
                    for gpu_type in gpu_types
                    for instance_id in instance_ids
                ) == 1
            )

        # Resolution scaling factor for HF/VAE/FT
        latency_ratio = self.workflow.get_resolution_scale(self.policy.use_upscaler)

        # Time constraints
        # Each model block is guarded by membership in model_names so that
        # the MILP can be built for a subset of models (e.g. Helix per-model).
        for gpu_type in gpu_types:
            # Gemma
            if Model.GEMMA in model_names and work[Model.GEMMA] > 0:
                model_name = Model.GEMMA
                for instance_id in instance_ids:
                    key = idx(gpu_type, model_name, instance_id)
                    # Makespan is the max time across all instances
                    # Linearized: use work_device instead of device_choice * work
                    if work[model_name] > 1:
                        # Parallel: each work unit = 1 scene
                        # Time for w scenes
                        # = gemma_first_scene + gemma_per_scene * (w - 1)
                        # = (gemma_first_scene - gemma_per_scene) * is_active + gemma_per_scene * work
                        # Using linearized variables:
                        # = (gemma_first_scene[d] - gemma_per_scene[d]) * \
                        # device_choice[d] + gemma_per_scene[d] * work_device[d]
                        m.constraints.add(
                            m.time[key] == sum(
                                (
                                    self.latency_data[gpu_type].gemma_first_scene[num_devices]
                                    - self.latency_data[gpu_type].gemma_per_scene[num_devices]
                                )
                                * m.device_choice[dev_idx(gpu_type, model_name, instance_id, num_devices)]
                                + self.latency_data[gpu_type].gemma_per_scene[num_devices]
                                * m.work_device[dev_idx(gpu_type, model_name, instance_id, num_devices)]
                                for num_devices in DEVICE_OPTIONS[model_name]
                            )
                        )
                    else:
                        m.constraints.add(
                            m.time[key] == sum(
                                (
                                    self.latency_data[gpu_type].gemma_first_scene[num_devices]
                                    + self.latency_data[gpu_type].gemma_per_scene[num_devices]
                                    * (self.workflow.total_scenes - 1)
                                )
                                * m.work_device[dev_idx(gpu_type, model_name, instance_id, num_devices)]
                                for num_devices in DEVICE_OPTIONS[model_name]
                            )
                        )
                    # TTFF is for 1 work unit
                    m.constraints.add(
                        m.ttff[key] == sum(
                            m.device_choice[dev_idx(gpu_type, model_name, instance_id, num_devices)]
                            * self.latency_data[gpu_type].gemma_first_scene[num_devices]
                            * 1  # TTFF for tokens in first scene
                            for num_devices in DEVICE_OPTIONS[model_name]
                        )
                    )

            # Flux
            if Model.FLUX in model_names and work[Model.FLUX] > 0:
                model_name = Model.FLUX
                for instance_id in instance_ids:
                    key = idx(gpu_type, model_name, instance_id)
                    # Makespan is the max time across all instances
                    # Linearized: use work_device instead of device_choice * work
                    if work[model_name] > 1:
                        # Parallel: each work unit = 1 scene
                        # Time for w scenes = latency * num_steps_flux * w
                        m.constraints.add(
                            m.time[key] == sum(
                                self.latency_data[gpu_type][model_name, num_devices]
                                * self.workflow.num_steps[model_name]
                                * m.work_device[dev_idx(gpu_type, model_name, instance_id, num_devices)]
                                for num_devices in DEVICE_OPTIONS[model_name]
                            )
                        )
                    else:
                        # Non-parallel: single work unit covers all scenes
                        m.constraints.add(
                            m.time[key] == sum(
                                self.latency_data[gpu_type][model_name, num_devices]
                                * self.workflow.num_steps[model_name]
                                * m.work_device[dev_idx(gpu_type, model_name, instance_id, num_devices)]
                                for num_devices in DEVICE_OPTIONS[model_name]
                            )
                        )
                    # TTFF is for 1 work unit
                    m.constraints.add(
                        m.ttff[key] == sum(
                            m.device_choice[dev_idx(gpu_type, model_name, instance_id, num_devices)]
                            * self.latency_data[gpu_type][model_name, num_devices]
                            * self.workflow.num_steps[model_name]
                            * 1  # TTFF for first work unit
                            for num_devices in DEVICE_OPTIONS[model_name]
                        )
                    )

            # Hunyuan FramePack
            if Model.HF in model_names and work[Model.HF] > 0:
                model_name = Model.HF
                for instance_id in instance_ids:
                    key = idx(gpu_type, model_name, instance_id)

                    """
                    from models import HFModelAllocation
                    HFModelAllocation(
                        gpu_type,
                        num_devices,
                        replicas=1,
                    )._calc_time_per_subscene(
                        self.policy,
                        self.workflow,
                        self.latency_data[gpu_type]
                    )
                    """

                    # Makespan is the max time across all instances
                    # Linearized: use work_device instead of device_choice * work
                    hf_time_expr = sum(
                        self.workflow.per_subscene_frames[model_name]
                        / self.workflow.hf_frames[self.workflow.frames_per_step_idx]
                        * self.latency_data[gpu_type][model_name, num_devices]
                        * latency_ratio
                        * self.workflow.num_steps[model_name]
                        * m.work_device[dev_idx(gpu_type, model_name, instance_id, num_devices)]
                        for num_devices in DEVICE_OPTIONS[model_name]
                    )
                    # When not disaggregated, VAE runs on the same instance
                    if not self.policy.is_disaggregated(Model.HF):
                        hf_vae_time_per_work = (
                            self.latency_data[gpu_type][Model.HF_VAE, 1]
                            * latency_ratio
                            / self.workflow.hf_frames[self.workflow.frames_per_step_idx]
                        )
                        hf_time_expr += hf_vae_time_per_work * m.work[key]
                    m.constraints.add(m.time[key] == hf_time_expr)
                    # TTFF is for first chunk (can be smaller than subscene when disaggregated)
                    ttff_frames_hf = min(
                        self.workflow.hf_frames[0],
                        self.workflow.per_subscene_frames[model_name])
                    hf_ttff_expr = sum(
                        m.device_choice[dev_idx(gpu_type, model_name, instance_id, num_devices)]
                        * ttff_frames_hf
                        / self.workflow.hf_frames[self.workflow.frames_per_step_idx]
                        * self.latency_data[gpu_type][model_name, num_devices]
                        * latency_ratio
                        * self.workflow.num_steps[model_name]
                        * 1  # TTFF for first chunk
                        for num_devices in DEVICE_OPTIONS[model_name]
                    )
                    # When not disaggregated, add VAE decode time for first chunk
                    if not self.policy.is_disaggregated(Model.HF):
                        hf_vae_ttff = (
                            ttff_frames_hf
                            / self.workflow.hf_frames[self.workflow.frames_per_step_idx]
                            * self.latency_data[gpu_type][Model.HF_VAE, 1]
                            * latency_ratio
                        )
                        hf_ttff_expr += hf_vae_ttff * m.is_active[key]
                    m.constraints.add(m.ttff[key] == hf_ttff_expr)

            # Hunyuan FramePack VAE
            if Model.HF_VAE in model_names and work[Model.HF_VAE] > 0:
                model_name = Model.HF_VAE
                for instance_id in instance_ids:
                    key = idx(gpu_type, model_name, instance_id)
                    # Makespan is the max time across all instances
                    # Linearized: use work_device instead of device_choice * work
                    m.constraints.add(
                        m.time[key] == sum(
                            self.latency_data[gpu_type][model_name, num_devices]
                            * latency_ratio
                            / self.workflow.hf_frames[self.workflow.frames_per_step_idx]
                            * m.work_device[dev_idx(gpu_type, model_name, instance_id, num_devices)]
                            for num_devices in DEVICE_OPTIONS[model_name]
                        )
                    )
                    # TTFF is for 1 subscene
                    m.constraints.add(
                        m.ttff[key] == sum(
                            m.device_choice[dev_idx(gpu_type, model_name, instance_id, num_devices)]
                            * self.workflow.per_subscene_frames[Model.HF]
                            * self.latency_data[gpu_type][model_name, num_devices]
                            * latency_ratio
                            / self.workflow.hf_frames[self.workflow.frames_per_step_idx]  # frames_per_step_hf
                            * 1  # TTFF for first subscene
                            for num_devices in DEVICE_OPTIONS[model_name]
                        )
                    )

            # Fantasy Talking
            if Model.FT in model_names and work[Model.FT] > 0:
                model_name = Model.FT
                for instance_id in instance_ids:
                    key = idx(gpu_type, model_name, instance_id)
                    # Makespan is the max time across all instances
                    # Linearized: use work_device instead of device_choice * work
                    ft_time_expr = sum(
                        self.workflow.per_subscene_frames[model_name]
                        / self.workflow.ft_frames[self.workflow.frames_per_step_idx]
                        * self.latency_data[gpu_type][model_name, num_devices]
                        * latency_ratio
                        * self.workflow.num_steps[model_name]
                        * m.work_device[dev_idx(gpu_type, model_name, instance_id, num_devices)]
                        for num_devices in DEVICE_OPTIONS[model_name]
                    )
                    # When not disaggregated, VAE runs on the same instance
                    if not self.policy.is_disaggregated(Model.FT):
                        ft_vae_time_per_work = (
                            self.latency_data[gpu_type][Model.FT_VAE, 1]
                            * latency_ratio
                            / self.workflow.ft_frames[self.workflow.frames_per_step_idx]
                        )
                        ft_time_expr += ft_vae_time_per_work * m.work[key]
                    m.constraints.add(m.time[key] == ft_time_expr)
                    # TTFF is for 1 work unit (e.g., subscene)
                    ft_ttff_expr = sum(
                        m.device_choice[dev_idx(gpu_type, model_name, instance_id, num_devices)]
                        * self.workflow.per_subscene_frames[model_name]
                        / self.workflow.ft_frames[self.workflow.frames_per_step_idx]
                        * self.latency_data[gpu_type][model_name, num_devices]
                        * latency_ratio
                        * self.workflow.num_steps[model_name]
                        * 1  # TTFF for first subscene
                        for num_devices in DEVICE_OPTIONS[model_name]
                    )
                    # When not disaggregated, add VAE decode time for first subscene
                    if not self.policy.is_disaggregated(Model.FT):
                        ft_vae_ttff = (
                            self.workflow.per_subscene_frames[Model.FT]
                            / self.workflow.ft_frames[self.workflow.frames_per_step_idx]
                            * self.latency_data[gpu_type][Model.FT_VAE, 1]
                            * latency_ratio
                        )
                        ft_ttff_expr += ft_vae_ttff * m.is_active[key]
                    m.constraints.add(m.ttff[key] == ft_ttff_expr)

            # Fantasy Talking VAE
            if Model.FT_VAE in model_names and work[Model.FT_VAE] > 0:
                model_name = Model.FT_VAE
                for instance_id in instance_ids:
                    key = idx(gpu_type, model_name, instance_id)
                    # Makespan is the max time across all instances
                    # Linearized: use work_device instead of device_choice * work
                    m.constraints.add(
                        m.time[key] == sum(
                            self.latency_data[gpu_type][model_name, num_devices]
                            * latency_ratio
                            / self.workflow.ft_frames[self.workflow.frames_per_step_idx]
                            * m.work_device[dev_idx(gpu_type, model_name, instance_id, num_devices)]
                            for num_devices in DEVICE_OPTIONS[model_name]
                        )
                    )
                    # TTFF is for 1 subscene
                    m.constraints.add(
                        m.ttff[key] == sum(
                            m.device_choice[dev_idx(gpu_type, model_name, instance_id, num_devices)]
                            * self.workflow.per_subscene_frames[Model.FT]
                            * self.latency_data[gpu_type][model_name, num_devices]
                            * latency_ratio
                            / self.workflow.ft_frames[self.workflow.frames_per_step_idx]  # frames_per_step_ft
                            * 1  # TTFF for first subscene
                            for num_devices in DEVICE_OPTIONS[model_name]
                        )
                    )

            # Upscaler
            if Model.UPSCALER in model_names and work[Model.UPSCALER] > 0 and self.policy.use_upscaler:
                model_name = Model.UPSCALER
                for instance_id in instance_ids:
                    key = idx(gpu_type, model_name, instance_id)
                    # Linearized: use work_device instead of device_choice * work
                    m.constraints.add(
                        m.time[key] == sum(
                            self.latency_data[gpu_type][model_name, num_devices]
                            * m.work_device[dev_idx(gpu_type, model_name, instance_id, num_devices)]
                            for num_devices in DEVICE_OPTIONS[model_name]
                        )
                    )
                    # TTFF is for 1 work unit (e.g., subscene)
                    m.constraints.add(
                        m.ttff[key] == sum(
                            m.device_choice[dev_idx(gpu_type, model_name, instance_id, num_devices)]
                            * self.latency_data[gpu_type][model_name, num_devices]
                            * self.workflow.per_subscene_frames[Model.FT]
                            * 1  # TTFF is for first subscene
                            for num_devices in DEVICE_OPTIONS[model_name]
                        )
                    )

            # Others
            if Model.OTHERS in model_names and work[Model.OTHERS] > 0:
                model_name = Model.OTHERS
                for instance_id in instance_ids:
                    key = idx(gpu_type, model_name, instance_id)
                    # Makespan is the max time across all instances
                    m.constraints.add(
                        m.time[key] == sum(
                            m.device_choice[dev_idx(gpu_type, model_name, instance_id, num_devices)]
                            * self.latency_data[gpu_type][model_name, num_devices]
                            * self.workflow.total_scenes
                            for num_devices in DEVICE_OPTIONS[model_name]
                        )
                    )
                    # TTFF is for 1 work unit
                    m.constraints.add(
                        m.ttff[key] == sum(
                            m.device_choice[dev_idx(gpu_type, model_name, instance_id, num_devices)]
                            * self.latency_data[gpu_type][model_name, num_devices]
                            * 1  # TTFF is for first scene
                            for num_devices in DEVICE_OPTIONS[model_name]
                        )
                    )

        # Total work to do for each model
        for model_name in model_names:
            m.constraints.add(
                sum(
                    m.work[idx(gpu_type, model_name, instance_id)]
                    for gpu_type in gpu_types
                    for instance_id in instance_ids
                ) == work[model_name]
            )

        # Number of GPUs per type
        # Add a variable to represent the number of servers for each GPU type
        m.num_servers = Var(m.GPU_TYPES, domain=NonNegativeIntegers)

        for gpu_type in gpu_types:
            total_gpus = sum(
                m.gpus[idx(gpu_type, model_name, instance_id)]
                for model_name in model_names
                for instance_id in instance_ids
            )
            if force_num_gpus:
                m.constraints.add(total_gpus == num_gpus[gpu_type])
            else:
                m.constraints.add(total_gpus <= num_gpus[gpu_type])

            # GPUs used must be a multiple of NUM_GPUS_PER_SERVER
            if not skip_server_constraint:
                m.constraints.add(total_gpus == m.num_servers[gpu_type.value] * NUM_GPUS_PER_SERVER[gpu_type])

        # Cost calculation
        # running_cost=True: cost based only on active model running time
        if running_cost:
            cost_expr = sum(
                self._get_latency_per_work(
                    gpu_type,
                    model_name,
                    num_devices,
                )
                * num_devices
                * m.work_device[dev_idx(gpu_type, model_name, instance_id, num_devices)]
                * self.policy.gpu_cost[gpu_type] / SECONDS_IN_HOUR
                for gpu_type in gpu_types
                for model_name in model_names
                for instance_id in instance_ids
                for num_devices in DEVICE_OPTIONS[model_name]
            )
        # running_cost=False: cost = makespan × total_GPUs_used (GPUs allocated for full job duration)
        else:
            cost_expr = m.makespan * sum(
                m.gpus[idx(gpu_type, model_name, instance_id)]
                * self.policy.gpu_cost[gpu_type] / SECONDS_IN_HOUR
                for gpu_type in gpu_types
                for model_name in model_names
                for instance_id in instance_ids
            )
        m.constraints.add(m.cost == cost_expr)

        # Energy: model-specific power * active time + idle power * (makespan - active time)
        if self.power_data is None:
            energy_expr = 0.0
        else:
            # Active energy: Use model-specific power values (not TDP)
            energy_expr = sum(
                self._get_latency_per_work(
                    gpu_type,
                    model_name,
                    num_devices,
                )
                * num_devices
                * m.work_device[dev_idx(gpu_type, model_name, instance_id, num_devices)]
                * (
                    self._get_power_per_work(
                        gpu_type,
                        model_name,
                        num_devices,
                    ) - self.power_data[gpu_type]["idle"]
                )
                for gpu_type in gpu_types
                for model_name in model_names
                for instance_id in instance_ids
                for num_devices in DEVICE_OPTIONS[model_name]
            )
            # Idle energy: idle power * num_gpus * makespan
            energy_expr += sum(
                self.power_data[gpu_type]["idle"] * num_gpus[gpu_type] * m.makespan
                for gpu_type in gpu_types
            )
        m.constraints.add(m.energy == energy_expr)

        # Bounds
        if max_cost is not None:
            m.constraints.add(m.cost <= max_cost)
        if max_ttff is not None:
            m.constraints.add(m.ttff_user <= max_ttff)
        if max_makespan is not None:
            m.constraints.add(m.makespan <= max_makespan)

        # Objective functions
        obj = get_objective(
            m=m,
            allocator=self.policy.objective,
            solver_name=solver_name,
        )
        if obj is not None:
            m.objective = obj

        # Solve
        solver = SolverFactory(solver_name)
        if solver_name == "gurobi" and time_limit:
            solver.options["TimeLimit"] = time_limit
        if solver_name == "highs" and time_limit:
            solver.options["time_limit"] = time_limit
        if self.policy.objective in QUADRATIC_OBJECTIVES and solver_name == "gurobi":
            solver.options['NonConvex'] = 2  # Option for bilinear objectives
        if solver_name == "highs":
            solver.options["time_limit"] = 50  # seconds

        if warm_start_path is not None:
            _load_warm_start(m, warm_start_path)

        if solver_name == "gurobi":
            opt_result = solver.solve(
                m,
                tee=verbose,
                warmstart=warm_start_path is not None,
            )
        else:
            opt_result = solver.solve(m, tee=verbose)

        if opt_result.solver.status != "ok":
            logging.error(f"Solver failed with status: {opt_result.solver.status}")

        if save_solution_path is not None:
            _save_solution(m, save_solution_path)

        models = milp_to_models_dict(
            m=m,
            gpu_types=gpu_types,
            model_names=model_names,
            instance_ids=instance_ids,
            idx=idx,
            workflow=self.workflow,
            power_data=self.power_data,
            policy=self.policy,
        )

        if not self._is_valid_result(m):
            return Result()

        tbf_s = 0.0
        if m.makespan.value and self.workflow.num_frames > 0:
            tbf_s = m.makespan.value / self.workflow.num_frames
        return Result(
            models=models,
            gpus_used=self._get_num_gpus(m, gpu_types, model_names, instance_ids),
            total_time_s=m.makespan.value,
            ttff_s=m.ttff_user.value,
            tbf_s=tbf_s,
            cost=m.cost.value,
            total_energy=m.energy.value,
        )

    def _is_valid_result(self, m: ConcreteModel) -> bool:
        for gpu_type in m.GPU_TYPES:
            for model_name in m.MODEL_NAMES:
                for instance_id in m.INSTANCES:
                    if m.gpus[gpu_type, model_name, instance_id].value is None:
                        return False
        return True

    def _get_num_gpus(
        self,
        m: ConcreteModel,
        gpu_types: list[GPUType],
        model_names: list[Model],
        instance_ids: list[int],
    ) -> dict[GPUType, int]:
        if not self._is_valid_result(m):
            return {}
        return {
            gpu_type: sum(
                # round() snaps solver float to nearest int (e.g. 1.9999 -> 2)
                int(round(m.gpus[idx(gpu_type, model_name, instance_id)].value))
                for model_name in model_names
                for instance_id in instance_ids
                if m.gpus[idx(gpu_type, model_name, instance_id)].value is not None
            )
            for gpu_type in gpu_types
        }

    def _get_latency_per_work(
        self,
        gpu_type: GPUType,
        model_name: Model,
        num_devices: int,
    ) -> float:
        """
        Cost per unit of work for a given model and GPU type, based on latency data.
        Cost: Linearized - sum of (latency * work_device * num_devices * ratio)
        This replaces the bilinear makespan * GPUs.
        """
        # Resolution scaling factor for HF/VAE/FT
        latency_ratio = self.workflow.get_resolution_scale(self.policy.use_upscaler)

        if model_name == Model.GEMMA:
            return (
                self.latency_data[gpu_type].gemma_first_scene[num_devices]
                + self.latency_data[gpu_type].gemma_per_scene[num_devices] * (self.workflow.total_scenes - 1)
            )

        if model_name == Model.FLUX:
            return (
                self.latency_data[gpu_type][model_name, num_devices]
                * self.workflow.num_steps[Model.FLUX]
            )

        if model_name == Model.HF:
            time_per_work = (
                self.workflow.per_subscene_frames[Model.HF]
                / self.workflow.hf_frames[self.workflow.frames_per_step_idx]
                * self.latency_data[gpu_type][model_name, num_devices]
                * latency_ratio
                * self.workflow.num_steps[Model.HF]
            )
            if not self.policy.is_disaggregated(Model.HF):
                time_per_work += self._get_latency_per_work(
                    gpu_type,
                    Model.HF_VAE,
                    1,  # VAE is single-device only in current policy
                )
            return time_per_work

        if model_name == Model.HF_VAE:
            return (
                self.latency_data[gpu_type][model_name, num_devices]
                * latency_ratio
                / self.workflow.hf_frames[self.workflow.frames_per_step_idx]
            )

        if model_name == Model.FT:
            time_per_work = (
                self.workflow.per_subscene_frames[Model.FT]
                / self.workflow.ft_frames[self.workflow.frames_per_step_idx]
                * self.latency_data[gpu_type][model_name, num_devices]
                * latency_ratio
                * self.workflow.num_steps[Model.FT]
            )
            if not self.policy.is_disaggregated(Model.FT):
                time_per_work += self._get_latency_per_work(
                    gpu_type,
                    Model.FT_VAE,
                    1,  # VAE is single-device only in current policy
                )
            return time_per_work

        if model_name == Model.FT_VAE:
            return (
                self.latency_data[gpu_type][model_name, num_devices]
                * latency_ratio
                / self.workflow.ft_frames[self.workflow.frames_per_step_idx]
            )

        if model_name == Model.UPSCALER:
            return self.latency_data[gpu_type][model_name, num_devices]

        if model_name == Model.OTHERS:
            return self.latency_data[gpu_type][model_name, num_devices] * self.workflow.total_scenes

        raise ValueError(f"Unknown model_name {model_name}")

    def _get_power_per_work(
        self,
        gpu_type: GPUType,
        model_name: Model,
        num_devices: int,
    ) -> float:
        """
        Average power per unit of work for a given model and GPU type.
        Returns the time-weighted average power consumption in watts.
        For energy calculation:
        energy = _get_latency_per_work(...) * _get_power_per_work(...) * num_devices * work
        """
        if self.power_data is None:
            return 0.0

        if model_name == Model.GEMMA:
            # For Gemma, power varies between first scene and subsequent scenes
            # Compute energy then divide by total time to get average power
            power_first = self.power_data[gpu_type].gemma_first_scene[num_devices]
            power_per_scene = self.power_data[gpu_type].gemma_per_scene[num_devices]
            latency_first = self.latency_data[gpu_type].gemma_first_scene[num_devices]
            latency_per_scene = self.latency_data[gpu_type].gemma_per_scene[num_devices]

            total_energy = (
                power_first * latency_first
                + power_per_scene * latency_per_scene * (self.workflow.total_scenes - 1)
            )
            total_time = latency_first + latency_per_scene * (self.workflow.total_scenes - 1)

            return total_energy / total_time if total_time > 0 else power_first

        if model_name == Model.FLUX:
            return self.power_data[gpu_type][model_name, num_devices]

        if model_name == Model.HF:
            return self.power_data[gpu_type][model_name, num_devices]

        if model_name == Model.HF_VAE:
            return self.power_data[gpu_type][model_name, num_devices]

        if model_name == Model.FT:
            return self.power_data[gpu_type][model_name, num_devices]

        if model_name == Model.FT_VAE:
            return self.power_data[gpu_type][model_name, num_devices]

        if model_name == Model.UPSCALER:
            return self.power_data[gpu_type][model_name, num_devices]

        if model_name == Model.OTHERS:
            # OTHERS model uses minimal GPU power (mostly idle)
            # See models.py OthersModelAllocation.calculate_energy - only uses idle power
            return self.power_data[gpu_type]["idle"]

        raise ValueError(f"Unknown model_name {model_name}")


def milp_to_models_dict(
    m: ConcreteModel,
    gpu_types: list[GPUType],
    model_names: list[Model],
    instance_ids: list[int],
    idx: Callable[[GPUType, Model, int], tuple[str, str, int]],
    workflow: WorkflowConfig,
    power_data: Optional[PowerData],
    policy: Policy,
) -> dict[GPUType, dict[Model, list[ModelAllocation]]]:
    """
    MILP result to models dictionary.
    """
    if m is None:
        return {}

    models: dict[GPUType, dict[Model, list[ModelAllocation]]] = {}
    for gpu_type in gpu_types:
        models[gpu_type] = {}
        for model_name in model_names:
            models[gpu_type][model_name] = []
            for instance_id in instance_ids:
                key = idx(gpu_type, model_name, instance_id)
                gpus_val = m.gpus[key].value
                work_val = m.work[key].value
                if gpus_val is None or work_val is None:
                    continue
                # round() snaps solver floats to nearest int (e.g. 1.9999 -> 2);
                # banker's rounding is irrelevant here since MILP values can be
                # near-integer, like 1.999 and 2.001
                gpus = int(round(gpus_val))
                work = int(round(work_val))
                if gpus > 0 and work > 0:
                    model_allocation = get_model_allocation(
                        model=model_name,
                        gpu_type=gpu_type,
                        devices=gpus,
                        replicas=1,
                    )
                    model_allocation.work = work
                    model_allocation.time = m.time[key].value
                    model_allocation.time_first = m.ttff[key].value
                    model_allocation.calculate_energy(
                        workflow=workflow,
                        power_data=power_data,
                        total_time_s=m.makespan.value
                    )
                    model_allocation.calculate_cost(
                        policy,
                        total_time_s=m.makespan.value
                    )
                    models[gpu_type][model_name].append(model_allocation)
    merged_models = models  # coalesce_models(models)
    return merged_models


def get_objective(
    m: ConcreteModel,
    allocator: Objective,
    solver_name: str,
) -> Optional[OptObjective]:
    if allocator == Objective.TIME:
        return OptObjective(expr=m.makespan, sense=minimize)

    if allocator == Objective.TTFF:
        return OptObjective(expr=m.ttff_user, sense=minimize)

    if allocator == Objective.TTFF_COST:
        # Note: This creates a bilinear (nonconvex) objective - requires Gurobi
        if solver_name == "gurobi":
            return OptObjective(expr=m.ttff_user * m.cost, sense=minimize)
        logging.warning("TTFF_COST using linear utility function.")
        a = 1.0
        b = 1.0
        return OptObjective(expr=a * m.ttff_user + b * m.cost, sense=minimize)

    if allocator == Objective.COST:
        return OptObjective(expr=m.cost, sense=minimize)

    if allocator == Objective.ENERGY:
        return OptObjective(expr=m.energy, sense=minimize)

    if allocator == Objective.TIME_ENERGY:
        # Note: This creates a bilinear objective - requires Gurobi
        if solver_name == "gurobi":
            return OptObjective(expr=m.makespan * m.energy, sense=minimize)
        logging.warning("TIME_ENERGY using linear utility function.")
        a = 1.0
        b = 1.0
        return OptObjective(expr=a * m.makespan + b * m.energy, sense=minimize)

    if allocator == Objective.ENERGY_COST:
        if solver_name == "gurobi":
            return OptObjective(expr=m.energy * m.cost, sense=minimize)
        logging.warning("ENERGY_COST using linear utility function.")
        a = 1.0
        b = 1.0
        return OptObjective(expr=a * m.energy + b * m.cost, sense=minimize)

    if allocator == Objective.FIFO:
        logging.error("FIFO not implemented in MILP")

    if allocator == Objective.RANDOM:
        return None  # No objective, just find a feasible solution

    if allocator == Objective.NONE:
        return None

    return OptObjective(expr=m.makespan, sense=minimize)


def _save_solution(
    m: ConcreteModel,
    save_solution_path: str,
) -> None:
    solution = {
        var.name: var.value
        for var in m.component_data_objects(Var, active=True)
        if var.value is not None
    }
    with open(save_solution_path, "w", encoding="utf-8") as output_file:
        json.dump(solution, output_file, indent=2)


def _load_warm_start(
    m: ConcreteModel,
    warm_start_path: str,
) -> None:
    """Load warm start values from a JSON file and apply them to the model variables."""
    with open(warm_start_path, "r", encoding="utf-8") as input_file:
        warm_start_values = json.load(input_file)

    warm_start_applied = 0
    for var in m.component_data_objects(Var, active=True):
        if var.name in warm_start_values:
            var.set_value(warm_start_values[var.name])
            warm_start_applied += 1

    logging.info(
        f"Warm start loaded from {warm_start_path}. "
        f"Applied values to {warm_start_applied} variables."
    )
