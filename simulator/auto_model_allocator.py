"""
Factory helpers for selecting the right model allocator implementation.
"""

from __future__ import annotations

import logging

from dataclasses import replace
from typing import Optional

from sim_types import Policy
from sim_types import WorkflowConfig
from sim_types import LatencyData
from sim_types import Model
from sim_types import PowerData
from sim_types import QualityLevel
from sim_types import Solver
from sim_types import GPUType
from sim_types import Result

from model_provisioner.policies import STREAMWISE_POLICY

from model_allocator import ModelAllocator


class AutoModelAllocator(ModelAllocator):
    """Allocator wrapper that routes to a concrete allocator by solver."""

    policy: Policy

    def __init__(
        self,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
        power_data: Optional[PowerData] = None,
        policy: Policy = STREAMWISE_POLICY,
    ) -> None:
        super().__init__(
            workflow=workflow,
            latency_data=latency_data,
            power_data=power_data,
            policy=policy,
        )
        self._allocator = self._build_allocator()

    def _build_allocator(self) -> ModelAllocator:
        """Create concrete allocator based on configured solver."""
        if self.policy.solver == Solver.GREEDY:
            from model_provisioner.greedy import GreedyAllocator
            return GreedyAllocator(
                workflow=self.workflow,
                latency_data=self.latency_data,
                power_data=self.power_data,
                policy=self.policy,
            )
        if self.policy.solver == Solver.NAIVE:
            from model_provisioner.naive_baseline import NaiveAllocator
            return NaiveAllocator(
                workflow=self.workflow,
                latency_data=self.latency_data,
                power_data=self.power_data,
                policy=self.policy,
            )
        if self.policy.solver in {Solver.GUROBI, Solver.HIGHS}:
            from model_provisioner.milp import MILPAllocator
            return MILPAllocator(
                workflow=self.workflow,
                latency_data=self.latency_data,
                power_data=self.power_data,
                policy=self.policy,
            )
        if self.policy.solver == Solver.HEXGEN:
            from model_provisioner.hexgen import HexGenAllocator
            return HexGenAllocator(
                workflow=self.workflow,
                latency_data=self.latency_data,
                power_data=self.power_data,
                policy=self.policy,
            )
        if self.policy.solver == Solver.HELIX:
            from model_provisioner.helix import HelixAllocator
            return HelixAllocator(
                workflow=self.workflow,
                latency_data=self.latency_data,
                power_data=self.power_data,
                policy=self.policy,
            )
        raise ValueError(f"Unsupported solver for allocator selection: {self.policy.solver}")

    def allocate(
        self,
        num_gpus: dict[GPUType, int],
        verbose: bool = False,
    ) -> Result:
        if self.policy.use_upscaler and self.workflow.target_resolution == QualityLevel.LOW:
            logging.warning(
                f"Policy {self.policy.name} uses upscaler, but workflow target resolution is LOW. "
                f"Disabling upscaler for this allocation.")
            self.policy = replace(self.policy, use_upscaler=False)
            self._allocator.policy = self.policy
            # Remove upscaler from model work
            self.workflow.model_work.pop(Model.UPSCALER, None)
            self._allocator.workflow = self.workflow

        return self._allocator.allocate(
            num_gpus=num_gpus,
            verbose=verbose,
        )
