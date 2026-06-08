"""
HexGen algorithm for the StreamWise workflow allocation problem.

Reference: https://arxiv.org/abs/2311.11514

HexGen treats each model in the workflow as an independent component for optimization.
It tracks metrics per model and optimizes models sequentially according to MODEL_ORDER.
When a model's metric converges (stops dropping), it moves to the next model.
After the last model converges, it cycles back to the first model and allocates
remaining GPUs until exhausted.
"""

from __future__ import annotations
import logging
from typing import Optional

from sim_types import Result
from sim_types import GPUType
from sim_types import WorkflowConfig
from sim_types import PowerData
from sim_types import LatencyData
from sim_types import Model
from sim_types import ModelAllocation
from sim_types import Policy
from sim_types import Solver
from sim_types import MODEL_ORDER

from utils import simplify_model_allocations

from evaluator import calc_used_gpus
from evaluator import evaluate_model_allocation

from .greedy import GreedyAllocator

from actions import gen_actions
from actions import choose_action
from actions import apply_action

try:
    from policies import HEXGEN_POLICY
    from policies import MAX_ITERATIONS
    from policies import USE_ALL_GPUS
except ModuleNotFoundError:
    from .policies import HEXGEN_POLICY
    from .policies import MAX_ITERATIONS
    from .policies import USE_ALL_GPUS


def _get_model_order(workflow: WorkflowConfig) -> list[Model]:
    """Get ordered list of models in the workflow, sorted by MODEL_ORDER."""
    return sorted(
        [m for m in workflow.models if m in MODEL_ORDER],
        key=lambda m: MODEL_ORDER[m],
    )


class HexGenAllocator(GreedyAllocator):
    """
    HexGen-style allocator that optimizes models one at a time,
    sequentially following MODEL_ORDER.

    Reference: https://arxiv.org/abs/2311.11514

    Key differences from GreedyAllocator:
    1. Each model is treated as an independent optimization target.
    2. Per-model metrics are tracked separately.
    3. Models are optimized in MODEL_ORDER sequence. When a model's metric
       converges, it moves to the next model. After the last model converges,
       it cycles back to the first and allocates remaining GPUs.
    """

    def __init__(
        self,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
        power_data: Optional[PowerData] = None,
        policy: Policy = HEXGEN_POLICY,
    ) -> None:
        super().__init__(
            workflow,
            latency_data,
            power_data,
            policy,
        )
        assert self.policy.solver == Solver.HEXGEN

    def _pick_from_single_device_mapping(
        self,
        num_gpus: int,
        gpu_type: GPUType,
        verbose: bool = False,
        allow_removal: bool = False,
        allow_merging: bool = False,
        look_ahead_replicas: int = 3,
    ) -> Result:
        """
        HexGen-style allocation for a single GPU type (>8 GPUs).
        Optimizes models one at a time following MODEL_ORDER.
        """
        from constants import NUM_GPUS_PER_SERVER

        assert num_gpus >= NUM_GPUS_PER_SERVER[gpu_type]

        # Initialize allocations (same as GreedyAllocator)
        models = self._init_single_device_models(gpu_type)

        remaining_gpus = num_gpus - calc_used_gpus(models)
        assert 0 <= remaining_gpus <= num_gpus

        # --- HexGen per-model sequential optimization ---
        model_order = _get_model_order(self.workflow)
        per_model_metrics: dict[Model, Optional[float]] = {m: None for m in model_order}

        it = 0
        current_model_idx = 0
        cycles_without_progress = 0  # track full cycles without any improvement
        total_models = len(model_order)

        while remaining_gpus > 0:
            if current_model_idx >= total_models:
                # Completed a full cycle, wrap around
                current_model_idx = 0
                cycles_without_progress += 1
                if cycles_without_progress >= 1:
                    logging.debug(
                        f"HexGen: No progress after {cycles_without_progress} full cycles.")
                    break

            current_model = model_order[current_model_idx]

            if verbose:
                print(f"--- HexGen: Optimizing {current_model.value} "
                      f"(model {current_model_idx + 1}/{total_models}) ---")

            # Inner loop: keep optimizing current model until convergence
            inner_it = 0
            while remaining_gpus > 0:
                # Evaluate current state
                evaluate_model_allocation(
                    models=models,
                    num_gpus={gpu_type: num_gpus},
                    workflow=self.workflow,
                    latency_data=self.latency_data,
                    power_data=self.power_data,
                    policy=self.policy,
                    round_up_cost_to_server=False,
                )

                # Generate actions only for the current model
                all_actions = gen_actions(
                    num_gpus={gpu_type: num_gpus},
                    latency_data=self.latency_data,
                    power_data=self.power_data,
                    workflow=self.workflow,
                    models=models,
                    policy=self.policy,
                )

                # Filter to actions targeting the current model only
                model_actions = [a for a in all_actions if a.model == current_model]

                if not model_actions:
                    logging.debug(
                        f"HexGen: No actions for {current_model.value} after {inner_it} inner iterations.")
                    break

                best_action = choose_action(model_actions, self.policy.objective)

                if not best_action:
                    logging.debug(f"HexGen: No action selected for {current_model.value}.")
                    break

                new_metric = best_action.get_metric(self.policy.objective)
                prev_metric = per_model_metrics[current_model]

                if self.policy.objective.is_monotonic() and prev_metric is not None and new_metric >= prev_metric:
                    msg = (
                        f"HexGen: {current_model.value} converged after {inner_it} inner iterations. "
                        f"Metric: {new_metric:.2f} >= previous {prev_metric:.2f}."
                    )
                    if verbose:
                        print(msg)
                    logging.debug(msg)
                    break

                per_model_metrics[current_model] = new_metric

                models = apply_action(best_action, models=models)
                models = simplify_model_allocations(models)

                remaining_gpus = num_gpus - calc_used_gpus(models)

                if verbose:
                    self._print_iteration(it, models, {gpu_type: num_gpus})
                    print(f"HexGen: Applied action for {current_model.value}, "
                          f"metric: {new_metric:.2f}, remaining: {remaining_gpus}")

                it += 1
                inner_it += 1

                if it > MAX_ITERATIONS:
                    logging.debug(f"HexGen: Reached max iterations ({MAX_ITERATIONS}). Stopping.")
                    break

            if it > MAX_ITERATIONS:
                break

            current_model_idx += 1

        # --- USE_ALL_GPUS: fill remaining GPUs by cycling through MODEL_ORDER ---
        remaining_gpus = num_gpus - calc_used_gpus(models)
        if USE_ALL_GPUS and remaining_gpus > 0:
            models = self._fill_remaining_gpus_single(
                models=models,
                num_gpus=num_gpus,
                gpu_type=gpu_type,
                model_order=model_order,
                it=it,
                verbose=verbose,
            )

        # Final evaluation
        result = evaluate_model_allocation(
            models=models,
            num_gpus={gpu_type: num_gpus},
            workflow=self.workflow,
            latency_data=self.latency_data,
            power_data=self.power_data,
            policy=self.policy,
            round_up_cost_to_server=True,
        )

        if verbose:
            self._print_final_allocation(
                models=models,
                used_devices=result.gpus_used,
                total_devices={gpu_type: num_gpus},
                power_data=self.power_data,
                total_time_s=result.total_time_s,
                ttff_s=result.ttff_s,
                first_chunk_time=result.first_chunk_time,
                tbf_s=result.tbf_s,
                total_energy=result.total_energy if self.power_data else 0.0,
                cost=result.cost,
            )

        if not self.policy.is_disaggregated(Model.HF):
            if models[gpu_type][Model.HF_VAE]:
                assert models[gpu_type][Model.HF_VAE][0].get_num_gpus() == 0, \
                    "HF_VAE must have 0 GPUs when HF disaggregation is disabled"
        if not self.policy.is_disaggregated(Model.FT):
            if models[gpu_type][Model.FT_VAE]:
                assert models[gpu_type][Model.FT_VAE][0].get_num_gpus() == 0, \
                    "FT_VAE must have 0 GPUs when FT disaggregation is disabled"

        num_gpus_used = result.gpus_used[gpu_type]
        assert num_gpus_used <= num_gpus, f"{num_gpus_used}>{num_gpus} for {gpu_type.value}"

        return Result(
            total_time_s=result.total_time_s,
            models=models,
            gpus_used={gpu_type: num_gpus_used},
            gpus_total={gpu_type: num_gpus},
            ttff_s=result.ttff_s,
            tbf_s=result.tbf_s,
            total_energy=result.total_energy if self.power_data else 0.0,
            cost=result.cost,
        )

    def _pick_from_both_devices_mapping(
        self,
        num_gpus: dict[GPUType, int],
        verbose: bool = False,
        allow_removal: bool = False,
        allow_merging: bool = False,
        look_ahead_replicas: int = 3,
    ) -> Result:
        """
        HexGen-style allocation for two GPU types.
        Optimizes models one at a time following MODEL_ORDER.
        """
        from constants import NUM_GPUS_PER_SERVER

        gpu_types = list(num_gpus.keys())
        assert len(gpu_types) == 2
        gpu_type1 = gpu_types[0]
        gpu_type2 = gpu_types[1]
        assert num_gpus[gpu_type1] >= NUM_GPUS_PER_SERVER[gpu_type1]
        assert num_gpus[gpu_type2] >= NUM_GPUS_PER_SERVER[gpu_type2]

        # Initialize allocations (same as GreedyAllocator)
        models = self._init_both_devices_models(gpu_type1, gpu_type2)

        remaining_gpus: dict[GPUType, int] = {}
        for gpu_type in num_gpus.keys():
            remaining_gpus[gpu_type] = num_gpus[gpu_type] - calc_used_gpus({gpu_type: models[gpu_type]})

        # --- HexGen per-model sequential optimization ---
        model_order = _get_model_order(self.workflow)
        per_model_metrics: dict[Model, Optional[float]] = {m: None for m in model_order}

        if verbose:
            evaluate_model_allocation(
                models=models,
                num_gpus=num_gpus,
                workflow=self.workflow,
                latency_data=self.latency_data,
                power_data=self.power_data,
                policy=self.policy,
                round_up_cost_to_server=True,
            )
            self._print_iteration(0, models, num_gpus)

        it = 1
        current_model_idx = 0
        cycles_without_progress = 0
        total_models = len(model_order)

        while sum(remaining_gpus.values()) > 0:
            if current_model_idx >= total_models:
                current_model_idx = 0
                cycles_without_progress += 1
                if cycles_without_progress >= 1:
                    logging.debug(
                        f"HexGen: No progress after {cycles_without_progress} full cycles.")
                    break

            current_model = model_order[current_model_idx]

            if verbose:
                print(f"--- HexGen: Optimizing {current_model.value} "
                      f"(model {current_model_idx + 1}/{total_models}) ---")

            inner_it = 0

            while sum(remaining_gpus.values()) > 0:
                evaluate_model_allocation(
                    models=models,
                    num_gpus=num_gpus,
                    workflow=self.workflow,
                    latency_data=self.latency_data,
                    power_data=self.power_data,
                    policy=self.policy,
                    round_up_cost_to_server=False,
                )

                all_actions = gen_actions(
                    workflow=self.workflow,
                    latency_data=self.latency_data,
                    power_data=self.power_data,
                    num_gpus=num_gpus,
                    models=models,
                    policy=self.policy,
                )

                # Filter to current model
                model_actions = [a for a in all_actions if a.model == current_model]

                if not model_actions:
                    logging.debug(
                        f"HexGen: No actions for {current_model.value} after {inner_it} inner iterations.")
                    break

                best_action = choose_action(model_actions, self.policy.objective)

                if not best_action:
                    logging.debug(f"HexGen: No action selected for {current_model.value}.")
                    break

                new_metric = best_action.get_metric(self.policy.objective)
                prev_metric = per_model_metrics[current_model]

                if self.policy.objective.is_monotonic() and prev_metric is not None and new_metric >= prev_metric:
                    msg = (
                        f"HexGen: {current_model.value} converged. "
                        f"Metric: {new_metric:.2f} >= previous {prev_metric:.2f}."
                    )
                    if verbose:
                        print(msg)
                    logging.debug(msg)
                    break

                per_model_metrics[current_model] = new_metric

                models = apply_action(best_action, models=models)
                models = simplify_model_allocations(models)

                remaining_gpus.clear()
                for gpu_type in num_gpus.keys():
                    remaining_gpus[gpu_type] = num_gpus[gpu_type] - calc_used_gpus({gpu_type: models[gpu_type]})

                if verbose:
                    self._print_iteration(it, models, num_gpus)
                    print(f"HexGen: Applied action for {current_model.value}, "
                          f"metric: {new_metric:.2f}")
                    print("Remaining devices:")
                    for gt in remaining_gpus:
                        print(f"  {remaining_gpus[gt]} x {gt.value}")

                it += 1
                inner_it += 1

                if it > MAX_ITERATIONS:
                    logging.debug(f"HexGen: Reached max iterations ({MAX_ITERATIONS}). Stopping.")
                    break

            if it > MAX_ITERATIONS:
                break

            current_model_idx += 1

        # --- USE_ALL_GPUS: fill remaining GPUs by cycling through MODEL_ORDER ---
        remaining_gpus_total = sum(
            num_gpus[gt] - calc_used_gpus({gt: models[gt]})
            for gt in num_gpus
        )
        if USE_ALL_GPUS and remaining_gpus_total > 0:
            models = self._fill_remaining_gpus_multi(
                models=models,
                num_gpus=num_gpus,
                model_order=model_order,
                it=it,
                verbose=verbose,
            )

        # Adjust for no disaggregation
        if not self.policy.is_disaggregated(Model.HF):
            for models_gpu in models.values():
                for instance_id in range(len(models_gpu[Model.HF_VAE])):
                    assert models_gpu[Model.HF_VAE][instance_id].get_num_gpus() == 0, \
                        "HF_VAE must have 0 GPUs when HF disaggregation is disabled"
        if not self.policy.is_disaggregated(Model.FT):
            for models_gpu in models.values():
                for instance_id in range(len(models_gpu[Model.FT_VAE])):
                    assert models_gpu[Model.FT_VAE][instance_id].get_num_gpus() == 0, \
                        "FT_VAE must have 0 GPUs when FT disaggregation is disabled"

        # Final evaluation
        result = evaluate_model_allocation(
            models=models,
            num_gpus=num_gpus,
            workflow=self.workflow,
            latency_data=self.latency_data,
            power_data=self.power_data,
            policy=self.policy,
            round_up_cost_to_server=True,
        )

        if verbose:
            self._print_final_allocation(
                models=models,
                used_devices=result.gpus_used,
                total_devices={
                    gpu_type1: num_gpus.get(gpu_type1, 0),
                    gpu_type2: num_gpus.get(gpu_type2, 0),
                },
                power_data=self.power_data,
                total_time_s=result.total_time_s,
                ttff_s=result.ttff_s,
                first_chunk_time=result.first_chunk_time,
                tbf_s=result.tbf_s,
                total_energy=result.total_energy if self.power_data else 0.0,
                cost=result.cost,
            )

        assert result.gpus_used[gpu_type1] <= num_gpus.get(gpu_type1, 0), \
            f"{gpu_type1.value}: {result.gpus_used[gpu_type1]} > {num_gpus.get(gpu_type1, 0)}"
        assert result.gpus_used[gpu_type2] <= num_gpus.get(gpu_type2, 0), \
            f"{gpu_type2.value}: {result.gpus_used[gpu_type2]} > {num_gpus.get(gpu_type2, 0)}"

        return Result(
            total_time_s=result.total_time_s,
            models=models,
            gpus_used=result.gpus_used,
            ttff_s=result.ttff_s,
            tbf_s=result.tbf_s,
            total_energy=result.total_energy if self.power_data else 0.0,
            cost=result.cost,
        )

    def _fill_remaining_gpus_single(
        self,
        models: dict[GPUType, dict[Model, list[ModelAllocation]]],
        num_gpus: int,
        gpu_type: GPUType,
        model_order: list[Model],
        it: int = 0,
        verbose: bool = False,
    ) -> dict[GPUType, dict[Model, list[ModelAllocation]]]:
        """
        Fill remaining GPUs by cycling through MODEL_ORDER (single GPU type).
        Applies any available action per model, ignoring metric convergence.
        Stops when all GPUs are used or no model can accept more.
        """
        remaining_gpus = num_gpus - calc_used_gpus(models)
        total_models = len(model_order)
        model_idx = 0
        models_exhausted: set[Model] = set()

        if verbose:
            print(f"--- HexGen: USE_ALL_GPUS fill phase, {remaining_gpus} remaining ---")

        while remaining_gpus > 0 and len(models_exhausted) < total_models:
            current_model = model_order[model_idx % total_models]
            model_idx += 1

            if current_model in models_exhausted:
                continue

            evaluate_model_allocation(
                models=models,
                num_gpus={gpu_type: num_gpus},
                workflow=self.workflow,
                latency_data=self.latency_data,
                power_data=self.power_data,
                policy=self.policy,
                round_up_cost_to_server=False,
            )

            all_actions = gen_actions(
                num_gpus={gpu_type: num_gpus},
                latency_data=self.latency_data,
                power_data=self.power_data,
                workflow=self.workflow,
                models=models,
                policy=self.policy,
            )
            model_actions = [a for a in all_actions if a.model == current_model]

            if not model_actions:
                models_exhausted.add(current_model)
                logging.debug(f"HexGen fill: {current_model.value} exhausted (no actions).")
                continue

            best_action = choose_action(model_actions, self.policy.objective)
            if not best_action:
                models_exhausted.add(current_model)
                logging.debug(f"HexGen fill: {current_model.value} exhausted (no action selected).")
                continue

            models = apply_action(best_action, models=models)
            models = simplify_model_allocations(models)
            remaining_gpus = num_gpus - calc_used_gpus(models)

            if verbose:
                self._print_iteration(it, models, {gpu_type: num_gpus})
                print(f"HexGen fill: Allocated to {current_model.value}, remaining: {remaining_gpus}")

            it += 1
            if it > MAX_ITERATIONS:
                logging.debug(f"HexGen fill: Reached max iterations ({MAX_ITERATIONS}). Stopping.")
                break

        return models

    def _fill_remaining_gpus_multi(
        self,
        models: dict[GPUType, dict[Model, list[ModelAllocation]]],
        num_gpus: dict[GPUType, int],
        model_order: list[Model],
        it: int = 0,
        verbose: bool = False,
    ) -> dict[GPUType, dict[Model, list[ModelAllocation]]]:
        """
        Fill remaining GPUs by cycling through MODEL_ORDER (multi GPU type).
        Applies any available action per model, ignoring metric convergence.
        Stops when all GPUs are used or no model can accept more.
        """
        total_remaining = sum(
            num_gpus[gt] - calc_used_gpus({gt: models[gt]})
            for gt in num_gpus
        )
        total_models = len(model_order)
        model_idx = 0
        models_exhausted: set[Model] = set()

        if verbose:
            print(f"--- HexGen: USE_ALL_GPUS fill phase, {total_remaining} remaining ---")

        while total_remaining > 0 and len(models_exhausted) < total_models:
            current_model = model_order[model_idx % total_models]
            model_idx += 1

            if current_model in models_exhausted:
                continue

            evaluate_model_allocation(
                models=models,
                num_gpus=num_gpus,
                workflow=self.workflow,
                latency_data=self.latency_data,
                power_data=self.power_data,
                policy=self.policy,
                round_up_cost_to_server=False,
            )

            all_actions = gen_actions(
                workflow=self.workflow,
                latency_data=self.latency_data,
                power_data=self.power_data,
                num_gpus=num_gpus,
                models=models,
                policy=self.policy,
            )
            model_actions = [a for a in all_actions if a.model == current_model]

            if not model_actions:
                models_exhausted.add(current_model)
                logging.debug(f"HexGen fill: {current_model.value} exhausted (no actions).")
                continue

            best_action = choose_action(model_actions, self.policy.objective)
            if not best_action:
                models_exhausted.add(current_model)
                logging.debug(f"HexGen fill: {current_model.value} exhausted (no action selected).")
                continue

            models = apply_action(best_action, models=models)
            models = simplify_model_allocations(models)
            total_remaining = sum(
                num_gpus[gt] - calc_used_gpus({gt: models[gt]})
                for gt in num_gpus
            )

            if verbose:
                self._print_iteration(it, models, num_gpus)
                print(f"HexGen fill: Allocated to {current_model.value}, remaining: {total_remaining}")

            it += 1
            if it > MAX_ITERATIONS:
                logging.debug(f"HexGen fill: Reached max iterations ({MAX_ITERATIONS}). Stopping.")
                break

        return models
