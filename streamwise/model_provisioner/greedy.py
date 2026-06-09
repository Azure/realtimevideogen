"""
Greedy algorithm for the StreamWise workflow allocation problem.
"""

from __future__ import annotations

import logging

from tabulate import tabulate

from typing import Optional

from operator import itemgetter

from constants import NUM_GPUS_PER_SERVER
from constants import SECONDS_IN_MINUTE
from constants import SECONDS_IN_HOUR

from sim_types import Result
from sim_types import GPUType
from sim_types import WorkflowConfig
from sim_types import LatencyData
from sim_types import PowerData
from sim_types import Model
from sim_types import ModelAllocation
from sim_types import Policy
from sim_types import Solver

from utils import simplify_model_allocations

from evaluator import calc_used_gpus
from evaluator import evaluate_model_allocation

from model_allocator import ModelAllocator

from .policies import STREAMWISE_POLICY
from .policies import MAX_ITERATIONS
from .policies import USE_ALL_GPUS

from actions import gen_actions
from actions import choose_action
from actions import apply_action


class GreedyAllocator(ModelAllocator):
    """
    Greedy allocator that iteratively applies the best action.
    """
    def __init__(
        self,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
        power_data: Optional[PowerData] = None,
        policy: Policy = STREAMWISE_POLICY,
    ) -> None:
        super().__init__(
            workflow,
            latency_data,
            power_data,
            policy,
        )
        assert self.policy.solver in {Solver.GREEDY, Solver.HEXGEN}

    def allocate(
        self,
        num_gpus: dict[GPUType, int],
        verbose: bool = False,
        # Greedy policy parameters
        allow_removal: bool = False,
        allow_merging: bool = False,
        look_ahead_replicas: int = 3,
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

        if len(gpu_types) == 1 and num_gpus[gpu_type1] == 8:
            # 8 x GPUs
            return self._pick_from_single_server(
                gpu_type=gpu_type1,
                verbose=verbose,
            )

        if len(gpu_types) == 1:
            # More than 8 x GPUs
            return self._pick_from_single_device_mapping(
                num_gpus.get(gpu_type1, 0),
                gpu_type=gpu_type1,
                verbose=verbose,
                allow_removal=allow_removal,
                allow_merging=allow_merging,
                look_ahead_replicas=look_ahead_replicas,
            )

        # Mixed setup of GPU types (e.g., A100 and H100)
        return self._pick_from_both_devices_mapping(
            num_gpus,
            verbose=verbose,
            allow_removal=allow_removal,
            allow_merging=allow_merging,
            look_ahead_replicas=look_ahead_replicas,
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
        Calculate based on two GPU types.
        """
        gpu_types = list(num_gpus.keys())
        assert len(gpu_types) == 2
        assert len(num_gpus) == 2
        gpu_type1 = gpu_types[0]
        gpu_type2 = gpu_types[1]
        assert num_gpus[gpu_type1] >= NUM_GPUS_PER_SERVER[gpu_type1]
        assert num_gpus[gpu_type2] >= NUM_GPUS_PER_SERVER[gpu_type2]

        # Initialize allocations with minimal setup
        models = self._init_both_devices_models(gpu_type1, gpu_type2)

        remaining_gpus = {}
        for gpu_type in num_gpus.keys():
            remaining_gpus[gpu_type] = num_gpus[gpu_type] - calc_used_gpus({gpu_type: models[gpu_type]})

        # Optimization loop
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
        prev_metric = None
        switch_objective = False
        while sum(remaining_gpus.values()) > 0:
            # Calculate current iteration times
            evaluate_model_allocation(
                models=models,
                num_gpus=num_gpus,
                workflow=self.workflow,
                latency_data=self.latency_data,
                power_data=self.power_data,
                policy=self.policy,
                round_up_cost_to_server=False,
            )

            # Calculate potential actions for each optimization option
            actions = gen_actions(
                workflow=self.workflow,
                latency_data=self.latency_data,
                power_data=self.power_data,
                num_gpus=num_gpus,
                models=models,
                policy=self.policy,
                allow_removal=allow_removal,
                allow_merging=allow_merging,
                look_ahead_replicas=look_ahead_replicas,
            )

            if not actions:
                logging.debug(f"No more actions possible after {it} iterations for {self.policy}.")
                break

            best_action = choose_action(actions, self.policy.objective, switch_objective=switch_objective)

            if not best_action:
                logging.debug("No actions selected.")
                break

            new_metric = best_action.get_metric(self.policy.objective, switch_objective=switch_objective)

            if self.policy.objective.is_monotonic() and prev_metric is not None and new_metric >= prev_metric:
                msg = f"No improvement after {it} iterations for {self.policy}."
                msg += f" Best action: {best_action}, metric: {new_metric:.2f} >= previous {prev_metric:.2f}."
                if verbose:
                    print(msg)
                logging.debug(msg)
                if not USE_ALL_GPUS:
                    logging.debug("Not using all GPUs as USE_ALL_GPUS is False. Stopping optimization loop.")
                    break
                switch_objective = True

            prev_metric = new_metric

            models = apply_action(best_action, models=models)

            models = simplify_model_allocations(models)

            remaining_gpus.clear()
            for gpu_type in num_gpus.keys():
                remaining_gpus[gpu_type] = num_gpus[gpu_type] - calc_used_gpus({gpu_type: models[gpu_type]})

            if verbose:
                self._print_iteration(it, models, num_gpus)
                print(f"{len(actions)} actions:")
                for action in actions:
                    if action == best_action:
                        print(f"* {action} (best)")
                    else:
                        print(f"  {action}")
                print(f"Metric: {new_metric:.2f}")
                print("Remaining devices:")
                for gpu_type in remaining_gpus.keys():
                    print(f"  {remaining_gpus[gpu_type]} x {gpu_type.value}")

            it += 1
            if it > MAX_ITERATIONS:
                logging.debug(f"Reached max iterations ({MAX_ITERATIONS}). Stopping optimization loop.")
                break

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

        # Final calculations
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

    def _pick_from_single_server(
        self,
        gpu_type: GPUType,
        verbose: bool = False,
    ) -> Result:
        """
        The minimal setup with a servers with a single server (8 GPUs or 4 for GB200).
        No parallelism across scenes/subscenes.
        """

        # Number of devices
        num_gpus = NUM_GPUS_PER_SERVER[gpu_type]
        models = self._init_single_server_models(gpu_type)

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
            model_device = models[gpu_type]
            print_data = [
                [Model.GEMMA.value, round(model_device[Model.GEMMA][0].time, 2)],
                [Model.FLUX.value, round(model_device[Model.FLUX][0].time, 2)],
                [Model.HF.value, round(model_device[Model.HF][0].time, 2)],
                [Model.HF_VAE.value, round(model_device[Model.HF_VAE][0].time, 2)],
                [Model.FT.value, round(model_device[Model.FT][0].time, 2)],
                [Model.FT_VAE.value, round(model_device[Model.FT_VAE][0].time, 2)],
            ]
            if self.policy.use_upscaler:
                print_data.append([Model.UPSCALER.value, round(model_device[Model.UPSCALER][0].time, 2)])
            print(f"Total time: {result.total_time_s:.2f} seconds")
            print(tabulate(
                print_data,
                headers=["Model", "Time (seconds)"],
                tablefmt="pretty",
                colalign=["left", "right"]
            ))
            self._print_final_allocation(
                models=models,
                used_devices={gpu_type: num_gpus},
                total_devices={gpu_type: num_gpus},
                power_data=self.power_data,
                total_time_s=result.total_time_s,
                ttff_s=result.ttff_s,
                first_chunk_time=result.first_chunk_time,
                tbf_s=result.tbf_s,
                total_energy=result.total_energy if self.power_data else 0.0,
                cost=result.cost,
            )

        return Result(
            total_time_s=result.total_time_s,
            models=models,
            gpus_used={gpu_type: num_gpus},
            ttff_s=result.ttff_s,
            tbf_s=result.tbf_s,
            total_energy=result.total_energy if self.power_data else 0.0,
            cost=result.cost,
        )

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
        Calculate time and energy based on a single GPU type.
        """
        assert num_gpus >= NUM_GPUS_PER_SERVER[gpu_type]
        latency_gpu_data = self.latency_data[gpu_type]
        assert gpu_type == latency_gpu_data.gpu_type

        if self.power_data is not None:
            power_gpu_data = self.power_data[gpu_type]
            assert gpu_type == power_gpu_data.gpu_type

        # Initialize allocations
        models = self._init_single_device_models(gpu_type)

        remaining_gpus = num_gpus - calc_used_gpus(models)

        assert 0 <= remaining_gpus <= num_gpus

        # Optimization loop
        it = 0
        prev_metric = None
        switch_objective = False
        while remaining_gpus > 0:
            # Calculate current iteration times
            evaluate_model_allocation(
                models=models,
                num_gpus={gpu_type: num_gpus},
                workflow=self.workflow,
                latency_data=self.latency_data,
                power_data=self.power_data,
                policy=self.policy,
                round_up_cost_to_server=False,
            )

            # Calculate potential actions for each optimization option
            actions = gen_actions(
                num_gpus={gpu_type: num_gpus},
                latency_data=self.latency_data,
                power_data=self.power_data,
                workflow=self.workflow,
                models=models,
                policy=self.policy,
                allow_removal=allow_removal,
                allow_merging=allow_merging,
                look_ahead_replicas=look_ahead_replicas,
            )

            if not actions:
                logging.debug(f"No more actions possible after {it} iterations for {self.policy}")
                break

            best_action = choose_action(
                actions,
                self.policy.objective,
                switch_objective=switch_objective)

            if not best_action:
                logging.debug("No action selected.")
                break

            new_metric = best_action.get_metric(self.policy.objective, switch_objective=switch_objective)
            if self.policy.objective.is_monotonic() and prev_metric is not None and new_metric >= prev_metric:
                msg = f"No improvement from actions after {it} iterations for {self.policy}."
                msg += f" Best action: {best_action}, metric: {new_metric:.2f} >= previous {prev_metric:.2f}."
                if verbose:
                    print(msg)
                logging.debug(msg)
                if not USE_ALL_GPUS:
                    logging.debug("Not using all GPUs as USE_ALL_GPUS is False. Stopping optimization loop.")
                    break
                switch_objective = True

            models = apply_action(best_action, models)

            models = simplify_model_allocations(models)

            remaining_gpus = num_gpus - calc_used_gpus(models)
            prev_metric = new_metric

            if verbose:
                self._print_iteration(it, models, {gpu_type: num_gpus})
                print(f"Metric: {new_metric:.2f}")
                print(f"{len(actions)} actions:")
                for action in actions:
                    if action == best_action:
                        print(f"  * {action} (best)")
                    else:
                        print(f"    {action}")
                print(f"Applied: {best_action}")
                print(f"Remaining devices: {remaining_gpus}x{gpu_type}")

            it += 1
            if it > MAX_ITERATIONS:
                logging.debug(f"Reached max iterations ({MAX_ITERATIONS}). Stopping optimization loop.")
                break

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

    def _print_iteration(
        self,
        it: int,
        models: dict[GPUType, dict[Model, list[ModelAllocation]]],
        num_gpus: dict[GPUType, int],
    ) -> None:
        print(f"--- Iteration {it} ---")

        for gpu_type in models.keys():
            total_gpus = calc_used_gpus({gpu_type: models[gpu_type]})
            print(f"Current {gpu_type.value} allocation: {total_gpus}/{num_gpus[gpu_type]} GPUs")
            for model in Model:
                for model_instance in models[gpu_type][model]:
                    if model_instance.get_num_gpus() > 0:
                        print(f"  {model.value:10s}:\t{model_instance}")

        # Find the bottleneck stage
        stage_times: dict[Model, float] = {}
        ttff_times: dict[Model, float] = {}
        for model_name in Model:
            times = []
            times_first = []
            for gpu_type in models.keys():
                for model_alloc in models[gpu_type][model_name]:
                    times.append(model_alloc.time)
                    times_first.append(model_alloc.time_first)
            stage_times[model_name] = max(times) if times else 0.0
            ttff_times[model_name] = max(times_first) if times_first else 0.0

        bottleneck_stage, bottleneck_time = max(
            stage_times.items(),
            key=itemgetter(1)
        )
        bottleneck_ttff_stage, bottleneck_ttff_time = max(
            ttff_times.items(),
            key=itemgetter(1)
        )
        print(f"Bottleneck: {bottleneck_stage} ({bottleneck_time:.2f}s)")
        print(f"Bottleneck TTFF: {bottleneck_ttff_stage} ({bottleneck_ttff_time:.2f}s)")
        # bottleneck stage is not necessarily the stage with the
        # highest potential gain from scaling up/out

    def _print_final_allocation(
        self,
        models: dict[GPUType, dict[Model, list[ModelAllocation]]],
        used_devices: dict[GPUType, int],
        total_devices: dict[GPUType, int],
        power_data: Optional[PowerData],
        total_time_s: float,
        ttff_s: float,
        first_chunk_time: float,
        tbf_s: float,
        total_energy: float,
        cost: float,
    ) -> None:
        print("=== FINAL ALLOCATION ===")
        print("Total devices used/available:")
        for gpu_type, total_device in total_devices.items():
            used_device = used_devices[gpu_type]
            print(f"  {gpu_type.value}: {used_device}/{total_device}")
        print("Model allocations:")
        for gpu_type in models.keys():
            print(f"  {gpu_type.value} ({used_devices[gpu_type]} used):")
            for model in Model:
                for model_alloc in models[gpu_type][model]:
                    print(f"    {model.value:10s}:\t{model_alloc}")
        print(f"Total time: {total_time_s:.2f} seconds ({total_time_s / SECONDS_IN_MINUTE:.2f} minutes)")
        print(f"TTFF: {ttff_s:.2f} seconds")
        print(f"First chunk time: {first_chunk_time:.2f} seconds")
        print(f"TBF: {tbf_s:.2f} seconds")
        print(f"Total cost: ${cost:.2f}")
        if power_data is not None:
            print(f"Total energy: {total_energy:.2f} Ws ({total_energy / SECONDS_IN_HOUR / 1000:.2f} kWh)")
