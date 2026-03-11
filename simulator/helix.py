"""
Helix algorithm for the StreamWise workflow allocation problem.

Reference: https://github.com/Thesys-lab/Helix-ASPLOS25

Helix optimizes models one-by-one following MODEL_ORDER, using MILP
for each model's resource allocation.  After each model reaches convergence
(solver optimality or per-model time limit), its allocation is fixed and the
remaining GPU budget is passed to the next model.

Design rationale:
    HelixAllocator does NOT inherit from MILPAllocator because the parent's
    allocate() builds a single joint MILP for all models simultaneously.
    Instead, HelixAllocator extends ModelAllocator and *composes*
    MILPAllocator instances — one per model in the workflow.

    For each model, a per-model WorkflowConfig is created where only the
    target model has non-zero work (all others set to 0).  The existing MILP
    constraints (is_active <= work, gpus <= num_gpus * is_active) naturally
    force 0 GPU allocation for those 0-work models, so no changes to
    milp.py are required.
"""

from __future__ import annotations

import logging

from dataclasses import replace
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

from model_allocator import ModelAllocator

from evaluator import evaluate_model_allocation

from milp import MILPAllocator

from policies import HELIX_POLICY
from policies import MAX_DEVICES

from constants import DEVICE_OPTIONS


# Default per-model MILP solver time limit in seconds.
# Each model gets this long to converge before the solver moves on.
DEFAULT_PER_MODEL_TIME_LIMIT = 30


def _compute_per_model_gpu_budget(
    model_order: list[Model],
    num_gpus: dict[GPUType, int],
    workflow: WorkflowConfig,
) -> dict[Model, dict[GPUType, int]]:
    """Compute a per-model GPU budget so every model gets a fair share.

    Budget is proportional to each model's ``MAX_DEVICES`` weight (capped
    by the model's actual maximum useful device count from ``DEVICE_OPTIONS``).
    Models not in ``MAX_DEVICES`` (e.g. OTHERS, UPSCALER) receive a minimum
    allocation of ``min(DEVICE_OPTIONS)`` GPUs.

    The allocations are floored per model, and any remainder is distributed
    round-robin starting from the first model.

    Returns:
        Mapping ``model -> {gpu_type -> max_gpus}`` that the model may use.
    """
    # Effective weight per model (max useful devices)
    weights: dict[Model, int] = {}
    for m in model_order:
        if workflow.model_work.get(m, 0) == 0:
            continue
        if m in MAX_DEVICES:
            weights[m] = MAX_DEVICES[m]
        else:
            # Models not in MAX_DEVICES (OTHERS, UPSCALER) get min allocation
            weights[m] = min(DEVICE_OPTIONS.get(m, [1]))

    total_weight = sum(weights.values())
    if total_weight == 0:
        # Fallback: equal split
        total_weight = len(weights) or 1
        weights = {m: 1 for m in weights}

    budget: dict[Model, dict[GPUType, int]] = {}
    for gpu_type, total in num_gpus.items():
        # Floor allocation per model
        allocated = 0
        per_model: dict[Model, int] = {}
        for m in model_order:
            if m not in weights:
                continue
            share = int(total * weights[m] / total_weight)
            # Ensure at least 1 GPU per model (if GPUs available)
            share = max(share, 1) if total - allocated >= 1 else 0
            per_model[m] = share
            allocated += share

        # Distribute remainder round-robin
        remainder = total - allocated
        idx = 0
        models_list = [m for m in model_order if m in per_model]
        while remainder > 0 and models_list:
            m = models_list[idx % len(models_list)]
            per_model[m] += 1
            remainder -= 1
            idx += 1

        for m in model_order:
            if m not in per_model:
                continue
            if m not in budget:
                budget[m] = {}
            budget[m][gpu_type] = per_model[m]

    return budget


class HelixAllocator(ModelAllocator):
    """
    Helix-style allocator that optimizes models one at a time
    using MILP, sequentially following MODEL_ORDER.

    Reference: https://github.com/Thesys-lab/Helix-ASPLOS25

    Key approach:
    1. For each model in MODEL_ORDER, create a per-model MILP sub-problem
       where only the target model has non-zero work.
    2. Solve the MILP with the remaining GPU budget and a per-model time limit.
    3. Fix the allocation for that model and subtract used GPUs.
    4. Move to the next model with the remaining GPU budget.
    5. Combine all per-model allocations into the final result.

    The HelixAllocator uses composition (not inheritance) with MILPAllocator,
    creating a separate MILPAllocator instance for each model's sub-problem.
    This avoids modifying the joint MILP formulation and allows per-model
    solver configurations.
    """

    def __init__(
        self,
        workflow: WorkflowConfig,
        latency_data: LatencyData,
        power_data: Optional[PowerData] = None,
        policy: Policy = HELIX_POLICY,
    ) -> None:
        super().__init__(
            workflow,
            latency_data,
            power_data,
            policy,
        )
        assert self.policy.solver == Solver.HELIX

    def allocate(
        self,
        num_gpus: dict[GPUType, int],
        verbose: bool = False,
        per_model_time_limit: int = DEFAULT_PER_MODEL_TIME_LIMIT,
        milp_solver: Solver = Solver.HIGHS,
    ) -> Result:
        """
        Allocate resources model-by-model following MODEL_ORDER.

        For each model, a MILPAllocator is created with a workflow where
        only the target model has non-zero work.  The MILP solver optimizes
        the allocation for that model within the remaining GPU budget.

        Args:
            num_gpus: Available GPUs per type.
            verbose: If True, print per-model allocation details.
            per_model_time_limit: Time limit (seconds) for each per-model MILP solve.
            milp_solver: MILP solver backend to use (GUROBI or HIGHS).

        Returns:
            Combined Result across all models.
        """
        assert milp_solver in (Solver.GUROBI, Solver.HIGHS), \
            f"milp_solver must be GUROBI or HIGHS, got {milp_solver}"

        model_order = self.workflow.get_model_order()
        if not self.policy.use_upscaler and Model.UPSCALER in model_order:
            # Remove UPSCALER from model_order if not using upscaler to avoid unnecessary MILP solve
            model_order.remove(Model.UPSCALER)
        remaining_gpus = dict(num_gpus)

        # ---- GPU budget partitioning ----
        # Pre-compute a per-model GPU budget proportional to MAX_DEVICES
        # so that early models cannot starve later ones.  Unused GPUs from
        # one model roll over to subsequent models.
        gpu_budget = _compute_per_model_gpu_budget(
            model_order, num_gpus, self.workflow,
        )

        if verbose:
            logging.info("Helix GPU budget per model:")
            for m in model_order:
                if m in gpu_budget:
                    logging.info(f"  {m.value}: {gpu_budget[m]}")

        # Accumulated per-model allocations and metrics
        all_model_allocations: dict[GPUType, dict[Model, list[ModelAllocation]]] = {}
        total_makespan = 0.0
        total_ttff = 0.0
        total_cost = 0.0
        total_energy = 0.0
        total_gpus_used: dict[GPUType, int] = {gt: 0 for gt in num_gpus}

        for model in model_order:
            work = self.workflow.model_work.get(model, 0)
            if work == 0:
                continue

            # Skip VAE models when disaggregation is disabled for the parent.
            # Their latency is folded into the parent model's time calculation.
            if model == Model.HF_VAE and not self.policy.is_disaggregated(Model.HF):
                continue
            if model == Model.FT_VAE and not self.policy.is_disaggregated(Model.FT):
                continue

            # Check if any GPUs remain
            if all(v <= 0 for v in remaining_gpus.values()):
                logging.warning(
                    f"Helix: No GPUs remaining for {model.value}. Skipping.")
                continue

            # Filter out GPU types with 0 remaining.
            # Cap per-model GPUs to the budget so later models are not starved.
            model_budget = gpu_budget.get(model, {})
            active_gpus = {
                gt: min(count, model_budget.get(gt, count))
                for gt, count in remaining_gpus.items()
                if count > 0 and model_budget.get(gt, count) > 0
            }

            if verbose:
                logging.info(
                    f"--- Helix: Optimizing {model.value} "
                    f"(work={work}) with remaining GPUs: {active_gpus} ---"
                )

            # ---- build per-model workflow ----
            # Only the target model has work; other models are excluded from
            # model_work so the MILP only builds variables/constraints for it.
            per_model_work = {model: self.workflow.model_work[model]}
            per_model_workflow = replace(
                self.workflow,
                model_work=per_model_work,
            )

            # ---- build MILP-compatible policy ----
            # The inner MILPAllocator requires solver ∈ {GUROBI, HIGHS}.
            # Force disaggregation / use_upscaler flags so that the inner
            # MILP's ``model_names`` list includes VAE / UPSCALER when those
            # are the target model.  Without this, the MILP would construct
            # an empty model set and produce a trivial (infeasible) problem.
            disag = {}  # dict(self.policy.disaggregation)
            if model == Model.HF_VAE and self.policy.is_disaggregated(Model.HF):
                disag[Model.HF] = True
            if model == Model.FT_VAE and self.policy.is_disaggregated(Model.FT):
                disag[Model.FT] = True
            milp_policy = Policy(
                name=self.policy.name,
                gpu_cost=self.policy.gpu_cost,
                objective=self.policy.objective,
                # disaggregation=self.policy.disaggregation or model == Model.HF_VAE,
                disaggregation=disag,
                use_upscaler=self.policy.use_upscaler or model == Model.UPSCALER,
                hardware=self.policy.hardware,
                solver=milp_solver,
            )

            # ---- solve per-model MILP ----
            milp_allocator = MILPAllocator(
                workflow=per_model_workflow,
                latency_data=self.latency_data,
                power_data=self.power_data,
                policy=milp_policy,
            )

            result = milp_allocator.allocate(
                num_gpus=active_gpus,
                verbose=verbose,
                time_limit=per_model_time_limit,
                # Use running_cost=True for linear cost formulation (HiGHS-compatible)
                running_cost=(milp_solver == Solver.HIGHS),
                # Skip server constraint: per-model allocations don't need
                # to be multiples of NUM_GPUS_PER_SERVER.
                skip_server_constraint=True,
            )

            if result.total_time_s == 0.0 and not result.models:
                logging.warning(
                    f"Helix: MILP failed for {model.value}. Skipping.")
                continue

            # ---- record allocations & snap devices to DEVICE_OPTIONS ----
            # The MILP constrains devices to DEVICE_OPTIONS, but floating-point
            # precision in the solver can occasionally produce off-by-one values
            # (e.g. 31 instead of 32).  Snap each replica to the nearest valid
            # option, adjusting the GPU accounting so we don't exceed the total
            # budget passed to evaluate_model_allocation at the end.
            for gpu_type, model_dict in result.models.items():
                if gpu_type not in all_model_allocations:
                    all_model_allocations[gpu_type] = {}
                for m_name, allocs in model_dict.items():
                    for alloc in allocs:
                        valid_devices = DEVICE_OPTIONS.get(m_name, [1])
                        if alloc.devices not in valid_devices:
                            nearest = min(valid_devices, key=lambda d: abs(d - alloc.devices))
                            diff = nearest - alloc.devices  # positive = round up
                            gpu_avail = remaining_gpus.get(gpu_type, 0) - result.gpus_used.get(gpu_type, 0)
                            if diff > 0 and gpu_avail < diff:
                                # Not enough spare GPUs to round up; round down instead
                                nearest = max(
                                    (d for d in valid_devices if d <= alloc.devices),
                                    default=valid_devices[0],
                                )
                                diff = nearest - alloc.devices
                            logging.info(
                                f"Helix: snapping {m_name.value} from "
                                f"{alloc.devices} to {nearest} devices "
                                f"(solver precision fix, diff={diff:+d})")
                            # Adjust GPU accounting for this model's result
                            result.gpus_used[gpu_type] = result.gpus_used.get(gpu_type, 0) + diff
                            alloc.devices = nearest
                    all_model_allocations[gpu_type][m_name] = allocs

            # ---- accumulate metrics ----
            total_makespan += result.total_time_s
            total_ttff += result.ttff_s
            total_cost += result.cost
            total_energy += result.total_energy
            if verbose:
                print(f'Model {model.value} - Time: {result.total_time_s:.2f}s,'
                      f'TTFF: {result.ttff_s:.2f}s, Cost: ${result.cost:.2f}')
                print(f'Total cost so far: ${total_cost:.2f}, Total time so far: {total_makespan:.2f}s,'
                      f'Total TTFF so far: {total_ttff:.2f}s')
                print(f'GPUs allocated for {model.value}: {result.gpus_used}')

            # ---- subtract used GPUs ----
            for gpu_type, used in result.gpus_used.items():
                remaining_gpus[gpu_type] = remaining_gpus.get(gpu_type, 0) - used
                total_gpus_used[gpu_type] = total_gpus_used.get(gpu_type, 0) + used

            # ---- roll over unused budget to next models ----
            # If this model used fewer GPUs than its budget, the surplus
            # is distributed evenly among the remaining models.
            remaining_models = [
                m for m in model_order
                if m in gpu_budget and MODEL_ORDER.get(m, 0) > MODEL_ORDER.get(model, 0)
            ]
            if remaining_models:
                for gpu_type in num_gpus:
                    budget_for_model = model_budget.get(gpu_type, 0)
                    used_by_model = result.gpus_used.get(gpu_type, 0)
                    surplus = budget_for_model - used_by_model
                    if surplus > 0:
                        per_model_extra = surplus // len(remaining_models)
                        leftover = surplus % len(remaining_models)
                        for i, rm in enumerate(remaining_models):
                            extra = per_model_extra + (1 if i < leftover else 0)
                            gpu_budget[rm][gpu_type] = gpu_budget[rm].get(gpu_type, 0) + extra

            if verbose:
                print(
                    f"Helix: {model.value} allocated.  "
                    f"Time: {result.total_time_s:.2f}s, "
                    f"TTFF: {result.ttff_s:.2f}s, "
                    f"GPUs used: {result.gpus_used}, "
                    f"Remaining: {remaining_gpus}"
                )

        result = evaluate_model_allocation(
            workflow=self.workflow,
            latency_data=self.latency_data,
            power_data=self.power_data,
            policy=self.policy,
            models=all_model_allocations,
            num_gpus=num_gpus,
        )

        if verbose:
            print(
                f"=== Helix final: "
                f"Makespan={result.total_time_s:.2f}s, "
                f"TTFF={result.ttff_s:.2f}s, "
                f"TBF={result.tbf_s:.4f}s, "
                f"Cost=${result.cost:.2f}, "
                f"Energy={result.total_energy:.2f}Ws, "
                f"GPUs used={result.gpus_used} ==="
            )

        return result
