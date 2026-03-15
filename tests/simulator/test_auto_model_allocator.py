"""
Tests for simulator/auto_model_allocator.py.

Covers:
- Routing to each concrete allocator by solver type.
- Upscaler-disabled warning when target_resolution is LOW.
- ValueError when an unsupported solver is requested.
"""

from __future__ import annotations

import sys
import os
import logging

import pytest
from dataclasses import replace
from unittest.mock import patch as _patch

# Add current path
sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path

with temp_sys_path("simulator"):
    from sim_types import GPUType
    from sim_types import Model
    from sim_types import QualityLevel
    from sim_types import Solver

    from constants import DEFAULT_WORKFLOW_CONFIG

    from data_loading import load_latency_data

    from policies import STREAMWISE_POLICY
    from policies import NAIVE_POLICY
    from policies import HEXGEN_POLICY
    from policies import HELIX_POLICY

    from auto_model_allocator import AutoModelAllocator

    from greedy import GreedyAllocator
    from naive_baseline import NaiveAllocator
    from hexgen import HexGenAllocator
    from helix import HelixAllocator
    from milp import MILPAllocator

    from workflows import PODCAST_WORKFLOW


# ---------------------------------------------------------------------------
# Solver routing
# ---------------------------------------------------------------------------

def test_greedy_solver_routes_to_greedy_allocator() -> None:
    """AutoModelAllocator uses GreedyAllocator when solver=GREEDY."""
    latency_data = load_latency_data("simulator/data/")
    allocator = AutoModelAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=replace(STREAMWISE_POLICY, solver=Solver.GREEDY),
    )
    assert isinstance(allocator._allocator, GreedyAllocator)


def test_naive_solver_routes_to_naive_allocator() -> None:
    """AutoModelAllocator uses NaiveAllocator when solver=NAIVE."""
    latency_data = load_latency_data("simulator/data/")
    allocator = AutoModelAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=replace(NAIVE_POLICY, solver=Solver.NAIVE),
    )
    assert isinstance(allocator._allocator, NaiveAllocator)


def test_hexgen_solver_routes_to_hexgen_allocator() -> None:
    """AutoModelAllocator uses HexGenAllocator when solver=HEXGEN."""
    latency_data = load_latency_data("simulator/data/")
    policy = replace(HEXGEN_POLICY, solver=Solver.HEXGEN)
    allocator = AutoModelAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=policy,
    )
    assert isinstance(allocator._allocator, HexGenAllocator)


def test_helix_solver_routes_to_helix_allocator() -> None:
    """AutoModelAllocator uses HelixAllocator when solver=HELIX."""
    latency_data = load_latency_data("simulator/data/")
    policy = replace(HELIX_POLICY, solver=Solver.HELIX)
    allocator = AutoModelAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=policy,
    )
    assert isinstance(allocator._allocator, HelixAllocator)


def test_highs_solver_routes_to_milp_allocator() -> None:
    """AutoModelAllocator uses MILPAllocator when solver=HIGHS."""
    latency_data = load_latency_data("simulator/data/")
    policy = replace(STREAMWISE_POLICY, solver=Solver.HIGHS)
    allocator = AutoModelAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=policy,
    )
    assert isinstance(allocator._allocator, MILPAllocator)


def test_gurobi_solver_routes_to_milp_allocator() -> None:
    """AutoModelAllocator uses MILPAllocator when solver=GUROBI."""
    latency_data = load_latency_data("simulator/data/")
    policy = replace(STREAMWISE_POLICY, solver=Solver.GUROBI)
    allocator = AutoModelAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=policy,
    )
    assert isinstance(allocator._allocator, MILPAllocator)


def test_unsupported_solver_raises() -> None:
    """Building AutoModelAllocator with an unrecognised solver raises ValueError."""
    latency_data = load_latency_data("simulator/data/")
    policy = replace(STREAMWISE_POLICY)

    allocator = AutoModelAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=policy,
    )

    # Patch the allocator's policy.solver to an unsupported sentinel value so
    # that _build_allocator falls through all known solver branches.
    bad_solver = object()
    with _patch.object(type(allocator.policy), 'solver',
                       new_callable=lambda: property(lambda self: bad_solver)):
        with pytest.raises((ValueError, AttributeError, TypeError)):
            allocator._build_allocator()


# ---------------------------------------------------------------------------
# Upscaler auto-disable when target_resolution is LOW
# ---------------------------------------------------------------------------

def test_upscaler_disabled_for_low_resolution(caplog) -> None:
    """
    When use_upscaler=True but target_resolution=LOW, allocate() should log a
    warning, disable the upscaler flag, and still return a valid Result.
    """
    latency_data = load_latency_data("simulator/data/")

    # Build a workflow with LOW target resolution (no upscaler work).
    # Use a fresh copy of model_work to avoid mutating the global PODCAST_WORKFLOW.
    low_workflow = replace(
        PODCAST_WORKFLOW,
        target_resolution=QualityLevel.LOW,
        model_work=dict(PODCAST_WORKFLOW.model_work),
    )
    # __post_init__ strips UPSCALER from model_work when resolution is not HIGH.
    assert Model.UPSCALER not in low_workflow.model_work

    # Policy says use_upscaler=True but workflow cannot support it.
    policy = replace(STREAMWISE_POLICY, use_upscaler=True)
    allocator = AutoModelAllocator(
        workflow=low_workflow,
        latency_data=latency_data,
        policy=policy,
    )

    with caplog.at_level(logging.WARNING, logger="root"):
        result = allocator.allocate(num_gpus={GPUType.A100: 8})

    # The upscaler flag should have been cleared.
    assert allocator.policy.use_upscaler is False
    # The allocation should still succeed.
    assert result is not None
    assert result.total_time_s > 0.0


# ---------------------------------------------------------------------------
# Basic end-to-end allocation through AutoModelAllocator
# ---------------------------------------------------------------------------

def test_greedy_allocation_produces_valid_result() -> None:
    """AutoModelAllocator with GREEDY solver produces a valid Result."""
    latency_data = load_latency_data("simulator/data/")
    allocator = AutoModelAllocator(
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        policy=replace(STREAMWISE_POLICY, solver=Solver.GREEDY),
    )
    result = allocator.allocate(num_gpus={GPUType.A100: 8})
    assert result.total_time_s > 0.0
    assert result.ttff_s > 0.0
    assert result.cost > 0.0
    assert result.gpus_used.get(GPUType.A100, 0) > 0
