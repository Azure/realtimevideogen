
"""
Test simulator policies.
"""

import sys
import os

# Add current path
sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path

with temp_sys_path("simulator", "streamwise"):
    from model_provisioner.policies import STREAMWISE_POLICY
    from model_provisioner.policies import BASELINE_POLICIES

    from constants import GPU_SPOT_COST

    from sim_types import GPUType
    from sim_types import Model
    from sim_types import Objective
    from sim_types import Solver


def test_streamwise_policies() -> None:
    policy = STREAMWISE_POLICY
    assert policy.name == "streamwise"
    assert policy.gpu_cost is not None
    assert policy.objective == Objective.TTFF_COST


def test_baseline_policies() -> None:
    assert len(BASELINE_POLICIES) == 7
    assert "naive" in BASELINE_POLICIES
    assert "naive disag" in BASELINE_POLICIES
    assert "naive ttff*cost allocator" in BASELINE_POLICIES
    assert "naive upscaler" in BASELINE_POLICIES
    assert "naive spot" in BASELINE_POLICIES
    assert "naive hardware" in BASELINE_POLICIES
    assert "naive combo" in BASELINE_POLICIES
    assert "fake" not in BASELINE_POLICIES


def test_naive_combo_policy() -> None:
    policy = BASELINE_POLICIES["naive combo"]

    assert policy.gpu_cost == GPU_SPOT_COST
    assert policy.objective == Objective.TTFF_COST
    assert policy.disaggregation == {
        Model.HF: True,
        Model.FT: False,
    }
    assert policy.use_upscaler is True
    assert policy.hardware == list(GPUType)
    assert policy.solver == Solver.HEXGEN
