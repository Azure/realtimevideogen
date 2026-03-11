
"""
Test simulator policies.
"""

import sys
import os

# Add current path
sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path

with temp_sys_path("simulator"):
    from policies import STREAMWISE_POLICY
    from policies import BASELINE_POLICIES

    from sim_types import Objective


def test_streamwise_policies() -> None:
    policy = STREAMWISE_POLICY
    assert policy.name == "streamwise"
    assert policy.gpu_cost is not None
    assert policy.objective == Objective.TTFF_COST


def test_baseline_policies() -> None:
    assert len(BASELINE_POLICIES) == 6
    assert "naive" in BASELINE_POLICIES
    assert "naive disag" in BASELINE_POLICIES
    assert "naive ttff*cost allocator" in BASELINE_POLICIES
    assert "naive upscaler" in BASELINE_POLICIES
    assert "naive spot" in BASELINE_POLICIES
    assert "naive hardware" in BASELINE_POLICIES
    assert "fake" not in BASELINE_POLICIES
