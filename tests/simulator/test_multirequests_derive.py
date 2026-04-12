import sys
import os

# Add current path
sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path
from tests.simulator.test_simulator_multirequests import assert_equals_approx

with temp_sys_path("simulator"):
    from multirequests import TIME_PER_REQ
    from multirequests import INIT_REPLICAS
    from multirequests import HARDWARE_BUDGET

    from multirequests_derive import derive_multirequest_params


def test_derived_constants_match_simulation() -> None:
    """Verify that the hardcoded constants in multirequests.py match a fresh simulation run.

    This test re-runs the StreamWise simulator at the documented hardware budget
    (HARDWARE_BUDGET) and checks that INIT_REPLICAS and TIME_PER_REQ still match.
    If this test fails, regenerate the constants by running:
        cd simulator/ && python multirequests_derive.py
    """
    derived_replicas, derived_time = derive_multirequest_params(
        budget=dict(HARDWARE_BUDGET),
        data_dir="simulator/data/",
    )

    # Verify INIT_REPLICAS matches
    for gpu_type in derived_replicas:
        assert gpu_type in INIT_REPLICAS, (
            f"Missing GPU type {gpu_type} in INIT_REPLICAS"
        )
        for model, count in derived_replicas[gpu_type].items():
            assert model in INIT_REPLICAS[gpu_type], (
                f"Missing model {model} in INIT_REPLICAS[{gpu_type}]"
            )
            assert INIT_REPLICAS[gpu_type][model] == count, (
                f"INIT_REPLICAS[{gpu_type}][{model}]: expected {count}, "
                f"got {INIT_REPLICAS[gpu_type][model]}"
            )

    # Verify TIME_PER_REQ matches (within tolerance)
    for gpu_type in derived_time:
        assert gpu_type in TIME_PER_REQ, (
            f"Missing GPU type {gpu_type} in TIME_PER_REQ"
        )
        for model, t in derived_time[gpu_type].items():
            assert model in TIME_PER_REQ[gpu_type], (
                f"Missing model {model} in TIME_PER_REQ[{gpu_type}]"
            )
            assert_equals_approx(TIME_PER_REQ[gpu_type][model], t)

    # Verify GPU totals match the documented budget
    for gpu_type, expected_count in HARDWARE_BUDGET.items():
        actual = sum(INIT_REPLICAS.get(gpu_type, {}).values())
        assert actual == expected_count, (
            f"INIT_REPLICAS {gpu_type.value} total: {actual} != {expected_count}"
        )
