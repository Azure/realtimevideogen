import sys
import os

# Add current path and simulator path so lazy imports (e.g. greedy inside
# auto_model_allocator) resolve correctly at test-execution time.
sys.path.append(os.getcwd())
sys.path.insert(0, "simulator")

from tests.test_utils import temp_sys_path
from tests.simulator.test_simulator_multirequests import assert_equals_approx

with temp_sys_path("simulator"):
    from multirequests import TIME_PER_REQ
    from multirequests import INIT_REPLICAS
    from multirequests import TIME_PER_REQ_ADAPTIVE
    from multirequests import INIT_REPLICAS_ADAPTIVE
    from multirequests import HARDWARE_BUDGET

    from multirequests import derive_multirequest_params
    from multirequests import derive_adaptive_params


def test_derived_constants_match_simulation() -> None:
    """Verify that the hardcoded constants in multirequests.py match a fresh simulation run.

    This test re-runs the StreamWise simulator at the documented hardware budget
    (HARDWARE_BUDGET) and checks that INIT_REPLICAS and TIME_PER_REQ still match.
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


def test_derived_adaptive_constants_match_simulation() -> None:
    """Verify that the adaptive-quality constants match a fresh simulation run.

    Re-runs the simulator at HIGH/MEDIUM/LOW quality levels and checks that
    INIT_REPLICAS_ADAPTIVE and TIME_PER_REQ_ADAPTIVE still match.
    """
    derived_replicas, derived_time = derive_adaptive_params(
        budget=dict(HARDWARE_BUDGET),
        data_dir="simulator/data/",
    )

    # Verify INIT_REPLICAS_ADAPTIVE matches (exact int comparison)
    for gpu_type in derived_replicas:
        assert gpu_type in INIT_REPLICAS_ADAPTIVE, (
            f"Missing GPU type {gpu_type} in INIT_REPLICAS_ADAPTIVE"
        )
        for model, count in derived_replicas[gpu_type].items():
            assert model in INIT_REPLICAS_ADAPTIVE[gpu_type], (
                f"Missing model {model} in INIT_REPLICAS_ADAPTIVE[{gpu_type}]"
            )
            assert INIT_REPLICAS_ADAPTIVE[gpu_type][model] == count, (
                f"INIT_REPLICAS_ADAPTIVE[{gpu_type}][{model}]: expected {count}, "
                f"got {INIT_REPLICAS_ADAPTIVE[gpu_type][model]}"
            )

    # Verify TIME_PER_REQ_ADAPTIVE matches (within tolerance, per quality level)
    for gpu_type in derived_time:
        assert gpu_type in TIME_PER_REQ_ADAPTIVE, (
            f"Missing GPU type {gpu_type} in TIME_PER_REQ_ADAPTIVE"
        )
        for model, quality_times in derived_time[gpu_type].items():
            assert model in TIME_PER_REQ_ADAPTIVE[gpu_type], (
                f"Missing model {model} in TIME_PER_REQ_ADAPTIVE[{gpu_type}]"
            )
            for quality, t in quality_times.items():
                assert quality in TIME_PER_REQ_ADAPTIVE[gpu_type][model], (
                    f"Missing quality {quality} in "
                    f"TIME_PER_REQ_ADAPTIVE[{gpu_type}][{model}]"
                )
                assert_equals_approx(
                    TIME_PER_REQ_ADAPTIVE[gpu_type][model][quality], t,
                )

    # Verify GPU totals match the documented budget
    for gpu_type, expected_count in HARDWARE_BUDGET.items():
        actual = sum(INIT_REPLICAS_ADAPTIVE.get(gpu_type, {}).values())
        assert actual == expected_count, (
            f"INIT_REPLICAS_ADAPTIVE {gpu_type.value} total: "
            f"{actual} != {expected_count}"
        )
