import sys
import os

# Add current path so test-package imports resolve correctly.
sys.path.append(os.getcwd())

from tests.test_utils import assert_equal_dict
from tests.test_utils import assert_equals_approx
from tests.test_utils import temp_sys_path

with temp_sys_path("simulator"):
    from sim_types import GPUType
    from sim_types import Model
    from sim_types import QualityLevel

    from multirequests import TIME_PER_REQ
    from multirequests import INIT_REPLICAS
    from multirequests import TIME_PER_REQ_ADAPTIVE
    from multirequests import INIT_REPLICAS_ADAPTIVE
    from multirequests import HARDWARE_BUDGET

    from multirequests import derive_multirequest_params
    from multirequests import derive_adaptive_params


def test_derived_constants_match_simulation() -> None:
    """Verify that the documented multirequest parameters match a fresh simulation run.
    This test re-runs the StreamWise simulator at the documented hardware budget
    (HARDWARE_BUDGET) and checks that the exported INIT_REPLICAS and TIME_PER_REQ
    values still match the simulator-derived results.
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


def test_init_replicas_snapshot() -> None:
    """Pin INIT_REPLICAS to known-good values so silent drift is detected."""
    expected = {
        GPUType.A100: {
            Model.GEMMA: 8,
            Model.FLUX: 8,
            Model.HF_VAE: 7,
            Model.FT: 192,
            Model.UPSCALER: 40,
            Model.OTHERS: 1,
        },
        GPUType.H100: {
            Model.HF: 14,
            Model.HF_VAE: 4,
            Model.FT: 38,
            Model.UPSCALER: 8,
        },
    }
    assert_equal_dict(INIT_REPLICAS, expected, name="INIT_REPLICAS")


def test_time_per_req_snapshot() -> None:
    """Pin TIME_PER_REQ to known-good values so silent drift is detected."""
    expected = {
        GPUType.A100: {
            Model.GEMMA: 8.57,
            Model.FLUX: 1.65,
            Model.HF_VAE: 21.80,
            Model.FT: 246.97,
            Model.UPSCALER: 49.40,
            Model.OTHERS: 25.80,
        },
        GPUType.H100: {
            Model.HF: 56.96,
            Model.HF_VAE: 21.80,
            Model.FT: 250.70,
            Model.UPSCALER: 49.40,
        },
    }
    assert_equal_dict(TIME_PER_REQ, expected, name="TIME_PER_REQ")


def test_init_replicas_adaptive_snapshot() -> None:
    """Pin INIT_REPLICAS_ADAPTIVE to known-good values so silent drift is detected."""
    expected = {
        GPUType.A100: {
            Model.GEMMA: 8,
            Model.FLUX: 8,
            Model.HF_VAE: 7,
            Model.FT: 192,
            Model.UPSCALER: 40,
            Model.OTHERS: 1,
        },
        GPUType.H100: {
            Model.HF: 14,
            Model.HF_VAE: 4,
            Model.FT: 38,
            Model.UPSCALER: 8,
        },
    }
    assert_equal_dict(INIT_REPLICAS_ADAPTIVE, expected, name="INIT_REPLICAS_ADAPTIVE")


def test_time_per_req_adaptive_snapshot() -> None:
    """Pin TIME_PER_REQ_ADAPTIVE to known-good values so silent drift is detected."""
    expected = {
        GPUType.A100: {
            Model.GEMMA: {
                QualityLevel.HIGH: 8.57,
                QualityLevel.MEDIUM: 8.57,
                QualityLevel.LOW: 8.57,
            },
            Model.FLUX: {
                QualityLevel.HIGH: 1.65,
                QualityLevel.MEDIUM: 0.41,
                QualityLevel.LOW: 0.10,
            },
            Model.HF_VAE: {
                QualityLevel.HIGH: 21.80,
                QualityLevel.MEDIUM: 2.79,
                QualityLevel.LOW: 1.25,
            },
            Model.FT: {
                QualityLevel.HIGH: 246.97,
                QualityLevel.MEDIUM: 57.57,
                QualityLevel.LOW: 22.99,
            },
            Model.UPSCALER: {
                QualityLevel.HIGH: 49.40,
                QualityLevel.MEDIUM: 8.50,
                QualityLevel.LOW: 3.50,
            },
            Model.OTHERS: {
                QualityLevel.HIGH: 25.80,
                QualityLevel.MEDIUM: 25.80,
                QualityLevel.LOW: 25.80,
            },
        },
        GPUType.H100: {
            Model.HF: {
                QualityLevel.HIGH: 56.96,
                QualityLevel.MEDIUM: 9.96,
                QualityLevel.LOW: 4.26,
            },
            Model.HF_VAE: {
                QualityLevel.HIGH: 21.80,
                QualityLevel.MEDIUM: 2.79,
                QualityLevel.LOW: 1.25,
            },
            Model.FT: {
                QualityLevel.HIGH: 250.70,
                QualityLevel.MEDIUM: 57.41,
                QualityLevel.LOW: 23.14,
            },
            Model.UPSCALER: {
                QualityLevel.HIGH: 49.40,
                QualityLevel.MEDIUM: 8.52,
                QualityLevel.LOW: 3.50,
            },
        },
    }
    assert_equal_dict(TIME_PER_REQ_ADAPTIVE, expected, name="TIME_PER_REQ_ADAPTIVE")
