import sys
import os

# Add current path so test-package imports resolve correctly.
sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path
from tests.simulator.test_simulator_multirequests import assert_equals_approx

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


# ---------------------------------------------------------------------------
# Snapshot tests — pin the concrete derived values so silent drift is caught.
# ---------------------------------------------------------------------------

EXPECTED_INIT_REPLICAS: dict[GPUType, dict[Model, int]] = {
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

EXPECTED_TIME_PER_REQ: dict[GPUType, dict[Model, float]] = {
    GPUType.A100: {
        Model.GEMMA: 8.574000000000002,
        Model.FLUX: 1.6500000000000001,
        Model.HF_VAE: 21.803283571428572,
        Model.FT: 246.96562067532471,
        Model.UPSCALER: 49.400000000000006,
        Model.OTHERS: 25.8,
    },
    GPUType.H100: {
        Model.HF: 56.956024691358024,
        Model.HF_VAE: 21.80495662654321,
        Model.FT: 250.69966349350648,
        Model.UPSCALER: 49.4046,
    },
}

EXPECTED_INIT_REPLICAS_ADAPTIVE: dict[GPUType, dict[Model, int]] = {
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

EXPECTED_TIME_PER_REQ_ADAPTIVE: dict[GPUType, dict[Model, dict[QualityLevel, float]]] = {
    GPUType.A100: {
        Model.GEMMA: {
            QualityLevel.HIGH: 8.574000000000002,
            QualityLevel.MEDIUM: 8.574000000000002,
            QualityLevel.LOW: 8.574000000000002,
        },
        Model.FLUX: {
            QualityLevel.HIGH: 1.6500000000000001,
            QualityLevel.MEDIUM: 0.41250000000000003,
            QualityLevel.LOW: 0.10312500000000001,
        },
        Model.HF_VAE: {
            QualityLevel.HIGH: 21.803283571428572,
            QualityLevel.MEDIUM: 2.7903365220959597,
            QualityLevel.LOW: 1.2475642875,
        },
        Model.FT: {
            QualityLevel.HIGH: 246.96562067532471,
            QualityLevel.MEDIUM: 57.57458318181818,
            QualityLevel.LOW: 22.986856840909095,
        },
        Model.UPSCALER: {
            QualityLevel.HIGH: 49.400000000000006,
            QualityLevel.MEDIUM: 8.495833333333334,
            QualityLevel.LOW: 3.4953125000000003,
        },
        Model.OTHERS: {
            QualityLevel.HIGH: 25.8,
            QualityLevel.MEDIUM: 25.8,
            QualityLevel.LOW: 25.8,
        },
    },
    GPUType.H100: {
        Model.HF: {
            QualityLevel.HIGH: 56.956024691358024,
            QualityLevel.MEDIUM: 9.962037037037037,
            QualityLevel.LOW: 4.261674382716049,
        },
        Model.HF_VAE: {
            QualityLevel.HIGH: 21.80495662654321,
            QualityLevel.MEDIUM: 2.7902749521604937,
            QualityLevel.LOW: 1.2478214344135803,
        },
        Model.FT: {
            QualityLevel.HIGH: 250.69966349350648,
            QualityLevel.MEDIUM: 57.40960875844156,
            QualityLevel.LOW: 23.139484528409092,
        },
        Model.UPSCALER: {
            QualityLevel.HIGH: 49.4046,
            QualityLevel.MEDIUM: 8.5176,
            QualityLevel.LOW: 3.4965,
        },
    },
}


def test_init_replicas_snapshot() -> None:
    """Pin INIT_REPLICAS to known-good values so silent drift is detected."""
    assert set(INIT_REPLICAS.keys()) == set(EXPECTED_INIT_REPLICAS.keys()), (
        f"GPU types differ: {set(INIT_REPLICAS.keys())} != {set(EXPECTED_INIT_REPLICAS.keys())}"
    )
    for gpu_type, models in EXPECTED_INIT_REPLICAS.items():
        assert set(INIT_REPLICAS[gpu_type].keys()) == set(models.keys()), (
            f"Models for {gpu_type} differ: "
            f"{set(INIT_REPLICAS[gpu_type].keys())} != {set(models.keys())}"
        )
        for model, expected_count in models.items():
            assert INIT_REPLICAS[gpu_type][model] == expected_count, (
                f"INIT_REPLICAS[{gpu_type}][{model}]: "
                f"expected {expected_count}, got {INIT_REPLICAS[gpu_type][model]}"
            )


def test_time_per_req_snapshot() -> None:
    """Pin TIME_PER_REQ to known-good values so silent drift is detected."""
    assert set(TIME_PER_REQ.keys()) == set(EXPECTED_TIME_PER_REQ.keys()), (
        f"GPU types differ: {set(TIME_PER_REQ.keys())} != {set(EXPECTED_TIME_PER_REQ.keys())}"
    )
    for gpu_type, models in EXPECTED_TIME_PER_REQ.items():
        assert set(TIME_PER_REQ[gpu_type].keys()) == set(models.keys()), (
            f"Models for {gpu_type} differ: "
            f"{set(TIME_PER_REQ[gpu_type].keys())} != {set(models.keys())}"
        )
        for model, expected_val in models.items():
            assert_equals_approx(TIME_PER_REQ[gpu_type][model], expected_val)


def test_init_replicas_adaptive_snapshot() -> None:
    """Pin INIT_REPLICAS_ADAPTIVE to known-good values so silent drift is detected."""
    assert set(INIT_REPLICAS_ADAPTIVE.keys()) == set(EXPECTED_INIT_REPLICAS_ADAPTIVE.keys()), (
        f"GPU types differ: "
        f"{set(INIT_REPLICAS_ADAPTIVE.keys())} != {set(EXPECTED_INIT_REPLICAS_ADAPTIVE.keys())}"
    )
    for gpu_type, models in EXPECTED_INIT_REPLICAS_ADAPTIVE.items():
        assert set(INIT_REPLICAS_ADAPTIVE[gpu_type].keys()) == set(models.keys()), (
            f"Models for {gpu_type} differ: "
            f"{set(INIT_REPLICAS_ADAPTIVE[gpu_type].keys())} != {set(models.keys())}"
        )
        for model, expected_count in models.items():
            assert INIT_REPLICAS_ADAPTIVE[gpu_type][model] == expected_count, (
                f"INIT_REPLICAS_ADAPTIVE[{gpu_type}][{model}]: "
                f"expected {expected_count}, got {INIT_REPLICAS_ADAPTIVE[gpu_type][model]}"
            )


def test_time_per_req_adaptive_snapshot() -> None:
    """Pin TIME_PER_REQ_ADAPTIVE to known-good values so silent drift is detected."""
    assert set(TIME_PER_REQ_ADAPTIVE.keys()) == set(EXPECTED_TIME_PER_REQ_ADAPTIVE.keys()), (
        f"GPU types differ: "
        f"{set(TIME_PER_REQ_ADAPTIVE.keys())} != {set(EXPECTED_TIME_PER_REQ_ADAPTIVE.keys())}"
    )
    for gpu_type, models in EXPECTED_TIME_PER_REQ_ADAPTIVE.items():
        assert set(TIME_PER_REQ_ADAPTIVE[gpu_type].keys()) == set(models.keys()), (
            f"Models for {gpu_type} differ: "
            f"{set(TIME_PER_REQ_ADAPTIVE[gpu_type].keys())} != {set(models.keys())}"
        )
        for model, quality_times in models.items():
            assert set(TIME_PER_REQ_ADAPTIVE[gpu_type][model].keys()) == set(quality_times.keys()), (
                f"Quality levels for {gpu_type}/{model} differ: "
                f"{set(TIME_PER_REQ_ADAPTIVE[gpu_type][model].keys())} != {set(quality_times.keys())}"
            )
            for quality, expected_val in quality_times.items():
                assert_equals_approx(
                    TIME_PER_REQ_ADAPTIVE[gpu_type][model][quality], expected_val,
                )
