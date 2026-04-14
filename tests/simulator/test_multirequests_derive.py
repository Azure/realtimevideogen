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
