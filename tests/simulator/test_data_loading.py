"""
Test data loading.
"""

import sys
import os
import pytest

# Add current path
sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path

with temp_sys_path("simulator", "streamwise"):
    from sim_types import QualityLevel

    from data_loading import load_latency_data
    from data_loading import load_power_data
    from data_loading import load_adaptive_quality_data


def test_latency() -> None:
    latency_data = load_latency_data("simulator/data/")
    assert latency_data is not None

    with pytest.raises(FileNotFoundError):
        load_latency_data("nonexisting")


def test_power() -> None:
    power_data = load_power_data("simulator/data/")
    assert power_data is not None

    with pytest.raises(FileNotFoundError):
        load_power_data("nonexisting")


def test_adaptive_quality() -> None:
    latency_quality_data = load_adaptive_quality_data(
        "simulator/data/",
        QualityLevel.LOW,
    )
    assert latency_quality_data is not None

    latency_quality_data = load_adaptive_quality_data(
        "simulator/data/",
        QualityLevel.MEDIUM,
    )
    assert latency_quality_data is not None

    latency_quality_data = load_adaptive_quality_data(
        "simulator/data/",
        QualityLevel.HIGH,  # ORIGINAL
    )
    assert latency_quality_data is not None

    with pytest.raises(AssertionError):
        load_adaptive_quality_data(
            "simulator/data/",
            "nonexisting"
        )
