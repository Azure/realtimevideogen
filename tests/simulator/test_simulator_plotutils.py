import sys
import os

# Add current path
sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path

with temp_sys_path("simulator"):
    from plot_utils import plot_ttff_vs_cost
    from plot_utils import plot_ttff_vs_energy
    from plot_utils import plot_adaptive_quality
    from plot_utils import plot_policies_ttff_vs_cost
    from plot_utils import plot_cost_vs_qpm
    from plot_utils import _get_time_ticklabels

    from sim_types import ProvisioningResult
    from sim_types import GPUType
    from sim_types import QualityLevel
    from sim_types import Model


def test_plot_ttff_vs_cost() -> None:
    """Dummy testing."""
    plot_ttff_vs_cost(
        ttffs=[10, 20, 30],
        costs=[10, 20, 30],
        provisions=[
            {GPUType.A100: 8},
            {GPUType.H100: 8},
            {GPUType.H200: 8},
        ],
        verbose=True,
    )


def test_plot_ttff_vs_energy() -> None:
    """Dummy testing."""
    plot_ttff_vs_energy(
        ttff_list=[10, 20, 30],
        energy_list=[100, 200, 300],
        actual_provision=[
            {GPUType.A100: 8},
            {GPUType.H100: 8},
            {GPUType.H200: 8},
        ],
        verbose=True,
    )


def test_plot_adaptive_quality() -> None:
    """Dummy testing."""
    provisioning_result_adaptive = ProvisioningResult(
        latencies=[10, 20, 30],
        costs=[100, 200, 300],
        ttffs=[50, 100, 150],
        tbfs=[0.5, 1.0, 1.5],
        actual_provision=[
            {GPUType.A100: 8},
            {GPUType.H100: 8},
            {GPUType.H200: 8},
        ],
        config_provision=[
            {GPUType.A100: 8},
            {GPUType.H100: 8},
            {GPUType.H200: 8},
        ],
        model_provision=[
            {GPUType.A100: {}},
            {GPUType.H100: {}},
            {GPUType.H200: {}},
        ],
    )

    # TODO
    provisioning_result_low = ProvisioningResult(
        latencies=[15, 25, 35],
        costs=[110, 210, 310],
        ttffs=[60, 110, 160],
        tbfs=[0.6, 1.1, 1.6],
        actual_provision=[
            {GPUType.A100: 8},
            {GPUType.H100: 8},
            {GPUType.H200: 8},
        ],
        config_provision=[
            {GPUType.A100: 8},
            {GPUType.H100: 8},
            {GPUType.H200: 8},
        ],
        model_provision=[
            {GPUType.A100: {}},
            {GPUType.H100: {}},
            {GPUType.H200: {}},
        ],
    )
    provisioning_result_medium = provisioning_result_low
    provisioning_result_high = provisioning_result_medium

    plot_adaptive_quality(
        provisioning_result_adaptive=provisioning_result_adaptive,
        provisioning_qualities={
            QualityLevel.LOW: provisioning_result_low,
            QualityLevel.MEDIUM: provisioning_result_medium,
            QualityLevel.HIGH: provisioning_result_high,
        }
    )


def test_plot_policies_ttff_vs_cost() -> None:
    plot_policies_ttff_vs_cost(
        provision_results={},
    )

    plot_policies_ttff_vs_cost(
        provision_results={
            "naive": ProvisioningResult(
                latencies=[10, 20, 30],
                costs=[100, 200, 300],
                ttffs=[50, 100, 150],
                tbfs=[0.5, 1.0, 1.5],
                actual_provision=[
                    {GPUType.A100: 8},
                    {GPUType.H100: 8},
                    {GPUType.H200: 8},
                ],
                config_provision=[
                    {GPUType.A100: 8},
                    {GPUType.H100: 8},
                    {GPUType.H200: 8},
                ],
                model_provision=[
                    {GPUType.A100: {}},
                    {GPUType.H100: {}},
                    {GPUType.H200: {}},
                ],
            ),
        },
    )


def test_plot_cost_vs_qpm() -> None:
    plot_cost_vs_qpm(
        costs={},
        qpms=[],
    )

    plot_cost_vs_qpm(
        costs={
            GPUType.A100: {
                Model.GEMMA: [10, 20, 30],
                Model.FT: [15, 25, 35],
            },
            GPUType.H100: {
                Model.FLUX: [18, 28, 38],
            },
        },
        qpms=[1, 2, 5],
    )


def test_get_time_ticklabels() -> None:
    ticks, tick_labels = _get_time_ticklabels()
    assert 18 == len(ticks) == len(tick_labels)
    assert tick_labels == [
        "1s", "2s", "5s", "10s", "15s", "30s",
        "1m", "2m", "5m", "10m", "20m", "40m",
        "1h", "3h", "5h", "8h", "12h", "1d"]

    ticks, tick_labels = _get_time_ticklabels(exclude_x=[17])
    assert 18 == len(ticks) == len(tick_labels)
    assert "5m" in tick_labels

    ticks, tick_labels = _get_time_ticklabels(exclude_x=[300])
    assert 17 == len(ticks) == len(tick_labels)
    assert "5m" not in tick_labels
