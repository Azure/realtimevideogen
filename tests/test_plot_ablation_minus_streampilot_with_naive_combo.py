from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.test_utils import temp_sys_path

with temp_sys_path("paper"):
    from plot_ablation_minus_streampilot_with_naive_combo import CURVE_SPECS
    from plot_ablation_minus_streampilot_with_naive_combo import get_pareto_frontier
    from plot_ablation_minus_streampilot_with_naive_combo import load_curve_points
    from plot_ablation_minus_streampilot_with_naive_combo import plot_ablation


def test_curve_specs_include_eight_expected_series() -> None:
    assert [(curve.label, curve.filename) for curve in CURVE_SPECS] == [
        ("StreamWise", "provisioning_streamwise.csv"),
        ("No Spot", "provisioning_streamwise_no_spot.csv"),
        ("No disaggregation", "provisioning_streamwise_no_disag.csv"),
        ("No upscaler", "provisioning_streamwise_no_upscaler.csv"),
        ("Naive", "provisioning_streamwise_naive.csv"),
        ("Static Allocation", "provisioning_streamwise_naive_allocator.csv"),
        ("Naive Combo", "provisioning_naive_combo.csv"),
        ("A100 Only", "provisioning_streamwise_A100.csv"),
    ]


def test_load_curve_points_rejects_missing_metrics(tmp_path: Path) -> None:
    data_path = tmp_path / "invalid.csv"
    pd.DataFrame({"ttff_s": [10.0]}).to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="missing required columns: cost"):
        load_curve_points(data_path)


def test_load_curve_points_rejects_non_numeric_metrics(tmp_path: Path) -> None:
    data_path = tmp_path / "invalid.csv"
    pd.DataFrame({"ttff_s": [10.0], "cost": ["invalid"]}).to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="non-numeric or non-finite"):
        load_curve_points(data_path)


def test_get_pareto_frontier_removes_dominated_points() -> None:
    points = np.array([
        [10.0, 100.0],
        [20.0, 80.0],
        [30.0, 90.0],
        [40.0, 60.0],
    ])

    frontier = get_pareto_frontier(points)

    assert [30.0, 90.0] not in frontier.tolist()
    assert [10.0, 100.0] in frontier.tolist()
    assert [20.0, 80.0] in frontier.tolist()
    assert [40.0, 60.0] in frontier.tolist()


def test_plot_ablation_creates_pdf(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for index, curve in enumerate(CURVE_SPECS):
        pd.DataFrame({
            "ttff_s": [10.0, 60.0, 600.0, 5_000.0],
            "cost": [200.0 + index, 120.0 + index, 70.0 + index, 40.0 + index],
        }).to_csv(data_dir / curve.filename, index=False)

    output_path = tmp_path / "figure.pdf"
    plot_ablation(data_dir, output_path)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
