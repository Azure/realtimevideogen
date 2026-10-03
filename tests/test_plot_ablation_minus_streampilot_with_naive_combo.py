from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from matplotlib.figure import Figure

from tests.test_utils import temp_sys_path

with temp_sys_path("paper"):
    import plot_ablation_minus_streampilot_with_naive_combo as plot_module
    from plot_ablation_minus_streampilot_with_naive_combo import CURVE_SPECS
    from plot_ablation_minus_streampilot_with_naive_combo import LEGEND_COLUMNS
    from plot_ablation_minus_streampilot_with_naive_combo import PRIMARY_LABEL
    from plot_ablation_minus_streampilot_with_naive_combo import get_pareto_frontier
    from plot_ablation_minus_streampilot_with_naive_combo import load_curve_points
    from plot_ablation_minus_streampilot_with_naive_combo import plot_ablation


def test_curve_specs_include_nine_expected_series() -> None:
    assert [(curve.label, curve.filename) for curve in CURVE_SPECS] == [
        ("StreamPilot", "provisioning_streamwise.csv"),
        ("No Spot", "provisioning_streamwise_no_spot.csv"),
        ("No disaggregation", "provisioning_streamwise_no_disag.csv"),
        ("No upscaler", "provisioning_streamwise_no_upscaler.csv"),
        ("Naive", "provisioning_streamwise_naive.csv"),
        ("Static Allocation", "provisioning_streamwise_naive_allocator.csv"),
        ("Naive Combo", "provisioning_naive_combo.csv"),
        ("StreamPilot (A100)", "provisioning_streamwise_A100.csv"),
        ("Optimal", "provisioning_streamwise_milp.csv"),
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


def test_plot_ablation_creates_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for index, curve in enumerate(CURVE_SPECS):
        pd.DataFrame({
            "ttff_s": [10.0, 60.0, 600.0, 5_000.0],
            "cost": [200.0 + index, 120.0 + index, 70.0 + index, 40.0 + index],
        }).to_csv(data_dir / curve.filename, index=False)

    closed_figures: list[Figure] = []
    original_close = plot_module.plt.close

    def capture_close(figure: Figure) -> None:
        closed_figures.append(figure)
        original_close(figure)

    monkeypatch.setattr(plot_module.plt, "close", capture_close)
    output_path = tmp_path / "figure.pdf"
    plot_ablation(data_dir, output_path)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    figure = closed_figures[0]
    legend = figure.axes[0].get_legend()
    assert legend is not None
    figure.canvas.draw()
    legend_columns = {round(text.get_window_extent().x0) for text in legend.get_texts()}
    assert len(legend_columns) == LEGEND_COLUMNS == 2
    assert [text.get_text() for text in legend.get_texts()] == [curve.label for curve in CURVE_SPECS]
    lines = {line.get_label(): line for line in figure.axes[0].get_lines()}
    assert lines[PRIMARY_LABEL].get_zorder() == 1
    assert {label: line.get_linestyle() for label, line in lines.items()} == {
        curve.label: "--" if curve.label == "Naive" else "-" for curve in CURVE_SPECS
    }
