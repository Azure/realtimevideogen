from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from matplotlib.figure import Figure

from tests.test_utils import temp_sys_path

with temp_sys_path("paper"):
    import plot_eval_cost_autoscaling as plot_module
    from plot_eval_cost_autoscaling import SYSTEM_STYLES
    from plot_eval_cost_autoscaling import get_ddit_cost_multiplier
    from plot_eval_cost_autoscaling import plot_autoscaling
    from prepare_azure_lmm_trace import aggregate_per_minute
    from prepare_azure_lmm_trace import prepare_trace

DATA_DIR = Path("paper") / "data"


def test_aggregate_per_minute_fills_empty_minutes(tmp_path: Path) -> None:
    trace = pd.DataFrame({"TIMESTAMP": [
        "2024-10-15T12:00:00.269Z",
        "2024-10-15T12:00:59.000Z",
        "2024-10-15T12:02:01.000Z",
    ]})
    per_minute = aggregate_per_minute(trace)
    assert per_minute["requests"].tolist() == [2, 0, 1]
    assert per_minute["minute"].tolist() == [0, 1, 2]

    source = tmp_path / "trace.csv"
    trace.to_csv(source, index=False)
    output = tmp_path / "per_minute.csv"
    prepare_trace(str(source), output)
    assert output.read_text(encoding="utf-8").startswith("# ")
    assert pd.read_csv(output, comment="#")["requests"].sum() == 3


def test_committed_trace_covers_one_week() -> None:
    per_minute = pd.read_csv(DATA_DIR / "azure_lmm_trace_2024_per_minute.csv", comment="#")
    assert len(per_minute) == 7 * 24 * 60
    assert per_minute["requests"].sum() == 1_000_000


def test_ddit_cost_multiplier_matches_steady_state_figure() -> None:
    multiplier = get_ddit_cost_multiplier(
        DATA_DIR / "provisioning_qpm.csv",
        DATA_DIR / "provisioning_streamwise.csv",
        DATA_DIR / "llm" / "provisioning_ddit.csv",
    )
    assert multiplier == pytest.approx(1405.62 / 687.61, rel=1e-3)


def test_plot_has_timeline_and_strategy_bars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Two days of the committed trace keep the test fast.
    per_minute = pd.read_csv(DATA_DIR / "azure_lmm_trace_2024_per_minute.csv", comment="#")
    trace_path = tmp_path / "trace.csv"
    per_minute.iloc[:2 * 24 * 60].to_csv(trace_path, index=False)
    figures: list[Figure] = []
    monkeypatch.setattr(plot_module.plt, "close", figures.append)
    output_path = tmp_path / "figure.pdf"

    results = plot_autoscaling(
        trace_path=trace_path,
        qpm_path=DATA_DIR / "provisioning_qpm.csv",
        stream_pilot_single_path=DATA_DIR / "provisioning_streamwise.csv",
        ddit_single_path=DATA_DIR / "llm" / "provisioning_ddit.csv",
        output_path=output_path,
    )

    assert output_path.stat().st_size > 0
    timeline_ax, bar_ax, load_ax = figures[0].axes
    systems = [style[0] for style in SYSTEM_STYLES]
    assert [line.get_label() for line in timeline_ax.get_lines()] == systems
    timeline_legend = timeline_ax.get_legend()
    assert timeline_legend is not None
    assert [text.get_text() for text in timeline_legend.get_texts()] == systems + ["Load"]
    assert load_ax.get_ylabel() == "Load (QPM)"
    assert load_ax.get_ylim() == (0, 30)
    assert timeline_ax.get_xlabel() == ""
    assert [tick.get_text() for tick in timeline_ax.get_xticklabels(minor=True)] == ["Day1", "Day2"]
    assert bar_ax.get_xlabel() == ""

    # Serving + warm-pool stacked bar per system.
    assert len(bar_ax.containers) == 2 * len(systems)
    assert all(len(container) == 3 for container in bar_ax.containers)
    bar_legend = bar_ax.get_legend()
    assert bar_legend is not None
    assert [text.get_text() for text in bar_legend.get_texts()] == systems + ["Warm pool"]
    assert [tick.get_text() for tick in bar_ax.get_xticklabels()] == ["Static", "Reactive", "Predictive"]
    for by_system in results.values():
        stream_pilot = by_system["StreamPilot"]
        assert stream_pilot.slo_attainment >= 0.99
        assert stream_pilot.avg_cost < by_system["DDiT"].avg_cost
    monkeypatch.undo()
    plt.close(figures[0])
