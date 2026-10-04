from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from matplotlib.figure import Figure

from tests.test_utils import temp_sys_path

with temp_sys_path("paper"):
    import plot_eval_cost_qpm_barplot_fcfs_vs_edf as plot_module
    from plot_eval_cost_qpm_barplot_fcfs_vs_edf import BAR_STYLES
    from plot_eval_cost_qpm_barplot_fcfs_vs_edf import plot_fcfs_vs_edf

DATA_DIR = Path("paper") / "data"


def test_plot_has_stage_lines_and_total_bars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    figures: list[Figure] = []
    monkeypatch.setattr(plot_module.plt, "close", figures.append)
    output_path = tmp_path / "figure.pdf"

    plot_fcfs_vs_edf(
        data_path=DATA_DIR / "provisioning_qpm.csv",
        stream_pilot_single_path=DATA_DIR / "provisioning_streamwise.csv",
        ddit_single_path=DATA_DIR / "llm" / "provisioning_ddit.csv",
        output_path=output_path,
    )

    assert output_path.stat().st_size > 0
    stage_ax, total_ax = figures[0].axes
    stage_labels = [line.get_label() for line in stage_ax.get_lines()]
    assert stage_labels == [
        "Kokoro",
        "Gemma",
        "Flux",
        "FramePack",
        "FramePack VAE",
        "Fantasy Talking",
        "Real-ESRGAN",
    ]
    assert stage_ax.get_xscale() == "log"
    stage_legend = stage_ax.get_legend()
    assert stage_legend is not None
    assert [text.get_text() for text in stage_legend.get_texts()] == stage_labels

    bar_labels = [container.get_label() for container in total_ax.containers]
    assert bar_labels == [style[0] for style in BAR_STYLES]
    assert all(len(container) == 9 for container in total_ax.containers)
    total_legend = total_ax.get_legend()
    assert total_legend is not None
    assert [text.get_text() for text in total_legend.get_texts()] == bar_labels
    monkeypatch.undo()
    plt.close(figures[0])
