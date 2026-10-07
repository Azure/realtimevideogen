from pathlib import Path

import pytest

from matplotlib.figure import Figure

from tests.test_utils import temp_sys_path

with temp_sys_path("paper"):
    import plot_eval_llm as plot_module
    from plot_eval_llm import plot_eval_llm


def test_plot_uses_stream_pilot_and_effective_ttff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    figures: list[Figure] = []
    monkeypatch.setattr(plot_module.plt, "close", figures.append)
    output_path = tmp_path / "figure.pdf"

    plot_eval_llm(Path("paper") / "data", output_path)

    assert output_path.stat().st_size > 0
    ax = figures[0].axes[0]
    assert ax.get_xlabel() == r"$TTFF_{eff}$"
    legend = ax.get_legend()
    assert legend is not None
    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert "StreamPilot" in legend_labels
    assert "StreamWise" not in legend_labels
