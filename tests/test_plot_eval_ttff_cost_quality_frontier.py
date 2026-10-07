from pathlib import Path

import pytest

from matplotlib.figure import Figure

from tests.test_utils import temp_sys_path

with temp_sys_path("paper"):
    import plot_eval_ttff_cost_quality_frontier as plot_module
    from plot_eval_ttff_cost_quality_frontier import plot_quality_frontier


def test_plot_uses_effective_ttff_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    figures: list[Figure] = []
    monkeypatch.setattr(plot_module.plt, "close", figures.append)
    output_path = tmp_path / "figure.pdf"

    plot_quality_frontier(Path("paper") / "data", output_path)

    assert output_path.stat().st_size > 0
    assert figures[0].axes[0].get_xlabel() == r"$TTFF_{eff}$"
