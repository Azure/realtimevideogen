from pathlib import Path

import numpy as np
import pytest

from matplotlib.figure import Figure

from tests.test_utils import temp_sys_path

with temp_sys_path("paper"):
    import plot_energy as plot_module
    from plot_energy import plot_energy


DATA_DIR = Path("paper") / "data"


def test_plot_includes_energy_aware_dashed_ddit_and_effective_ttff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    figures: list[Figure] = []
    monkeypatch.setattr(plot_module.plt, "close", figures.append)
    output_path = tmp_path / "energy.pdf"

    plot_energy(
        stream_pilot_path=DATA_DIR / "provisioning_streamwise.csv",
        naive_path=DATA_DIR / "provisioning_naive.csv",
        ddit_path=DATA_DIR / "llm" / "provisioning_ddit_spot_upscaler.csv",
        output_path=output_path,
    )

    assert output_path.stat().st_size > 0
    energy_ax = figures[0].axes[0]
    assert energy_ax.get_xlabel() == r"$TTFF_{eff}$"
    lines = {line.get_label(): line for line in energy_ax.get_lines()}
    assert lines["DDiT"].get_linestyle() == "--"
    assert np.asarray(lines["DDiT"].get_xdata()).size > 1
    assert np.asarray(lines["DDiT"].get_ydata()).min() > 0
