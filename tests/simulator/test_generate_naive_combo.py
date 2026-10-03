from pathlib import Path

import pandas as pd
import pytest

from tests.test_utils import temp_sys_path

with temp_sys_path("simulator"):
    from generate_naive_combo import CSV_COLUMNS
    from generate_naive_combo import combine_checkpoints
    from generate_naive_combo import get_batch_path
    from generate_naive_combo import generate_naive_combo
    from generate_naive_combo import result_to_frame
    from sim_types import GPUType
    from sim_types import ProvisioningResult


def test_result_to_frame_uses_standard_provisioning_columns() -> None:
    result = ProvisioningResult(
        latencies=[20.123],
        costs=[4.567],
        energies=[100.789],
        ttffs=[10.234],
        tbfs=[0.345],
        actual_provision=[{GPUType.A100: 8, GPUType.H100: 16}],
        config_provision=[{GPUType.A100: 8, GPUType.H100: 16}],
        model_provision=[{}],
    )

    frame = result_to_frame(result)

    assert list(frame.columns) == CSV_COLUMNS
    assert frame.to_dict(orient="records") == [{
        "num_a100": 8,
        "num_h100": 16,
        "num_h200": 0,
        "num_gb200": 0,
        "ttff_s": 10.23,
        "tbf_s": 0.34,
        "cost": 4.57,
        "total_time": 20.12,
        "energy": 100.79,
    }]


def test_combine_checkpoints_preserves_batch_order(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    first_path = get_batch_path(checkpoint_dir, 0, 2)
    second_path = get_batch_path(checkpoint_dir, 2, 4)
    output_path = tmp_path / "combined.csv"

    pd.DataFrame([
        {column: index for column in CSV_COLUMNS}
        for index in [0, 1]
    ]).to_csv(first_path, index=False)
    pd.DataFrame([
        {column: index for column in CSV_COLUMNS}
        for index in [2, 3]
    ]).to_csv(second_path, index=False)

    combine_checkpoints([first_path, second_path], output_path)

    combined = pd.read_csv(output_path)
    assert combined["num_a100"].tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("batch_size,max_batches", [(0, None), (1, 0)])
def test_generate_naive_combo_rejects_invalid_batch_limits(
    tmp_path: Path,
    batch_size: int,
    max_batches: int | None,
) -> None:
    with pytest.raises(ValueError):
        generate_naive_combo(
            output_path=tmp_path / "output.csv",
            checkpoint_dir=tmp_path / "checkpoints",
            batch_size=batch_size,
            max_workers=1,
            timeout=1.0,
            max_batches=max_batches,
        )
