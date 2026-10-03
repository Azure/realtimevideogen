from __future__ import annotations

import os

from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import pandas as pd

from data_loading import load_latency_data
from data_loading import load_power_data
from policies import BASELINE_POLICIES
from provisioning import get_provisioning_results
from provisioning import get_provisions
from sim_types import GPUType
from sim_types import Provision
from sim_types import ProvisioningResult
from workflows import PODCAST_WORKFLOW


CSV_COLUMNS = [
    "num_a100",
    "num_h100",
    "num_h200",
    "num_gb200",
    "ttff_s",
    "tbf_s",
    "cost",
    "total_time",
    "energy",
]


def result_to_frame(result: ProvisioningResult) -> pd.DataFrame:
    frame = pd.DataFrame({
        "num_a100": [provision.get(GPUType.A100, 0) for provision in result.actual_provision],
        "num_h100": [provision.get(GPUType.H100, 0) for provision in result.actual_provision],
        "num_h200": [provision.get(GPUType.H200, 0) for provision in result.actual_provision],
        "num_gb200": [provision.get(GPUType.GB200, 0) for provision in result.actual_provision],
        "ttff_s": result.ttffs,
        "tbf_s": result.tbfs,
        "cost": result.costs,
        "total_time": result.latencies,
        "energy": result.energies,
    })
    frame[["ttff_s", "tbf_s", "cost", "total_time", "energy"]] = (
        frame[["ttff_s", "tbf_s", "cost", "total_time", "energy"]].round(2)
    )
    return frame.loc[:, CSV_COLUMNS]


def get_batch_path(checkpoint_dir: Path, start: int, end: int) -> Path:
    return checkpoint_dir / f"batch_{start:05d}_{end:05d}.csv"


def write_frame_atomically(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)


def combine_checkpoints(
    checkpoint_paths: list[Path],
    output_path: Path,
) -> None:
    frames = [pd.read_csv(path) for path in checkpoint_paths]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CSV_COLUMNS)
    write_frame_atomically(combined, output_path)


def generate_naive_combo(
    output_path: Path,
    checkpoint_dir: Path,
    batch_size: int,
    max_workers: Optional[int],
    timeout: float,
    max_batches: Optional[int] = None,
) -> None:
    if batch_size <= 0:
        raise ValueError("Batch size must be greater than zero")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("Maximum batches must be greater than zero")

    simulator_dir = Path(__file__).resolve().parent
    data_dir = f"{simulator_dir / 'data'}{os.sep}"
    latency_data = load_latency_data(data_dir=data_dir)
    power_data = load_power_data(data_dir=data_dir)
    policy = BASELINE_POLICIES["naive combo"]
    provisions = get_provisions(policy.hardware)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths: list[Path] = []
    generated_batches = 0
    total = len(provisions)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        checkpoint_path = get_batch_path(checkpoint_dir, start, end)
        checkpoint_paths.append(checkpoint_path)
        if checkpoint_path.is_file():
            print(f"Skipping completed batch {start}:{end} ({end}/{total})")
            continue

        batch: list[Provision] = provisions[start:end]
        result = get_provisioning_results(
            workflow=PODCAST_WORKFLOW,
            latency_data=latency_data,
            power_data=power_data,
            policy=policy,
            provisions=batch,
            verbose=False,
            max_workers=max_workers,
            timeout=timeout,
        )
        write_frame_atomically(result_to_frame(result), checkpoint_path)
        print(f"Checkpointed batch {start}:{end} ({end}/{total})")
        generated_batches += 1
        if max_batches is not None and generated_batches >= max_batches:
            print("Stopped after the requested number of new batches; rerun the same command to resume.")
            return

    combine_checkpoints(checkpoint_paths, output_path)
    print(f"Saved {output_path} from {len(checkpoint_paths)} checkpoint batches")


def parse_args() -> ArgumentParser:
    repository_root = Path(__file__).resolve().parents[1]
    output_path = repository_root / "paper" / "data" / "provisioning_naive_combo.csv"

    parser = ArgumentParser(description="Generate resumable Naive Combo provisioning data.")
    parser.add_argument("--output", type=Path, default=output_path, help="Final output CSV path.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=output_path.parent / f"{output_path.stem}.checkpoints",
        help="Directory containing atomic per-batch checkpoint CSVs.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Provisions saved per checkpoint batch.")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Stop after this many new batches without promoting a final CSV; useful for resume testing.",
    )
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum worker processes per batch.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Timeout in seconds for each provision.")
    return parser


if __name__ == "__main__":
    args = parse_args().parse_args()
    generate_naive_combo(
        output_path=args.output,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        timeout=args.timeout,
        max_batches=args.max_batches,
    )
