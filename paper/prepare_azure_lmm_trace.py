"""Aggregate the Azure LMM inference trace 2025 into per-minute request counts.

Source: https://github.com/Azure/AzurePublicDataset/blob/master/AzureLMMInferenceDataset2025.md
(ModServe, SoCC'25; CC-BY license). The raw trace (1M requests, Oct 15-22 2024) is not committed;
only the derived per-minute arrival counts are.
"""
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


TRACE_URL = (
    "https://github.com/Azure/AzurePublicDataset/raw/master/data/"
    "AzureLMMInferenceTrace_multimodal.csv.gz"
)
HEADER = (
    "# Per-minute request arrivals of the Azure LMM Inference Trace 2025 (ModServe, SoCC'25).\n"
    "# Source: " + TRACE_URL + "\n"
    "# License: CC-BY (https://github.com/Azure/AzurePublicDataset/blob/master/LICENSE)\n"
)


def aggregate_per_minute(trace: pd.DataFrame) -> pd.DataFrame:
    """Count request arrivals per minute, including empty minutes."""
    timestamps = pd.to_datetime(trace["TIMESTAMP"], utc=True)
    counts = timestamps.dt.floor("min").value_counts().sort_index()
    full_index = pd.date_range(counts.index.min(), counts.index.max(), freq="min")
    counts = counts.reindex(full_index, fill_value=0)
    return pd.DataFrame({
        "minute": range(len(counts)),
        "timestamp": counts.index.strftime("%Y-%m-%dT%H:%MZ"),
        "requests": counts.to_numpy(),
    })


def prepare_trace(source: str, output_path: Path) -> pd.DataFrame:
    trace = pd.read_csv(source, usecols=["TIMESTAMP"])
    per_minute = aggregate_per_minute(trace)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(HEADER)
        per_minute.to_csv(output_file, index=False)
    return per_minute


def parse_args() -> ArgumentParser:
    paper_dir = Path(__file__).resolve().parent
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=TRACE_URL, help="Trace CSV(.gz) path or URL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=paper_dir / "data" / "azure_lmm_trace_2024_per_minute.csv",
        help="Output per-minute CSV path.",
    )
    return parser


if __name__ == "__main__":
    args = parse_args().parse_args()
    result = prepare_trace(args.source, args.output)
    print(f"Wrote {len(result)} minutes, {result['requests'].sum()} requests to {args.output}")
