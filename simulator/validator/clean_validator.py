"""
Clean validator for external simulators — no dependency on the ground truth simulator.

This module provides:
1. Loading of pre-computed ground truth from JSON.
2. Validation of external simulator outputs against the ground truth.

This file is self-contained and can be copied into any project that needs to
validate its simulator implementation against the StreamWise ground truth.

Usage:
    from clean_validator import load_ground_truth, validate, validate_all_passed

    ground_truth = load_ground_truth("path/to/validator_ground_truth.json")
    results = validate(my_simulator_outputs, ground_truth)
    for r in results:
        if not r.passed:
            print(f"FAIL: {r.scenario_name}")
            for err in r.errors:
                print(f"  {err}")
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


RELATIVE_TOLERANCE = 0.01  # 1% relative error

_DEFAULT_GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "validator_ground_truth.json"


@dataclass
class ValidationResult:
    """Result of validating one scenario."""
    scenario_name: str
    passed: bool
    errors: list[str]


def load_ground_truth(
    path: Optional[str | Path] = None,
) -> dict[str, dict]:
    """
    Load previously generated ground truth from JSON.

    Args:
        path: Path to the ground truth JSON file.
              Defaults to validator_ground_truth.json in the same directory.

    Returns:
        Dictionary mapping scenario name to scenario data (allocations, num_gpus, expected).
    """
    if path is None:
        path = _DEFAULT_GROUND_TRUTH_PATH
    with open(path, "r") as f:
        return json.load(f)


def _check_relative_error(
    actual: float,
    expected: float,
    metric_name: str,
    tolerance: float = RELATIVE_TOLERANCE,
) -> Optional[str]:
    """Check if actual is within tolerance of expected. Returns error message or None."""
    if expected == 0.0:
        if actual == 0.0:
            return None
        return (
            f"{metric_name}: expected 0.0, got {actual:.6f}"
        )
    relative_error = abs(actual - expected) / abs(expected)
    if relative_error > tolerance:
        return (
            f"{metric_name}: expected {expected:.6f}, got {actual:.6f} "
            f"(relative error: {relative_error:.4%}, tolerance: {tolerance:.2%})"
        )
    return None


def validate(
    simulator_outputs: dict[str, dict[str, float]],
    ground_truth: Optional[dict[str, dict]] = None,
) -> list[ValidationResult]:
    """
    Validate external simulator outputs against ground truth.

    Args:
        simulator_outputs: Dict mapping scenario name to metrics dict.
            Each metrics dict must have keys: "cost", "ttff_s", "tbf_s", "total_time_s".
        ground_truth: Ground truth dict (loaded from JSON). If None, loads from default path.

    Returns:
        List of ValidationResult for each scenario in the ground truth.
    """
    if ground_truth is None:
        ground_truth = load_ground_truth()

    results: list[ValidationResult] = []

    for scenario_name, gt_data in ground_truth.items():
        if scenario_name not in simulator_outputs:
            results.append(ValidationResult(
                scenario_name=scenario_name,
                passed=False,
                errors=[f"Scenario '{scenario_name}' not found in simulator outputs"],
            ))
            continue

        output = simulator_outputs[scenario_name]
        expected = gt_data["expected"]
        errors: list[str] = []

        for metric in ("cost", "ttff_s", "tbf_s", "total_time_s"):
            if metric not in output:
                errors.append(f"Missing metric: {metric}")
                continue
            error = _check_relative_error(
                actual=output[metric],
                expected=expected[metric],
                metric_name=metric,
            )
            if error is not None:
                errors.append(error)

        results.append(ValidationResult(
            scenario_name=scenario_name,
            passed=len(errors) == 0,
            errors=errors,
        ))

    return results


def validate_all_passed(
    simulator_outputs: dict[str, dict[str, float]],
    ground_truth: Optional[dict[str, dict]] = None,
) -> bool:
    """Convenience function: returns True if all scenarios pass validation."""
    results = validate(simulator_outputs, ground_truth)
    return all(r.passed for r in results)


def get_scenario_allocations(
    ground_truth: Optional[dict[str, dict]] = None,
) -> dict[str, dict]:
    """
    Extract allocations and num_gpus from the ground truth for each scenario.

    This is the input specification your simulator needs to process.

    Returns:
        Dict mapping scenario name to {"allocations": [...], "num_gpus": {...}}.
    """
    if ground_truth is None:
        ground_truth = load_ground_truth()
    return {
        name: {"allocations": data["allocations"], "num_gpus": data["num_gpus"]}
        for name, data in ground_truth.items()
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate external simulator results against ground truth"
    )
    parser.add_argument(
        "results_json",
        help="Path to JSON file with simulator results to validate",
    )
    parser.add_argument(
        "--ground-truth", type=str, default=None,
        help="Path to ground truth JSON (default: validator_ground_truth.json)",
    )
    args = parser.parse_args()

    gt_path = Path(args.ground_truth) if args.ground_truth else None
    gt = load_ground_truth(gt_path)

    with open(args.results_json, "r") as f:
        external_results = json.load(f)

    results = validate(external_results, gt)
    all_passed = True
    for r in results:
        if r.passed:
            print(f"  PASS: {r.scenario_name}")
        else:
            all_passed = False
            print(f"  FAIL: {r.scenario_name}")
            for err in r.errors:
                print(f"        {err}")

    print(f"\n{'ALL PASSED' if all_passed else 'SOME FAILED'} ({sum(r.passed for r in results)}/{len(results)})")
    sys.exit(0 if all_passed else 1)
