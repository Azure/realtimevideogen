"""
Utility functions for tests.
Mainly manage the sys.path temporary modifications.
"""

import sys

from contextlib import contextmanager

from typing import Iterator
from typing import Any


@contextmanager
def temp_sys_path(
    *paths: Any
) -> Iterator[None]:
    """Temporarily add paths to sys.path."""
    old_sys_path = sys.path.copy()
    sys.path[:0] = paths
    try:
        yield
    finally:
        sys.path = old_sys_path


def assert_equals_approx(
    value: float,
    expected: float,
    delta: float = 0.01,
) -> None:
    """Assert that two floats are approximately equal within a tolerance."""
    assert abs(value - expected) < delta, (
        f"Expected {value:.2f} to be approximately equal to {expected:.2f} within tolerance {delta}"
    )


def assert_equal_dict(
    actual: dict[Any, Any],
    expected: dict[Any, Any],
    name: str = "dict",
    delta: float = 0.01,
    _path: str = "",
) -> None:
    """Recursively compare two nested dicts.

    At each level the key sets must match exactly.  Leaf values are compared
    with ``assert_equals_approx`` when they are floats, otherwise with ``==``.
    """
    label = _path or name
    assert set(actual.keys()) == set(expected.keys()), (
        f"{label}: keys differ: {set(actual.keys())} != {set(expected.keys())}"
    )
    for key in expected:
        exp_val = expected[key]
        act_val = actual[key]
        child_path = f"{label}[{key}]"
        if isinstance(exp_val, dict):
            assert_equal_dict(act_val, exp_val, name=name, delta=delta, _path=child_path)
        elif isinstance(exp_val, float):
            assert abs(act_val - exp_val) < delta, (
                f"{child_path}: expected {exp_val:.2f}, got {act_val:.2f} (delta {delta})"
            )
        else:
            assert act_val == exp_val, (
                f"{child_path}: expected {exp_val}, got {act_val}"
            )
