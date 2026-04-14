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
