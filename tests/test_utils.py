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
