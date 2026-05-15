"""
Shared fixtures for streamwise tests.
"""
import sys

import pytest


@pytest.fixture(autouse=True)
def _ensure_simulator_path() -> None:  # type: ignore[misc]
    """Keep simulator/ on sys.path during each test for lazy policy imports."""
    added = False
    if "simulator" not in sys.path:
        sys.path.insert(0, "simulator")
        added = True
    yield  # type: ignore[misc]
    if added and "simulator" in sys.path:
        sys.path.remove("simulator")
