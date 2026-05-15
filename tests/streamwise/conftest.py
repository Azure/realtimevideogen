"""
Shared fixtures for streamwise tests.
"""
import sys
from typing import Generator

import pytest


@pytest.fixture(autouse=True)
def _ensure_simulator_path() -> Generator[None, None, None]:
    """Keep simulator/ on sys.path during each test for lazy policy imports."""
    added = False
    if "simulator" not in sys.path:
        sys.path.insert(0, "simulator")
        added = True
    yield
    if added and "simulator" in sys.path:
        sys.path.remove("simulator")
