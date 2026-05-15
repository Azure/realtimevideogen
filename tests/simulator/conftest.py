"""
Conftest for simulator tests.

Sets PYTHONPATH so that child processes spawned by ProcessPoolExecutor
can find the simulator and streamwise modules.
"""
import os
import sys

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SIMULATOR_DIR = os.path.join(_REPO_ROOT, "simulator")
_STREAMWISE_DIR = os.path.join(_REPO_ROOT, "streamwise")

# Propagate paths to child processes via PYTHONPATH.
_EXTRA = os.pathsep.join((_REPO_ROOT, _SIMULATOR_DIR, _STREAMWISE_DIR))
_EXISTING = os.environ.get("PYTHONPATH", "")
if _SIMULATOR_DIR not in _EXISTING:
    os.environ["PYTHONPATH"] = (
        _EXTRA + os.pathsep + _EXISTING if _EXISTING else _EXTRA
    )

for _p in (_REPO_ROOT, _SIMULATOR_DIR, _STREAMWISE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
