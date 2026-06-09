"""
Model Provisioner — allocation policy implementations for GPU resource distribution.

Contains greedy, naive, MILP, HexGen, and Helix allocation strategies.
The foundation types (sim_types, constants, models, etc.) live in simulator/.
"""
import os
import sys

# Add simulator/ to sys.path so policy files can import foundation modules.
# Supports both local dev layout (../../simulator) and Docker layout (../simulator).
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.path.normpath(os.path.join(_HERE, "..", "..", "simulator")),
    os.path.normpath(os.path.join(_HERE, "..", "simulator")),
]
for _path in _CANDIDATES:
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)
        break
