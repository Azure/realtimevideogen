"""
Model Provisioner — allocation policy implementations for GPU resource distribution.

Contains greedy, naive, MILP, HexGen, and Helix allocation strategies.
The foundation types (sim_types, constants, models, etc.) live in simulator/.
"""
import os
import sys

# Add simulator/ to sys.path so policy files can import foundation modules.
_SIMULATOR_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "simulator")
)
if _SIMULATOR_DIR not in sys.path:
    sys.path.insert(0, _SIMULATOR_DIR)
