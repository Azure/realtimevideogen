"""
Simulator package.

The core allocation logic lives in ``streamwise.model_provisioner``.
This package adds provisioning sweeps, multi-request analysis, and plotting
on top of that shared foundation.
"""
import os
import sys

# Make model_provisioner importable for simulator modules and child processes.
_STREAMWISE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "streamwise")
_STREAMWISE_DIR = os.path.normpath(_STREAMWISE_DIR)
if _STREAMWISE_DIR not in sys.path:
    sys.path.insert(0, _STREAMWISE_DIR)
