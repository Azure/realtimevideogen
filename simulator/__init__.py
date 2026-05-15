"""
Simulator package — provisioning sweeps, multi-request analysis, and plotting
on top of the model_provisioner allocation policies.

The allocation policy implementations live in ``streamwise/model_provisioner/``.
"""
import os
import sys

# Make model_provisioner importable for simulator modules.
_STREAMWISE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "streamwise")
)
if _STREAMWISE_DIR not in sys.path:
    sys.path.insert(0, _STREAMWISE_DIR)
