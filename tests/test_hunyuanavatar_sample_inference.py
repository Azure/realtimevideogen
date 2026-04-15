#!/usr/bin/env python3
"""
Tests for wrapper/hunyuanavatar/sample_inference_audio.py
"""

import sys

from unittest.mock import patch, MagicMock
from tests.torch_mock import TorchMock

mock_torch = TorchMock()

mock_modules = {
    'nvidia_smi': MagicMock(),
    'torch': mock_torch,
    'torchvision': MagicMock(),
    'torchvision.transforms': MagicMock(),
    'loguru': MagicMock(),
    'einops': MagicMock(),
    'hymm_sp': MagicMock(),
    'hymm_sp.diffusion': MagicMock(),
    'hymm_sp.helpers': MagicMock(),
    'hymm_sp.inference': MagicMock(),
    'hymm_sp.data_kits': MagicMock(),
    'hymm_sp.data_kits.audio_preprocessor': MagicMock(),
}
mock_modules.update(mock_torch.get_sub_modules())

sys.path.append("wrapper")
sys.path.append("wrapper/hunyuanavatar")

with patch.dict(sys.modules, mock_modules):
    from hunyuanavatar.sample_inference_audio import align_to


def test_align_to_already_aligned() -> None:
    """align_to should return value unchanged when already divisible."""
    assert align_to(16, 16) == 16
    assert align_to(32, 8) == 32
    assert align_to(100, 10) == 100


def test_align_to_rounds_up() -> None:
    """align_to rounds up to the next multiple of alignment."""
    assert align_to(17, 16) == 32
    assert align_to(1, 16) == 16
    assert align_to(9, 8) == 16


def test_align_to_float_input() -> None:
    """align_to handles float inputs."""
    assert align_to(16.0, 16) == 16
    assert align_to(15.5, 8) == 16


def test_align_to_zero() -> None:
    """align_to with zero returns zero."""
    assert align_to(0, 16) == 0
