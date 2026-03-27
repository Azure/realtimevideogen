#!/usr/bin/env python3
"""
Tests for wrapper/hunyuanavatar/config.py and encode_data.py
"""

import sys
import argparse
import pytest

from unittest.mock import patch, MagicMock
from tests.torch_mock import TorchMock

mock_torch = TorchMock()

# Build mock modules for hymm_sp.constants used by config.py
_mock_constants = MagicMock()
_mock_constants.TEXT_ENCODER_PATH = {"llava-llama-3-8b": "/path/llava", "clipL": "/path/clip"}
_mock_constants.TOKENIZER_PATH = {"llava-llama-3-8b": "/path/tokenizer", "clipL": "/path/clip_tok"}
_mock_constants.PROMPT_TEMPLATE = ["li-dit-encode-video"]
_mock_constants.TEXT_PROJECTION = ["single_refiner"]
_mock_constants.PRECISIONS = ["fp32", "fp16", "bf16"]

mock_modules = {
    'nvidia_smi': MagicMock(),
    'torch': mock_torch,
    'torchvision': MagicMock(),
    'torchvision.transforms': MagicMock(),
    'torchvision.transforms.functional': MagicMock(),
    'transformers': MagicMock(),
    'einops': MagicMock(),
    'hymm_sp': MagicMock(),
    'hymm_sp.constants': _mock_constants,
    'hymm_sp.config': MagicMock(),
}
mock_modules.update(mock_torch.get_sub_modules())

sys.path.append("wrapper")
sys.path.append("wrapper/hunyuanavatar")

with patch.dict(sys.modules, mock_modules):
    from hunyuanavatar.config import as_tuple
    from hunyuanavatar.config import parse_args
    from hunyuanavatar.config import sanity_check_args
    from hunyuanavatar.config import add_extra_args
    from hunyuanavatar.config import add_network_args
    from hunyuanavatar.config import add_extra_models_args
    from hunyuanavatar.config import add_denoise_schedule_args
    from hunyuanavatar.config import add_evaluation_args


# ── as_tuple ──────────────────────────────────────────────────────────────────

def test_as_tuple_with_list() -> None:
    assert as_tuple([1, 2, 3]) == (1, 2, 3)


def test_as_tuple_with_tuple() -> None:
    assert as_tuple((4, 5)) == (4, 5)


def test_as_tuple_with_int() -> None:
    assert as_tuple(7) == (7,)


def test_as_tuple_with_float() -> None:
    assert as_tuple(3.14) == (3.14,)


def test_as_tuple_with_string() -> None:
    assert as_tuple("hello") == ("hello",)


def test_as_tuple_with_none() -> None:
    assert as_tuple(None) == (None,)


def test_as_tuple_with_unknown_type() -> None:
    class _Obj:
        pass
    with pytest.raises(ValueError, match="Unknown type"):
        as_tuple(_Obj())


# ── parse_args ────────────────────────────────────────────────────────────────

def test_parse_args_returns_namespace() -> None:
    args = parse_args()
    assert isinstance(args, argparse.Namespace)


def test_parse_args_defaults() -> None:
    args = parse_args()
    assert args.vae == "884-16c-hy0801"
    assert args.latent_channels == 16
    assert args.rope_theta == 256
    assert args.flow_solver == "euler"


def test_parse_args_with_namespace() -> None:
    ns = argparse.Namespace()
    args = parse_args(namespace=ns)
    assert isinstance(args, argparse.Namespace)


# ── sanity_check_args ─────────────────────────────────────────────────────────

def test_sanity_check_args_valid() -> None:
    args = argparse.Namespace(vae="884-16c-hy0801", latent_channels=None)
    result = sanity_check_args(args)
    assert result.latent_channels == 16


def test_sanity_check_args_latent_channels_mismatch() -> None:
    args = argparse.Namespace(vae="884-16c-hy0801", latent_channels=8)
    with pytest.raises(ValueError, match="Latent.*must match VAE"):
        sanity_check_args(args)


def test_sanity_check_args_invalid_vae_format() -> None:
    args = argparse.Namespace(vae="invalid-vae", latent_channels=None)
    with pytest.raises(ValueError, match="Invalid VAE model"):
        sanity_check_args(args)


def test_sanity_check_args_latent_matches_vae() -> None:
    args = argparse.Namespace(vae="884-16c-hy0801", latent_channels=16)
    result = sanity_check_args(args)
    assert result.latent_channels == 16


# ── add_*_args helpers ────────────────────────────────────────────────────────

def test_add_network_args() -> None:
    parser = argparse.ArgumentParser()
    result = add_network_args(parser)
    assert result is parser


def test_add_extra_models_args() -> None:
    parser = argparse.ArgumentParser()
    result = add_extra_models_args(parser)
    assert result is parser


def test_add_denoise_schedule_args() -> None:
    parser = argparse.ArgumentParser()
    result = add_denoise_schedule_args(parser)
    assert result is parser


def test_add_evaluation_args() -> None:
    parser = argparse.ArgumentParser()
    result = add_evaluation_args(parser)
    assert result is parser


def test_add_extra_args() -> None:
    parser = argparse.ArgumentParser()
    result = add_extra_args(parser)
    assert result is parser
