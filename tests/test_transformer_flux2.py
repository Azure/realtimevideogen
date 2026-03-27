#!/usr/bin/env python3
"""Tests for wrapper/flux2/transformer_flux2.py."""

import sys

from typing import Any
from unittest.mock import patch, MagicMock
from tests.torch_mock import TorchMock
from tests.diffusers_mock import DiffusersMock

mock_torch = TorchMock()
mock_diffusers = DiffusersMock()

sys.path.append("wrapper")
sys.path.append("wrapper/flux2")

# ---------------------------------------------------------------------------
# Stub parent classes
# These replace the diffusers / xfuser base classes so that the real code in
# transformer_flux2.py can be imported, instantiated, and exercised without
# requiring the actual GPU libraries.
# ---------------------------------------------------------------------------


class _FakeFlux2AttnProcessor:
    """Minimal stub for diffusers.Flux2AttnProcessor."""

    def __init__(self) -> None:
        pass


class _FakeFlux2ParallelSelfAttnProcessor:
    """Minimal stub for diffusers.Flux2ParallelSelfAttnProcessor."""

    def __init__(self) -> None:
        pass


class _FakeFlux2Transformer2DModel:
    """Minimal stub for diffusers.Flux2Transformer2DModel."""

    def __init__(self, **kwargs: Any) -> None:
        # Provide two mock blocks in each list so __init__ loops execute.
        self.transformer_blocks = [MagicMock(), MagicMock()]
        self.single_transformer_blocks = [MagicMock(), MagicMock()]

    def forward(
        self,
        hidden_states: Any,
        encoder_hidden_states: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # Return a plain tuple so the wrapper's return_dict logic takes the
        # tuple branch (not the dict branch).
        return (hidden_states,)


class _FakeTransformerOutput:
    """Fake non-tuple transformer output for testing the return_dict=True branch."""

    def __init__(self, sample: Any = None) -> None:
        self._sample = sample

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, slice):
            # Return an empty list so that `*output[1:]` unpacking produces no
            # extra arguments when the code under test calls
            # `output.__class__(sample, *output[1:])` in the return_dict branch.
            return []
        return self._sample


class _FakeXFuserAttentionBaseWrapper:
    """Minimal stub for xfuser xFuserAttentionBaseWrapper."""

    def __init__(self, attention: Any) -> None:
        self.attention = attention

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return MagicMock()


# ---------------------------------------------------------------------------
# Decorator helpers
# The @register(...) decorators run at import time.  We must make them act as
# identity decorators so the class bodies in transformer_flux2.py survive.
# ---------------------------------------------------------------------------


def _identity_register(base_cls: Any) -> Any:
    """Return a decorator that leaves the decorated class unchanged."""
    def decorator(cls: Any) -> Any:
        return cls
    return decorator


# ---------------------------------------------------------------------------
# Mock modules
# ---------------------------------------------------------------------------

_mock_attn_proc_register = MagicMock()
_mock_attn_proc_register.register.side_effect = _identity_register
_mock_attn_proc_register.get_processor.return_value = MagicMock(return_value=MagicMock())

_mock_attn_proc_module = MagicMock()
_mock_attn_proc_module.xFuserAttentionBaseWrapper = _FakeXFuserAttentionBaseWrapper
_mock_attn_proc_module.xFuserAttentionProcessorRegister = _mock_attn_proc_register

_mock_layer_wrappers_register = MagicMock()
_mock_layer_wrappers_register.register.side_effect = _identity_register

_mock_layers_module = MagicMock()
_mock_layers_module.xFuserLayerWrappersRegister = _mock_layer_wrappers_register

_mock_sp_group = MagicMock()
_mock_sp_group.all_gather.side_effect = lambda x, dim: x  # identity

_mock_cfg_group = MagicMock()
_mock_cfg_group.all_gather.side_effect = lambda x, dim: x  # identity

_mock_distributed = MagicMock()
_mock_distributed.get_sequence_parallel_world_size.return_value = 1
_mock_distributed.get_sequence_parallel_rank.return_value = 0
_mock_distributed.get_classifier_free_guidance_world_size.return_value = 1
_mock_distributed.get_classifier_free_guidance_rank.return_value = 0
_mock_distributed.get_sp_group.return_value = _mock_sp_group
_mock_distributed.get_cfg_group.return_value = _mock_cfg_group

_mock_transformer_module = MagicMock()
_mock_transformer_module.Flux2Attention = MagicMock
_mock_transformer_module.Flux2AttnProcessor = _FakeFlux2AttnProcessor
_mock_transformer_module.Flux2Transformer2DModel = _FakeFlux2Transformer2DModel
_mock_transformer_module.Flux2ParallelSelfAttention = MagicMock
_mock_transformer_module.Flux2ParallelSelfAttnProcessor = _FakeFlux2ParallelSelfAttnProcessor
_mock_transformer_module._get_qkv_projections = MagicMock(
    return_value=(
        MagicMock(), MagicMock(), MagicMock(),
        MagicMock(), MagicMock(), MagicMock(),
    )
)

_mock_embeddings = MagicMock()
_mock_embeddings.apply_rotary_emb.side_effect = lambda q, emb, sequence_dim: q  # identity

_mock_usp_module = MagicMock()

# Build module dict from DiffusersMock and override the two entries that need
# custom behaviour for these transformer tests.
_diffusers_sub_modules = mock_diffusers.get_sub_modules()
_diffusers_sub_modules["diffusers.models.transformers.transformer_flux2"] = _mock_transformer_module
_diffusers_sub_modules["diffusers.models.embeddings"] = _mock_embeddings

mock_modules = {
    'torch': mock_torch,
    'xfuser': MagicMock(),
    'xfuser.config': MagicMock(),
    'xfuser.core': MagicMock(),
    'xfuser.core.distributed': _mock_distributed,
    'xfuser.model_executor': MagicMock(),
    'xfuser.model_executor.models': MagicMock(),
    'xfuser.model_executor.models.transformers': MagicMock(),
    'xfuser.model_executor.layers': _mock_layers_module,
    'xfuser.model_executor.layers.attention_processor': _mock_attn_proc_module,
    'xfuser.model_executor.layers.usp': _mock_usp_module,
}
mock_modules.update(mock_torch.get_sub_modules())
mock_modules.update(_diffusers_sub_modules)

with patch.dict(sys.modules, mock_modules):
    from transformer_flux2 import (
        xFuserFlux2AttnProcessor,
        xFuserFlux2ParallelSelfAttnProcessor,
        xFuserFlux2ParallelSelfAttention,
        xFuserFlux2Transformer2DWrapper,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_attn_processor_no_encoder_no_rotary() -> None:
    """xFuserFlux2AttnProcessor: no encoder states, no rotary embedding."""
    processor = xFuserFlux2AttnProcessor()
    assert processor is not None

    attn = MagicMock()
    attn.added_kv_proj_dim = None   # skip encoder branch

    result = processor(
        attn,
        hidden_states=MagicMock(),
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
    )
    assert result is not None


def test_attn_processor_with_encoder_and_rotary() -> None:
    """xFuserFlux2AttnProcessor: encoder states + rotary embedding present."""
    processor = xFuserFlux2AttnProcessor()

    attn = MagicMock()
    attn.added_kv_proj_dim = 64   # enter encoder branch

    encoder_hidden_states = MagicMock()
    encoder_hidden_states.shape = [1, 10, 64]  # list — supports indexing

    result = processor(
        attn,
        hidden_states=MagicMock(),
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=None,
        image_rotary_emb=MagicMock(),
    )
    assert result is not None


def test_parallel_self_attn_processor_no_rotary() -> None:
    """xFuserFlux2ParallelSelfAttnProcessor.__call__ without rotary embedding."""
    processor = xFuserFlux2ParallelSelfAttnProcessor()
    assert processor is not None

    attn = MagicMock()
    hidden_states = MagicMock()

    # torch.split must return an iterable of exactly 2 elements.
    qkv_mock = MagicMock()
    qkv_mock.chunk.return_value = [MagicMock(), MagicMock(), MagicMock()]
    mock_torch.split = MagicMock(return_value=(qkv_mock, MagicMock()))

    result = processor(attn, hidden_states, attention_mask=None, image_rotary_emb=None)
    assert result is not None


def test_parallel_self_attn_processor_with_rotary() -> None:
    """xFuserFlux2ParallelSelfAttnProcessor.__call__ with rotary embedding."""
    processor = xFuserFlux2ParallelSelfAttnProcessor()

    attn = MagicMock()
    hidden_states = MagicMock()

    qkv_mock = MagicMock()
    qkv_mock.chunk.return_value = [MagicMock(), MagicMock(), MagicMock()]
    mock_torch.split = MagicMock(return_value=(qkv_mock, MagicMock()))

    result = processor(
        attn,
        hidden_states,
        attention_mask=None,
        image_rotary_emb=MagicMock(),
    )
    assert result is not None


def test_parallel_self_attention_init_and_forward() -> None:
    """xFuserFlux2ParallelSelfAttention: init sets processor; forward delegates."""
    mock_attention = MagicMock()
    wrapper = xFuserFlux2ParallelSelfAttention(mock_attention)

    assert wrapper is not None
    assert wrapper.attention is mock_attention

    hidden_states = MagicMock()
    result = wrapper.forward(hidden_states, attention_mask=None, image_rotary_emb=None)
    assert result is not None


def test_transformer_wrapper_init() -> None:
    """xFuserFlux2Transformer2DWrapper.__init__ wires processors onto each block."""
    wrapper = xFuserFlux2Transformer2DWrapper()
    assert wrapper is not None

    # Both block lists are populated by _FakeFlux2Transformer2DModel (2 each).
    for block in wrapper.transformer_blocks:
        assert isinstance(block.attn.processor, xFuserFlux2AttnProcessor)
    for block in wrapper.single_transformer_blocks:
        assert isinstance(block.attn.processor, xFuserFlux2ParallelSelfAttnProcessor)


def test_pad_to_sp_divisible() -> None:
    """_pad_to_sp_divisible appends zeros along the specified dimension."""
    wrapper = xFuserFlux2Transformer2DWrapper()

    tensor = MagicMock()
    tensor.shape = [2, 5, 16]
    tensor.dtype = MagicMock()
    tensor.device = MagicMock()

    result = wrapper._pad_to_sp_divisible(tensor, padding_length=3, dim=1)
    assert result is not None


def test_transformer_wrapper_forward() -> None:
    """xFuserFlux2Transformer2DWrapper.forward runs end-to-end with sp_world_size=1."""
    wrapper = xFuserFlux2Transformer2DWrapper()

    # Use a list for shape so integer indexing returns real ints.
    hidden_states = MagicMock()
    hidden_states.shape = [1, 4, 64]  # sequence_length=4, divisible by sp_world_size=1

    encoder_hidden_states = MagicMock()
    img_ids = MagicMock()
    txt_ids = MagicMock()
    timestep = MagicMock()  # not a torch.Tensor instance → skips CFG-chunk branch

    result = wrapper.forward(
        hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
    )
    assert result is not None


def test_transformer_wrapper_forward_with_padding() -> None:
    """forward() pads hidden_states / img_ids when sequence length % sp_world_size != 0."""
    wrapper = xFuserFlux2Transformer2DWrapper()

    # sp_world_size=2, sequence_length=3 → padding_length=1, triggering the
    # padding path: hidden_states and img_ids are padded to a length divisible
    # by sp_world_size, and the extra padding tokens are stripped from the
    # gathered output before returning.
    _mock_distributed.get_sequence_parallel_world_size.return_value = 2
    try:
        hidden_states = MagicMock()
        hidden_states.shape = [1, 3, 64]  # seq_len=3 is not divisible by 2

        result = wrapper.forward(
            hidden_states,
            encoder_hidden_states=MagicMock(),
            timestep=MagicMock(),
            img_ids=MagicMock(),
            txt_ids=MagicMock(),
        )
        assert result is not None
    finally:
        _mock_distributed.get_sequence_parallel_world_size.return_value = 1


def test_transformer_wrapper_forward_timestep_tensor() -> None:
    """forward() chunks the timestep tensor when it is a real Tensor with ndim > 0."""
    wrapper = xFuserFlux2Transformer2DWrapper()

    # Build a timestep that satisfies isinstance(timestep, torch.Tensor) check.
    timestep = mock_torch.Tensor()
    timestep.ndim = 1
    timestep.shape = [1]  # shape[0]=1 matches hidden_states.shape[0]=1

    hidden_states = MagicMock()
    hidden_states.shape = [1, 4, 64]

    result = wrapper.forward(
        hidden_states,
        encoder_hidden_states=MagicMock(),
        timestep=timestep,
        img_ids=MagicMock(),
        txt_ids=MagicMock(),
    )
    assert result is not None


def test_transformer_wrapper_forward_return_dict() -> None:
    """forward() returns output.__class__(sample, ...) when output is not a tuple."""
    wrapper = xFuserFlux2Transformer2DWrapper()

    hidden_states = MagicMock()
    hidden_states.shape = [1, 4, 64]

    # Patch super().forward to return a _FakeTransformerOutput (not a tuple)
    # → return_dict=True, which exercises line 298.
    non_tuple_output = _FakeTransformerOutput(sample=MagicMock())
    with patch.object(_FakeFlux2Transformer2DModel, "forward", return_value=non_tuple_output):
        result = wrapper.forward(
            hidden_states,
            encoder_hidden_states=MagicMock(),
            timestep=MagicMock(),
            img_ids=MagicMock(),
            txt_ids=MagicMock(),
        )
    assert result is not None
