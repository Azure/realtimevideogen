#!/usr/bin/env python3

import sys
import pytest

from typing import Any

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.torch_mock import TorchMock
from tests.diffusers_mock import DiffusersMock

mock_torch = TorchMock()
mock_diffusers = DiffusersMock()

sys.path.append("fluxkrea")
sys.path.append("flux")

mock_modules = {
    'nvidia_smi': MagicMock(),
    'imageio': MagicMock(),
    'cv2': MagicMock(),
    'torch': mock_torch,
    'xfuser': MagicMock(),
    'xfuser.config': MagicMock(),
    'xfuser.core': MagicMock(),
    'xfuser.core.distributed': MagicMock(),
    'xfuser.model_executor': MagicMock(),
    'xfuser.model_executor.models': MagicMock(),
    'xfuser.model_executor.models.transformers.transformer_flux': MagicMock(),
    'xfuser.model_executor.layers': MagicMock(),
    'xfuser.model_executor.layers.attention_processor': MagicMock(),
}
mock_modules.update(mock_torch.get_sub_modules())
mock_modules.update(mock_diffusers.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from fluxkrea.wrapper_fluxkrea import FluxKreaGeneration
    _fluxkrea_module = sys.modules['fluxkrea.wrapper_fluxkrea']


@pytest.mark.asyncio
async def test_wrapper_fluxkrea() -> None:
    model = FluxKreaGeneration()
    assert model is not None
    assert model.model_name == "fluxkrea"
    assert model.status == "initializing"

    with pytest.raises(ValueError, match="Model not initialized"):
        await model.generate(
            width=128,
            height=80,
            prompt="test prompt")

    model.init()
    assert model.status == "ok"

    # Mock pipeline return object
    mock_output = MagicMock()
    mock_output.images = ["image"]
    model.pipeline = MagicMock(return_value=mock_output)
    model.pipeline.vae_scale_factor = 8

    health = model.get_health()
    assert health is not None
    timestamps = model.get_timestamps()
    assert timestamps is not None

    with pytest.raises(ValueError):
        await model.get_rest_args(None)
    with pytest.raises(ValueError):
        await model.get_rest_args({})
    await model.get_rest_args({
        "job_id": "unittest",
        "prompt": "Test prompt",
        "width": 80,
        "height": 60,
    })

    await model.warmup()

    # TODO this should fail
    await model.generate(
        width=17,
        height=13,
        prompt="Test prompt")

    image = await model.generate(
        width=256,
        height=256,
        prompt="Test prompt")
    assert image is not None

    image = await model.generate(
        width=256,
        height=160,
        prompt="Test prompt")
    assert image is not None

    # 15x17 not supported for 2 GPUs.
    model.world_size = 2
    # TODO
    # with pytest.raises(ValueError, msg="15x17 not supported for 2 GPUs."):
    image = await model.generate(
        width=15,
        height=17,
        prompt="Test prompt")

    del model


@pytest.mark.asyncio
async def test_wrapper_fluxkrea_additional_coverage() -> None:
    """Cover seed path, step callbacks, parallelism init, and compile-disabled path."""
    model = FluxKreaGeneration()
    model.init()
    assert model.status == "ok"

    image = await model.generate(
        width=256,
        height=256,
        prompt="Seed coverage test",
        seed=42)
    assert image is not None

    pipeline_instance = model.pipeline

    def _pipeline_with_callback(*args: Any, **kwargs: Any) -> Any:
        n_steps = kwargs.get("num_inference_steps", 2)
        callback = kwargs.get("callback_on_step_end")
        if callback:
            for step in range(n_steps):
                callback(pipeline_instance, step, 0, {})
        out = MagicMock()
        out.images = [MagicMock()]
        return out

    pipeline_instance.side_effect = _pipeline_with_callback
    image = await model.generate(
        width=256,
        height=256,
        prompt="Callback coverage test",
        sampling_steps=2)
    assert image is not None

    model.world_size = 2
    with patch.object(_fluxkrea_module, 'parallelize_transformer'):
        model.init_model_parallelism()

    model.torch_compile = False
    model.model_compile()

    del model


def test_wrapper_fluxkrea_model_compile_no_pipeline() -> None:
    """model_compile() with pipeline=None returns early (pipeline not yet loaded)."""
    model = FluxKreaGeneration()
    assert model.pipeline is None
    model.model_compile()
    del model
