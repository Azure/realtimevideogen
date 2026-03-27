#!/usr/bin/env python3

import sys
import pytest

from typing import Any

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.torch_mock import TorchMock
from tests.diffusers_mock import DiffusersMock

from PIL import Image

mock_torch = TorchMock()
mock_diffusers = DiffusersMock()

sys.path.append("wrapper")
sys.path.append("wrapper/flux2")
sys.path.append("wrapper/flux")

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
    'xfuser.model_executor.models.transformers.transformer_flux2': MagicMock(),
    'xfuser.model_executor.layers': MagicMock(),
    'xfuser.model_executor.layers.attention_processor': MagicMock(),
}
mock_modules.update(mock_torch.get_sub_modules())
mock_modules.update(mock_diffusers.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from flux2.wrapper_flux2 import Flux2Generation


@pytest.mark.asyncio
async def test_wrapper_flux2() -> None:
    model = Flux2Generation()
    assert model is not None
    assert model.model_name == "flux2"
    assert model.status == "initializing"

    with pytest.raises(ValueError, match="Model not initialized"):
        await model.generate(
            width=128,
            height=80,
            prompt="test prompt")

    # Capture the mock Flux2Pipeline and transformer wrapper so we can assert
    # the new sharding behaviour: device_map="balanced" on both the transformer
    # and the pipeline (so VAE and text encoders are distributed too), and
    # pipeline.to() is never called.
    mock_pipeline_cls = mock_modules['diffusers'].Flux2Pipeline
    mock_transformer_cls = mock_modules[
        'xfuser.model_executor.models.transformers.transformer_flux2'
    ].xFuserFlux2Transformer2DWrapper

    # Pre-access mock sub-component attributes so we hold stable references for
    # assertions after init() calls .to() on each of them.
    mock_pipeline_instance = mock_pipeline_cls.from_pretrained.return_value

    model.init()
    assert model.status == "ok"

    # Verify transformer was loaded with device_map="balanced"
    _, transformer_kwargs = mock_transformer_cls.from_pretrained.call_args
    assert transformer_kwargs.get("device_map") == "balanced", (
        "Transformer must be loaded with device_map='balanced' to shard across GPUs"
    )

    # Verify the pipeline was also loaded with device_map="balanced" so that
    # VAE and text encoders are distributed rather than crammed onto one GPU.
    _, pipeline_kwargs = mock_pipeline_cls.from_pretrained.call_args
    assert pipeline_kwargs.get("device_map") == "balanced", (
        "Pipeline must be loaded with device_map='balanced' to distribute VAE and text encoders"
    )

    # Verify the full pipeline was NOT moved to a single device (would cause OOM)
    mock_pipeline_instance.to.assert_not_called()

    health = model.get_health()
    assert health is not None
    timestamps = model.get_timestamps()
    assert timestamps is not None

    with pytest.raises(ValueError, match="Missing JSON body"):
        await model.get_rest_args(None)
    with pytest.raises(ValueError, match="Missing 'prompt' parameter"):
        await model.get_rest_args({})
    args = await model.get_rest_args({
        "job_id": "unittest",
        "prompt": "Test prompt",
        "width": 80,
        "height": 60,
    })
    assert "args" in args

    await model.warmup()

    image = await model.generate(
        width=256,
        height=320,
        prompt="Test prompt")
    assert image is not None
    assert isinstance(image, Image.Image)
    assert image.size == (256, 320)

    image = await model.generate(
        width=480,
        height=320,
        prompt="Test prompt")
    assert image is not None
    assert isinstance(image, Image.Image)
    assert image.size == (480, 320)

    # 48x48 not supported for 2 GPUs (latent shape 9, odd).
    model.world_size = 2
    with pytest.raises(ValueError, match="48x48 not supported for 2 GPUs"):
        await model.generate(
            width=48,
            height=48,
            prompt="Test prompt")

    del model


@pytest.mark.asyncio
async def test_wrapper_flux2_additional_coverage() -> None:
    """Cover lines missed by test_wrapper_flux2: seed path, callbacks, parallelism, compile."""
    model = Flux2Generation()
    model.init()
    assert model.status == "ok"

    # Cover line 138: generate() with an explicit seed >= 0 triggers set_seed().
    image = await model.generate(
        width=256,
        height=320,
        prompt="Seed coverage test",
        seed=42)
    assert isinstance(image, Image.Image)

    # Cover callback_gen_timer body: replace the pipeline side-effect so it
    # invokes callback_on_step_end each step.
    pipeline_instance = model.pipeline

    def _pipeline_with_callback(*args: Any, **kwargs: Any) -> Any:
        n_steps = kwargs.get("num_inference_steps", 2)
        callback = kwargs.get("callback_on_step_end")
        if callback:
            for step in range(n_steps):
                callback(pipeline_instance, step, 0, {})
        out = MagicMock()
        out.images = [Image.new("RGB", (kwargs.get("width", 64), kwargs.get("height", 64)))]
        return out

    pipeline_instance.side_effect = _pipeline_with_callback
    image = await model.generate(
        width=256,
        height=320,
        prompt="Callback coverage test",
        sampling_steps=2)
    assert isinstance(image, Image.Image)

    # Cover lines 82-89: init_model_parallelism() body when world_size > 1.
    # dist.is_initialized() returns a truthy MagicMock from the mock setup.
    model.world_size = 2
    model.init_model_parallelism()

    # Cover line 94: model_compile() returns early when torch_compile is False.
    model.torch_compile = False
    model.model_compile()

    # Cover line 96: model_compile() returns early when pipeline is None.
    model2 = Flux2Generation()
    assert model2.pipeline is None
    model2.model_compile()  # torch_compile=True (default), pipeline=None → returns at line 96

    del model
    del model2
