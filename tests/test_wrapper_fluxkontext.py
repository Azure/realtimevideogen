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
sys.path.append("wrapper/fluxkontext")
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
    'xfuser.model_executor.layers': MagicMock(),
    'xfuser.model_executor.layers.attention_processor': MagicMock(),
}
mock_modules.update(mock_torch.get_sub_modules())
mock_modules.update(mock_diffusers.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from image_utils import img_to_base64
    from fluxkontext.wrapper_fluxkontext import FluxKontextGeneration
    _fluxkontext_module = sys.modules['fluxkontext.wrapper_fluxkontext']


@pytest.mark.asyncio
async def test_wrapper_fluxkontext() -> None:
    model = FluxKontextGeneration()
    assert model is not None
    assert model.model_name == "fluxkontext"
    assert model.status == "initializing"

    img = Image.new("RGB", (40, 30))
    img_base64 = img_to_base64(img)

    with pytest.raises(ValueError, match="Model not initialized"):
        await model.generate(
            img=img,
            width=128,
            height=80,
            prompt="test prompt")

    model.init()
    assert model.status == "ok"

    health = model.get_health()
    assert health is not None
    timestamps = model.get_timestamps()
    assert timestamps is not None

    with pytest.raises(ValueError, match="Missing JSON body"):
        await model.get_rest_args(None)
    with pytest.raises(ValueError, match="Missing 'img' parameter"):
        await model.get_rest_args({})
    with pytest.raises(ValueError, match="Missing 'prompt' parameter"):
        # Missing prompt
        await model.get_rest_args({
            "img": img_base64,
        })

    # Success case
    args = await model.get_rest_args({
        "job_id": "unittest",
        "prompt": "Test prompt",
        "width": 80,
        "height": 60,
        "img": img_base64
    })
    assert "args" in args

    await model.warmup()

    image = await model.generate(
        img=img,
        width=256,
        height=160,
        prompt="Test prompt",
    )
    assert image is not None
    assert isinstance(image, Image.Image)
    assert image.size == (256, 160)

    model.world_size = 4
    with pytest.raises(ValueError, match="48x48 not supported for 4 GPUs"):
        await model.generate(
            img=img,
            width=48,
            height=48,
            prompt="Test prompt",
        )

    del model


@pytest.mark.asyncio
async def test_additional_coverage() -> None:
    """Cover seed path, step callbacks, parallelism init, and compile-disabled path."""
    model = FluxKontextGeneration()
    model.init()
    assert model.status == "ok"

    img = Image.new("RGB", (256, 160))
    image = await model.generate(
        img=img,
        width=256,
        height=160,
        prompt="Seed coverage test",
        seed=42)
    assert isinstance(image, Image.Image)

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
        img=img,
        width=256,
        height=160,
        prompt="Callback coverage test",
        sampling_steps=2)
    assert isinstance(image, Image.Image)

    model.world_size = 2
    with patch.object(_fluxkontext_module, 'parallelize_transformer'):
        model.init_model_parallelism()

    model.torch_compile = False
    model.model_compile()

    del model
