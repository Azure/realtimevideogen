#!/usr/bin/env python3

import sys
import pytest

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.torch_mock import TorchMock
from tests.diffusers_mock import DiffusersMock

mock_torch = TorchMock()
mock_diffusers = DiffusersMock()

sys.path.append("hunyuanimage")

mock_modules = {
    "torch": mock_torch,
    "diffusers": mock_diffusers,
    "transformers": MagicMock(),
    "xfuser": MagicMock(),
    "xfuser.config": MagicMock(),
}
mock_modules.update(mock_torch.get_sub_modules())
mock_modules.update(mock_diffusers.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from hunyuanimage.wrapper_hunyuanimage import HunyuanImageGeneration


@pytest.mark.asyncio
async def test_wrapper_hunyuanimage() -> None:
    model = HunyuanImageGeneration()
    assert model is not None
    assert model.model_name == "hunyuanimage"
    assert model.status == "initializing"

    with pytest.raises(ValueError, match="Model not initialized. Current status: initializing."):
        await model.generate(64, 48, "test prompt")

    model.status = "ok"
    with pytest.raises(ValueError, match="HunyuanImage model not loaded."):
        await model.generate(64, 48, "test prompt")
    model.status == "initializing"

    model.init()
    assert model.status == "ok"

    # Mock model init()
    model.model.vae.config.ffactor_spatial = 8

    health = model.get_health()
    assert health is not None

    with pytest.raises(ValueError, match="Missing JSON body"):
        await model.get_rest_args(None)
    with pytest.raises(ValueError, match="Missing 'prompt' parameter"):
        await model.get_rest_args({})

    rest_args = await model.get_rest_args({
        "prompt": "test prompt",
    })
    assert rest_args == {
        "task": "hunyuanimage",
        "args": {
            "width": 640,
            "height": 480,
            "prompt": "test prompt",
            "sampling_steps": 25,
            "seed": None,
        }
    }

    await model.warmup()

    image = await model.generate(
        width=1024,
        height=1024,
        prompt="Test prompt")
    assert image is not None

    with pytest.raises(ValueError, match="2048x1024 too large. Max is 1024 x 1024."):
        await model.generate(width=2048, height=1024, prompt="Prompt")

    with pytest.raises(ValueError, match="Width 1027 not supported. Must be multiple of 8."):
        await model.generate(width=1027, height=512, prompt="Prompt")

    with pytest.raises(ValueError, match="Height 513 not supported. Must be multiple of 8."):
        await model.generate(width=512, height=513, prompt="Prompt")


@pytest.mark.asyncio
async def test_wrapper_hunyuanimage_parallel() -> None:
    model = HunyuanImageGeneration()
    model.init()
    assert model.status == "ok"
    model.rank = 1  # Simulate distributed rank
    image = await model.generate(
        width=128,
        height=128,
        prompt="Test prompt")
    assert image is None  # Only rank 0 generates images
