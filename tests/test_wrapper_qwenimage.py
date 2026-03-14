#!/usr/bin/env python3

import sys
import pytest

from PIL import Image

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.torch_mock import TorchMock

mock_torch = TorchMock()

sys.path.append("wrapper")

mock_modules = {
    "torch": mock_torch,
    "diffusers": MagicMock(),
    "diffusers.utils": MagicMock(),
    "diffusers.models": MagicMock(),
    "diffusers.utils.torch_utils": MagicMock(),
    "diffusers.models.autoencoders": MagicMock(),
    "diffusers.models.autoencoders.autoencoder_kl": MagicMock(),
    "diffusers.models.activations": MagicMock(),
    "diffusers.models.attention_processor": MagicMock(),
    "diffusers.models.unets": MagicMock(),
    "diffusers.models.unets.unet_2d_blocks": MagicMock(),
    "diffusers.models.normalization": MagicMock(),
    "diffusers.models.resnet": MagicMock(),
    "diffusers.models.transformers": MagicMock(),
    "diffusers.models.transformers.dual_transformer_2d": MagicMock(),
    "diffusers.models.transformers.transformer_2d": MagicMock(),
    "diffusers.models.downsampling": MagicMock(),
    "diffusers.models.upsampling": MagicMock(),
    "distvae.modules.adapters.vae.decoder_adapters": MagicMock(),
    "xfuser.config": MagicMock(),
    "xfuser.core.distributed.group_coordinator": MagicMock(),
}
mock_modules.update(mock_torch.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from qwenimage.wrapper_qwenimage import QwenImageGeneration


@pytest.mark.asyncio
async def test_wrapper_qwenimage() -> None:
    model = QwenImageGeneration()
    assert model is not None
    assert model.model_name == "qwenimage"
    assert model.status == "initializing"

    with pytest.raises(ValueError, match="Model not initialized."):
        await model.generate(64, 48, "test prompt")

    model.init()
    assert model.status == "ok"

    # Mock pipeline return object
    mock_output = MagicMock()
    mock_output.images = [
        Image.new("RGB", (64, 48), color="red")
    ]
    model.pipeline = MagicMock(return_value=mock_output)
    model.pipeline.vae_scale_factor = 8

    health = model.get_health()
    assert health is not None
    assert health["model_name"] == "qwenimage"
    assert health["running"] is False
    assert health["status"] == "ok"
    assert "load_timer" in health
    assert "gen_timer" in health

    timestamps = model.get_timestamps()
    assert timestamps is not None

    with pytest.raises(ValueError):
        await model.get_rest_args(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        await model.get_rest_args({})
    await model.get_rest_args({
        "job_id": "unittest",
        "prompt": "Test prompt",
        "width": 80,
        "height": 60,
        "seed": 7,
    })

    await model.warmup()

    image = await model.generate(
        prompt="Test prompt",
        height=1024,
        width=1024)
    assert image is not None
    assert image.size == (64, 48)  # Returns the mock value

    del model


@pytest.mark.asyncio
async def test_wrapper_qwenimage_assert_args() -> None:
    """_assert_args raises for image sizes not evenly divisible across GPUs."""
    model = QwenImageGeneration()
    model.init()
    model.pipeline = MagicMock(return_value=MagicMock(images=[
        Image.new("RGB", (64, 48), color="red")
    ]))
    model.pipeline.vae_scale_factor = 8  # latent factor = 8

    # Single GPU (world_size=1): any size accepted
    model.world_size = 1
    model.rank = 0
    image = await model.generate(prompt="test", height=512, width=512)
    assert image is not None

    # Multi-GPU (world_size=3): latent shape 1024 is not divisible by 3 → raises
    model.world_size = 3
    with pytest.raises(ValueError, match="not supported for"):
        await model.generate(prompt="test", height=512, width=512)

    # Multi-GPU (world_size=4): latent shape 1024 is divisible by 4 → OK
    model.world_size = 4
    image = await model.generate(prompt="test", height=512, width=512)
    assert image is not None

    del model
