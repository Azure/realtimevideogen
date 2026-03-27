#!/usr/bin/env python3

import sys
import pytest

from PIL import Image

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.torch_mock import TorchMock
from tests.diffusers_mock import DiffusersMock

mock_torch = TorchMock()
mock_diffusers = DiffusersMock()

sys.path.append("wrapper")

mock_modules = {
    "distvae.modules.adapters.vae.decoder_adapters": MagicMock(),
    "xfuser.config": MagicMock(),
    "xfuser.core.distributed.group_coordinator": MagicMock(),
}
mock_modules.update(mock_torch.get_sub_modules())
mock_modules.update(mock_diffusers.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from image_utils import img_to_base64
    from qwenimageedit.wrapper_qwenimageedit import QwenImageEditGeneration


@pytest.mark.asyncio
async def test_wrapper_qwenimage() -> None:
    model = QwenImageEditGeneration()
    assert model is not None
    assert model.model_name == "qwenimageedit"
    assert model.status == "initializing"

    img_input = Image.new("RGB", (64, 48), color="red")
    img_base64 = img_to_base64(img_input)

    with pytest.raises(ValueError, match="Model not initialized."):
        await model.generate(
            img=img_input,
            width=64,
            height=48,
            prompt="test prompt")

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
    assert health["model_name"] == "qwenimageedit"
    assert health["running"] is False
    assert health["status"] == "ok"
    assert "load_timer" in health
    assert "gen_timer" in health

    timestamps = model.get_timestamps()
    assert timestamps is not None

    with pytest.raises(ValueError):
        await model.get_rest_args(None)
    with pytest.raises(ValueError):
        await model.get_rest_args({})
    with pytest.raises(ValueError, match="Missing 'img' parameter"):
        await model.get_rest_args({
            "job_id": "unittest",
            "prompt": "Test prompt",
            "width": 80,
            "height": 60,
            "seed": 7,
        })
    await model.get_rest_args({
        "job_id": "unittest",
        "img": img_base64,
        "prompt": "Test prompt",
        "width": 80,
        "height": 60,
        "seed": 7,
    })

    await model.warmup()

    image = await model.generate(
        img=img_input,
        prompt="Test prompt",
        height=1024,
        width=1024)
    assert image is not None
    assert image.size == (64, 48)  # Returns the mock value

    del model
