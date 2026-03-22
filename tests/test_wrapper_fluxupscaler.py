#!/usr/bin/env python3

import sys
import pytest

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.torch_mock import TorchMock

from PIL import Image

mock_torch = TorchMock()

sys.path.append("fluxupscaler")
sys.path.append("flux")

mock_modules = {
    'nvidia_smi': MagicMock(),
    'colorlog': MagicMock(),
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
    'diffusers': MagicMock(),
    'diffusers.pipelines': MagicMock(),
}
mock_modules.update(mock_torch.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from image_utils import img_to_base64
    from fluxupscaler.wrapper_fluxupscaler import FluxUpscalerGeneration


@pytest.mark.asyncio
async def test_wrapper_fluxupscaler() -> None:
    model = FluxUpscalerGeneration()
    assert model is not None
    assert model.model_name == "fluxupscaler"
    assert model.status == "initializing"

    img = Image.new("RGB", (40, 30))
    img_base64 = img_to_base64(img)

    with pytest.raises(ValueError):
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

    await model.get_rest_args({})

    await model.get_rest_args({
        "img": img_base64,
    })

    with pytest.raises(ValueError):
        await model.get_rest_args(None)

    # Success case
    await model.get_rest_args({
        "job_id": "unittest",
        "prompt": "Test prompt",
        "width": 80,
        "height": 60,
        "img": img_base64
    })

    with pytest.raises(ValueError):
        await model.warmup()

    with pytest.raises(ValueError):
        await model.generate(
            img=img,
            width=256,
            height=160,
            prompt="Test prompt",
        )

    del model
