#!/usr/bin/env python3

import sys
import pytest

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.torch_mock import TorchMock

from PIL import Image

mock_torch = TorchMock()

sys.path.append("wrapper")
sys.path.append("wrapper/fluxkontext")
sys.path.append("wrapper/flux")

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
}
mock_modules.update(mock_torch.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from image_utils import img_to_base64
    from fluxkontext.wrapper_fluxkontext import FluxKontextGeneration


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

    with pytest.raises(ValueError):
        await model.get_rest_args(None)
    with pytest.raises(ValueError):
        await model.get_rest_args({})
    with pytest.raises(ValueError):
        # Missing prompt
        await model.get_rest_args({
            "img": img_base64,
        })

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
