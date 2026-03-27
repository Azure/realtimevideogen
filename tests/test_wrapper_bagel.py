#!/usr/bin/env python3

import sys
import pytest

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.torch_mock import TorchMock
from tests.diffusers_mock import DiffusersMock

from PIL import Image

mock_torch = TorchMock()
mock_diffusers = DiffusersMock()

sys.path.append("wrapper")
sys.path.append("wrapper/bagel")

mock_modules = {
    'nvidia_smi': MagicMock(),
    'imageio': MagicMock(),
    'cv2': MagicMock(),
    'xfuser': MagicMock(),
    'xfuser.config': MagicMock(),
    'xfuser.core': MagicMock(),
    'xfuser.core.distributed': MagicMock(),
    'xfuser.model_executor': MagicMock(),
    'xfuser.model_executor.layers': MagicMock(),
    'xfuser.model_executor.layers.attention_processor': MagicMock(),
    'modeling': MagicMock(),
    'modeling.autoencoder': MagicMock(),
    'modeling.qwen2': MagicMock(),
    'modeling.bagel': MagicMock(),
    'modeling.bagel.qwen2_navit': MagicMock(),
    'data': MagicMock(),
    'data.transforms': MagicMock(),
    'data.data_utils': MagicMock(),
    'accelerate': MagicMock(),
    'safetensors': MagicMock(),
    'safetensors.torch': MagicMock(),
}
mock_modules.update(mock_torch.get_sub_modules())
mock_modules.update(mock_diffusers.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from image_utils import img_to_base64
    from bagel.wrapper_bagel import BagelGeneration


@pytest.mark.asyncio
async def test_basic() -> None:
    model = BagelGeneration()
    assert model is not None
    assert model.model_name == "bagel"
    assert model.status == "initializing"

    img = Image.new("RGB", (40, 30))
    img_base64 = img_to_base64(img)

    with pytest.raises(ValueError, match="Model not initialized"):
        await model.generate(
            imgs=[img],
            width=128,
            height=80,
            prompt="test prompt")

    with pytest.raises(ValueError):
        model.init()
    assert model.status == "failed"

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
            "imgs": img_base64,
        })

    # Success case
    args = await model.get_rest_args({
        "job_id": "unittest",
        "prompt": "Test prompt",
        "width": 80,
        "height": 60,
        "imgs": [img_base64]
    })
    assert "args" in args
    assert args["task"] == "bagel"

    with pytest.raises(ValueError, match="Model not initialized"):
        await model.warmup()

    with pytest.raises(ValueError, match="Model not initialized"):
        await model.generate(
            imgs=[img],
            width=256,
            height=160,
            prompt="Test prompt",
        )

    del model
