#!/usr/bin/env python3

import sys
import pytest

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.torch_mock import TorchMock

from PIL import Image

mock_torch = TorchMock()

sys.path.append("wrapper")
sys.path.append("wrapper/hidream")

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
    'xfuser.model_executor.layers': MagicMock(),
    'xfuser.model_executor.layers.attention_processor': MagicMock(),
    'diffusers': MagicMock(),
    'transformers': MagicMock(),
}
mock_modules.update(mock_torch.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from hidream.wrapper_hidream import HiDreamGeneration


@pytest.mark.asyncio
async def test_basic() -> None:
    model = HiDreamGeneration()
    assert model is not None
    assert model.model_name == "hidream"
    assert model.status == "initializing"

    with pytest.raises(ValueError):
        await model.generate(64, 48, "test prompt")

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
    await model.get_rest_args({
        "job_id": "unittest",
        "prompt": "Test prompt",
        "width": 80,
        "height": 60,
        "seed": 7,
    })

    await model.warmup()

    image = await model.generate(
        width=1280,
        height=800,
        prompt="Test prompt")
    assert image is not None
    assert isinstance(image, Image.Image)
    assert image.size == (1280, 800)

    del model
