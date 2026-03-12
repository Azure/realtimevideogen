#!/usr/bin/env python3

import sys
import pytest

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.torch_mock import TorchMock

from PIL import Image

mock_torch = TorchMock()

sys.path.append("realesrgan")

with patch.dict(sys.modules, {
    'nvidia_smi': MagicMock(),
    'colorlog': MagicMock(),
    'imageio': MagicMock(),
    'cv2': MagicMock(),
    'RealESRGAN': MagicMock(),
    'torch': mock_torch,
}):
    from image_utils import img_to_base64
    from realesrgan.wrapper_realesrgan import RealESRGANGeneration


@pytest.mark.asyncio
async def test_wrapper_realesrgan() -> None:
    model = RealESRGANGeneration()
    assert model is not None
    assert model.model_name == "realesrgan"
    assert model.status == "initializing"
    assert len(model.models) == 0

    with pytest.raises(ValueError):
        await model.generate(image=None)

    model.init()
    assert model.status == "ok"
    assert len(model.models) == 3

    health = model.get_health()
    assert health is not None
    assert len(health) >= 5
    timestamps = model.get_timestamps()
    assert timestamps is not None
    assert len(timestamps) >= 7

    with pytest.raises(ValueError):
        await model.get_rest_args(None)
    await model.get_rest_args({})
    img = Image.new("RGB", (40, 30))
    img_base64 = img_to_base64(img)
    await model.get_rest_args({
        "job_id": "unittest",
        "img": img_base64,
        "width": 80,
        "height": 60,
    })

    await model.warmup()

    image_resized = await model.generate(
        image=img,
        width=160,
        height=120,
        output_type="pil"
    )
    assert image_resized is not None

    del model
