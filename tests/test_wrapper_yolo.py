#!/usr/bin/env python3

import sys
import gc
import pytest

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.torch_mock import TorchMock

from PIL import Image

mock_torch = TorchMock()

sys.path.append("yolo")

with patch.dict(sys.modules, {
    'ultralytics': MagicMock(),
    'ultralytics.YOLO': MagicMock(),
    'cv2': MagicMock(),
    'nvidia_smi': MagicMock(),
    'imageio': MagicMock(),
    'torch': mock_torch,
}):
    from image_utils import img_to_base64
    from yolo.wrapper_yolo import ImageCharacterExtractor
    from yolo.wrapper_yolo import zoom_image
    from yolo.wrapper_yolo import take_top_characters


@pytest.mark.asyncio
async def test_wrapper_yolo() -> None:
    model = ImageCharacterExtractor()
    assert model is not None
    assert model.model_name == "yolo"

    model.init()
    health = model.get_health()
    assert health is not None
    timestamps = model.get_timestamps()
    assert timestamps is not None

    with pytest.raises(ValueError):
        await model.get_rest_args({})
    img = Image.new("RGB", (40, 30))
    img_base64 = img_to_base64(img)
    await model.get_rest_args({
        "img": img_base64
    })

    await model.warmup()

    await model.generate(img=img)

    del model
    del img
    del img_base64
    gc.collect()


@pytest.mark.asyncio
async def test_zoom() -> None:
    image = zoom_image(
        image=Image.new("RGB", (100, 100)),
        x=50, y=50,
        w=10, h=10,
        zoom_factor=2.0)
    assert image is not None
    assert image.size == (50, 50)

    image = zoom_image(
        image=Image.new("RGB", (100, 100)),
        x=0, y=0,
        w=20, h=20,
        zoom_factor=1.2)
    assert image is not None
    assert image.size == (83, 83)


@pytest.mark.asyncio
async def test_take_top_characters() -> None:
    person_zoom_images = [
        (0, 0.7, Image.new("RGB", (32, 32), color=(255, 0, 0))),
        (1, 0.9, Image.new("RGB", (64, 48), color=(0, 255, 0))),
        (2, 0.85, Image.new("RGB", (32, 16), color=(0, 0, 255)))
    ]

    top_characters = take_top_characters(
        person_zoom_images,
        num_characters=2)
    assert len(top_characters) == 2
    assert top_characters[0].size == (64, 48)
    assert top_characters[1].size == (32, 16)

    top_characters = take_top_characters(
        person_zoom_images,
        num_characters=1)
    assert len(top_characters) == 1
    assert top_characters[0].size == (64, 48)
