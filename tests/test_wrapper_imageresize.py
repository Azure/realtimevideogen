#!/usr/bin/env python3

import sys
import pytest

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.torch_mock import TorchMock

from PIL import Image

mock_torch = TorchMock()

with patch.dict(sys.modules, {
    'nvidia_smi': MagicMock(),
    'colorlog': MagicMock(),
    'imageio': MagicMock(),
    'cv2': MagicMock(),
    'torch': mock_torch,
}):
    from image_utils import img_to_base64
    from media_utils import video_frames_to_base64
    from imageresize.wrapper_imageresize import ImageResize


@pytest.mark.asyncio
async def test_e2e() -> None:
    model = ImageResize()
    assert model is not None
    assert model.model_name == "imageresize"

    model.init()

    health = model.get_health()
    assert health is not None

    await model.warmup()

    image = await model.generate(image=Image.new('RGB', (100, 100)))
    assert image is not None


@pytest.mark.asyncio
async def test_get_health() -> None:
    model = ImageResize()
    assert model is not None
    assert model.model_name == "imageresize"
    assert model.status == "initializing"

    model.init()
    health = model.get_health()
    assert health is not None
    assert health["status"] == "ok"
    timestamps = model.get_timestamps()
    assert timestamps is not None


@pytest.mark.asyncio
async def test_get_rest_args() -> None:
    model = ImageResize()

    with pytest.raises(ValueError):
        await model.get_rest_args(None)
    rest_args = await model.get_rest_args({})
    assert rest_args is not None
    assert len(rest_args) == 2
    assert rest_args["task"] == "imageresize"
    rest_args = await model.get_rest_args({
        "img": img_to_base64(Image.new('RGB', (100, 100))),
    })
    assert len(rest_args) == 2
    rest_args = await model.get_rest_args({
        "video": video_frames_to_base64([
            Image.new('RGB', (100, 100)),
            Image.new('RGB', (100, 100)),
        ]),
    })
    assert len(rest_args) == 2


@pytest.mark.asyncio
async def test_generate() -> None:
    model = ImageResize()

    with pytest.raises(TypeError):
        await model.generate()
    image_resize = await model.generate(
        image=Image.new('RGB', (100, 100)),
        width=200,
        height=200)
    assert image_resize is not None
    assert image_resize.size == (200, 200)

    video_resize = await model.generate(
        image=None,
        video=[
            Image.new('RGB', (50, 25)),
            Image.new('RGB', (50, 25)),
        ],
        width=100,
        height=50)
    assert video_resize is not None
    assert len(video_resize) == 2
    assert video_resize[0].size == (100, 50)

    with pytest.raises(ValueError):
        await model.generate(image=None)
