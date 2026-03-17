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
    from mock.wrapper_mock import MockGeneration  # type: ignore[import-untyped]


@pytest.mark.asyncio
async def test_e2e() -> None:
    model = MockGeneration()
    assert model is not None
    assert model.model_name == "mock"
    assert model.status == "initializing"

    model.init()

    health = model.get_health()
    assert health is not None

    await model.warmup()

    image = await model.generate(
        image=Image.new('RGB', (100, 100)),
        output_type="pil"
    )
    assert image is not None
    assert isinstance(image, Image.Image)


@pytest.mark.asyncio
async def test_generate() -> None:
    model = MockGeneration()

    image = await model.generate(
        image=Image.new('RGB', (100, 100)),
        output_type="pil"
    )
    assert image is not None
    assert isinstance(image, Image.Image)

    jsonl = await model.generate(output_type="jsonl")
    assert jsonl is not None
    assert isinstance(jsonl, str)

    tensor = await model.generate(output_type="tensor")
    assert tensor is not None
    assert isinstance(tensor, str)


# TODO fix this test
"""
@pytest.mark.asyncio
async def test_http_server() -> None:
    with patch.dict(sys.modules, {
        'nvidia_smi': MagicMock(),
        'torch': mock_torch,
        'torch.distributed': MagicMock(),
        'colorlog': MagicMock(),
        'imageio': MagicMock(),
        'xfuser': MagicMock(),
        'xfuser.config': MagicMock(),
        'xfuser.core': MagicMock(),
        'xfuser.core.distributed': MagicMock(),
    }):
        from run_httpserver import main

    test_args = ["run_http_server.py", "--mock"]
    with patch.object(sys, "argv", test_args):
        await main()
"""
