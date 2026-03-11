#!/usr/bin/env python3

import sys
import gc
import pytest

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.torch_mock import TorchMock

mock_torch = TorchMock()

sys.path.append("kokoro")

mock_modules = {
    'kokoro.KPipeline': MagicMock(),
    'nvidia_smi': MagicMock(),
    'colorlog': MagicMock(),
    'imageio': MagicMock(),
    'cv2': MagicMock(),
    'torch': mock_torch,
}
mock_modules.update(mock_torch.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from kokoro.wrapper_kokoro import KokoroGeneration


@pytest.mark.asyncio
async def test_basic() -> None:
    model = KokoroGeneration()
    assert model is not None
    assert model.model_name == "kokoro"
    assert model.status == "initializing"

    model.init()
    assert model.status == "ok"
    health = model.get_health()
    assert health is not None
    assert len(health) >= 5
    timestamps = model.get_timestamps()
    assert timestamps is not None
    assert len(timestamps) >= 4

    with pytest.raises(ValueError):
        await model.get_rest_args(None)
    with pytest.raises(ValueError):
        await model.get_rest_args({})
    await model.get_rest_args({
        "text": "Test text"
    })

    await model.warmup()

    audio_path = await model.generate(
        text="Test text",
        output_type="audio_path")
    # TODO make it not None
    assert audio_path is None

    del model
    gc.collect()
