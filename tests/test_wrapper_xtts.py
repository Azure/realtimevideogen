#!/usr/bin/env python3

import sys
import pytest

from unittest.mock import patch, MagicMock
from tests.torch_mock import TorchMock

mock_torch = TorchMock()

sys.path.append("wrapper")
sys.path.append("wrapper/xtts")

mock_modules = {
    'nvidia_smi': MagicMock(),
    'numpy': MagicMock(),
    'TTS': MagicMock(),
    'TTS.tts': MagicMock(),
    'TTS.tts.configs': MagicMock(),
    'TTS.tts.configs.xtts_config': MagicMock(),
    'TTS.tts.models': MagicMock(),
    'TTS.tts.models.xtts': MagicMock(),
    'torch': mock_torch,
}
mock_modules.update(mock_torch.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from xtts.wrapper_xtts import XTTSGeneration


@pytest.mark.asyncio
async def test_xtts_basic() -> None:
    model = XTTSGeneration()
    assert model is not None
    assert model.model_name == "xtts"
    assert model.status == "initializing"


@pytest.mark.asyncio
async def test_xtts_init() -> None:
    model = XTTSGeneration()
    # init succeeds since all heavy deps are mocked
    model.init()
    assert model.status == "ok"


@pytest.mark.asyncio
async def test_xtts_get_rest_args_validation() -> None:
    model = XTTSGeneration()

    with pytest.raises(ValueError):
        await model.get_rest_args(None)

    with pytest.raises(ValueError):
        await model.get_rest_args({})

    result = await model.get_rest_args({"text": "hello"})
    assert result["task"] == "xtts"
    assert result["args"]["text"] == "hello"


@pytest.mark.asyncio
async def test_xtts_get_health() -> None:
    model = XTTSGeneration()
    health = model.get_health()
    assert isinstance(health, dict)
