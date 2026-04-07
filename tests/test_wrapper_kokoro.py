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
    'imageio': MagicMock(),
    'cv2': MagicMock(),
    'torch': mock_torch,
}
mock_modules.update(mock_torch.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from kokoro.wrapper_kokoro import KokoroGeneration
    from kokoro.wrapper_kokoro import Language


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


def test_language_enum() -> None:
    """All Language enum members should have their correct string codes."""
    assert Language.AMERICAN_ENGLISH.value == "a"
    assert Language.BRITISH_ENGLISH.value == "b"
    assert Language.SPANISH.value == "e"
    assert Language.FRENCH.value == "f"
    assert Language.HINDI.value == "h"
    assert Language.ITALIAN.value == "i"
    assert Language.BRAZILIAN_PORTUGUESE.value == "p"
    assert Language.JAPANESE.value == "j"
    assert Language.MANDARIN_CHINESE.value == "z"

    # Language is a str enum — its value is equal to the plain string
    assert Language.AMERICAN_ENGLISH == "a"


@pytest.mark.asyncio
async def test_get_rest_args_optional_params() -> None:
    """get_rest_args should return defaults and accept custom voice/speed/lang."""
    model = KokoroGeneration()
    model.init()

    # Defaults: voice=af_heart, speed=1.0, lang_code=a (American English)
    args = await model.get_rest_args({"text": "hello"})
    assert args["task"] == "kokoro"
    inner = args["args"]
    assert inner["text"] == "hello"
    assert inner["voice"] == "af_heart"
    assert inner["speed"] == 1.0
    assert inner["lang_code"] == "a"
    assert inner["job_id"] is None

    # Custom voice, speed, and language
    args = await model.get_rest_args({
        "job_id": "j1",
        "text": "bonjour",
        "voice": "bf_emma",
        "speed": "0.8",
        "lang_code": "f",
    })
    inner = args["args"]
    assert inner["job_id"] == "j1"
    assert inner["text"] == "bonjour"
    assert inner["voice"] == "bf_emma"
    assert inner["speed"] == pytest.approx(0.8)
    assert inner["lang_code"] == "f"

    # Pass a Language enum value directly (should be treated as its string value)
    args = await model.get_rest_args({
        "text": "nihao",
        "lang_code": Language.MANDARIN_CHINESE,
    })
    assert args["args"]["lang_code"] == Language.MANDARIN_CHINESE

    del model
    gc.collect()
