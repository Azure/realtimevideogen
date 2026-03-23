#!/usr/bin/env python3

import sys
import gc
import pytest

from unittest.mock import patch
from unittest.mock import MagicMock
from unittest.mock import AsyncMock

from PIL import Image

from tests.torch_mock import TorchMock

mock_torch = TorchMock()

sys.path.append("wrapper")
sys.path.append("wrapper/wan")
sys.path.append("wrapper/wan22")

# Build a mock for media_utils with async base64_to_audio_file
mock_media_utils = MagicMock()
mock_media_utils.base64_to_audio_file = AsyncMock(return_value="/tmp/test_audio.wav")
mock_media_utils.empty_audio_file = MagicMock(return_value="/tmp/warmup.wav")

mock_modules = {
    'nvidia_smi': MagicMock(),
    'colorlog': MagicMock(),
    'imageio': MagicMock(),
    'cv2': MagicMock(),
    'torch': mock_torch,
    'torchvision': MagicMock(),
    'torchvision.transforms': MagicMock(),
    'torchvision.transforms.functional': MagicMock(),
    'xfuser': MagicMock(),
    'xfuser.config': MagicMock(),
    'xfuser.core': MagicMock(),
    'xfuser.core.distributed': MagicMock(),
    'transformers': MagicMock(),
    'wan': MagicMock(),
    'wan.configs': MagicMock(),
    'wan.modules': MagicMock(),
    'wan.modules.t5': MagicMock(),
    'wan.modules.clip': MagicMock(),
    'wan.modules.vae': MagicMock(),
    'wan.modules.model': MagicMock(),
    'wan.utils': MagicMock(),
    'wan.utils.utils': MagicMock(),
    'wan.utils.fm_solvers_unipc': MagicMock(),
    'wan.distributed': MagicMock(),
    'wan.distributed.fsdp': MagicMock(),
    'wan.distributed.sequence_parallel': MagicMock(),
    'media_utils': mock_media_utils,
}
mock_modules.update(mock_torch.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from image_utils import img_to_base64
    from wan22.wrapper_wan22 import Wan22VideoGeneration


@pytest.mark.asyncio
async def test_wrapper_wan22_init() -> None:
    """Test basic initialization and health checks."""
    model = Wan22VideoGeneration()
    assert model is not None
    assert model.model_name == "wan22"

    model.init()
    health = model.get_health()
    assert health is not None
    timestamps = model.get_timestamps()
    assert timestamps is not None

    del model
    gc.collect()


@pytest.mark.asyncio
async def test_wrapper_wan22_get_rest_args_missing_body() -> None:
    """Test that get_rest_args raises for missing JSON body."""
    model = Wan22VideoGeneration()
    model.init()

    with pytest.raises(ValueError, match="Missing JSON body"):
        await model.get_rest_args(None)


@pytest.mark.asyncio
async def test_wrapper_wan22_get_rest_args_missing_img() -> None:
    """Test that get_rest_args raises when img is absent."""
    model = Wan22VideoGeneration()
    model.init()

    with pytest.raises(ValueError, match="Missing 'img' parameter"):
        await model.get_rest_args({})


@pytest.mark.asyncio
async def test_wrapper_wan22_get_rest_args_missing_prompt() -> None:
    """Test that get_rest_args raises when prompt is absent."""
    model = Wan22VideoGeneration()
    model.init()

    img = Image.new("RGB", (40, 30))
    img_base64 = img_to_base64(img)

    with pytest.raises(ValueError, match="Missing 'prompt' parameter"):
        await model.get_rest_args({"img": img_base64})


@pytest.mark.asyncio
async def test_wrapper_wan22_get_rest_args_missing_audio() -> None:
    """Test that get_rest_args raises when audio is absent and TTS is disabled."""
    model = Wan22VideoGeneration()
    model.init()

    img = Image.new("RGB", (40, 30))
    img_base64 = img_to_base64(img)

    with pytest.raises(ValueError, match="Missing 'audio' parameter"):
        await model.get_rest_args({
            "img": img_base64,
            "prompt": "test prompt",
        })


@pytest.mark.asyncio
async def test_wrapper_wan22_get_rest_args_with_audio() -> None:
    """Test get_rest_args with a base64-encoded audio file."""
    model = Wan22VideoGeneration()
    model.init()

    img = Image.new("RGB", (40, 30))
    img_base64 = img_to_base64(img)

    result = await model.get_rest_args({
        "img": img_base64,
        "prompt": "test prompt",
        "audio": "dGVzdA==",  # base64("test") - placeholder
    })

    assert "args" in result
    assert result["task"] == "wan22"
    args = result["args"]
    assert args["prompt"] == "test prompt"
    assert args["neg_prompt"] == ""
    assert args["enable_tts"] is False
    assert args["audio_path"] is not None
    assert args["audio_path"].endswith(".wav")
    assert args["max_area"] == 1024 * 704
    assert args["sampling_steps"] == 40
    assert args["infer_frames"] == 80
    assert args["num_clip"] is None

    del model
    gc.collect()


@pytest.mark.asyncio
async def test_wrapper_wan22_get_rest_args_with_tts() -> None:
    """Test get_rest_args with TTS parameters."""
    model = Wan22VideoGeneration()
    model.init()

    img = Image.new("RGB", (40, 30))
    img_base64 = img_to_base64(img)

    result = await model.get_rest_args({
        "img": img_base64,
        "prompt": "Summer beach scene",
        "enable_tts": True,
        "tts_prompt_audio": "dGVzdA==",
        "tts_prompt_text": "Hello world",
        "tts_text": "Generated speech text",
    })

    args = result["args"]
    assert args["enable_tts"] is True
    assert args["tts_text"] == "Generated speech text"
    assert args["tts_prompt_text"] == "Hello world"
    assert args["tts_prompt_audio"] is not None
    assert args["tts_prompt_audio"].endswith(".wav")
    assert args["audio_path"] is None

    del model
    gc.collect()


@pytest.mark.asyncio
async def test_wrapper_wan22_get_rest_args_custom_params() -> None:
    """Test get_rest_args honours custom resolution, steps, and clip count."""
    model = Wan22VideoGeneration()
    model.init()

    img = Image.new("RGB", (40, 30))
    img_base64 = img_to_base64(img)

    result = await model.get_rest_args({
        "img": img_base64,
        "prompt": "test",
        "audio": "dGVzdA==",
        "max_area": 720 * 1280,
        "sampling_steps": 20,
        "infer_frames": 48,
        "num_clip": 3,
        "neg_prompt": "blurry",
        "output_type": "video_path",
    })

    args = result["args"]
    assert args["max_area"] == 720 * 1280
    assert args["sampling_steps"] == 20
    assert args["infer_frames"] == 48
    assert args["num_clip"] == 3
    assert args["neg_prompt"] == "blurry"
    assert args["output_type"] == "video_path"

    del model
    gc.collect()


@pytest.mark.asyncio
async def test_wrapper_wan22_generate_raises_without_init() -> None:
    """Test that generate raises when model has not been initialized."""
    model = Wan22VideoGeneration()
    # NOTE: intentionally NOT calling model.init() here

    img = Image.new("RGB", (40, 30))

    with pytest.raises(ValueError, match="Model not initialized"):
        await model.generate(
            img=img,
            prompt="test prompt",
            audio_path="/tmp/fake_audio.wav",
        )

    del model
    gc.collect()
