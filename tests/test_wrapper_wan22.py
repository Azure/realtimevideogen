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
async def test_init() -> None:
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
async def test_get_rest_args_missing_body() -> None:
    """Test that get_rest_args raises for missing JSON body."""
    model = Wan22VideoGeneration()
    model.init()

    with pytest.raises(ValueError, match="Missing JSON body"):
        await model.get_rest_args(None)


@pytest.mark.asyncio
async def test_get_rest_args_missing_img() -> None:
    """Test that get_rest_args raises when img is absent."""
    model = Wan22VideoGeneration()
    model.init()

    with pytest.raises(ValueError, match="Missing 'img' parameter"):
        await model.get_rest_args({})


@pytest.mark.asyncio
async def test_get_rest_args_missing_prompt() -> None:
    """Test that get_rest_args raises when prompt is absent."""
    model = Wan22VideoGeneration()
    model.init()

    img = Image.new("RGB", (40, 30))
    img_base64 = img_to_base64(img)

    with pytest.raises(ValueError, match="Missing 'prompt' parameter"):
        await model.get_rest_args({"img": img_base64})


@pytest.mark.asyncio
async def test_get_rest_args_missing_audio() -> None:
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
async def test_get_rest_args_with_audio() -> None:
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
async def test_get_rest_args_with_tts() -> None:
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
async def test_get_rest_args_custom_params() -> None:
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
async def test_generate_raises_without_init() -> None:
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


def test_assert_model_init_wan_s2v_none() -> None:
    """Test _assert_model_init raises when wan_s2v is None after init."""
    model = Wan22VideoGeneration()
    model.init()
    model.wan_s2v = None
    with pytest.raises(ValueError, match="WanS2V model not initialized"):
        model._assert_model_init()


def test_model_compile_no_op_no_torch_compile() -> None:
    """Test model_compile is no-op when torch_compile=False."""
    model = Wan22VideoGeneration()
    model.init()
    model.torch_compile = False
    model.model_compile()  # should not raise


def test_model_compile_no_op_no_wan_s2v() -> None:
    """Test model_compile is no-op when wan_s2v is None."""
    model = Wan22VideoGeneration()
    model.init()
    model.torch_compile = True
    model.wan_s2v = None
    model.model_compile()  # should not raise (wan_s2v is None guard)


def test_model_compile_with_torch_compile() -> None:
    """Test model_compile calls torch.compile when torch_compile=True and wan_s2v is set."""
    model = Wan22VideoGeneration()
    model.init()
    model.torch_compile = True
    assert model.wan_s2v is not None
    model.model_compile()  # torch.compile is mocked, should not raise


@pytest.mark.asyncio
async def test_get_rest_args_img_not_string() -> None:
    """Test get_rest_args raises when img is not a string."""
    model = Wan22VideoGeneration()
    model.init()

    with pytest.raises(ValueError, match="'img' parameter must be a base64-encoded string"):
        await model.get_rest_args({"img": 12345})


@pytest.mark.asyncio
async def test_get_rest_args_audio_not_string() -> None:
    """Test get_rest_args raises when audio is not a string."""
    model = Wan22VideoGeneration()
    model.init()

    img = Image.new("RGB", (40, 30))
    img_base64 = img_to_base64(img)

    with pytest.raises(ValueError, match="'audio' parameter must be a base64-encoded string"):
        await model.get_rest_args({
            "img": img_base64,
            "prompt": "test",
            "audio": 99999,
        })


@pytest.mark.asyncio
async def test_get_rest_args_tts_prompt_audio_not_string() -> None:
    """Test get_rest_args raises when tts_prompt_audio is not a string."""
    model = Wan22VideoGeneration()
    model.init()

    img = Image.new("RGB", (40, 30))
    img_base64 = img_to_base64(img)

    with pytest.raises(ValueError, match="'tts_prompt_audio' must be a base64-encoded string"):
        await model.get_rest_args({
            "img": img_base64,
            "prompt": "test",
            "enable_tts": True,
            "tts_prompt_audio": 99999,
        })


@pytest.mark.asyncio
async def test_get_rest_args_with_job_id() -> None:
    """Test get_rest_args passes a deterministic audio destination when job_id is given."""
    model = Wan22VideoGeneration()
    model.init()

    img = Image.new("RGB", (40, 30))
    img_base64 = img_to_base64(img)

    mock_media_utils.base64_to_audio_file.reset_mock()

    await model.get_rest_args({
        "img": img_base64,
        "prompt": "test",
        "audio": "dGVzdA==",
        "job_id": "myjob123",
    })

    # Verify that the audio file helper was called with the expected destination path
    mock_media_utils.base64_to_audio_file.assert_called_once_with(
        "dGVzdA==", audio_path="/tmp/myjob123.wav"
    )
