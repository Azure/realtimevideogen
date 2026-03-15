#!/usr/bin/env python3

import sys
import pytest

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.torch_mock import TorchMock

from PIL import Image

mock_torch = TorchMock()

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
    'sample_inference_audio': MagicMock(),
    'transformers': MagicMock(),
    'encode_data': MagicMock(),
    'hymm_sp.config': MagicMock(),
    'hymm_sp.data_kits.face_align': MagicMock(),
    'hymm_sp.modules.parallel_states': MagicMock(),
}
mock_modules.update(mock_torch.get_sub_modules())

sys.path.append("wrapper")
sys.path.append("wrapper/hunyuanavatar")

with patch.dict(sys.modules, mock_modules):
    from image_utils import img_to_base64
    from hunyuanavatar.wrapper_hunyuanavatar import HunyuanAvatarGeneration


@pytest.mark.asyncio
async def test_hunyuan_avater() -> None:
    model = HunyuanAvatarGeneration()
    assert model is not None
    assert model.model_name == "hunyuanavatar"
    assert model.status == "initializing"

    with pytest.raises(TypeError):
        model.init()
    assert model.status == "failed"
    health = model.get_health()
    assert health is not None
    timestamps = model.get_timestamps()
    assert timestamps is not None

    with pytest.raises(ValueError):
        await model.get_rest_args({})
    img = Image.new("RGB", (40, 30))
    img_base64 = img_to_base64(img)
    await model.get_rest_args({
        "audio": "test",
        "img": img_base64,
        "prompt": "test prompt",
    })

    with pytest.raises(AssertionError):
        # TODO improve the mocking
        await model.warmup()

    with pytest.raises(AssertionError):
        # TODO improve the mocking
        await model.generate(
            img=Image.new('RGB', (100, 100)),
            audio_path="test_audio.wav",
            prompt="test prompt")
        # assert video_frames is not None

    del model


@pytest.mark.asyncio
async def test_hunyuan_avatar_get_rest_args_validation() -> None:
    model = HunyuanAvatarGeneration()

    with pytest.raises(ValueError):
        await model.get_rest_args(None)

    # Missing img
    with pytest.raises(ValueError):
        await model.get_rest_args({})

    # Missing audio
    img = Image.new("RGB", (40, 30))
    img_base64 = img_to_base64(img)
    with pytest.raises(ValueError):
        await model.get_rest_args({"img": img_base64})

    # Missing prompt - use valid base64 audio ("test" decodes cleanly)
    with pytest.raises(ValueError):
        await model.get_rest_args({"img": img_base64, "audio": "test"})

    # All required params succeeds
    result = await model.get_rest_args({
        "img": img_base64,
        "audio": "test",
        "prompt": "test prompt",
    })
    assert result is not None
    assert "args" in result


@pytest.mark.asyncio
async def test_hunyuan_avatar_get_rest_args_extra_params() -> None:
    """Test get_rest_args with all optional parameters."""
    model = HunyuanAvatarGeneration()
    img = Image.new("RGB", (40, 30))
    img_base64 = img_to_base64(img)

    result = await model.get_rest_args({
        "img": img_base64,
        "audio": "test",
        "prompt": "test prompt",
        "height": 720,
        "width": 1280,
        "sampling_steps": 5,
        "audio_scale": 0.8,
        "cfg_scale": 3.0,
        "audio_cfg_scale": 2.0,
        "job_id": "test_job_001",
    })
    assert result["task"] == "hunyuanavatar"
    assert result["args"]["height"] == 720
    assert result["args"]["width"] == 1280
    assert result["args"]["sampling_steps"] == 5
    assert result["args"]["audio_scale"] == 0.8
    assert result["args"]["cfg_scale"] == 3.0
    assert result["args"]["audio_cfg_scale"] == 2.0
    assert result["args"]["job_id"] == "test_job_001"


def test_hunyuan_avatar_assert_model_init() -> None:
    """Test _assert_model_init raises before model components are loaded."""
    model = HunyuanAvatarGeneration()
    # Components are None, so raises AssertionError
    with pytest.raises(AssertionError, match="HunyuanVideoSampler is not initialized"):
        model._assert_model_init()
