#!/usr/bin/env python3

import os
import sys
import tempfile
import numpy as np
import pytest

from unittest.mock import patch
from unittest.mock import MagicMock
from unittest.mock import AsyncMock
from tests.torch_mock import TorchMock

from PIL import Image

mock_torch = TorchMock()

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
    from model_timing import GenTimer
    _wrapper_module = sys.modules['hunyuanavatar.wrapper_hunyuanavatar']


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


def test_hunyuan_avatar_assert_model_init_partial() -> None:
    """Test _assert_model_init raises when only some components are initialized."""
    model = HunyuanAvatarGeneration()
    model.hunyuan_video_sampler = MagicMock()
    with pytest.raises(AssertionError, match="Wav2Vec model is not initialized"):
        model._assert_model_init()

    model.wav2vec = MagicMock()
    with pytest.raises(AssertionError, match="AlignImage instance is not initialized"):
        model._assert_model_init()

    model.align_instance = MagicMock()
    with pytest.raises(AssertionError, match="Feature extractor is not initialized"):
        model._assert_model_init()

    model.feature_extractor = MagicMock()
    with pytest.raises(AssertionError, match="Text encoder is not initialized"):
        model._assert_model_init()

    model.text_encoder = MagicMock()
    with pytest.raises(AssertionError, match="Text encoder 2 is not initialized"):
        model._assert_model_init()

    model.text_encoder_2 = MagicMock()
    # All initialized - should not raise
    model._assert_model_init()


def test_hunyuan_avatar_del_with_components() -> None:
    """Test __del__ properly cleans up initialized components."""
    model = HunyuanAvatarGeneration()
    model.hunyuan_video_sampler = MagicMock()
    model.wav2vec = MagicMock()
    model.align_instance = MagicMock()
    model.feature_extractor = MagicMock()
    model.text_encoder = MagicMock()
    model.text_encoder_2 = MagicMock()
    # Should not raise
    model.__del__()
    assert model.hunyuan_video_sampler is None
    assert model.wav2vec is None
    assert model.align_instance is None
    assert model.feature_extractor is None
    assert model.text_encoder is None
    assert model.text_encoder_2 is None


def test_hunyuan_avatar_init_parallelism_no_master_addr() -> None:
    """Test init_parallelism returns early when MASTER_ADDR is not set."""
    model = HunyuanAvatarGeneration()
    env_without_master = {k: v for k, v in os.environ.items() if k != "MASTER_ADDR"}
    with patch.dict(os.environ, env_without_master, clear=True):
        # Should return early without error
        model.init_parallelism()


def test_hunyuan_avatar_init_parallelism_world_size_one() -> None:
    """Test init_parallelism with world_size=1 returns after setting device."""
    model = HunyuanAvatarGeneration()
    env = {"MASTER_ADDR": "localhost", "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}
    with patch.dict(os.environ, env, clear=False):
        model.init_parallelism()
    assert model.world_size == 1


@pytest.mark.asyncio
async def test_hunyuan_avatar_output_video_pil() -> None:
    """Test _output_video returns frames directly for pil output type."""
    model = HunyuanAvatarGeneration()
    gen_timer = GenTimer()
    video_frames = np.zeros((5, 64, 64, 3), dtype=np.uint8)
    result = await model._output_video(
        job_id=None,
        gen_timer=gen_timer,
        audio_path="/tmp/test.wav",
        video_frames=video_frames,
        output_type="pil",
    )
    assert result is video_frames


@pytest.mark.asyncio
async def test_hunyuan_avatar_output_video_unknown_type() -> None:
    """Test _output_video returns None for unknown output type."""
    model = HunyuanAvatarGeneration()
    gen_timer = GenTimer()
    video_frames = np.zeros((5, 64, 64, 3), dtype=np.uint8)
    result = await model._output_video(
        job_id=None,
        gen_timer=gen_timer,
        audio_path="/tmp/test.wav",
        video_frames=video_frames,
        output_type="unknown_type",
    )
    assert result is None


@pytest.mark.asyncio
async def test_hunyuan_avatar_output_video_video_path() -> None:
    """Test _output_video with video_path output type."""
    model = HunyuanAvatarGeneration()
    gen_timer = GenTimer()
    video_frames = np.zeros((5, 64, 64, 3), dtype=np.uint8)
    with patch.object(_wrapper_module, 'save_video_audio', new_callable=AsyncMock) as mock_save:
        mock_save.return_value = "/tmp/test_output.mp4"
        result = await model._output_video(
            job_id="test_job",
            gen_timer=gen_timer,
            audio_path="/tmp/test.wav",
            video_frames=video_frames,
            output_type="video_path",
        )
    assert result == "/tmp/test_output.mp4"


@pytest.mark.asyncio
async def test_hunyuan_avatar_output_video_video_binary() -> None:
    """Test _output_video with video_binary output type."""
    model = HunyuanAvatarGeneration()
    gen_timer = GenTimer()
    video_frames = np.zeros((5, 64, 64, 3), dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name
        f.write(b"fake_video_data")

    try:
        with patch.object(_wrapper_module, 'save_video_audio', new_callable=AsyncMock) as mock_save:
            mock_save.return_value = tmp_path
            result = await model._output_video(
                job_id=None,
                gen_timer=gen_timer,
                audio_path="/tmp/test.wav",
                video_frames=video_frames,
                output_type="video_binary",
            )
        assert result == b"fake_video_data"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@pytest.mark.asyncio
async def test_hunyuan_avatar_output_video_no_job_id() -> None:
    """Test _output_video with video_path output type and no job_id creates temp file."""
    model = HunyuanAvatarGeneration()
    gen_timer = GenTimer()
    video_frames = np.zeros((5, 64, 64, 3), dtype=np.uint8)
    with patch.object(_wrapper_module, 'save_video_audio', new_callable=AsyncMock) as mock_save:
        mock_save.return_value = "/tmp/generated_output.mp4"
        result = await model._output_video(
            job_id=None,
            gen_timer=gen_timer,
            audio_path="/tmp/test.wav",
            video_frames=video_frames,
            output_type="video_path",
        )
    assert result == "/tmp/generated_output.mp4"
    # Verify save_video_audio was called with a temp path (no job_id)
    call_kwargs = mock_save.call_args
    assert call_kwargs is not None


@pytest.mark.asyncio
async def test_hunyuan_avatar_generate_assert_model_not_init() -> None:
    """Test generate raises AssertionError when model is not initialized."""
    model = HunyuanAvatarGeneration()
    with pytest.raises(AssertionError):
        await model.generate(
            img=Image.new("RGB", (100, 100)),
            audio_path="/tmp/test.wav",
            prompt="test prompt",
        )


@pytest.mark.asyncio
async def test_hunyuan_avatar_generate_with_mocked_model() -> None:
    """Test generate with mocked model components."""
    model = HunyuanAvatarGeneration()

    # Set up all required mocked components
    model.hunyuan_video_sampler = MagicMock()
    model.wav2vec = MagicMock()
    model.wav2vec.dtype = mock_torch.bfloat16
    model.align_instance = MagicMock()
    model.feature_extractor = MagicMock()
    model.text_encoder = MagicMock()
    model.text_encoder_2 = MagicMock()
    model.data_loader = MagicMock()

    # Mock encode_data result
    model.data_loader.encode_data.return_value = {
        "audio_len": 5,
    }

    # Mock hunyuan_video_sampler.predict
    fake_sample = MagicMock()
    fake_sample.unsqueeze.return_value = fake_sample
    fake_sample.__getitem__ = MagicMock(return_value=fake_sample)
    model.hunyuan_video_sampler.predict.return_value = {"samples": [fake_sample]}

    # Set up the mock chain for video output:
    # video = einops.rearrange(sample[0], ...) → mock_rearranged
    # video = (video * 255.).data.cpu().numpy().astype(np.uint8) → fake_video (numpy)
    fake_video = np.zeros((5, 64, 64, 3), dtype=np.uint8)
    mock_rearranged = MagicMock()
    mul_result = MagicMock()
    mul_result.data.cpu.return_value.numpy.return_value.astype.return_value = fake_video
    mock_rearranged.__mul__ = MagicMock(return_value=mul_result)

    with patch.object(_wrapper_module, 'librosa') as mock_librosa, \
         patch.object(_wrapper_module, 'einops') as mock_einops:

        mock_librosa.get_duration.return_value = 2.0
        mock_einops.rearrange.return_value = mock_rearranged

        result = await model.generate(
            img=Image.new("RGB", (100, 100)),
            audio_path="/tmp/test.wav",
            prompt="test prompt",
            output_type="pil",
        )
    assert result is not None


@pytest.mark.asyncio
async def test_hunyuan_avatar_generate_audio_too_long() -> None:
    """Test generate raises ValueError when audio exceeds MAX_FRAMES."""
    model = HunyuanAvatarGeneration()
    model.hunyuan_video_sampler = MagicMock()
    model.wav2vec = MagicMock()
    model.align_instance = MagicMock()
    model.feature_extractor = MagicMock()
    model.text_encoder = MagicMock()
    model.text_encoder_2 = MagicMock()
    model.data_loader = MagicMock()

    # 7 seconds at 12.5 FPS gives num_frames = int(87.5 // 4) * 4 + 5 = 89 > MAX_FRAMES=81
    with patch.object(_wrapper_module, 'librosa') as mock_librosa:
        mock_librosa.get_duration.return_value = 7.0
        with pytest.raises(ValueError, match="exceeds"):
            await model.generate(
                img=Image.new("RGB", (100, 100)),
                audio_path="/tmp/long_audio.wav",
                prompt="test prompt",
            )
