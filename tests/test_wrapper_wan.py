#!/usr/bin/env python3

import sys
import gc
import pytest

from unittest.mock import patch
from unittest.mock import MagicMock

from PIL import Image

from tests.torch_mock import TorchMock

mock_torch = TorchMock()

sys.path.append("wrapper")
sys.path.append("wrapper/wan")

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
    'wan.distributed.xdit_context_parallel': MagicMock(),
}
mock_modules.update(mock_torch.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from image_utils import img_to_base64
    from wan.wrapper_wan21 import Wan21VideoGeneration
    from wan.vae import WanVAE


@pytest.mark.asyncio
async def test_wrapper_wan() -> None:
    model = Wan21VideoGeneration()
    assert model is not None
    assert model.model_name == "wan"

    model.init()
    health = model.get_health()
    assert health is not None
    timestamps = model.get_timestamps()
    assert timestamps is not None

    with pytest.raises(ValueError):
        await model.get_rest_args(None)
    with pytest.raises(ValueError):
        await model.get_rest_args({})
    img = Image.new("RGB", (40, 30))
    img_base64 = img_to_base64(img)
    await model.get_rest_args({
        "img": img_base64,
        "prompt": "test prompt",
    })

    with pytest.raises(ValueError):
        # TODO implement fixture for torch tensor
        # TF.to_tensor(img_resized).sub_(0.5).div_(0.5).to(self.device)
        await model.warmup()

    with pytest.raises(ValueError, match="not enough values to unpack"):
        # TODO implement fixture for torch tensor
        # TF.to_tensor(img_resized).sub_(0.5).div_(0.5).to(self.device)
        await model.generate(
            img=img,
            prompt="test prompt",
            output_type="video_frames")

    del model
    gc.collect()


@pytest.mark.asyncio
async def test_vae() -> None:
    """Test the VAE wrapper."""
    vae = WanVAE()
    assert vae is not None

    mock_tensor = mock_torch.randn(1, 3, 4, 64, 64)

    encoded_ret = vae.encode(
        videos=[mock_tensor],
        start_frames=1,
        end_frames=0)
    assert encoded_ret is not None
    assert len(encoded_ret) == 1

    decoded_ret = vae.decode(zs=[mock_tensor])
    assert decoded_ret is not None
    assert len(decoded_ret) == 1

    for ret in vae.decode_stream(z=mock_tensor):
        assert ret is not None


@pytest.mark.asyncio
async def test_wan_assert_args() -> None:
    with patch.dict(sys.modules, mock_modules):
        from wan.wrapper_wan21 import Wan21VideoGeneration as _Wan21

    model = _Wan21()
    model.init()
    model.vae_stride = (4, 8, 8)

    # Valid args: 480%8=0, 640%8=0, (5-1)%4=0
    model._assert_args(height=480, width=640, num_frames=5)

    # Height not divisible by vae_stride[1]=8
    with pytest.raises(ValueError, match="Height"):
        model._assert_args(height=481, width=640, num_frames=5)

    # Width not divisible by vae_stride[2]=8
    with pytest.raises(ValueError, match="Width"):
        model._assert_args(height=480, width=641, num_frames=5)

    # num_frames: (2-1)%4 != 0
    with pytest.raises(ValueError):
        model._assert_args(height=480, width=640, num_frames=2)

    # Too many frames: > 1+80
    with pytest.raises(ValueError):
        model._assert_args(height=480, width=640, num_frames=100)

    # vae_stride not set
    model.vae_stride = None
    with pytest.raises(ValueError, match="VAE stride"):
        model._assert_args(height=480, width=640, num_frames=5)


@pytest.mark.asyncio
async def test_wan_get_rest_args_full() -> None:
    with patch.dict(sys.modules, mock_modules):
        from wan.wrapper_wan21 import Wan21VideoGeneration as _Wan21
        from image_utils import img_to_base64 as _img_to_base64

    model = _Wan21()
    model.init()

    img = Image.new("RGB", (40, 30))
    img_base64 = _img_to_base64(img)

    # Missing img still raises ValueError
    with pytest.raises(ValueError):
        await model.get_rest_args({"prompt": "test"})

    # With num_frames, width, height, seed
    result = await model.get_rest_args({
        "img": img_base64,
        "prompt": "test prompt",
        "num_frames": 17,
        "width": 640,
        "height": 480,
    })
    assert result["args"]["num_frames"] == 17
    assert result["args"]["width"] == 640
    assert result["args"]["height"] == 480
    assert result["args"]["prompt"] == "test prompt"


@pytest.mark.asyncio
async def test_wan_get_rest_args_extra_params() -> None:
    """Test get_rest_args with neg_prompt, sampling_steps, output_type and video_seconds."""
    with patch.dict(sys.modules, mock_modules):
        from wan.wrapper_wan21 import Wan21VideoGeneration as _Wan21
        from image_utils import img_to_base64 as _img_to_base64

    model = _Wan21()
    model.init()

    img = Image.new("RGB", (40, 30))
    img_base64 = _img_to_base64(img)

    # neg_prompt, sampling_steps, output_type
    result = await model.get_rest_args({
        "img": img_base64,
        "prompt": "test prompt",
        "neg_prompt": "bad quality",
        "sampling_steps": 20,
        "output_type": "video_path",
    })
    assert result["args"]["neg_prompt"] == "bad quality"
    assert result["args"]["sampling_steps"] == 20
    assert result["args"]["output_type"] == "video_path"

    # video_seconds branch: converts seconds to num_frames using FPS and vae_stride
    result_secs = await model.get_rest_args({
        "img": img_base64,
        "prompt": "test prompt",
        "video_seconds": 2.0,
    })
    # FPS=16, vae_stride[0]=4: num_frames = 1 + ((int(2.0*16)-1) // 4) * 4 = 29
    assert result_secs["args"]["num_frames"] == 29


def test_wan_assert_model_init() -> None:
    """Test that _assert_model_init raises when model components are not loaded."""
    with patch.dict(sys.modules, mock_modules):
        from wan.wrapper_wan21 import Wan21VideoGeneration as _Wan21

    model = _Wan21()
    # Status is "initializing" (init() not called yet), base raises ValueError
    with pytest.raises(ValueError, match="Model not initialized"):
        model._assert_model_init()


def test_wan_model_compile_no_op() -> None:
    """Test model_compile returns immediately when torch_compile=False."""
    with patch.dict(sys.modules, mock_modules):
        from wan.wrapper_wan21 import Wan21VideoGeneration as _Wan21

    model = _Wan21()
    model.init()
    model.torch_compile = False
    model.model_compile()  # should not raise


def test_wan_model_compile_with_torch_compile() -> None:
    """Test model_compile calls torch.compile when torch_compile=True."""
    with patch.dict(sys.modules, mock_modules):
        from wan.wrapper_wan21 import Wan21VideoGeneration as _Wan21

    model = _Wan21()
    model.init()
    model.torch_compile = True
    model.model_compile()  # should call torch.compile (mocked, no-op)


@pytest.mark.asyncio
async def test_wan_assert_model_init_text_encoder_none() -> None:
    """Test _assert_model_init raises when text_encoder is None."""
    with patch.dict(sys.modules, mock_modules):
        from wan.wrapper_wan21 import Wan21VideoGeneration as _Wan21

    model = _Wan21()
    model.init()
    model.text_encoder = None
    with pytest.raises(ValueError, match="Text encoder"):
        model._assert_model_init()


@pytest.mark.asyncio
async def test_wan_assert_model_init_vae_none() -> None:
    """Test _assert_model_init raises when vae is None."""
    with patch.dict(sys.modules, mock_modules):
        from wan.wrapper_wan21 import Wan21VideoGeneration as _Wan21

    model = _Wan21()
    model.init()
    model.vae = None
    with pytest.raises(ValueError, match="VAE"):
        model._assert_model_init()


@pytest.mark.asyncio
async def test_wan21_assert_model_init_image_encoder_none() -> None:
    """Test Wan21 _assert_model_init raises when image_encoder is None."""
    with patch.dict(sys.modules, mock_modules):
        from wan.wrapper_wan21 import Wan21VideoGeneration as _Wan21

    model = _Wan21()
    model.init()
    model.image_encoder = None
    with pytest.raises(ValueError, match="Image encoder"):
        model._assert_model_init()


@pytest.mark.asyncio
async def test_wan21_assert_model_init_image_encoder_model_none() -> None:
    """Test Wan21 _assert_model_init raises when image_encoder.model is None."""
    with patch.dict(sys.modules, mock_modules):
        from wan.wrapper_wan21 import Wan21VideoGeneration as _Wan21

    model = _Wan21()
    model.init()
    # Use a fresh MagicMock to avoid polluting the shared mock state
    fresh_encoder = MagicMock()
    fresh_encoder.model = None
    model.image_encoder = fresh_encoder
    with pytest.raises(ValueError, match="Image encoder"):
        model._assert_model_init()


@pytest.mark.asyncio
async def test_wan21_assert_model_init_dit_model_none() -> None:
    """Test Wan21 _assert_model_init raises when dit model is None."""
    with patch.dict(sys.modules, mock_modules):
        from wan.wrapper_wan21 import Wan21VideoGeneration as _Wan21

    model = _Wan21()
    model.init()
    # Ensure image_encoder and its .model remain truthy; only null out the DiT model
    model.image_encoder = MagicMock()
    model.image_encoder.model = MagicMock()
    model.model = None
    with pytest.raises(ValueError, match="DiT model"):
        model._assert_model_init()


@pytest.mark.asyncio
async def test_wan_output_video_tensor() -> None:
    """Test _output_video returns tensor directly for output_type='tensor'."""
    with patch.dict(sys.modules, mock_modules):
        from wan.wrapper_wan21 import Wan21VideoGeneration as _Wan21

    model = _Wan21()
    model.init()
    gen_timer = model._new_gen_timer(None)
    mock_video = MagicMock()

    result = await model._output_video(None, gen_timer, mock_video, "tensor")
    assert result is mock_video


@pytest.mark.asyncio
async def test_wan_output_video_unknown_type() -> None:
    """Test _output_video returns None for an unknown output_type."""
    with patch.dict(sys.modules, mock_modules):
        from wan.wrapper_wan21 import Wan21VideoGeneration as _Wan21

    model = _Wan21()
    model.init()
    gen_timer = model._new_gen_timer(None)
    mock_video = MagicMock()

    result = await model._output_video(None, gen_timer, mock_video, "not_a_real_type")
    assert result is None


@pytest.mark.asyncio
async def test_wan_get_rest_args_steps_fallback() -> None:
    """Test get_rest_args uses 'steps' when sampling_steps=0."""
    with patch.dict(sys.modules, mock_modules):
        from wan.wrapper_wan21 import Wan21VideoGeneration as _Wan21
        from image_utils import img_to_base64 as _img_to_base64

    model = _Wan21()
    model.init()

    img = Image.new("RGB", (40, 30))
    img_base64 = _img_to_base64(img)

    result = await model.get_rest_args({
        "img": img_base64,
        "prompt": "test",
        "sampling_steps": 0,
        "steps": 15,
    })
    assert result["args"]["sampling_steps"] == 15


@pytest.mark.asyncio
async def test_wan_get_rest_args_video_seconds_no_vae_stride() -> None:
    """Test get_rest_args raises when video_seconds is set but vae_stride is None."""
    with patch.dict(sys.modules, mock_modules):
        from wan.wrapper_wan21 import Wan21VideoGeneration as _Wan21
        from image_utils import img_to_base64 as _img_to_base64

    model = _Wan21()
    model.init()
    model.vae_stride = None

    img = Image.new("RGB", (40, 30))
    img_base64 = _img_to_base64(img)

    with pytest.raises(ValueError, match="VAE stride"):
        await model.get_rest_args({
            "img": img_base64,
            "prompt": "test",
            "video_seconds": 2.0,
        })


@pytest.mark.asyncio
async def test_wan_assert_args_num_frames_too_large() -> None:
    """Test _assert_args raises when num_frames is too large (passes modulo check)."""
    with patch.dict(sys.modules, mock_modules):
        from wan.wrapper_wan21 import Wan21VideoGeneration as _Wan21

    model = _Wan21()
    model.init()
    model.vae_stride = (4, 8, 8)

    # num_frames=85: (85-1)%4=0 and 85 > 1+80=81 → hits the upper-bound error
    with pytest.raises(ValueError, match="num_frames"):
        model._assert_args(height=480, width=640, num_frames=85)


@pytest.mark.asyncio
async def test_wan_get_rest_args_img_not_string() -> None:
    """Test get_rest_args raises when img is truthy but not a string."""
    with patch.dict(sys.modules, mock_modules):
        from wan.wrapper_wan21 import Wan21VideoGeneration as _Wan21

    model = _Wan21()
    model.init()

    with pytest.raises(ValueError, match="'img' parameter must be a base64-encoded string"):
        await model.get_rest_args({"img": 12345, "prompt": "test"})


@pytest.mark.asyncio
async def test_wan_get_rest_args_missing_prompt() -> None:
    """Test get_rest_args raises when prompt is missing."""
    with patch.dict(sys.modules, mock_modules):
        from wan.wrapper_wan21 import Wan21VideoGeneration as _Wan21
        from image_utils import img_to_base64 as _img_to_base64

    model = _Wan21()
    model.init()

    img = Image.new("RGB", (40, 30))
    img_base64 = _img_to_base64(img)

    with pytest.raises(ValueError, match="Missing 'prompt' parameter"):
        await model.get_rest_args({"img": img_base64})


@pytest.mark.asyncio
async def test_wan_output_video_pil() -> None:
    """Test _output_video calls _tensor_to_pil for output_type='pil'."""
    with patch.dict(sys.modules, mock_modules):
        from wan.wrapper_wan21 import Wan21VideoGeneration as _Wan21

    model = _Wan21()
    model.init()
    gen_timer = model._new_gen_timer(None)
    mock_video = MagicMock()
    mock_pil_frames = [MagicMock()]

    with patch.object(model, '_tensor_to_pil', return_value=mock_pil_frames) as mock_pil:
        result = await model._output_video(None, gen_timer, mock_video, "pil")
        mock_pil.assert_called_once_with(mock_video)
        assert result is mock_pil_frames


@pytest.mark.asyncio
async def test_wan_output_video_video_path() -> None:
    """Test _output_video returns a video path for output_type='video_path'."""
    with patch.dict(sys.modules, mock_modules):
        from wan.wrapper_wan21 import Wan21VideoGeneration as _Wan21

    model = _Wan21()
    model.init()
    gen_timer = model._new_gen_timer(None)
    mock_video = MagicMock()

    with patch.object(model, '_save_video', return_value=None):
        result = await model._output_video("test_job", gen_timer, mock_video, "video_path")
        assert result == "/tmp/test_job.mp4"


@pytest.mark.asyncio
async def test_wan_output_video_video_binary() -> None:
    """Test _output_video returns video bytes for output_type='video_binary'."""
    import tempfile
    import os
    with patch.dict(sys.modules, mock_modules):
        from wan.wrapper_wan21 import Wan21VideoGeneration as _Wan21

    model = _Wan21()
    model.init()
    gen_timer = model._new_gen_timer(None)
    mock_video = MagicMock()

    # Create a temp file so aiofiles.open can read it
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(b"fake video data")
        tmp_path = tmp.name

    try:
        with patch.object(model, '_save_video', return_value=None):
            # Patch NamedTemporaryFile so _output_video uses our pre-created file
            mock_ntf_instance = MagicMock()
            mock_ntf_instance.name = tmp_path
            with patch('tempfile.NamedTemporaryFile', return_value=mock_ntf_instance):
                result = await model._output_video(None, gen_timer, mock_video, "video_binary")
                assert isinstance(result, bytes)
                assert result == b"fake video data"
    finally:
        os.unlink(tmp_path)


def test_wan_vae_decode() -> None:
    """Test vae_decode passes through correctly with a properly mocked tensor."""
    with patch.dict(sys.modules, mock_modules):
        from wan.wrapper_wan21 import Wan21VideoGeneration as _Wan21

    model = _Wan21()
    model.init()

    # Create a fake tensor that passes isinstance(x, torch.Tensor)
    class FakeLatent(mock_torch.Tensor):  # type: ignore[name-defined]
        def to(self, *args: object, **kwargs: object) -> 'FakeLatent':
            return self

        def dim(self) -> int:
            return 4

        @property
        def shape(self) -> tuple:
            return (20, 21, 68, 90)

    fake_lat = FakeLatent()
    mock_pixel = MagicMock()
    model.vae.decode = MagicMock(return_value=[mock_pixel])

    result = model.vae_decode(fake_lat)
    assert result is mock_pixel


def test_wan_vae_encode() -> None:
    """Test vae_encode passes through correctly with a properly mocked tensor."""
    with patch.dict(sys.modules, mock_modules):
        from wan.wrapper_wan21 import Wan21VideoGeneration as _Wan21

    model = _Wan21()
    model.init()

    # Create a fake tensor that passes isinstance(x, torch.Tensor)
    class FakePixels(mock_torch.Tensor):  # type: ignore[name-defined]
        def to(self, *args: object, **kwargs: object) -> 'FakePixels':
            return self

        def dim(self) -> int:
            return 4

        @property
        def shape(self) -> tuple:
            return (21, 3, 68, 90)  # shape[1] == 3 (RGB channels)

    fake_pix = FakePixels()
    mock_latent = MagicMock()
    model.vae.encode = MagicMock(return_value=[mock_latent])

    result = model.vae_encode(fake_pix)
    assert result is mock_latent
