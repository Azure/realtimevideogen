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
    # num_frames should be computed from video_seconds
    assert result_secs["args"]["num_frames"] >= 1


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
