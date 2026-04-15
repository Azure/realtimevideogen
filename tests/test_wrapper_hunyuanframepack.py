#!/usr/bin/env python3

import sys
import pytest

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.torch_mock import TorchMock
from tests.diffusers_mock import DiffusersMock

from PIL import Image

sys.path.append("wrapper")
sys.path.append("wrapper/hunyuanframepackf1")
sys.path.append("wrapper/hunyuanframepack")

mock_torch = TorchMock()
mock_diffusers = DiffusersMock()

mock_modules = {
    'nvidia_smi': MagicMock(),
    'imageio': MagicMock(),
    'cv2': MagicMock(),
    'torch': mock_torch,
    'torchvision': MagicMock(),
    'torchvision.transforms': MagicMock(),
    'torchvision.transforms.functional': MagicMock(),
    'xfuser': MagicMock(),
    'xfuser.envs': MagicMock(),
    'xfuser.config': MagicMock(),
    'xfuser.core': MagicMock(),
    'xfuser.core.distributed': MagicMock(),
    'xfuser.core.cache_manager': MagicMock(),
    'xfuser.core.cache_manager.cache_manager': MagicMock(),
    'xfuser.core.long_ctx_attention': MagicMock(),
    'xfuser.model_executor': MagicMock(),
    'xfuser.model_executor.layers': MagicMock(),
    'xfuser.model_executor.layers.attention_processor': MagicMock(),
    'transformers': MagicMock(),
    'einops': MagicMock(),
    'flash_attn': MagicMock(),
}
mock_modules.update({
    'diffusers_helper': MagicMock(),
    'diffusers_helper.hunyuan': MagicMock(),
    'diffusers_helper.utils': MagicMock(),
    'diffusers_helper.clip_vision': MagicMock(),
    'diffusers_helper.models': MagicMock(),
    'diffusers_helper.models.hunyuan_video_packed': MagicMock(),
    'diffusers_helper.pipelines': MagicMock(),
    'diffusers_helper.pipelines.k_diffusion_hunyuan': MagicMock(),
    'diffusers_helper.k_diffusion': MagicMock(),
    'diffusers_helper.k_diffusion.uni_pc_fm': MagicMock(),
    'diffusers_helper.k_diffusion.wrapper': MagicMock(),
})
mock_modules.update(mock_torch.get_sub_modules())
mock_modules.update(mock_diffusers.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from image_utils import img_to_base64
    from hunyuanframepack.wrapper_hunyuanframepack import HunyuanFramepackGeneration


@pytest.mark.asyncio
async def test_hunyuan_framepack() -> None:
    model = HunyuanFramepackGeneration()
    assert model is not None
    assert model.model_name == "hunyuanframepack"
    assert model.status == "initializing"

    model.init()
    assert model.status == "ok"
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

    with pytest.raises(ValueError, match="not enough values to unpack"):
        # TODO implement fixtures for diffusers_helper
        await model.warmup()

    with pytest.raises(ValueError, match="not enough values to unpack"):
        # TODO implement fixtures for diffusers_helper
        await model.generate(
            img=img,
            prompt="test prompt",
            output_type="video_frames")
        # assert video_frames is not None

    del model


@pytest.mark.asyncio
async def test_hunyuan_framepack_get_rest_args_negative_values() -> None:
    """get_rest_args raises ValueError for non-positive numeric parameters."""
    model = HunyuanFramepackGeneration()
    model.init()

    img = Image.new("RGB", (40, 30))
    img_base64 = img_to_base64(img)
    base = {"img": img_base64, "prompt": "test prompt"}

    with pytest.raises(ValueError, match="num_frames"):
        await model.get_rest_args({**base, "num_frames": -3})

    with pytest.raises(ValueError, match="num_frames"):
        await model.get_rest_args({**base, "num_frames": 0})

    with pytest.raises(ValueError, match="height"):
        await model.get_rest_args({**base, "height": -480})

    with pytest.raises(ValueError, match="height"):
        await model.get_rest_args({**base, "height": 0})

    with pytest.raises(ValueError, match="width"):
        await model.get_rest_args({**base, "width": -640})

    with pytest.raises(ValueError, match="width"):
        await model.get_rest_args({**base, "width": 0})

    with pytest.raises(ValueError, match="sampling_steps"):
        await model.get_rest_args({**base, "sampling_steps": -10})

    with pytest.raises(ValueError, match="sampling_steps"):
        await model.get_rest_args({**base, "sampling_steps": 0})

    with pytest.raises(ValueError, match="latent_window_size"):
        await model.get_rest_args({**base, "latent_window_size": -1})

    with pytest.raises(ValueError, match="latent_window_size"):
        await model.get_rest_args({**base, "latent_window_size": 0})

    with pytest.raises(ValueError, match="video_seconds"):
        await model.get_rest_args({**base, "video_seconds": -1.0})

    with pytest.raises(ValueError, match="video_seconds"):
        await model.get_rest_args({**base, "video_seconds": 0.0})

    del model


@pytest.mark.asyncio
async def test_hunyuan_framepack_generate_proceeds_past_text_encoding() -> None:
    """generate() proceeds past _encode_text when it is patched to return mock values."""
    model = HunyuanFramepackGeneration()
    model.init()

    img = Image.new("RGB", (768, 512))
    six_mocks = tuple(MagicMock() for _ in range(6))

    with patch.object(model, "_encode_text", return_value=six_mocks):
        with pytest.raises(ValueError):
            await model.generate(
                img=img,
                prompt="test prompt",
                height=512,
                width=768,
            )

    del model
