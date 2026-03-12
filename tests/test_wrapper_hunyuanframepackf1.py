#!/usr/bin/env python3

import sys
import pytest

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.torch_mock import TorchMock

from PIL import Image

mock_torch = TorchMock()

sys.path.append("wrapper")
sys.path.append("wrapper/hunyuanframepackf1")
sys.path.append("wrapper/hunyuanframepack")

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
    'diffusers': MagicMock(),
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
    'diffusers.models': MagicMock(),
    'diffusers.models.attention': MagicMock(),
    'diffusers.models.transformers': MagicMock(),
    'diffusers.models.transformers.transformer_hunyuan_video': MagicMock(),
    'diffusers.models.transformers.transformer_2d': MagicMock(),
    'flash_attn': MagicMock(),
}
mock_modules.update(mock_torch.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from image_utils import img_to_base64
    from hunyuanframepackf1.wrapper_hunyuanframepackf1 import HunyuanFramepackF1Generation


@pytest.mark.asyncio
async def test_hunyuanframepackf1() -> None:
    model = HunyuanFramepackF1Generation()
    assert model is not None
    assert model.model_name == "hunyuanframepackf1"
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

    with pytest.raises(ValueError):
        # TODO implement fixtures for diffusers_helper
        await model.warmup()

    with pytest.raises(ValueError):
        # TODO implement fixtures for diffusers_helper
        await model.generate(
            img=img,
            prompt="test prompt",
            output_type="video_frames")

    del model
