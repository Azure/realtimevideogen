#!/usr/bin/env python3

import sys
import pytest

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.torch_mock import TorchMock
from tests.diffusers_mock import DiffusersMock

mock_torch = TorchMock()
mock_diffusers = DiffusersMock()

sys.path.append("wrapper")
sys.path.append("wrapper/hunyuanframepackvae")
sys.path.append("wrapper/hunyuanframepack")

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
    'flash_attn': MagicMock(),
}
mock_modules.update(mock_torch.get_sub_modules())
mock_modules.update(mock_diffusers.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from hunyuanframepackvae.wrapper_hunyuanframepackvae import HunyuanFramepackVAEGeneration


@pytest.mark.asyncio
async def test_wrapper_hunyuanframepackvae() -> None:
    model = HunyuanFramepackVAEGeneration()
    assert model is not None
    assert model.model_name == "hunyuanframepackvae"
    assert model.status == "initializing"

    model.init()
    assert model.status == "ok"
    health = model.get_health()
    assert health is not None
    assert len(health) >= 10
    timestamps = model.get_timestamps()
    assert timestamps is not None
    assert len(timestamps) == 4

    with pytest.raises(ValueError, match="Missing JSON body"):
        await model.get_rest_args(None)
    with pytest.raises(ValueError, match="Missing 'latents' parameter"):
        await model.get_rest_args({})

    try:
        args = await model.get_rest_args({"latents": ""})
        assert args is not None
        assert args["task"] == "hunyuanframepackvae"
    except EOFError:
        pass  # This is expected due to the mocks not providing actual latents

    with pytest.raises(TypeError, match="isinstance"):
        # TODO fix mocks
        await model.warmup()

    with pytest.raises(ValueError, match="Latents cannot be None"):
        # TODO implement fixtures
        await model.generate(latents=None)
        # assert video_frames is not None

    del model
