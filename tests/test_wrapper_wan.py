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
