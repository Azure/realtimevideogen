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
sys.path.append("wrapper/wan22")

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
    "wan.distributed.sequence_parallel": MagicMock(),
}
mock_modules.update(mock_torch.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from image_utils import img_to_base64
    from wan22.wrapper_wan22 import Wan22VideoGeneration


@pytest.mark.asyncio
async def test_wrapper_wan() -> None:
    model = Wan22VideoGeneration()
    assert model is not None
    assert model.model_name == "wan22"

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

    with pytest.raises(ValueError):
        # TODO implement fixture for torch tensor
        # TF.to_tensor(img_resized).sub_(0.5).div_(0.5).to(self.device)
        await model.generate(
            img=img,
            prompt="test prompt",
            output_type="video_frames")

    del model
    gc.collect()
