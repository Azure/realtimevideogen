#!/usr/bin/env python3

import sys
import pytest

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.torch_mock import TorchMock
from tests.diffusers_mock import DiffusersMock

from PIL import Image

mock_torch = TorchMock()
mock_diffusers = DiffusersMock()

sys.path.append("wrapper")
sys.path.append("wrapper/flux")
sys.path.append("wrapper/fluxupscaler")

mock_modules = {
    'nvidia_smi': MagicMock(),
    'imageio': MagicMock(),
    'cv2': MagicMock(),
    'torch': mock_torch,
    'xfuser': MagicMock(),
    'xfuser.config': MagicMock(),
    'xfuser.core': MagicMock(),
    'xfuser.core.distributed': MagicMock(),
    'xfuser.model_executor': MagicMock(),
    'xfuser.model_executor.models': MagicMock(),
    'xfuser.model_executor.models.transformers.transformer_flux': MagicMock(),
    'xfuser.model_executor.layers': MagicMock(),
    'xfuser.model_executor.layers.attention_processor': MagicMock(),
}
mock_modules.update(mock_torch.get_sub_modules())
mock_modules.update(mock_diffusers.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from image_utils import img_to_base64
    from fluxupscaler.wrapper_fluxupscaler import FluxUpscalerGeneration


@pytest.mark.asyncio
async def test_wrapper_fluxupscaler() -> None:
    model = FluxUpscalerGeneration()
    assert model is not None
    assert model.model_name == "fluxupscaler"
    assert model.status == "initializing"

    img = Image.new("RGB", (40, 30))
    img_base64 = img_to_base64(img)

    with pytest.raises(ValueError, match="Model not initialized"):
        await model.generate(
            img=img,
            width=128,
            height=80,
            prompt="test prompt")

    model.init()
    assert model.status == "ok"

    health = model.get_health()
    assert health is not None
    timestamps = model.get_timestamps()
    assert timestamps is not None

    await model.get_rest_args({})

    await model.get_rest_args({
        "img": img_base64,
    })

    with pytest.raises(ValueError, match="Missing JSON body"):
        await model.get_rest_args(None)

    # Success case
    await model.get_rest_args({
        "job_id": "unittest",
        "prompt": "Test prompt",
        "width": 80,
        "height": 60,
        "img": img_base64
    })

    await model.warmup()

    image = await model.generate(
        img=img,
        width=256,
        height=160,
        prompt="Test prompt",
    )
    assert image is not None
    assert isinstance(image, Image.Image)

    # Bad cases: _assert_args only rejects when world_size > 1
    model.world_size = 2
    with pytest.raises(ValueError, match="48x48 not supported for 2 GPUs"):
        await model.generate(
            img=img,
            width=48,
            height=48,
            prompt="Test prompt",
        )
    with pytest.raises(ValueError, match="208x116 not supported for 2 GPUs"):
        await model.generate(
            img=img,
            width=208,
            height=116,
            prompt="Test prompt",
        )

    del model
