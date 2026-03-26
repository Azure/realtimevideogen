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
sys.path.append("wrapper/flux")

mock_modules = {
    'torch': mock_torch,
    'nvidia_smi': MagicMock(),
    'colorlog': MagicMock(),
    'imageio': MagicMock(),
    'cv2': MagicMock(),
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
    from flux.wrapper_flux import FluxGeneration


@pytest.mark.asyncio
async def test_wrapper_flux() -> None:
    model = FluxGeneration()
    assert model is not None
    assert model.model_name == "flux"
    assert model.status == "initializing"

    with pytest.raises(ValueError, match="Model not initialized. Current status: initializing."):
        await model.generate(64, 48, "test prompt")

    model.init()
    assert model.status == "ok"

    # Mock pipeline return object
    mock_output = MagicMock()
    mock_output.images = ["image"]
    model.pipeline = MagicMock(return_value=mock_output)
    model.pipeline.vae_scale_factor = 8

    health = model.get_health()
    assert health is not None
    timestamps = model.get_timestamps()
    assert timestamps is not None

    with pytest.raises(ValueError, match="Missing JSON body"):
        await model.get_rest_args(None)
    with pytest.raises(ValueError, match="Missing 'prompt' parameter"):
        await model.get_rest_args({})
    await model.get_rest_args({
        "job_id": "unittest",
        "prompt": "Test prompt",
        "width": 80,
        "height": 60,
        "seed": 7,
    })

    await model.warmup()

    image = await model.generate(
        width=1024,
        height=1024,
        prompt="Test prompt")
    assert image is not None

    # 15x17 not supported for 2 GPUs.
    model.world_size = 2
    # TODO fix
    # with pytest.raises(ValueError, msg="15x17 not supported for 2 GPUs."):
    image = await model.generate(
        width=15,
        height=17,
        prompt="Test prompt")

    del model
