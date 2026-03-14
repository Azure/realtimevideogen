#!/usr/bin/env python3

import sys
import pytest

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.torch_mock import TorchMock

mock_torch = TorchMock()

sys.path.append("fluxkrea")
sys.path.append("flux")

mock_modules = {
    'nvidia_smi': MagicMock(),
    'colorlog': MagicMock(),
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
    'diffusers': MagicMock(),
}
mock_modules.update(mock_torch.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from fluxkrea.wrapper_fluxkrea import FluxKreaGeneration


@pytest.mark.asyncio
async def test_wrapper_fluxkrea() -> None:
    model = FluxKreaGeneration()
    assert model is not None
    assert model.model_name == "fluxkrea"
    assert model.status == "initializing"

    with pytest.raises(AttributeError):
        await model.generate(
            width=128,
            height=80,
            prompt="test prompt")

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

    with pytest.raises(ValueError):
        await model.get_rest_args(None)
    with pytest.raises(ValueError):
        await model.get_rest_args({})
    await model.get_rest_args({
        "job_id": "unittest",
        "prompt": "Test prompt",
        "width": 80,
        "height": 60,
    })

    await model.warmup()

    # TODO this should fail
    await model.generate(
        width=17,
        height=13,
        prompt="Test prompt")

    image = await model.generate(
        width=256,
        height=256,
        prompt="Test prompt")
    assert image is not None

    image = await model.generate(
        width=256,
        height=160,
        prompt="Test prompt")
    assert image is not None

    # 15x17 not supported for 2 GPUs.
    model.world_size = 2
    # TODO
    # with pytest.raises(ValueError, msg="15x17 not supported for 2 GPUs."):
    image = await model.generate(
        width=15,
        height=17,
        prompt="Test prompt")

    del model
