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
sys.path.append("wrapper/januspro")

mock_modules = {
    'nvidia_smi': MagicMock(),
    'imageio': MagicMock(),
    'cv2': MagicMock(),
    'torch': mock_torch,
    'torch.amp': MagicMock(),
    'torch.distributed': MagicMock(),
    'xfuser': MagicMock(),
    'xfuser.config': MagicMock(),
    'xfuser.core': MagicMock(),
    'xfuser.core.distributed': MagicMock(),
    'xfuser.model_executor': MagicMock(),
    'xfuser.model_executor.layers': MagicMock(),
    'xfuser.model_executor.layers.attention_processor': MagicMock(),
    'transformers': MagicMock(),
    'janus': MagicMock(),
    'janus.models': MagicMock(),
}
mock_modules.update(mock_diffusers.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from januspro.wrapper_januspro import JanusProGeneration


@pytest.mark.asyncio
async def test_wrapper_januspro() -> None:
    model = JanusProGeneration()
    assert model is not None
    assert model.model_name == "januspro"
    assert model.status == "initializing"

    with pytest.raises(ValueError):
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

    with pytest.raises(ValueError):
        await model.get_rest_args(None)
    with pytest.raises(ValueError):
        await model.get_rest_args({})
    args = await model.get_rest_args({
        "job_id": "unittest",
        "prompt": "Test prompt",
        "width": 80,
        "height": 60,
        "seed": 7,
    })
    assert args is not None
    assert "args" in args
    assert args["args"]["prompt"] == "Test prompt"

    with pytest.raises(ValueError, match="could not broadcast input array from shape"):
        await model.warmup()

    with pytest.raises(ValueError, match="could not broadcast input array from shape"):
        await model.generate(prompt="Test prompt")

    del model
