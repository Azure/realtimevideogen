#!/usr/bin/env python3

import sys
import pytest

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.torch_mock import TorchMock

mock_torch = TorchMock()

sys.path.append("flux")

with patch.dict(sys.modules, {
    'nvidia_smi': MagicMock(),
    'colorlog': MagicMock(),
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
    'diffusers': MagicMock(),
    'transformers': MagicMock(),
    'janus': MagicMock(),
    'janus.models': MagicMock(),
}):
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
    await model.get_rest_args({
        "job_id": "unittest",
        "prompt": "Test prompt",
        "width": 80,
        "height": 60,
        "seed": 7,
    })

    with pytest.raises(ValueError):
        await model.warmup()

    with pytest.raises(ValueError):
        await model.generate(prompt="Test prompt")

    del model
