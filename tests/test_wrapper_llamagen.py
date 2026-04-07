#!/usr/bin/env python3

import sys
import pytest

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.torch_mock import TorchMock

mock_torch = TorchMock()

sys.path.append("wrapper")
with patch.dict(sys.modules, {
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
    'tokenizer.tokenizer_image.vq_model': MagicMock(),
    'language.t5': MagicMock(),
    'autoregressive.models.gpt': MagicMock(),
    'autoregressive.models.generate': MagicMock(),
}):
    from llamagen.wrapper_llamagen import LlamaGenGeneration


@pytest.mark.asyncio
async def test_wrapper_llamagen() -> None:
    model = LlamaGenGeneration()
    assert model is not None
    assert model.model_name == "llamagen"
    assert model.status == "initializing"

    with pytest.raises(AssertionError):  # vq_model not set
        await model.generate(64, 48, "test prompt")

    with pytest.raises(AttributeError, match="'LlamaGenGeneration' object has no attribute 'gpt_type'"):
        model.init()
    assert model.status == "failed"

    # Mock pipeline return object
    mock_output = MagicMock()
    mock_output.images = ["image"]
    model.pipeline = MagicMock(return_value=mock_output)
    model.pipeline.vae_scale_factor = 8
    model.pipeline.gpt_type = "llama"
    model.vq_model_name = "vq_model"
    model.gpt_model_name = "gpt_model"
    model.t5_model_type = "t5_model"
    model.gpt_type = "llama"

    with pytest.raises(FileNotFoundError, match="T5 model directory 'google' does not exist."):
        model.init()

    health = model.get_health()
    assert health is not None
    assert "dtype" in health
    assert "gpu" in health
    assert "world_size" in health

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

    with pytest.raises(AssertionError):
        await model.warmup()

    with pytest.raises(AssertionError):
        await model.generate(prompt="Test prompt")
        # assert image is not None

    del model
