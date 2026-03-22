#!/usr/bin/env python3

import sys
import pytest

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.torch_mock import TorchMock

mock_torch = TorchMock()

sys.path.append("wrapper")
sys.path.append("wrapper/flux2")
sys.path.append("wrapper/flux")

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
    'xfuser.model_executor.models.transformers.transformer_flux2': MagicMock(),
    'xfuser.model_executor.layers': MagicMock(),
    'xfuser.model_executor.layers.attention_processor': MagicMock(),
    'diffusers': MagicMock(),
}
mock_modules.update(mock_torch.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from flux2.wrapper_flux2 import Flux2Generation


@pytest.mark.asyncio
async def test_wrapper_flux2() -> None:
    model = Flux2Generation()
    assert model is not None
    assert model.model_name == "flux2"
    assert model.status == "initializing"

    with pytest.raises(AttributeError):
        await model.generate(
            width=128,
            height=80,
            prompt="test prompt")

    # Capture the mock Flux2Pipeline and transformer wrapper so we can assert
    # the new sharding behaviour (device_map="balanced" on the transformer;
    # individual .to() calls for VAE / text encoders; NO full pipeline.to()).
    mock_pipeline_cls = mock_modules['diffusers'].Flux2Pipeline
    mock_transformer_cls = mock_modules[
        'xfuser.model_executor.models.transformers.transformer_flux2'
    ].xFuserFlux2Transformer2DWrapper

    # Pre-access mock sub-component attributes so we hold stable references for
    # assertions after init() calls .to() on each of them.
    mock_pipeline_instance = mock_pipeline_cls.from_pretrained.return_value
    vae_mock = mock_pipeline_instance.vae
    text_encoder_mock = mock_pipeline_instance.text_encoder
    text_encoder_2_mock = mock_pipeline_instance.text_encoder_2

    model.init()
    assert model.status == "ok"

    # Verify transformer was loaded with device_map="balanced"
    _, transformer_kwargs = mock_transformer_cls.from_pretrained.call_args
    assert transformer_kwargs.get("device_map") == "balanced", (
        "Transformer must be loaded with device_map='balanced' to shard across GPUs"
    )

    # Verify the full pipeline was NOT moved to a single device (would cause OOM)
    mock_pipeline_instance.to.assert_not_called()

    # Verify non-transformer components were individually moved to the primary device
    vae_mock.to.assert_called_once()
    text_encoder_mock.to.assert_called_once()
    text_encoder_2_mock.to.assert_called_once()

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
    image = await model.generate(
        width=15,
        height=17,
        prompt="Test prompt")

    del model
