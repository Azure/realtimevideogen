#!/usr/bin/env python3

import sys
import pytest

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.torch_mock import TorchMock

from PIL import Image

mock_torch = TorchMock()

sys.path.append("realesrgan")

with patch.dict(sys.modules, {
    'nvidia_smi': MagicMock(),
    'imageio': MagicMock(),
    'cv2': MagicMock(),
    'RealESRGAN': MagicMock(),
    'torch': mock_torch,
}):
    from image_utils import img_to_base64
    from realesrgan.wrapper_realesrgan import RealESRGANGeneration


@pytest.mark.asyncio
async def test_wrapper_realesrgan() -> None:
    model = RealESRGANGeneration()
    assert model is not None
    assert model.model_name == "realesrgan"
    assert model.status == "initializing"
    assert len(model.models) == 0

    with pytest.raises(ValueError):
        await model.generate(image=None)

    model.init()
    assert model.status == "ok"
    assert len(model.models) == 3

    health = model.get_health()
    assert health is not None
    assert len(health) >= 5
    timestamps = model.get_timestamps()
    assert timestamps is not None
    assert len(timestamps) >= 7

    with pytest.raises(ValueError):
        await model.get_rest_args(None)
    await model.get_rest_args({})
    img = Image.new("RGB", (40, 30))
    img_base64 = img_to_base64(img)
    await model.get_rest_args({
        "job_id": "unittest",
        "img": img_base64,
        "width": 80,
        "height": 60,
    })

    await model.warmup()

    image_resized = await model.generate(
        image=img,
        width=160,
        height=120,
        output_type="pil"
    )
    assert image_resized is not None

    del model


def test_get_model_scaling_factor() -> None:
    """Test that get_model_scaling_factor picks the right (next-larger) scale."""
    model = RealESRGANGeneration()
    model.init()

    # 2x — exact match
    sf = model.get_model_scaling_factor(100, 100, 200, 200)
    assert sf == 2

    # 3x ratio → should round up to 4x
    sf = model.get_model_scaling_factor(100, 100, 300, 300)
    assert sf == 4

    # 4x — exact match
    sf = model.get_model_scaling_factor(100, 100, 400, 400)
    assert sf == 4

    # 5x ratio → should round up to 8x (largest available)
    sf = model.get_model_scaling_factor(100, 100, 500, 500)
    assert sf == 8

    # Asymmetric: width doubles, height quadruples → max factor 4
    sf = model.get_model_scaling_factor(100, 100, 200, 400)
    assert sf == 4

    # Output smaller than input raises ValueError
    with pytest.raises(ValueError, match="must be larger than input"):
        model.get_model_scaling_factor(200, 200, 100, 100)

    # Scaling factor exceeds max (8x)
    with pytest.raises(ValueError, match="Scaling factor"):
        model.get_model_scaling_factor(10, 10, 1000, 1000)

    del model


def test_chunk_list_image() -> None:
    """Test _chunk_list_image distributes frames across ranks correctly."""
    from PIL import Image

    model = RealESRGANGeneration()

    # Single rank (world_size == 1): all frames returned unchanged
    model.world_size = 1
    model.rank = 0
    frames = [Image.new("RGB", (10, 10)) for _ in range(4)]
    result = model._chunk_list_image(frames)
    assert result == frames

    # Multi-rank (world_size == 2): rank 0 gets even-indexed frames
    model.world_size = 2
    model.rank = 0
    result = model._chunk_list_image(frames)
    assert len(result) == 4
    assert result[0] is frames[0]   # index 0 → rank 0
    assert result[1] is None         # index 1 → rank 1
    assert result[2] is frames[2]   # index 2 → rank 0
    assert result[3] is None         # index 3 → rank 1

    # Multi-rank (world_size == 2): rank 1 gets odd-indexed frames
    model.rank = 1
    result = model._chunk_list_image(frames)
    assert result[0] is None
    assert result[1] is frames[1]
    assert result[2] is None
    assert result[3] is frames[3]

    # Empty list always returns empty list
    model.world_size = 2
    model.rank = 0
    result = model._chunk_list_image([])
    assert result == []

    del model
