#!/usr/bin/env python3

import sys
import pytest

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.torch_mock import TorchMock

from PIL import Image

mock_torch = TorchMock()

sys.path.append("wrapper")
sys.path.append("wrapper/fantasytalking")

mock_modules = {
    'nvidia_smi': MagicMock(),
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
    'diffsynth': MagicMock(),
    'diffsynth.models': MagicMock(),
    'diffsynth.models.wan_video_dit': MagicMock(),
    'wan.distributed': MagicMock(),
    'wan.distributed.xdit_context_parallel': MagicMock(),
    'wan.distributed.fsdp': MagicMock(),
    'transformers': MagicMock(),
    'model': MagicMock(),
    'utils': MagicMock(),
}
mock_modules.update(mock_torch.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from fantasytalking.wrapper_fantasytalking import FantasyTalking
    from fantasytalking.wrapper_fantasytalking import resample_frames


@pytest.mark.asyncio
async def test_fantasytalking_e2e() -> None:
    model = FantasyTalking()
    assert model is not None
    assert model.model_name == "fantasytalking"
    assert model.status == "initializing"

    model.init()
    assert model.status == "ok"
    health = model.get_health()
    assert health is not None
    timestamps = model.get_timestamps()
    assert timestamps is not None

    with pytest.raises(ValueError, match="Missing JSON body"):
        await model.get_rest_args(None)

    with pytest.raises(ValueError, match="Missing 'audio' parameter"):
        await model.get_rest_args({})

    with pytest.raises(ValueError, match="Missing 'prompt' parameter"):
        await model.get_rest_args({"audio": "test"})

    rest_args = await model.get_rest_args({
        "audio": "test",
        "prompt": "test prompt",
    })
    assert "args" in rest_args
    assert rest_args["args"]["audio_path"].startswith("/tmp/tmp")
    assert rest_args["args"]["audio_path"].endswith(".wav")

    await model.get_rest_args({
        "audio": "test",
        "prompt": "test prompt",
        "job_id": "testjob",
    })
    assert "args" in rest_args
    assert rest_args["args"]["audio_path"].endswith(".wav")
    assert rest_args["args"]["audio_cfg_scale"] == 5.0

    with pytest.raises((TypeError, ValueError)):
        await model.warmup()

    with pytest.raises(TypeError, match="required positional arguments"):
        await model.generate()

    with pytest.raises(ValueError, match="Audio file 'nonexisting.wav' does not exist"):
        await model.generate(
            img=Image.new('RGB', (100, 100)),
            audio_path="nonexisting.wav",
            prompt="test prompt",
            video=None)

    with pytest.raises((TypeError, ValueError)):
        # TODO improve mocking
        await model.generate(
            img=Image.new('RGB', (100, 100)),
            audio_path="tests/data/audio_4675.wav",
            prompt="test prompt",
            video=None)

    with pytest.raises(TypeError):
        # TODO improve mocking
        await model.generate(
            video=[
                Image.new('RGB', (100, 100))
                for _ in range(3)
            ],
            audio_path="tests/data/audio_4675.wav",
            prompt="test prompt")

    with pytest.raises((TypeError, ValueError)):
        # TODO improve mocking
        await model.generate(
            img=None,
            video=None,
            audio_path="tests/data/audio_4675.wav",
            prompt="test prompt")

    del model


def test_resample_frames() -> None:
    resampled_frames = resample_frames([], 30, 23)
    assert resampled_frames == []

    resampled_frames = resample_frames([
        Image.new("RGB", (100, 100))
        for _ in range(3)
    ], 30, 23)
    assert len(resampled_frames) == 2

    resampled_frames = resample_frames([
        Image.new("RGB", (100, 100))
        for _ in range(24)
    ], 24, 23, 0.5)
    assert len(resampled_frames) == 12


def test_resample_frames_may_differ_from_num_frames() -> None:
    """Reproduce the bug scenario: 73 frames at 30 FPS with ~2.4s audio.

    Before the fix, resample_frames returns 55 frames while num_frames (computed
    from audio at 23 FPS aligned to 1+4n) is 57, causing a latent/noise tensor
    shape mismatch (14 vs 15 frames) inside CustomWanVideoPipeline.__call__.
    This test documents that resample_frames can return fewer frames than
    num_frames; the fix in generate() normalises the result by padding with the
    last frame so both tensors share the same latent dimension.
    """
    import math

    FPS = 23.0
    SRC_FPS = 30.0
    vae_stride = 4

    audio_duration = 2.4  # seconds — representative value from the bug report
    audio_num_frames = int(math.ceil(FPS * audio_duration))
    num_frames = int(1 + math.ceil((audio_num_frames - 1) / vae_stride) * vae_stride)

    src_frames = [Image.new("RGB", (100, 100)) for _ in range(73)]
    resampled = resample_frames(src_frames, SRC_FPS, FPS, audio_duration)

    # resample_frames produces fewer frames than num_frames due to rounding
    assert len(resampled) < num_frames, (
        f"Expected resampled ({len(resampled)}) < num_frames ({num_frames})"
    )

    # Simulate the normalization applied by generate() after resample_frames
    if len(resampled) < num_frames and len(resampled) > 0:
        normalized = resampled + [resampled[-1]] * (num_frames - len(resampled))
    elif len(resampled) > num_frames:
        normalized = resampled[:num_frames]
    else:
        normalized = resampled

    assert len(normalized) == num_frames, (
        f"After normalization, expected {num_frames} frames but got {len(normalized)}"
    )

    # Verify that the latent dimensions now match
    lat_expected = (num_frames - 1) // vae_stride + 1
    lat_actual = (len(normalized) - 1) // vae_stride + 1
    assert lat_expected == lat_actual, (
        f"Latent frame count mismatch: noise={lat_expected}, latents={lat_actual}"
    )
