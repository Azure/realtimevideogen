#!/usr/bin/env python3

import sys
import os
import pytest
import logging

from PIL import Image

from typing import Tuple

from unittest.mock import patch
from unittest.mock import MagicMock

# Add current path
sys.path.append(os.getcwd())

from file_utils import read_file_base64

from tests.torch_mock import TorchMock
from tests.test_utils import temp_sys_path

mock_torch = TorchMock()

mock_modules = {}
mock_modules.update(mock_torch.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    with temp_sys_path("apps"):
        from apps.gen_video_chunked import GenVideoChunked
        from apps.gen_video_chunked import SubVideoInfo

        from media_utils import get_video_frames
        from media_utils import empty_audio_file
        from media_utils import get_video_file_info

        from video import HUNYUANFRAMEPACK_FPS

        from tests.streamwise_app.lmm_generator_mock import LMMGeneratorMock
        from tests.streamwise_app.lmm_generator_mock import MOCK_COLORS_RGB


@pytest.mark.asyncio
async def test_gen_slide_video_chunks_noservice() -> None:
    gen = LMMGeneratorMock()
    logger = logging
    gen_video_chunked = GenVideoChunked(
        video_id=0,
        gen=gen,
        job_path="tests/data",
        logger=logger,  # type: ignore[arg-type]
    )

    with pytest.raises(FileNotFoundError, match="Audio file not found: audio.wav"):
        await gen_video_chunked.gen_video_chunked(
            audio_path="audio.wav",
            image=Image.new("RGB", (640, 480), color="blue"),
            prompt="A test slide",
            neg_prompt="",
            width=640,
            height=480,
            num_steps=10,
        )

    # TODO
    with pytest.raises(Exception):  # , match="Error generating video.+hunyuanframepackf1"):
        await gen_video_chunked.gen_video_chunked(
            audio_path="tests/data/audio_4675.wav",
            image=Image.new("RGB", (640, 480), color="blue"),
            prompt="A test slide",
            neg_prompt="",
            width=640,
            height=480,
            num_steps=10,
        )


def test_scene_info() -> None:
    video = []
    video.append(SubVideoInfo(0, 2))
    video.append(SubVideoInfo(2, 5))

    scene0 = video[0]
    assert scene0 is not None
    assert scene0.get_seconds() == 2
    assert scene0.get_frames(fps=23) == (0, 46)
    assert str(scene0) == "SubVideoInfo(0.000-2.000s)"

    scene1 = video[1]
    assert scene1 is not None
    assert scene1.get_seconds() == 3
    assert scene1.get_frames(fps=20) == (40, 100)
    assert str(scene1) == "SubVideoInfo(2.000-5.000s)"

    # assert video.get_num_audio_frames() == 118


@pytest.mark.asyncio
async def test_gen_subvideo() -> None:
    service_manager = MagicMock()
    service_manager.get_service_url = MagicMock(
        return_value="http://mock_service_url:1234"
    )

    # Chunk generator
    gen = LMMGeneratorMock()
    logger = logging
    gen_video_chunked = GenVideoChunked(
        video_id=1,
        gen=gen,
        job_path="tests/data",
        logger=logger,  # type: ignore[arg-type]
    )

    # Chunk components
    subvideo_info = SubVideoInfo(
        start_seconds=0.0,
        end_seconds=4.0,
    )
    assert subvideo_info.get_start_frame(HUNYUANFRAMEPACK_FPS) == 0
    assert subvideo_info.get_end_frame(HUNYUANFRAMEPACK_FPS) == 120  # 4 secs x 30 FPS

    audio_base64 = await read_file_base64("tests/data/audio_4675.wav")
    video_frames = [
        Image.new("RGB", (640, 480), color=color)
        for color in ["blue", "green", "red", "yellow"]
    ]

    # Generation
    subvideo_task = await gen_video_chunked._gen_subvideo(
        subvideo_id=0,
        subvideo_info=subvideo_info,
        video_frames=video_frames,
        audio_base64=audio_base64,
        width=640,
        height=480,
        num_steps=10,
    )
    assert subvideo_task is not None
    subvideo_binary = await subvideo_task
    assert subvideo_binary is not None

    # Check data
    subvideo_frames = await get_video_frames(subvideo_binary)
    assert 90 <= len(subvideo_frames) <= 94  # == 1 + (4.0 * 23)  # 4 secs x 23 FPS = 92
    frame0 = subvideo_frames[0]
    assert frame0.size == (640, 480)

    # Check metadata
    video_file_info = get_video_file_info(subvideo_binary)
    video_info = video_file_info["video"]
    assert video_info["codec"] == "h264"
    assert video_info["duration_seconds"] is not None
    assert 3.9 <= video_info["duration_seconds"] <= 4.1  # 4 seconds


@pytest.mark.asyncio
async def test_gen_slide_video_chunks() -> None:
    service_manager = MagicMock()
    service_manager.get_service_url = MagicMock(
        return_value="http://mock_service_url:1234"
    )
    # job_id = "test_gen_slide_video_chunks"
    # job = StreamPersonaJob(job_id, service_manager)
    # Mocking generation
    # job = await _mock_generation(job)

    gen = LMMGeneratorMock()
    logger = logging
    gen_video_chunked = GenVideoChunked(
        video_id=2,
        gen=gen,
        job_path="tests/data",
        logger=logger,  # type: ignore[arg-type]
    )

    # Failure case: audio too short (4.6s < 5s minimum)
    with pytest.raises(ValueError, match="Audio too short for chunked generation"):
        await gen_video_chunked.gen_video_chunked(
            audio_path="tests/data/audio_4675.wav",
            image=Image.new("RGB", (320, 200), color="pink"),
            prompt="A test slide 1",
            neg_prompt="",
            width=1280,
            height=800,
            num_steps=20,
        )

    # Success case with 2 chunks
    audio_path = empty_audio_file(duration_seconds=7.0)
    video_binary = await gen_video_chunked.gen_video_chunked(
        audio_path=audio_path,
        image=Image.new("RGB", (640, 480), color="pink"),
        prompt="A test slide 1",
        neg_prompt="",
        width=1280,
        height=800,
        num_steps=20,
    )
    assert video_binary is not None

    # Check metadata
    video_file_info = get_video_file_info(video_binary)
    video_info = video_file_info["video"]
    assert video_info["codec"] == "h264"
    assert video_info["num_frames"] is not None
    assert 1 + 161 <= video_info["num_frames"] <= 1 + 161 + 4  # 7 secs * 23 FPS = 161
    assert video_info["width"] == 1280
    assert video_info["height"] == 800
    assert video_info["fps"] == 23
    assert video_info["duration_seconds"] is not None
    assert 6.9 <= video_info["duration_seconds"] <= 7.1  # 7 seconds
    audio_info = video_file_info["audio"]
    assert audio_info["codec"] == "aac"
    assert audio_info["duration_seconds"] is not None
    assert 6.9 <= audio_info["duration_seconds"] <= 7.1  # 7 seconds

    # Check actual data
    video_frames = await get_video_frames(video_binary)
    assert 1 + 161 <= len(video_frames) <= 1 + 161 + 4  # 7 secs * 23 FPS = 161

    expected_color = MOCK_COLORS_RGB["gen_video_audio_from_video"]
    frame = video_frames[0]
    assert frame.size == (1280, 800)
    pixel_color = frame.convert("RGB").getpixel((0, 0))
    assert is_pixel_color(pixel_color, expected_color), (
        f"Frame 0: {pixel_color}!={expected_color}")
    for ix, frame in enumerate(video_frames):
        assert frame.size == (1280, 800)
        pixel_color = frame.convert("RGB").getpixel((0, 0))
        assert is_pixel_color(pixel_color, expected_color), (
            f"Frame {ix}: {pixel_color}!={expected_color}")

    # TODO test with upscaling
    # TODO test with debug


def is_pixel_color(
    pixel_color: Tuple[int, int, int],
    expected_color: Tuple[int, int, int],
) -> bool:
    """Check if pixel_color is approximately equal to expected_color."""
    r, g, b = pixel_color
    er, eg, eb = expected_color
    if abs(r - er) > 2:
        return False
    if abs(g - eg) > 2:
        return False
    if abs(b - eb) > 2:
        return False
    return True


"""
@pytest.mark.timeout(5)
@pytest.mark.asyncio
async def test_gen_scene_chunks() -> None:
    service_manager = MagicMock()

    job_id = "gen_scene_chunks"
    job = StreamCastJob(job_id, service_manager)

    with pytest.raises(Exception, match="Service 'hunyuanframepackf1' not found"):
        await job.gen_scene_chunks(
            scene_id=0,
            audio_path="tests/data/audio_4675.wav",
            image=Image.new("RGB", (80, 80), color="blue"),
            video_prompt="Video prompt.")

    # Mocking the LMM generator
    job.gen = await _get_gen_mock(job.gen)

    # TODO go further than this
    with pytest.raises(ValueError, match="No sub-scenes generated for scene 0"):
        video_binary = await job.gen_scene_chunks(
            scene_id=0,
            audio_path="tests/data/audio_4675.wav",
            image=Image.new("RGB", (80, 80), color="blue"),
            video_prompt="Video prompt.")
        assert video_binary is not None

    del job
    del service_manager


def _get_soundfile_mock() -> MagicMock:
    soundfile_mock = MagicMock()
    sample_rate = 16 * 1000
    audio_data = [1.0] * sample_rate * 5  # 5 seconds of dummy audio
    soundfile_mock.read = MagicMock(return_value=(audio_data, sample_rate))
    soundfile_mock.write = MagicMock()
    return soundfile_mock


@pytest.mark.asyncio
async def test_podcast_gen_sub_scene() -> None:
    # TODO
    service_manager = MagicMock()
    job = StreamCastJob("test_podcast_gen_sub_scene", service_manager)
    audio_base64 = await read_file_base64("tests/data/audio_4675.wav")

    job.gen = await _get_gen_mock(job.gen)

    scene_frames = [
        Image.new("RGB", (80, 50), color="pink")
        for _ in range(109)
    ]

    with patch.dict(sys.modules, {
        'soundfile': _get_soundfile_mock(),
    }):
        sub_scene_task = await job.gen_sub_scene(
            scene_id=1,
            sub_scene_id=2,
            sub_scene_info=SubVideoInfo(1, 2),
            scene_frames=scene_frames,
            audio_base64=audio_base64,
        )
        assert sub_scene_task is not None
        await sub_scene_task

    # No frames should return None
    with patch.dict(sys.modules, {
        'soundfile': _get_soundfile_mock(),
    }):
        sub_scene_task = await job.gen_sub_scene(
            scene_id=2,
            sub_scene_id=3,
            sub_scene_info=SubSceneInfo(2, 3),
            scene_frames=[],
            audio_base64=audio_base64,
        )
        assert sub_scene_task is None

    await job.close()
    del service_manager
"""
