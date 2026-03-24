#!/usr/bin/env python3

from __future__ import annotations

import os
import logging
import pytest
import aiofiles
import aiofiles.os

import numpy as np

from typing import List
from typing import Optional

from PIL import Image

from media_utils import get_frame_with_text
from media_utils import video_frames_to_base64
from media_utils import base64_to_video_frames
from media_utils import chunk_video_binary
from media_utils import save_video_frames
from media_utils import save_video_audio
from media_utils import add_text_to_frame
from media_utils import concatenate_videos
from media_utils import change_video_fps
from media_utils import get_video_file_info
from media_utils import get_video_duration
from media_utils import get_video_frames
from media_utils import get_video_frame
from media_utils import get_video_fps
from media_utils import get_video_num_frames
from media_utils import get_video_with_text
from media_utils import get_video_frames_at_fps
from media_utils import get_video_size
from media_utils import get_font_size
from media_utils import get_audio_duration
from media_utils import get_ffmpeg_version

from file_utils import read_file_base64


@pytest.mark.asyncio
async def test_video() -> None:
    video_frames: List[np.ndarray] = [
        get_frame_with_text(100, 60, f"frame{frame_id:02d}", output_type="numpy")  # type: ignore[misc]
        for frame_id in range(24)
    ]
    video_path = await save_video_frames(video_frames, fps=24)
    video_file_info = get_video_file_info(video_path)

    video_overall_info = video_file_info.get("overall")
    assert video_overall_info is not None
    assert video_overall_info["num_bytes"] > 1000
    assert video_overall_info["duration_seconds"] == 1.0

    video_info = video_file_info.get("video")
    assert video_info is not None
    assert video_info.get("duration_seconds") == 1.0
    assert video_info.get("width") == 100
    assert video_info.get("height") == 60
    assert video_info.get("num_frames") == 24
    assert video_info.get("fps") == 24
    assert video_info.get("codec") == "h264"

    pix_fmt = video_info.get("pix_fmt")
    logging.info(f"Pixel format: {pix_fmt}")

    del video_frames
    del video_path
    del video_info


@pytest.mark.asyncio
async def test_base64() -> None:
    """Test video frames to/from base64 conversion."""
    video_frames: List[np.ndarray] = [
        get_frame_with_text(80, 64, f"frame{frame_id:02d}", output_type="pil")  # type: ignore[misc]
        for frame_id in range(12)
    ]
    video_base64 = video_frames_to_base64(video_frames)  # type: ignore[arg-type]
    video_frames_2 = base64_to_video_frames(video_base64)
    assert len(video_frames) == len(video_frames_2)
    for frame1, frame2 in zip(video_frames, video_frames_2):
        assert frame1.size == frame2.size
    assert get_video_size(video_frames_2) == 184320

    with pytest.raises(TypeError):
        video_frames_to_base64(12345)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        video_frames_to_base64([])
    with pytest.raises(TypeError):
        base64_to_video_frames(12345)  # type: ignore[arg-type]

    assert get_video_size(None) == 0  # type: ignore[arg-type]
    assert get_video_size([]) == 0

    del video_frames
    del video_frames_2


def test_get_video_frames_at_fps() -> None:
    """Test getting a video frames at a specific FPS."""
    video_frames: list[Image.Image] = [
        get_frame_with_text(128, 64, f"frame{frame_id:02d}", output_type="pil")  # type: ignore[misc]
        for frame_id in range(10)
    ]
    new_video_frames = get_video_frames_at_fps(video_frames, src_fps=30, dst_fps=30)
    assert len(new_video_frames) == 10
    new_video_frames = get_video_frames_at_fps(video_frames, src_fps=30, dst_fps=24)
    assert len(new_video_frames) == 8  # 10 * 24 / 30 = 8
    assert get_video_frames_at_fps([], src_fps=24) == []
    assert get_video_frames_at_fps(None, dst_fps=16) == []  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        get_video_frames_at_fps(video_frames, src_fps=0, dst_fps=0)
    with pytest.raises(ValueError):
        get_video_frames_at_fps(video_frames, src_fps=30, dst_fps=-16)

    del video_frames
    del new_video_frames


@pytest.mark.asyncio
async def test_chunk() -> None:
    """Test chunking a video binary."""
    video_binary = await get_video_with_text(50, 30, "Test, video", duration_seconds=3.0, fps=30)
    video_binary_chunked = chunk_video_binary(video_binary, 0.5, 1.5)
    duration = await get_video_duration(video_binary_chunked)
    assert duration == 1.0

    video_binary_chunked = chunk_video_binary(video_binary, 0.0, 1.5)
    duration = await get_video_duration(video_binary_chunked)
    assert duration == 1.5

    video_binary_chunked = chunk_video_binary(video_binary_chunked, 0.5, 2.5)
    duration = await get_video_duration(video_binary_chunked)
    assert duration == 1.0

    with pytest.raises(ValueError, match="end_seconds must be greater than start_seconds"):
        chunk_video_binary(video_binary, 0.5, 0.4)
    with pytest.raises(ValueError, match="must be non-negative"):
        chunk_video_binary(video_binary, -0.5, 0.4)
    with pytest.raises(TypeError):
        chunk_video_binary(12345, 0.5, 1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Video input is empty"):
        chunk_video_binary(None, 0.5, 1.5)  # type: ignore[arg-type]

    del video_binary


@pytest.mark.asyncio
async def test_get_frame() -> None:
    """Test getting a specific frame from a video binary."""
    video_frames = [
        get_frame_with_text(160, 90, f"frame{frame_id:02d}", output_type="pil")
        for frame_id in range(12)
    ]
    video_path = await save_video_frames(video_frames, fps=24)
    async with aiofiles.open(video_path, "rb") as f:
        video_binary = await f.read()
    frame = await get_video_frame(video_binary, 5)
    assert frame.size == (160, 90)
    duration_seconds = await get_video_duration(video_binary)
    assert duration_seconds == 0.5
    duration_seconds = await get_video_duration(video_path)
    assert duration_seconds == 0.5

    with pytest.raises(TypeError):
        await get_video_frame(video_path, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        await get_video_frame(video_binary, 512)
    with pytest.raises(ValueError):
        await get_video_frame(video_binary, -1)
    with pytest.raises(ValueError):
        await get_video_frame(b"", 5)

    await aiofiles.os.remove(video_path)
    del video_frames
    del video_binary


def test_add_text() -> None:
    """Test adding text to video frames."""
    NUM_FRAMES = 12
    video_frames = [
        get_frame_with_text(64, 64, "", output_type="pil")
        for frame_id in range(NUM_FRAMES)
    ]
    assert len(video_frames) == NUM_FRAMES
    for frame in video_frames:
        assert frame.size == (64, 64)
        assert np.abs(np.array(frame)).sum() == 0  # All black

    video_frames = [
        add_text_to_frame(frame, text="Sample Text", position="top-left")
        for frame in video_frames
    ]
    assert len(video_frames) == NUM_FRAMES
    for frame in video_frames:
        assert frame.size == (64, 64)
        assert np.abs(np.array(frame)).sum() > 0  # Not all black

    # Centered text
    video_frames = [
        add_text_to_frame(frame, text="Sample Text 2", font_color="blue", position="center")
        for frame in video_frames
    ]
    assert len(video_frames) == NUM_FRAMES
    for frame in video_frames:
        assert frame.size == (64, 64)
        assert np.abs(np.array(frame)).sum() > 0  # Not all black

    # Specific text position
    font_size = get_font_size(64, 64)
    video_frames = [
        add_text_to_frame(frame, text="Sample Text 3", font_size=font_size, position=(0, 0))
        for frame in video_frames
    ]
    assert len(video_frames) == NUM_FRAMES
    for frame in video_frames:
        assert frame.size == (64, 64)
        assert np.abs(np.array(frame)).sum() > 0  # Not all black

    del video_frames


@pytest.mark.asyncio
async def test_concatenate() -> None:
    """Test concatenating multiple video binaries."""
    video_frames1 = [
        get_frame_with_text(120, 60, f"frame1_{frame_id:02d}", output_type="numpy")
        for frame_id in range(5)
    ]
    video_frames2 = [
        get_frame_with_text(120, 60, f"frame2_{frame_id:02d}", output_type="numpy")
        for frame_id in range(10)
    ]
    video_path = await save_video_frames(video_frames1, fps=30)
    async with aiofiles.open(video_path, 'rb') as file:
        video_binary1 = await file.read()
    video_path = await save_video_frames(video_frames2, fps=30)
    async with aiofiles.open(video_path, 'rb') as file:
        video_binary2 = await file.read()
    video_binary = await concatenate_videos([
        video_binary1,
        video_binary2
    ])
    num_video_frames = await get_video_num_frames(video_binary)
    assert num_video_frames == 15  # 5 + 10

    video_frames = await get_video_frames(video_binary)
    assert len(video_frames) == 15
    video_frame = await get_video_frame(video_binary, 5)
    assert video_frame.size == (120, 60)

    video_frames = await get_video_frames(None)
    assert video_frames == []
    with pytest.raises(ValueError):
        await concatenate_videos([])
    with pytest.raises(TypeError, match="Video input 0 is not bytes or str"):
        await concatenate_videos([video_frames1, video_frames2])  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        await concatenate_videos(["abc", "def"])
    with pytest.raises(ValueError, match="One of the inputs is corrupted"):
        await concatenate_videos([b"abc", b"def"])

    with pytest.raises(FileNotFoundError, match="Video file does not exist"):
        await get_video_num_frames("abc")
    with pytest.raises(ValueError, match="The video binary is corrupted or has an unsupported format"):
        await get_video_num_frames(b"abc")

    await aiofiles.os.remove(video_path)

    del video_frames1
    del video_frames2
    del video_binary1
    del video_binary2
    del video_binary


@pytest.mark.asyncio
async def test_get_video_frames_empty() -> None:
    video_frames = await get_video_frames(None)
    assert video_frames == []

    video_frames = await get_video_frames(b"")
    assert video_frames == []

    video_frames = await get_video_frames([])  # type: ignore[arg-type]
    assert video_frames == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "num_frames,fps",
    [
        (5, 30.0),  # 5 frames at 30 fps
        (11, 23.0),  # 11 frames at 23 fps
        (17, 23.0),  # 17 frames at 23 fps
        (23, 23.0),  # 23 frames at 23 fps
        (32, 23.0),  # 32 frames at 23 fps
        (33, 23.0),  # 33 frames at 23 fps
        (81, 23.0),  # 81 frames at 23 fps
        (5, 16.0),  # 5 frames at 16 fps
    ],
)
async def test_get_video_frames(num_frames: int, fps: float) -> None:
    width, height = 120, 60
    video_frames = [
        get_frame_with_text(width, height, f"frame{frame_id:02d}", output_type="numpy")
        for frame_id in range(num_frames)
    ]
    video_path = await save_video_frames(
        video_frames,
        fps=fps)
    async with aiofiles.open(video_path, "rb") as file:
        video_binary = await file.read()
    video_frames_out = await get_video_frames(video_binary)
    assert len(video_frames_out) == num_frames

    video_file_info = get_video_file_info(video_binary)
    assert "video" in video_file_info
    video_info = video_file_info["video"]
    assert video_info is not None
    assert video_info.get("width") == width
    assert video_info.get("height") == height
    assert video_info.get("num_frames") == num_frames
    assert video_info.get("num_frames") == len(video_frames_out)
    assert video_info.get("fps") == fps
    duration_seconds = video_info.get("duration_seconds")
    assert duration_seconds is not None
    assert abs(duration_seconds - (num_frames / fps)) < 0.01
    video_overall_info = video_file_info.get("overall")
    assert video_overall_info is not None
    num_bytes = video_overall_info.get("num_bytes")
    assert num_bytes is not None
    assert num_bytes > 1000

    os.remove(video_path)
    del video_frames


@pytest.mark.asyncio
async def test_save_video_frames() -> None:
    NUM_FRAMES = 81
    WIDTH, HEIGHT = 160, 100
    FPS = 23.0

    video_frames = [
        get_frame_with_text(WIDTH, HEIGHT, f"frame{frame_id:02d}", output_type="numpy")
        for frame_id in range(NUM_FRAMES)
    ]

    video_path = await save_video_frames(video_frames, fps=FPS)
    video_file_info = get_video_file_info(video_path)
    video_info = video_file_info.get("video")
    assert video_info is not None
    assert video_info.get("fps") == FPS
    assert video_info.get("num_frames") == 81
    assert video_info.get("duration_seconds") == 3.521739  # 81 / 23
    os.remove(video_path)

    """
    # Test with trimming
    video_path = await save_video_frames(video_frames, fps=FPS, time_in_seconds=2.0)
    video_info = get_video_file_info(video_path)
    assert video_info["video"]["fps"] == FPS
    assert video_info["video"]["num_frames"] == 46  # 2.0 * 23
    assert video_info["video"]["duration_seconds"] == 2.0
    os.remove(video_path)

    # Test with expansion adds empty frames
    video_path = await save_video_frames(video_frames, fps=FPS, time_in_seconds=5.0)
    video_info = get_video_file_info(video_path)
    assert video_info["video"]["fps"] == FPS
    assert video_info["video"]["num_frames"] == 115
    assert video_info["video"]["duration_seconds"] == 5.0
    os.remove(video_path)
    """

    with pytest.raises(ValueError):
        await save_video_frames(None, fps=FPS)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        await save_video_frames([], fps=FPS)
    with pytest.raises(TypeError):
        await save_video_frames(b"BLAH", fps=FPS)  # type: ignore[arg-type]


def assert_approx(
    a: float,
    b: float,
    tol: float = 1e-3,
    msg: Optional[str] = None
) -> None:
    if msg:
        assert abs(a - b) < tol, f"{a} !~= {b}: {msg}"
    else:
        assert abs(a - b) < tol, f"{a} !~= {b}"


@pytest.mark.asyncio
async def test_save_video_frames_audio() -> None:
    # Get the audio
    audio_path = "tests/data/audio_4675.wav"
    audio_base64 = await read_file_base64(audio_path)
    audio_duration_secs = get_audio_duration(audio_base64)
    assert audio_duration_secs == 4.675

    # Get video only with the video frames
    NUM_FRAMES = 113
    WIDTH, HEIGHT = 180, 100
    FPS = 23.0
    video_frames: list[Image.Image] = [
        get_frame_with_text(  # type: ignore[misc]
            WIDTH, HEIGHT,
            text=f"frame{frame_id:02d}",
            font_size=24,
            output_type="pil")
        for frame_id in range(NUM_FRAMES)
    ]
    assert len(video_frames) == NUM_FRAMES

    # Video without audio
    video_path = await save_video_frames(
        video_frames=video_frames,
        fps=FPS)
    video_file_info = get_video_file_info(video_path)
    video_overall_info = video_file_info.get("overall")
    assert video_overall_info is not None
    num_bytes = video_overall_info.get("num_bytes")
    assert num_bytes is not None
    assert num_bytes > 1000
    duration_seconds = video_overall_info.get("duration_seconds")
    assert duration_seconds is not None
    assert_approx(duration_seconds, 4.913)

    assert "video" in video_file_info
    video_info = video_file_info["video"]
    assert video_info is not None
    assert video_info.get("fps") == FPS, f"Video info: {video_file_info}"
    assert video_info.get("num_frames") == 113, f"Video info: {video_file_info}"
    assert video_info.get("width") == 180, f"Video info: {video_file_info}"
    assert video_info.get("height") == 100, f"Video info: {video_file_info}"
    video_duration_seconds = video_info.get("duration_seconds")
    assert video_duration_seconds is not None
    assert_approx(video_duration_seconds, 4.913)  # 113 / 23

    assert "audio" not in video_file_info

    os.remove(video_path)

    # Video + audio (video longer than audio)
    video_path = await save_video_audio(
        video_content=video_frames,
        audio_path=audio_path,
        fps=FPS)
    video_file_info = get_video_file_info(video_path)
    assert video_file_info is not None
    video_overall_info = video_file_info.get("overall")
    assert video_overall_info is not None
    num_bytes = video_overall_info.get("num_bytes")
    assert num_bytes is not None
    assert num_bytes > 1000
    duration_seconds = video_overall_info.get("duration_seconds")
    assert duration_seconds is not None
    assert_approx(duration_seconds, 4.913)

    assert "video" in video_file_info
    video_info = video_file_info["video"]
    assert video_info is not None
    assert video_info.get("fps") == FPS, f"Video info: {video_file_info}"
    # TODO the following works with ffmpeg 4.2 but not with 6.0
    """
    assert video_info.get("num_frames") == 113, f"Video info: {video_file_info}"
    assert_approx(
        video_info.get("duration_seconds"), 4.913,
        msg=f"Video info: {video_file_info}")
    """
    logging.warning(f"Video info: {video_file_info}")
    num_frames = video_info.get("num_frames")
    assert num_frames is not None
    assert num_frames > 100
    video_duration_seconds = video_info.get("duration_seconds")
    assert video_duration_seconds is not None
    assert video_duration_seconds > 4.0

    assert "audio" in video_file_info
    audio_info = video_file_info["audio"]
    assert audio_info is not None
    audio_duration_seconds = audio_info.get("duration_seconds")
    assert audio_duration_seconds is not None
    assert audio_duration_seconds > 0
    """
    # assert_approx(audio_info["duration_seconds"], 4.913)
    assert_approx(
        audio_duration_seconds, 4.864,
        msg=f"Video info: {video_file_info}")  # TODO this is not good
    """
    logging.info(f"Audio info: {audio_info}")
    assert audio_info["duration_seconds"] is not None and audio_info["duration_seconds"] > 4.0

    os.remove(video_path)


@pytest.mark.asyncio
async def test_ffmpeg_version() -> None:
    """Test getting the FFmpeg version."""
    ffmpeg_version = get_ffmpeg_version()
    logging.info(f"FFmpeg version: {get_ffmpeg_version()}")
    assert ffmpeg_version is not None


@pytest.mark.asyncio
async def test_change_fps() -> None:
    """Test changing the FPS of a video binary."""
    NUM_FRAMES = 30
    FPS = 30
    video_frames = [
        get_frame_with_text(320, 240, f"frame{frame_id:02d}", output_type="numpy")
        for frame_id in range(NUM_FRAMES)
    ]
    video_path = await save_video_frames(video_frames, fps=FPS)
    async with aiofiles.open(video_path, 'rb') as f:
        video_binary = await f.read()
    video_file_info = get_video_file_info(video_path)
    assert "video" in video_file_info
    video_info = video_file_info["video"]
    assert video_info.get("fps") == FPS
    assert video_info.get("num_frames") == 30
    assert video_info.get("duration_seconds") == 1.0
    video_num_frames = await get_video_num_frames(video_binary)
    assert video_num_frames == 30

    # Change FPS to 15
    new_video_binary = change_video_fps(video_binary, 15)
    new_video_size = len(new_video_binary)
    assert new_video_size > 100
    video_file_info = get_video_file_info(new_video_binary)
    assert "video" in video_file_info
    video_info = video_file_info["video"]
    assert video_info.get("fps") == 15
    assert video_info.get("num_frames") == 15
    assert video_info.get("duration_seconds") == 1.0
    assert get_video_fps(new_video_binary) == 15

    with pytest.raises(TypeError):
        change_video_fps(video_path, 15)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        change_video_fps(video_binary, 0)

    with pytest.raises(ValueError, match="No video frames provided."):
        await save_video_frames([], fps=30)

    with pytest.raises(ValueError, match="No video frames provided."):
        await save_video_frames(None, fps=30)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        get_video_fps(12345)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        get_video_fps(b"")

    os.remove(video_path)
    del video_frames
    del video_binary
