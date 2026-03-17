#!/usr/bin/env python3
"""
Unit tests for StreamEdit.
"""

import base64
import os
import sys
import tempfile
import pytest

from http import HTTPStatus

from PIL import Image

from quart import Quart

from unittest.mock import patch
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

# Add current path
sys.path.append(os.getcwd())

from media_utils import video_frames_to_base64

from tests.test_utils import temp_sys_path
from tests.torch_mock import TorchMock

mock_torch = TorchMock()

mock_modules = {}
mock_modules.update(mock_torch.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    with temp_sys_path("apps", "apps/streamedit"):
        from apps.streamedit.streamedit import StreamEditApp

scene_mocks_base = {
    'scenedetect': MagicMock(),
    'scenedetect.detectors': MagicMock(),
    'scenedetect.stats_manager': MagicMock(),
}

with patch.dict(sys.modules, {**mock_modules, **scene_mocks_base}):
    with temp_sys_path("apps", "apps/streamedit"):
        from apps.streamedit.streamedit_job import StreamEditJob
        import apps.streamedit.streamedit_job as _sei_module  # keep reference for patch.object

with temp_sys_path("apps"):
    from scene import SceneSegment


streamedit_app = StreamEditApp()


@pytest.fixture(name="test_app")
def _test_app() -> Quart:
    return streamedit_app.app


@pytest.mark.asyncio
async def test_app(test_app: Quart) -> None:
    """Check that GET / returns 200."""
    client = test_app.test_client()
    response = await client.get("/")
    assert response is not None
    assert response.status_code == HTTPStatus.OK
    assert "text/html; charset=utf-8" == response.content_type
    response_html = await response.get_data(as_text=True)
    assert response_html.startswith("<!DOCTYPE html>\n<html lang=\"en\">")
    assert "StreamEdit" in response_html


@pytest.mark.asyncio
async def test_submit_job(test_app: Quart) -> None:
    """Check the API for job requests."""
    client = test_app.test_client()

    response = await client.post("/api/job", json={"video_base64": "AAAA"})
    assert response.status_code == HTTPStatus.BAD_REQUEST
    response_json = await response.get_json()
    assert "error" in response_json
    assert response_json["error"] == "Service manager not initialized"

    # Mock the service manager
    streamedit_app.service_manager = MagicMock()
    streamedit_app.service_manager.get_service_url = MagicMock(
        return_value="http://mock_service_url:1234"
    )

    # Bad video data – scenedetect raises "Ensure file is valid video" (or similar)
    response = await client.post("/api/job", json={"video_base64": "AAAA"})
    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    response_json = await response.get_json()
    assert response_json["status"] == "error"
    assert "error" in response_json

    # Generate fake video for testing
    video_frames = [
        Image.new("RGB", (640, 480), color="blue")
        for _ in range(8)
    ]
    video_base64 = video_frames_to_base64(video_frames)

    response = await client.post("/api/job", json={"video_base64": video_base64})
    assert response.status_code == HTTPStatus.OK
    response_json = await response.get_json()
    assert response_json["status"] == "success"
    assert "job_id" in response_json


def test_edit_prompt() -> None:
    """Test that EDIT_PROMPT is imported and non-empty."""
    from apps.streamedit.edit_prompts import EDIT_PROMPT
    assert EDIT_PROMPT
    assert "video editor" in EDIT_PROMPT.lower() or "edit" in EDIT_PROMPT.lower()


@pytest.mark.asyncio
async def test_streamedit_job_no_video() -> None:
    """StreamEditJob.gen_edit with missing video raises ValueError."""
    job = StreamEditJob(
        job_id="test_no_video",
        service_manager=MagicMock(),
    )
    with pytest.raises(ValueError, match="Missing 'video_base64'"):
        await job.gen_edit(video_base64=None)


@pytest.mark.asyncio
async def test_streamedit_job_detect_scenes_missing_file() -> None:
    """StreamEditJob.detect_scenes raises FileNotFoundError when video file is absent."""
    job = StreamEditJob(
        job_id="test_detect_no_file",
        service_manager=MagicMock(),
    )
    with pytest.raises(FileNotFoundError):
        await job.detect_scenes("/nonexistent/path/video.mp4")


@pytest.mark.asyncio
async def test_streamedit_job_chunk_audio_into_scenes() -> None:
    """chunk_audio_into_scenes saves scene_{id:03d}.wav and sets scene.audio_path."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        job = StreamEditJob(
            job_id="test_chunk_audio",
            service_manager=MagicMock(),
        )
        job.job_path = tmp_dir

        # Create a fake input video file
        video_path = f"{tmp_dir}/video.mp4"
        with open(video_path, "wb") as f:
            f.write(b"fake_video_data")

        fake_audio_base64 = base64.b64encode(b"dummy_audio").decode()

        job.scenes = [
            SceneSegment(scene_id=0, start_frame=0, end_frame=30, start_sec=0.0, end_sec=1.0),
            SceneSegment(scene_id=1, start_frame=30, end_frame=60, start_sec=1.0, end_sec=2.0),
        ]

        with patch.object(_sei_module, "extract_audio_from_video", new_callable=AsyncMock,
                          return_value=f"{tmp_dir}/audio.wav"), \
             patch.object(_sei_module, "read_file_base64", new_callable=AsyncMock,
                          return_value=fake_audio_base64), \
             patch.object(_sei_module, "chunk_audio_base64", return_value=fake_audio_base64):

            chunks = await job.chunk_audio_into_scenes()

        # Both scenes should have audio_path set with consistent naming
        assert job.scenes[0].audio_path == "scene_000.wav"
        assert job.scenes[1].audio_path == "scene_001.wav"

        # Per-scene WAV files must exist on disk
        assert os.path.exists(f"{tmp_dir}/scene_000.wav")
        assert os.path.exists(f"{tmp_dir}/scene_001.wav")

        # Two audio chunks returned
        assert len(chunks) == 2


@pytest.mark.asyncio
async def test_streamedit_job_gen_edit_scene_saves_video() -> None:
    """gen_edit_scene reads audio from scene.audio_path and saves the edited video."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        job = StreamEditJob(
            job_id="test_scene_edit",
            service_manager=MagicMock(),
        )
        job.job_path = tmp_dir

        # Create a fake input video file so read_file_bytes can read it
        video_path = f"{tmp_dir}/video.mp4"
        with open(video_path, "wb") as f:
            f.write(b"fake_video_data")

        # Pre-create the per-scene audio file (as chunk_audio_into_scenes would)
        fake_audio_base64 = base64.b64encode(b"dummy_audio").decode()
        with open(f"{tmp_dir}/scene_002.wav", "wb") as f:
            f.write(b"dummy_audio")

        scene = SceneSegment(
            scene_id=2,
            start_frame=60,
            end_frame=120,
            start_sec=2.0,
            end_sec=4.0,
            audio_path="scene_002.wav",
        )

        with patch.object(_sei_module, "chunk_video_binary", return_value=b"scene_vid"), \
             patch.object(_sei_module, "get_video_frames", new_callable=AsyncMock, return_value=[]), \
             patch.object(_sei_module, "read_file_base64", new_callable=AsyncMock,
                          return_value=fake_audio_base64), \
             patch.object(job.gen, "gen_video_audio_from_video", new_callable=AsyncMock,
                          return_value=b"edited_video"):

            result_path = await job.gen_edit_scene(scene)

        # The edited scene file must be saved with consistent naming
        assert os.path.exists(f"{tmp_dir}/scene_002_edit.mp4")

        # The returned path must point to the edited scene file
        assert result_path == f"{tmp_dir}/scene_002_edit.mp4"
