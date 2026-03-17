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
async def test_streamedit_job_gen_edit_scene_saves_audio() -> None:
    """gen_edit_scene saves per-scene audio WAV file and sets scene.audio_path."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        job = StreamEditJob(
            job_id="test_scene_audio",
            service_manager=MagicMock(),
        )
        job.job_path = tmp_dir

        # Create a fake input video file so read_file_bytes can read it
        video_path = f"{tmp_dir}/video.mp4"
        with open(video_path, "wb") as f:
            f.write(b"fake_video_data")

        scene = SceneSegment(
            scene_id=2,
            start_frame=60,
            end_frame=120,
            start_sec=2.0,
            end_sec=4.0,
        )

        # Return valid base64-encoded bytes from chunk_audio_base64
        fake_audio_base64 = base64.b64encode(b"dummy_audio").decode()

        with patch.object(_sei_module, "chunk_video_binary", return_value=b"scene_vid"), \
             patch.object(_sei_module, "get_video_frames", new_callable=AsyncMock, return_value=[]), \
             patch.object(_sei_module, "chunk_audio_base64", return_value=fake_audio_base64), \
             patch.object(job.gen, "gen_video_audio_from_video", new_callable=AsyncMock, return_value=b"edited_video"):

            result_path = await job.gen_edit_scene(scene, "full_audio_b64")

        # scene.audio_path must be set to the relative per-scene filename
        assert scene.audio_path == "002_audio.wav"

        # The per-scene audio file must exist on disk
        assert os.path.exists(f"{tmp_dir}/002_audio.wav")

        # The edited scene file must also be saved
        assert os.path.exists(f"{tmp_dir}/002_edit.mp4")

        # The returned path must point to the edited scene file
        assert result_path == f"{tmp_dir}/002_edit.mp4"
