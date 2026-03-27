#!/usr/bin/env python3
"""
Unit tests for StreamShort.
"""

import os
import sys
import json
import pytest
import aiofiles

from http import HTTPStatus

from quart import Quart

from openai import APIConnectionError

from unittest.mock import patch
from unittest.mock import MagicMock
from unittest.mock import AsyncMock

# Add current path
sys.path.append(os.getcwd())

from file_utils import binary_to_base64
from media_utils import get_video_with_text

from tests.test_utils import temp_sys_path
from tests.torch_mock import TorchMock

mock_torch = TorchMock()

mock_cv2 = MagicMock()

mock_scenedetect = MagicMock()
mock_scenedetect.VideoManager = MagicMock()
mock_scenedetect.SceneManager = MagicMock()
mock_scenedetect.detectors = MagicMock()
mock_scenedetect.detectors.ContentDetector = MagicMock()
mock_scenedetect.stats_manager = MagicMock()
mock_scenedetect.stats_manager.StatsManager = MagicMock()

mock_modules = {
    "cv2": mock_cv2,
    "scenedetect": mock_scenedetect,
    "scenedetect.detectors": mock_scenedetect.detectors,
    "scenedetect.stats_manager": mock_scenedetect.stats_manager,
}
mock_modules.update(mock_torch.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    with temp_sys_path("apps", "apps/streamshort"):
        from apps.streamshort.streamshort import StreamShortApp
        from apps.streamshort.streamshort_job import StreamShortJob
        from apps.streamshort.streamshort_job import SceneSegment

streamshort_app = StreamShortApp()


@pytest.fixture(name="test_app")
def _test_app() -> Quart:
    return streamshort_app.app


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
    assert "StreamShort" in response_html


@pytest.mark.asyncio
async def test_health(test_app: Quart) -> None:
    """Check /health."""
    client = test_app.test_client()
    response = await client.get("/health")
    assert response is not None
    assert response.status_code == HTTPStatus.OK
    response_json = await response.get_json()
    assert response_json == {
        "host": None,
        "jobs": {},
        "k8s_cluster": None,
        "port": None,
        "services": {},
        "status": "ok"
    }


@pytest.mark.asyncio
async def test_files(test_app: Quart) -> None:
    """Check /files endpoint."""
    client = test_app.test_client()

    if not os.path.exists("/tmp/streamshort"):
        os.makedirs("/tmp/streamshort", exist_ok=True)

    response = await client.get("/files")
    assert response.status_code == HTTPStatus.OK
    response_json = await response.get_json()
    assert "files" in response_json

    response = await client.get("/file/testfile.txt")
    assert response.status_code == HTTPStatus.NOT_FOUND
    response_json = await response.get_json()
    assert response_json == {"error": "File '/tmp/streamshort/testfile.txt' not found"}

    response = await client.get("/file_stream/job_id/testfile2.txt")
    assert response.status_code == HTTPStatus.NOT_FOUND
    response_json = await response.get_json()
    assert response_json == {"error": "File not found"}

    response = await client.get("/file_view/job_id/testfile3.txt")
    assert response.status_code == HTTPStatus.OK
    assert response.content_type == "text/html; charset=utf-8"
    response_text = await response.get_data(as_text=True)
    assert response_text.startswith("<!DOCTYPE html>\n<html>")
    assert "<title>File viewer: testfile3.txt</title>" in response_text


@pytest.mark.asyncio
async def test_unknown_route(test_app: Quart) -> None:
    """Check that an unknown route returns 404."""
    client = test_app.test_client()
    response = await client.get("/does-not-exist")
    assert response.status_code == HTTPStatus.NOT_FOUND


@pytest.mark.asyncio
async def test_job_submit_page(test_app: Quart) -> None:
    """Check the web page for job submission."""
    client = test_app.test_client()
    response = await client.get("/job")
    assert response.status_code == HTTPStatus.OK
    text = await response.get_data(as_text=True)
    assert len(text) > 0


@pytest.mark.asyncio
async def test_job_status_page(test_app: Quart) -> None:
    """Check the web page for job status."""
    client = test_app.test_client()
    job_id = "testjobid"
    response = await client.get(f"/job/{job_id}")
    assert response.status_code == HTTPStatus.OK
    text = await response.get_data(as_text=True)
    assert len(text) > 0


@pytest.mark.asyncio
async def test_api_job_status(test_app: Quart) -> None:
    """Check the API for job status (returns UNKNOWN for nonexistent jobs)."""
    client = test_app.test_client()
    job_id = "nonexistent_job_id"
    response = await client.get(f"/api/job/{job_id}/status")
    assert response.status_code == HTTPStatus.OK
    response_json = await response.get_json()
    assert "status" in response_json


@pytest.mark.asyncio
async def test_api_job_requests(test_app: Quart) -> None:
    """Check the API for job requests listing (returns empty for nonexistent jobs)."""
    client = test_app.test_client()
    job_id = "nonexistent_job_id"
    response = await client.get(f"/api/job/{job_id}/requests")
    assert response.status_code == HTTPStatus.OK


@pytest.mark.asyncio
async def test_submit_job(test_app: Quart) -> None:
    """Check the API for job requests."""
    client = test_app.test_client()

    # Mock the service manager
    streamshort_app.service_manager = MagicMock()
    streamshort_app.service_manager.get_service_url = MagicMock(
        return_value="http://mock_service_url:1234"
    )

    response = await client.post("/api/job", json={"pdf_base64": "AAAA"})
    assert response.status_code == HTTPStatus.BAD_REQUEST
    response_json = await response.get_json()
    assert "error" in response_json
    assert response_json["error"] == "Missing 'video_base64' in request"

    # Bad video data
    video_base64 = binary_to_base64(b"binary")
    response = await client.post("/api/job", json={"video_base64": video_base64})
    assert response.status_code == HTTPStatus.OK
    response_json = await response.get_json()
    assert response_json is not None
    # assert "job_id" in response_json
    # assert "error" in response_json
    # assert "Ensure file is valid video" in response_json["error"]

    # Success case with a valid video file
    video_binary = await get_video_with_text(
        width=100, height=100,
        text="Test Video",
        duration_seconds=1,
    )
    video_base64 = binary_to_base64(video_binary)
    response = await client.post("/api/job", json={"video_base64": video_base64})
    assert response.status_code == HTTPStatus.OK
    response_json = await response.get_json()
    assert "job_id" in response_json
    assert response_json["status"] == "success"


@pytest.mark.asyncio
async def test_pick_key_frames() -> None:
    service_manager = MagicMock()
    service_manager.get_service_url = MagicMock(
        return_value="http://mock_service_url:1234"
    )
    job_id = "test_pick_key_frames"
    job = StreamShortJob(job_id, service_manager)

    job.scenes = [
        SceneSegment(scene_id=0, start_frame=0, end_frame=1, start_sec=0.0, end_sec=1.0),
        SceneSegment(scene_id=1, start_frame=2, end_frame=3, start_sec=1.0, end_sec=2.0),
        SceneSegment(scene_id=2, start_frame=4, end_frame=5, start_sec=2.0, end_sec=3.0),
    ]

    key_frames = job.pick_key_frames()
    assert key_frames == [0, 1, 2, 3, 4]

    await job.close()


@pytest.mark.asyncio
async def test_find_scene_for_frame() -> None:
    service_manager = AsyncMock()
    service_manager.get_service_url = MagicMock(
        return_value="http://mock_service_url:1234"
    )
    job_id = "tert_find_scene_for_frame"
    job = StreamShortJob(job_id, service_manager)

    scene = job.find_scene_for_frame(1)
    assert scene is None

    job.scenes = [
        SceneSegment(scene_id=0, start_frame=0, end_frame=10, start_sec=0.0, end_sec=1.0),
        SceneSegment(scene_id=1, start_frame=10, end_frame=20, start_sec=1.0, end_sec=2.0),
        SceneSegment(scene_id=2, start_frame=20, end_frame=30, start_sec=2.0, end_sec=3.0),
    ]
    scene = job.find_scene_for_frame(15)
    assert scene is not None
    assert scene.scene_id == 1

    scene = job.find_scene_for_frame(0)
    assert scene is not None
    assert scene.scene_id == 0

    scene = job.find_scene_for_frame(25)
    assert scene is not None
    assert scene.scene_id == 2

    scene = job.find_scene_for_frame(-10)
    assert scene is None

    scene = job.find_scene_for_frame(100)
    assert scene is None

    await job.close()


@pytest.mark.asyncio
async def test_describe_frames() -> None:
    service_manager = AsyncMock()
    service_manager.get_service_url = MagicMock(
        return_value="http://mock_service_url:1234"
    )
    job_id = "test_describe_frames"
    job = StreamShortJob(job_id, service_manager)

    with pytest.raises(FileNotFoundError, match="frame_0001.jpg"):
        await job.describe_frames([1, 2, 3])

    await job.close()


@pytest.mark.asyncio
async def test_choose_scenes_for_highlight() -> None:
    service_manager = AsyncMock()
    service_manager.get_service_url = MagicMock(
        return_value="http://mock_service_url:1234"
    )
    job_id = "test_choose_scenes_for_highlight"
    job = StreamShortJob(job_id, service_manager)

    with pytest.raises(APIConnectionError):
        await job.choose_scenes_for_highlight()

    job.scenes = [
        SceneSegment(scene_id=0, start_frame=0, end_frame=10, start_sec=0.0, end_sec=1.0),
        SceneSegment(scene_id=1, start_frame=10, end_frame=20, start_sec=1.0, end_sec=2.0),
    ]
    with pytest.raises(APIConnectionError):
        await job.choose_scenes_for_highlight()

    await job.close()


@pytest.mark.asyncio
async def test_save_highlight_short() -> None:
    service_manager = AsyncMock()
    service_manager.get_service_url = MagicMock(
        return_value="http://mock_service_url:1234"
    )
    job_id = "test_save_highlight_short"
    job = StreamShortJob(job_id, service_manager)

    with pytest.raises(FileNotFoundError, match="video.mp4"):
        await job.save_highlight_short([])

    await job.close()


@pytest.mark.asyncio
async def test_save_selected_scenes() -> None:
    service_manager = AsyncMock()
    service_manager.get_service_url = MagicMock(
        return_value="http://mock_service_url:1234"
    )
    job_id = "test_save_selected_scenes"
    job = StreamShortJob(job_id, service_manager)
    try:
        chosen = [0, 2, 4]
        await job.save_selected_scenes(chosen)

        selected_path = f"{job.job_path}/selected_scenes.json"
        async with aiofiles.open(selected_path) as f:
            data = json.loads(await f.read())
        assert data == chosen

        # Overwrite with a new selection
        chosen2 = [1, 3]
        await job.save_selected_scenes(chosen2)
        async with aiofiles.open(selected_path) as f:
            data2 = json.loads(await f.read())
        assert data2 == chosen2

        # Empty selection is also valid
        await job.save_selected_scenes([])
        async with aiofiles.open(selected_path) as f:
            data3 = json.loads(await f.read())
        assert data3 == []
    finally:
        await job.close()


@pytest.mark.asyncio
async def test_chunk_audio_into_scenes() -> None:
    service_manager = AsyncMock()
    service_manager.get_service_url = MagicMock(
        return_value="http://mock_service_url:1234"
    )
    job_id = "test_chunk_audio_into_scenes"
    job = StreamShortJob(job_id, service_manager)

    await job.chunk_audio_into_scenes()

    await job.close()


@pytest.mark.asyncio
async def test_describe_frames_batch() -> None:
    service_manager = AsyncMock()
    service_manager.get_service_url = MagicMock(
        return_value="http://mock_service_url:1234"
    )
    job_id = "test_describe_frames_batch"
    job = StreamShortJob(job_id, service_manager)
    try:
        with pytest.raises(FileNotFoundError, match="frame_0007.jpg"):
            await job.describe_frames_batch([7, 8, 9])

        # Mock frame files
        for frame_id in [1, 2, 3, 4, 5]:
            frame_path = f"{job.job_path}/frame_{frame_id:04d}.jpg"
            with open(frame_path, "wb") as f:
                f.write(b"binary")
        with pytest.raises(APIConnectionError):
            await job.describe_frames_batch([1, 2, 3, 4, 5])
    finally:
        await job.close()


@pytest.mark.asyncio
async def test_transcribe_audio() -> None:
    service_manager = AsyncMock()
    service_manager.get_service_url = MagicMock(
        return_value="http://mock_service_url:1234"
    )
    job_id = "test_transcribe_audio"
    job = StreamShortJob(job_id, service_manager)
    try:
        transcription = await job.transcribe_audio()
        assert transcription == ""

        # No audio
        job.scenes = [
            SceneSegment(scene_id=0, start_frame=0, end_frame=10, start_sec=0.0, end_sec=1.0),
            SceneSegment(scene_id=1, start_frame=10, end_frame=20, start_sec=1.0, end_sec=2.0),
        ]
        transcription = await job.transcribe_audio()
        assert transcription == ""

        # Non-existent audio files
        for scene in job.scenes:
            scene.audio_path = "non_existent_audio.wav"
        transcription = await job.transcribe_audio()
        assert transcription == ""

        # Mock audio files
        for scene in job.scenes:
            scene_id = scene.scene_id
            scene.audio_path = f"{job.job_path}/scene_{scene_id:04d}_audio.wav"
            async with aiofiles.open(scene.audio_path, "wb") as f:
                await f.write(b"binary")
        transcription = await job.transcribe_audio()
        assert transcription == ""

        # Mock transcript generation
        job.gen.gen_audio_transcript = AsyncMock(return_value=("This is a test transcription.", "en"))
        transcription = await job.transcribe_audio()
        assert transcription == "This is a test transcription.\nThis is a test transcription.\n"
        for scene in job.scenes:
            if scene.audio_path:
                assert scene.language == "en"
    finally:
        await job.close()


def test_scene_segement() -> None:
    scene = SceneSegment(
        scene_id=1,
        start_frame=10,
        end_frame=20,
        start_sec=1.0,
        end_sec=2.0
    )
    assert scene.scene_id == 1
    assert scene.start_frame == 10
    assert scene.end_frame == 20
    assert scene.start_sec == 1.0
    assert scene.end_sec == 2.0
    assert scene.duration_sec == 1
    assert scene.transcript is None

    assert scene.get_start() == "00:00:01"
    assert scene.get_end() == "00:00:02"
    assert str(scene) == "[  10-  20,  1.0- 2.0]"

    scene.add_image_path("/path/to/image.jpg")
    assert scene.frame_image_paths == ["/path/to/image.jpg"]

    assert scene.descriptions == []
    scene.add_description("")
    assert scene.descriptions == []

    scene.add_description("A sample description.")
    assert scene.descriptions == ["A sample description."]

    assert str(scene) == "[  10-  20,  1.0- 2.0] | A sample description.... | 1 images"
