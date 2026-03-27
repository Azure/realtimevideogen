#!/usr/bin/env python3
"""
Unit tests for StreamAnimate.
"""

import os
import sys
import pytest

from http import HTTPStatus

from PIL import Image  # noqa: F401 - import before patch.dict to keep PIL in sys.modules

from quart import Quart

from unittest.mock import patch
from unittest.mock import MagicMock

# Add current path
sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path
from tests.torch_mock import TorchMock

mock_torch = TorchMock()

mock_modules = {}
mock_modules.update(mock_torch.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    with temp_sys_path("apps", "apps/streamanimate"):
        from apps.streamanimate.streamanimate import StreamAnimateApp
        from apps.streamanimate.streamanimate_job import StreamAnimateJob
        from apps.streamanimate.streamanimate_job import JobStatus
        from apps.streamanimate.animate_prompts import IMG_PROMPT
        from apps.streamanimate.animate_prompts import VIDEO_PROMPT
    from tests.streamwise_app.lmm_generator_mock import LMMGeneratorMock


streamanimate_app = StreamAnimateApp()


@pytest.fixture(name="test_app")
def _test_app() -> Quart:
    return streamanimate_app.app


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
    assert "StreamAnimate" in response_html


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

    if not os.path.exists("/tmp/streamanimate"):
        os.makedirs("/tmp/streamanimate", exist_ok=True)

    response = await client.get("/files")
    assert response.status_code == HTTPStatus.OK
    response_json = await response.get_json()
    assert "files" in response_json

    response = await client.get("/file/testfile.txt")
    assert response.status_code == HTTPStatus.NOT_FOUND
    response_json = await response.get_json()
    assert response_json == {"error": "File '/tmp/streamanimate/testfile.txt' not found"}

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

    response = await client.post("/api/job", json={"text_prompt": "A bird flying"})
    assert response.status_code == HTTPStatus.BAD_REQUEST
    response_json = await response.get_json()
    assert "error" in response_json
    assert response_json["error"] == "Service manager not initialized"

    # Mock the service manager
    streamanimate_app.service_manager = MagicMock()
    streamanimate_app.service_manager.get_service_url = MagicMock(
        return_value="http://mock_service_url:1234"
    )

    response = await client.post("/api/job", json={"text_prompt": "A bird flying over mountains"})
    # The job is accepted; it may fail later if external services are unavailable.
    response_json = await response.get_json()
    assert response_json is not None
    assert "status" in response_json
    # If the image generation service responds before 0.1s we may get an error;
    # if the job is still running after 0.1s we get job_id back.
    if response.status_code == HTTPStatus.OK:
        assert "job_id" in response_json
        assert response_json["status"] == "success"
    else:
        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
        assert "error" in response_json


def test_animate_prompts() -> None:
    """Test that animate prompts are defined and non-empty."""
    assert IMG_PROMPT
    assert len(IMG_PROMPT) > 0
    assert VIDEO_PROMPT
    assert "animation" in VIDEO_PROMPT.lower() or "motion" in VIDEO_PROMPT.lower()


@pytest.mark.asyncio
async def test_gen_animate_no_video_no_audio() -> None:
    """StreamAnimateJob.gen_animate raises ValueError when no video/audio generated."""
    service_manager = MagicMock()
    job = StreamAnimateJob(
        job_id="test_gen_animate_no_video_no_audio",
        service_manager=service_manager,
    )
    job.gen = LMMGeneratorMock()
    job.config["output_mode"] = "audio_only"
    # narration_text is empty and audio_only => neither video nor audio generated
    with pytest.raises(ValueError, match="Neither video nor audio was generated"):
        await job.gen_animate(
            image_base64=None,
            text_prompt="A bird flying",
            narration_text="",
        )
    job_status = await job.get_status()
    assert job_status == JobStatus.FAILED

    del job
    del service_manager


@pytest.mark.asyncio
async def test_gen_animate_text_only() -> None:
    """StreamAnimateJob.gen_animate with text prompt generates video or fails gracefully."""
    service_manager = MagicMock()
    job = StreamAnimateJob(
        job_id="test_gen_animate_text_only",
        service_manager=service_manager,
    )
    job.gen = LMMGeneratorMock()
    job.config["output_mode"] = "video_audio_unsynced"

    try:
        await job.gen_animate(
            image_base64=None,
            text_prompt="A bird flying over mountains",
            narration_text="",
        )
        job_status = await job.get_status()
        assert job_status == JobStatus.COMPLETED
    except FileNotFoundError:
        # ffmpeg not available in this environment
        job_status = await job.get_status()
        assert job_status == JobStatus.FAILED

    del job
    del service_manager


@pytest.mark.asyncio
async def test_gen_animate_with_narration() -> None:
    """StreamAnimateJob.gen_animate with narration generates video and audio or fails gracefully."""
    service_manager = MagicMock()
    job = StreamAnimateJob(
        job_id="test_gen_animate_with_narration",
        service_manager=service_manager,
    )
    job.gen = LMMGeneratorMock()
    job.config["output_mode"] = "video_audio_unsynced"

    try:
        await job.gen_animate(
            image_base64=None,
            text_prompt="A sunset over the ocean",
            narration_text="Watch the beautiful sunset over the ocean.",
        )
        job_status = await job.get_status()
        assert job_status == JobStatus.COMPLETED
    except FileNotFoundError:
        # ffmpeg not available in this environment
        job_status = await job.get_status()
        assert job_status == JobStatus.FAILED

    del job
    del service_manager
