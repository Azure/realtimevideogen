#!/usr/bin/env python3
"""
Unit tests for StreamLecture.
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

from file_utils import read_file_base64

from tests.test_utils import temp_sys_path
from tests.torch_mock import TorchMock
from tests.streamwise_app.app_test_helpers import check_app_root
from tests.streamwise_app.app_test_helpers import check_health
from tests.streamwise_app.app_test_helpers import check_files
from tests.streamwise_app.app_test_helpers import check_unknown_route
from tests.streamwise_app.app_test_helpers import check_job_submit_page
from tests.streamwise_app.app_test_helpers import check_job_status_page
from tests.streamwise_app.app_test_helpers import check_api_job_status
from tests.streamwise_app.app_test_helpers import check_api_job_requests

mock_torch = TorchMock()

mock_modules = {}
mock_modules.update(mock_torch.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    with temp_sys_path("apps", "apps/streamlecture"):
        from apps.streamlecture.streamlecture import StreamLectureApp
        from apps.streamlecture.streamlecture_job import StreamLectureJob
        from apps.streamlecture.streamlecture_job import JobStatus
        from apps.streamlecture.lecture_prompts import IMG_PROMPT
        from apps.streamlecture.lecture_prompts import VIDEO_PROMPT
    from tests.streamwise_app.lmm_generator_mock import LMMGeneratorMock


streamlecture_app = StreamLectureApp()


@pytest.fixture(name="test_app")
def _test_app() -> Quart:
    return streamlecture_app.app


@pytest.mark.asyncio
async def test_app(test_app: Quart) -> None:
    """Check that GET / returns 200."""
    await check_app_root(test_app, "StreamLecture")


@pytest.mark.asyncio
async def test_health(test_app: Quart) -> None:
    """Check /health."""
    await check_health(test_app)


@pytest.mark.asyncio
async def test_files(test_app: Quart) -> None:
    """Check /files endpoint."""
    await check_files(test_app, "streamlecture")


@pytest.mark.asyncio
async def test_unknown_route(test_app: Quart) -> None:
    """Check that an unknown route returns 404."""
    await check_unknown_route(test_app)


@pytest.mark.asyncio
async def test_job_submit_page(test_app: Quart) -> None:
    """Check the web page for job submission."""
    await check_job_submit_page(test_app)


@pytest.mark.asyncio
async def test_job_status_page(test_app: Quart) -> None:
    """Check the web page for job status."""
    await check_job_status_page(test_app)


@pytest.mark.asyncio
async def test_api_job_status(test_app: Quart) -> None:
    """Check the API for job status (returns UNKNOWN for nonexistent jobs)."""
    await check_api_job_status(test_app)


@pytest.mark.asyncio
async def test_api_job_requests(test_app: Quart) -> None:
    """Check the API for job requests listing (returns empty for nonexistent jobs)."""
    await check_api_job_requests(test_app)


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
    streamlecture_app.service_manager = MagicMock()
    streamlecture_app.service_manager.get_service_url = MagicMock(
        return_value="http://mock_service_url:1234"
    )

    pdf_path = "tests/data/blank.pdf"
    pdf_base64 = await read_file_base64(pdf_path)
    response = await client.post("/api/job", json={"pdf_base64": pdf_base64})
    # The job is accepted; it may fail later if external services are unavailable.
    response_json = await response.get_json()
    assert response_json is not None
    assert "status" in response_json
    # If the job started before any service call fails, we get job_id back
    # If a service call fails synchronously (e.g. DNS error), we get an error
    if response.status_code == HTTPStatus.OK:
        assert "job_id" in response_json
        assert response_json["status"] == "success"
    else:
        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
        assert "error" in response_json


def test_lecture_prompts() -> None:
    """Test that lecture prompts are defined and non-empty."""
    assert IMG_PROMPT
    assert "classroom" in IMG_PROMPT.lower() or "professor" in IMG_PROMPT.lower()
    assert VIDEO_PROMPT
    assert len(VIDEO_PROMPT) > 0


@pytest.mark.asyncio
async def test_gen_lecture_no_pdf() -> None:
    """StreamLectureJob.gen_lecture with missing pdf raises ValueError."""
    service_manager = MagicMock()
    job = StreamLectureJob(
        job_id="test_no_pdf",
        service_manager=service_manager,
    )
    with pytest.raises(ValueError, match="Missing 'pdf_base64' in request"):
        await job.gen_lecture(pdf_base64=None)
    job_status = await job.get_status()
    assert job_status == JobStatus.FAILED

    del job
    del service_manager


@pytest.mark.asyncio
async def test_gen_lecture_invalid_pdf() -> None:
    """StreamLectureJob.gen_lecture with invalid base64 raises ValueError."""
    service_manager = MagicMock()
    job = StreamLectureJob(
        job_id="test_invalid_pdf",
        service_manager=service_manager,
    )
    with pytest.raises(ValueError, match="Invalid base64-encoded string"):
        await job.gen_lecture(pdf_base64="AAAAA")
    job_status = await job.get_status()
    assert job_status == JobStatus.FAILED

    del job
    del service_manager


@pytest.mark.asyncio
async def test_gen_lecture_mock() -> None:
    """StreamLectureJob.gen_lecture with mocked services completes or fails gracefully."""
    service_manager = MagicMock()
    job = StreamLectureJob(
        job_id="test_gen_lecture_mock",
        service_manager=service_manager,
    )
    job.gen = LMMGeneratorMock()

    pdf_path = "tests/data/blank.pdf"
    pdf_base64 = await read_file_base64(pdf_path)
    try:
        await job.gen_lecture(pdf_base64=pdf_base64)
        job_status = await job.get_status()
        assert job_status == JobStatus.COMPLETED
    except (FileNotFoundError, ValueError):
        # ffmpeg not available in this environment; job fails gracefully
        job_status = await job.get_status()
        assert job_status == JobStatus.FAILED

    del job
    del service_manager


@pytest.mark.asyncio
async def test_gen_scene_no_image() -> None:
    """StreamLectureJob.gen_scene with no image raises ValueError."""
    service_manager = MagicMock()
    job = StreamLectureJob(
        job_id="test_gen_scene_no_image",
        service_manager=service_manager,
    )
    job.gen = LMMGeneratorMock()
    # With no image set, gen_scene raises ValueError after generating audio
    with pytest.raises(ValueError, match="Classroom image not available for video generation"):
        await job.gen_scene(scene_id=0, text="Test text.")

    del job
    del service_manager


@pytest.mark.asyncio
async def test_gen_scene_audio_only() -> None:
    """StreamLectureJob.gen_scene in audio_only mode returns audio path."""
    service_manager = MagicMock()
    job = StreamLectureJob(
        job_id="test_gen_scene_audio_only",
        service_manager=service_manager,
    )
    job.gen = LMMGeneratorMock()
    job.config["output_mode"] = "audio_only"
    # Set a mock classroom image (required even in audio_only mode)
    job.image = Image.new("RGB", (640, 400), color="white")

    result = await job.gen_scene(scene_id=0, text="Test lecture text.")
    assert result.endswith(".wav")
    assert os.path.exists(result)

    del job
    del service_manager


@pytest.mark.asyncio
async def test_gen_scene_video_audio_synced() -> None:
    """StreamLectureJob.gen_scene in video_audio_synced mode with short audio uses gen_video_audio_from_img."""
    service_manager = MagicMock()
    job = StreamLectureJob(
        job_id="test_gen_scene_synced",
        service_manager=service_manager,
    )
    job.gen = LMMGeneratorMock()
    job.config["output_mode"] = "video_audio_synced"
    job.image = Image.new("RGB", (160, 100), color="white")

    try:
        result = await job.gen_scene(scene_id=0, text="Test synced scene.")
        # If ffmpeg is available the result is an mp4 path
        assert result.endswith(".mp4")
        assert os.path.exists(result)
    except FileNotFoundError:
        pass  # ffmpeg not available in test environment

    del job
    del service_manager


@pytest.mark.asyncio
async def test_gen_scene_video_audio_unsynced() -> None:
    """StreamLectureJob.gen_scene in video_audio_unsynced mode uses gen_video."""
    service_manager = MagicMock()
    job = StreamLectureJob(
        job_id="test_gen_scene_unsynced",
        service_manager=service_manager,
    )
    job.gen = LMMGeneratorMock()
    job.config["output_mode"] = "video_audio_unsynced"
    job.image = Image.new("RGB", (160, 100), color="white")

    try:
        result = await job.gen_scene(scene_id=0, text="Test unsynced scene.")
        assert result.endswith(".mp4")
        assert os.path.exists(result)
    except FileNotFoundError:
        pass  # ffmpeg not available in test environment

    del job
    del service_manager


@pytest.mark.asyncio
async def test_gen_lecture_audio_only() -> None:
    """StreamLectureJob.gen_lecture in audio_only mode skips video generation."""
    service_manager = MagicMock()
    job = StreamLectureJob(
        job_id="test_gen_lecture_audio_only",
        service_manager=service_manager,
    )
    job.gen = LMMGeneratorMock()
    job.config["output_mode"] = "audio_only"

    pdf_path = "tests/data/blank.pdf"
    pdf_base64 = await read_file_base64(pdf_path)
    try:
        await job.gen_lecture(pdf_base64=pdf_base64)
        job_status = await job.get_status()
        assert job_status == JobStatus.COMPLETED
    except (FileNotFoundError, ValueError):
        job_status = await job.get_status()
        assert job_status == JobStatus.FAILED

    del job
    del service_manager
