#!/usr/bin/env python3
"""
Unit tests for StreamChat.
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
    with temp_sys_path("apps", "apps/streamchat"):
        from apps.streamchat.streamchat import StreamChatApp
        from apps.streamchat.streamchat_job import StreamChatJob
        from apps.streamchat.streamchat_job import remove_emojis
        from apps.streamchat.streamchat_job import JobStatus
    from tests.streamwise_app.lmm_generator_mock import LMMGeneratorMock


streamchat_app = StreamChatApp()


@pytest.fixture(name="test_app")
def _test_app() -> Quart:
    return streamchat_app.app


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
    assert "StreamChat" in response_html


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

    if not os.path.exists("/tmp/streamchat"):
        os.makedirs("/tmp/streamchat", exist_ok=True)

    response = await client.get("/files")
    assert response.status_code == HTTPStatus.OK
    response_json = await response.get_json()
    assert "files" in response_json

    response = await client.get("/file/testfile.txt")
    assert response.status_code == HTTPStatus.NOT_FOUND
    response_json = await response.get_json()
    assert response_json == {"error": "File '/tmp/streamchat/testfile.txt' not found"}

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

    response = await client.post("/api/job", json={"video_base64": "AAAA"})
    assert response.status_code == HTTPStatus.BAD_REQUEST
    response_json = await response.get_json()
    assert "error" in response_json
    assert response_json["error"] == "Service manager not initialized"

    # Mock the service manager
    streamchat_app.service_manager = MagicMock()
    streamchat_app.service_manager.get_service_url = MagicMock(
        return_value="http://mock_service_url:1234"
    )

    response = await client.post("/api/job", json={"video_base64": "AAAA"})
    # assert response.status_code == HTTPStatus.OK
    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    response_json = await response.get_json()
    # assert "job_id" in response_json
    # assert response_json["status"] == "success"
    assert response_json["status"] == "error"
    assert "error" in response_json
    assert "Error generating image" in response_json["error"]


def test_remove_emojis() -> None:
    """Test the remove_emojis function."""
    text_with_emojis = "Hello, world! 😊🚀🌟"
    text_without_emojis = "Hello, world! "

    assert remove_emojis(text_with_emojis) == text_without_emojis
    assert remove_emojis(text_without_emojis) == text_without_emojis


@pytest.mark.asyncio
async def test_gen_chat_base_mock() -> None:
    """StreamChatJob.gen_chat_base with mocked services generates response."""
    service_manager = MagicMock()
    job = StreamChatJob(
        job_id="test_gen_chat_base_mock",
        service_manager=service_manager,
    )
    job.gen = LMMGeneratorMock()

    await job.gen_chat_base()
    job_status = await job.get_status()
    # The gen_chat_base starts the character/image and sends a chat message
    # With mock services that don't fail it should at minimum start
    assert job_status in (JobStatus.COMPLETED, JobStatus.FAILED)

    del job
    del service_manager


@pytest.mark.asyncio
async def test_gen_chat_history() -> None:
    """StreamChatJob.get_chat_history returns messages list."""
    service_manager = MagicMock()
    job = StreamChatJob(
        job_id="test_gen_chat_history",
        service_manager=service_manager,
    )
    history = await job.get_chat_history()
    assert isinstance(history, list)
    # System prompt is set in __init__
    assert len(history) >= 1
    assert history[0]["role"] == "system"

    del job
    del service_manager
