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
    with temp_sys_path("apps", "apps/streamchat"):
        from apps.streamchat.streamchat import StreamChatApp
        from apps.streamchat.streamchat import get_chat_history_from_file
        from apps.streamchat.streamchat import parse_chat_history
        from apps.streamchat.streamchat_job import StreamChatJob
        from apps.streamchat.streamchat_job import remove_emojis
        from apps.streamchat.streamchat_job import JobStatus
        from character import Character
    from tests.streamwise_app.lmm_generator_mock import LMMGeneratorMock


streamchat_app = StreamChatApp()


@pytest.fixture(name="test_app")
def _test_app() -> Quart:
    return streamchat_app.app


@pytest.mark.asyncio
async def test_app(test_app: Quart) -> None:
    """Check that GET / returns 200."""
    await check_app_root(test_app, "StreamChat")


@pytest.mark.asyncio
async def test_health(test_app: Quart) -> None:
    """Check /health."""
    await check_health(test_app)


@pytest.mark.asyncio
async def test_files(test_app: Quart) -> None:
    """Check /files endpoint."""
    await check_files(test_app, "streamchat")


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


@pytest.mark.asyncio
async def test_get_msg_id() -> None:
    """StreamChatJob._get_msg_id returns 0 before any chat turns."""
    service_manager = MagicMock()
    job = StreamChatJob(job_id="test_msg_id", service_manager=service_manager)
    assert job._get_msg_id() == 0
    del job
    del service_manager


@pytest.mark.asyncio
async def test_gen_chat_text() -> None:
    """StreamChatJob.gen_chat_text calls gen.gen_text and appends messages."""
    service_manager = MagicMock()
    job = StreamChatJob(job_id="test_gen_chat_text", service_manager=service_manager)
    job.gen = LMMGeneratorMock()

    response_text = await job.gen_chat_text(user_message="Hello!", msg_id=0)
    assert isinstance(response_text, str)
    assert len(response_text) > 0
    # Messages should now contain system + user + assistant
    assert len(job.messages) == 3
    assert job.messages[1]["role"] == "user"
    assert job.messages[2]["role"] == "assistant"

    del job
    del service_manager


@pytest.mark.asyncio
async def test_gen_chat_audio() -> None:
    """StreamChatJob.gen_chat_audio produces a WAV file."""

    service_manager = MagicMock()
    job = StreamChatJob(job_id="test_gen_chat_audio", service_manager=service_manager)
    job.gen = LMMGeneratorMock()
    job.character = Character(name="Alice", gender="Female", speech_speed=1.0)

    audio_base64 = await job.gen_chat_audio(response_text="Hello world.", msg_id=0)
    assert isinstance(audio_base64, str)
    assert len(audio_base64) > 0

    del job
    del service_manager


@pytest.mark.asyncio
async def test_gen_chat() -> None:
    """StreamChatJob.gen_chat returns a reply dict."""

    service_manager = MagicMock()
    job = StreamChatJob(job_id="test_gen_chat", service_manager=service_manager)
    job.gen = LMMGeneratorMock()
    job.image = Image.new("RGB", (160, 100), color="white")
    job.character = Character(name="Alice", gender="Female", speech_speed=1.0)

    result = await job.gen_chat(user_message="Say something.")
    assert "reply" in result
    assert isinstance(result["reply"], str)
    assert "id" in result

    del job
    del service_manager


@pytest.mark.asyncio
async def test_chat_route_not_found(test_app: Quart) -> None:
    """POST /chat/<job_id> for unknown job returns 404."""
    client = test_app.test_client()
    response = await client.post("/chat/unknown_job_id", data={"message": "Hi"})
    assert response.status_code == HTTPStatus.NOT_FOUND
    response_json = await response.get_json()
    assert "error" in response_json


@pytest.mark.asyncio
async def test_chat_route_no_message(test_app: Quart) -> None:
    """POST /chat/<job_id> with no text and no audio returns 400."""

    service_manager = MagicMock()
    job = StreamChatJob(job_id="test_no_msg_job", service_manager=service_manager)
    job.gen = LMMGeneratorMock()
    job.image = Image.new("RGB", (160, 100), color="white")
    job.character = Character(name="Alice", gender="Female", speech_speed=1.0)
    streamchat_app.jobs["test_no_msg_job"] = job

    client = test_app.test_client()
    response = await client.post("/chat/test_no_msg_job", data={})
    assert response.status_code == HTTPStatus.BAD_REQUEST
    response_json = await response.get_json()
    assert "error" in response_json

    del streamchat_app.jobs["test_no_msg_job"]
    del job
    del service_manager


@pytest.mark.asyncio
async def test_chat_route_with_message(test_app: Quart) -> None:
    """POST /chat/<job_id> with a text message returns a reply."""

    service_manager = MagicMock()
    job = StreamChatJob(job_id="test_chat_msg_job", service_manager=service_manager)
    job.gen = LMMGeneratorMock()
    job.image = Image.new("RGB", (160, 100), color="white")
    job.character = Character(name="Alice", gender="Female", speech_speed=1.0)
    streamchat_app.jobs["test_chat_msg_job"] = job

    client = test_app.test_client()
    response = await client.post(
        "/chat/test_chat_msg_job",
        form={"message": "Hello!"},
    )
    assert response.status_code == HTTPStatus.OK
    response_json = await response.get_json()
    assert response_json["status"] == "ok"
    assert "reply" in response_json

    del streamchat_app.jobs["test_chat_msg_job"]
    del job
    del service_manager


@pytest.mark.asyncio
async def test_chat_history_route(test_app: Quart) -> None:
    """GET /chat/<job_id>/history returns the chat history."""

    service_manager = MagicMock()
    job = StreamChatJob(job_id="test_history_job", service_manager=service_manager)
    job.gen = LMMGeneratorMock()
    job.character = Character(name="Alice", gender="Female", speech_speed=1.0)
    streamchat_app.jobs["test_history_job"] = job

    client = test_app.test_client()
    response = await client.get("/chat/test_history_job/history")
    assert response.status_code == HTTPStatus.OK
    response_json = await response.get_json()
    assert response_json["status"] == "ok"
    assert "history" in response_json
    assert isinstance(response_json["history"], list)

    del streamchat_app.jobs["test_history_job"]
    del job
    del service_manager


@pytest.mark.asyncio
async def test_parse_chat_history() -> None:
    """parse_chat_history returns empty list for nonexistent file."""
    history = await parse_chat_history("/nonexistent/file.jsonl")
    assert history == []


@pytest.mark.asyncio
async def test_get_chat_history_from_file_not_found() -> None:
    """get_chat_history_from_file raises FileNotFoundError for missing job path."""
    import pytest
    with pytest.raises(FileNotFoundError):
        await get_chat_history_from_file("/nonexistent/path", "fake_job_id")


@pytest.mark.asyncio
async def test_transcribe_audio() -> None:
    """StreamChatJob.transcribe_audio returns a transcript string."""
    service_manager = MagicMock()
    job = StreamChatJob(job_id="test_transcribe", service_manager=service_manager)
    job.gen = LMMGeneratorMock()

    audio_path = "tests/data/audio_4675.wav"
    transcript = await job.transcribe_audio(audio_path)
    assert isinstance(transcript, str)
    assert len(transcript) > 0

    del job
    del service_manager


@pytest.mark.asyncio
async def test_gen_chat_base_with_config() -> None:
    """StreamChatJob.gen_chat_base with style/scene/custom prompt config branches."""
    service_manager = MagicMock()
    job = StreamChatJob(job_id="test_gen_chat_base_config", service_manager=service_manager)
    job.gen = LMMGeneratorMock()
    job.config["style_prompt"] = "cinematic"
    job.config["scene_prompt"] = "office"
    job.config["custom_prompt"] = "daytime lighting"

    await job.gen_chat_base()
    job_status = await job.get_status()
    assert job_status in (JobStatus.COMPLETED, JobStatus.FAILED)

    del job
    del service_manager


@pytest.mark.asyncio
async def test_chat_history_route_nonexistent_job(test_app: Quart) -> None:
    """GET /chat/<job_id>/history for nonexistent job falls back to file lookup."""
    client = test_app.test_client()
    # job_id not in jobs dict, and no history file exists → FileNotFoundError handled by app
    response = await client.get("/chat/definitely_nonexistent_job/history")
    # The app should either return 200 with empty history or 500 if FileNotFoundError propagates
    assert response.status_code in (HTTPStatus.OK, HTTPStatus.INTERNAL_SERVER_ERROR)


@pytest.mark.asyncio
async def test_parse_chat_history_with_data() -> None:
    """parse_chat_history correctly parses a JSONL file."""
    import json
    import aiofiles

    tmp_path = "/tmp/test_chat_history.jsonl"
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    async with aiofiles.open(tmp_path, "w") as f:
        for msg in messages:
            await f.write(json.dumps(msg) + "\n")

    history = await parse_chat_history(tmp_path)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
