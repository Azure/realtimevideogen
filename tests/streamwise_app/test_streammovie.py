#!/usr/bin/env python3
"""
Unit tests for StreamMovie.
"""

import os
import sys
import json
import pytest
import aiofiles

from PIL import Image  # noqa: F401 - import before patch.dict to keep PIL in sys.modules

from http import HTTPStatus

from typing import AsyncGenerator
from typing import Dict
from typing import Any

from quart import Quart

from unittest.mock import patch
from unittest.mock import MagicMock

# Add current path
sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path
from tests.torch_mock import TorchMock
from tests.k8s_mock import K8sMock
from tests.fantasytalking_mock import FantasyTalkingMock

mock_torch = TorchMock()
mock_k8s = K8sMock()
mock_ft = FantasyTalkingMock()

mock_modules = {
    "imageio": MagicMock(),
    "tabulate": MagicMock(),
    "soundfile": MagicMock(),
}
mock_modules.update(mock_torch.get_sub_modules())
mock_modules.update(mock_k8s.get_sub_modules())
mock_modules.update(mock_ft.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    with temp_sys_path("apps", "apps/streammovie"):
        from apps.streammovie.streammovie import StreamMovieApp
        from apps.streammovie.streammovie_job import StreamMovieJob
        from apps.streammovie.streammovie_job import JobStatus

    from tests.streamwise_app.lmm_generator_mock import LMMGeneratorMock


streammovie_app = StreamMovieApp()


@pytest.fixture(name="test_app")
def _test_app() -> Quart:
    return streammovie_app.app


# ---------------------------------------------------------------------------
# Helper: a mock LMMGeneratorMock that also supports gen_text_stream
# ---------------------------------------------------------------------------

# Sample JSONL script returned by the LLM mock
MOCK_SHOT_0 = {
    "type": "shot_description",
    "shot_id": "S001",
    "visual_prompt": "A wide-angle cinematic shot of a cyberpunk city at night.",
    "negative_prompt": "blur, watermark",
    "dialogue": None,
    "technical_specs": {"duration_seconds": 4},
}

MOCK_SHOT_1 = {
    "type": "shot_description",
    "shot_id": "S002",
    "visual_prompt": "Close-up of the hero's determined face.",
    "negative_prompt": "blur",
    "dialogue": "I will stop them.",
    "technical_specs": {"duration_seconds": 3},
}


def _make_mock_script_chunks():
    """Yield JSONL lines as streaming chunks."""
    lines = [
        json.dumps({"type": "movie_metadata", "title": "Test Movie"}),
        json.dumps(MOCK_SHOT_0),
        json.dumps(MOCK_SHOT_1),
    ]
    for line in lines:
        yield line + "\n"


def _make_mock_script_chunks_with_noise():
    """Yield a mix of valid JSONL and prose/noise lines."""
    items = [
        "Okay, here's the Movie Bible for a test film.",  # prose noise
        json.dumps({"type": "movie_metadata", "title": "Noisy Movie"}),
        "```jsonl",  # markdown fence noise
        json.dumps(MOCK_SHOT_0),
        "This is an explanatory paragraph that should be filtered out.",  # prose noise
        json.dumps(MOCK_SHOT_1),
        "```",  # markdown fence noise
        "And now, some notes about the next act...",  # prose noise
    ]
    for item in items:
        yield item + "\n"


class LMMGeneratorMovieMock(LMMGeneratorMock):
    """LMMGeneratorMock that handles gen_text_stream for movie script generation."""

    def __init__(self) -> None:
        super().__init__()
        self.last_messages: list = []

    async def gen_text_stream(
        self,
        *args,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        self.last_messages = kwargs.get("messages", [])
        for chunk in _make_mock_script_chunks():
            yield chunk

    async def gen_text(
        self,
        *args,
        **kwargs,
    ) -> str:
        return ""

    async def stop(self) -> None:
        pass


class LMMGeneratorMovieNoisyMock(LMMGeneratorMock):
    """LMMGeneratorMock that emits prose noise mixed with valid JSONL."""

    async def gen_text_stream(
        self,
        *args,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        for chunk in _make_mock_script_chunks_with_noise():
            yield chunk

    async def gen_text(
        self,
        *args,
        **kwargs,
    ) -> str:
        return ""

    async def stop(self) -> None:
        pass


def _make_job(job_id: str, config: Dict[str, Any] | None = None) -> StreamMovieJob:
    """Create a StreamMovieJob with a mocked service manager and generator."""
    service_manager = MagicMock()
    service_manager.get_service_url = MagicMock(return_value="http://mock:1234")
    job = StreamMovieJob(job_id=job_id, config=config or {}, service_manager=service_manager)
    job.gen = LMMGeneratorMovieMock()
    return job


def _make_noisy_job(job_id: str, config: Dict[str, Any] | None = None) -> StreamMovieJob:
    """Create a StreamMovieJob backed by the noisy LLM mock."""
    service_manager = MagicMock()
    service_manager.get_service_url = MagicMock(return_value="http://mock:1234")
    job = StreamMovieJob(job_id=job_id, config=config or {}, service_manager=service_manager)
    job.gen = LMMGeneratorMovieNoisyMock()
    return job


# ---------------------------------------------------------------------------
# HTTP route tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_app(test_app: Quart) -> None:
    """Check that GET / returns 200 with the correct content."""
    client = test_app.test_client()
    response = await client.get("/")
    assert response is not None
    assert response.status_code == HTTPStatus.OK
    assert "text/html; charset=utf-8" == response.content_type
    response_html = await response.get_data(as_text=True)
    assert response_html.startswith("<!DOCTYPE html>\n<html lang=\"en\">")
    assert "StreamMovie" in response_html


@pytest.mark.asyncio
async def test_submit_job_page(test_app: Quart) -> None:
    """Check that GET /job returns the submit job form."""
    client = test_app.test_client()
    response = await client.get("/job")
    assert response.status_code == HTTPStatus.OK
    response_html = await response.get_data(as_text=True)
    assert "movie_description" in response_html


@pytest.mark.asyncio
async def test_submit_job_no_service_manager(test_app: Quart) -> None:
    """POST /api/job without service manager should return 400."""
    # Reset service manager to None
    original = streammovie_app.service_manager
    streammovie_app.service_manager = None
    try:
        client = test_app.test_client()
        response = await client.post("/api/job", json={"movie_description": "Test movie"})
        assert response.status_code == HTTPStatus.BAD_REQUEST
        response_json = await response.get_json()
        assert "error" in response_json
        assert response_json["error"] == "Service manager not initialized"
    finally:
        streammovie_app.service_manager = original


@pytest.mark.asyncio
async def test_submit_job_no_description(test_app: Quart) -> None:
    """POST /api/job without movie_description triggers job failure, returning 400."""
    streammovie_app.service_manager = MagicMock()
    streammovie_app.service_manager.get_service_url = MagicMock(
        return_value="http://mock_service_url:1234"
    )
    client = test_app.test_client()
    response = await client.post("/api/job", json={"video_base64": "AAAA"})
    # missing movie_description causes the job to fail with ValueError → 400 BAD REQUEST
    assert response.status_code == HTTPStatus.BAD_REQUEST
    response_json = await response.get_json()
    assert response_json is not None
    assert response_json["status"] == "error"
    assert "error" in response_json
    assert "movie_description" in response_json["error"]

# ---------------------------------------------------------------------------
# StreamMovieJob unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_shot_deadline() -> None:
    """Deadline should equal submission_time + offset + buffer."""
    job = _make_job("test_deadline")
    t0 = job.get_submission_time()

    deadline_0 = job._get_shot_deadline(0, 4.0)
    deadline_1 = job._get_shot_deadline(1, 4.0)
    deadline_5 = job._get_shot_deadline(5, 4.0)

    assert deadline_0 == pytest.approx(t0 + 0 * 4.0 + 120.0)
    assert deadline_1 == pytest.approx(t0 + 1 * 4.0 + 120.0)
    assert deadline_5 == pytest.approx(t0 + 5 * 4.0 + 120.0)

    # Later shots have a later deadline (gives earlier shots higher priority)
    assert deadline_1 > deadline_0
    assert deadline_5 > deadline_1

    await job.close()


@pytest.mark.asyncio
async def test_try_parse_json() -> None:
    """Valid JSON parses correctly; invalid JSON returns None."""
    job = _make_job("test_parse_json")

    result = job._try_parse_json('{"type": "shot_description", "shot_id": 1}', 1)
    assert result is not None
    assert result["type"] == "shot_description"

    result = job._try_parse_json("not json at all", 2)
    assert result is None

    result = job._try_parse_json("", 3)
    assert result is None

    await job.close()


@pytest.mark.asyncio
async def test_stream_movie_script() -> None:
    """_stream_movie_script should return shot_description objects from LLM stream."""
    job = _make_job("test_stream_script")

    shots = await job._stream_movie_script("A cyberpunk heist movie.")

    # Two shot_description objects are yielded by the mock
    assert len(shots) == 2
    assert shots[0]["shot_id"] == "S001"
    assert shots[1]["shot_id"] == "S002"

    # Script file should be saved
    script_path = f"{job.job_path}/movie_script.jsonl"
    assert await aiofiles.os.path.exists(script_path)
    async with aiofiles.open(script_path) as f:
        content = await f.read()
    # All three JSONL lines should be present (metadata + 2 shots)
    lines = [line for line in content.strip().splitlines() if line]
    assert len(lines) == 3

    await job.close()


@pytest.mark.asyncio
async def test_stream_movie_script_returns_all_shots() -> None:
    """_stream_movie_script returns ALL parsed shots; max_shots cap is applied in gen_movie."""
    job = _make_job("test_all_shots", config={"max_shots": 1})

    shots = await job._stream_movie_script("A sci-fi adventure.")
    # The LLM mock emits 2 shots; _stream_movie_script collects all of them.
    # The max_shots cap (1) is enforced by gen_movie, not _stream_movie_script.
    assert len(shots) == 2

    await job.close()


@pytest.mark.asyncio
async def test_gen_shot_no_dialogue() -> None:
    """_gen_shot without dialogue should produce a plain video."""
    job = _make_job("test_gen_shot_no_dialogue", config={"output_mode": "video_audio_synced"})

    shot = {
        "visual_prompt": "A wide panoramic shot of mountains.",
        "negative_prompt": "",
        "dialogue": None,
        "technical_specs": {"duration_seconds": 4.0},
    }

    shot_path = await job._gen_shot(0, shot)
    assert shot_path is not None
    assert shot_path.endswith(".mp4")
    assert await aiofiles.os.path.exists(shot_path)

    # Image for the shot should also have been saved
    image_path = f"{job.job_path}/shot_000.png"
    assert await aiofiles.os.path.exists(image_path)

    await job.close()


@pytest.mark.asyncio
async def test_gen_shot_with_dialogue_synced() -> None:
    """_gen_shot with dialogue in VIDEO_AUDIO_SYNCED mode uses gen_video_audio_from_img."""
    job = _make_job("test_gen_shot_synced", config={"output_mode": "video_audio_synced"})

    shot = {
        "visual_prompt": "Close-up of the hero.",
        "negative_prompt": "blur",
        "dialogue": "I will stop them.",
        "technical_specs": {"duration_seconds": 3.0},
    }

    shot_path = await job._gen_shot(0, shot)
    assert shot_path is not None
    assert shot_path.endswith(".mp4")
    assert await aiofiles.os.path.exists(shot_path)

    await job.close()


@pytest.mark.asyncio
async def test_gen_shot_with_dialogue_unsynced() -> None:
    """_gen_shot with dialogue in VIDEO_AUDIO_UNSYNCED mode merges audio separately."""
    job = _make_job("test_gen_shot_unsynced", config={"output_mode": "video_audio_unsynced"})

    shot = {
        "visual_prompt": "Hero walking through rain.",
        "negative_prompt": "",
        "dialogue": "We have to go now.",
        "technical_specs": {"duration_seconds": 3.0},
    }

    shot_path = await job._gen_shot(0, shot)
    assert shot_path is not None
    assert shot_path.endswith(".mp4")
    assert await aiofiles.os.path.exists(shot_path)

    await job.close()


@pytest.mark.asyncio
async def test_gen_shot_with_dialogue_audio_only() -> None:
    """In AUDIO_ONLY mode, dialogue presence doesn't trigger video+audio from img path."""
    job = _make_job("test_gen_shot_audio_only", config={"output_mode": "audio_only"})

    shot = {
        "visual_prompt": "Hero close-up.",
        "negative_prompt": "",
        "dialogue": "Audio only dialogue.",
        "technical_specs": {"duration_seconds": 4.0},
    }

    shot_path = await job._gen_shot(0, shot)
    assert shot_path is not None
    assert shot_path.endswith(".mp4")
    assert await aiofiles.os.path.exists(shot_path)

    await job.close()


@pytest.mark.asyncio
async def test_gen_movie_missing_description() -> None:
    """gen_movie with no description should fail immediately with ValueError."""
    job = _make_job("test_gen_movie_no_desc")

    with pytest.raises(ValueError, match="Missing 'movie_description'"):
        await job.gen_movie(None)

    job_status = await job.get_status()
    assert job_status == JobStatus.FAILED

    await job.close()


@pytest.mark.asyncio
async def test_gen_movie_success() -> None:
    """gen_movie end-to-end: produces a final .mp4 file."""
    job = _make_job("test_gen_movie_success", config={"output_mode": "video_audio_synced"})

    await job.gen_movie("A short heist movie in Neo-Tokyo.")

    job_status = await job.get_status()
    assert job_status == JobStatus.COMPLETED

    final_path = f"{job.job_path}/{job.job_id}.mp4"
    assert await aiofiles.os.path.exists(final_path)

    await job.close()


@pytest.mark.asyncio
async def test_gen_movie_max_shots_limits_output() -> None:
    """When max_shots=1, only one shot should be generated."""
    job = _make_job("test_gen_movie_max1", config={"max_shots": 1})

    await job.gen_movie("A one-shot thriller.")

    # Only shot_000.mp4 should exist; shot_001 should not
    assert await aiofiles.os.path.exists(f"{job.job_path}/shot_000.mp4")
    assert not await aiofiles.os.path.exists(f"{job.job_path}/shot_001.mp4")

    job_status = await job.get_status()
    assert job_status == JobStatus.COMPLETED

    await job.close()


@pytest.mark.asyncio
async def test_gen_movie_script_saved() -> None:
    """After gen_movie, the raw JSONL script should be on disk."""
    job = _make_job("test_gen_movie_script_saved")

    await job.gen_movie("A space adventure.")

    script_path = f"{job.job_path}/movie_script.jsonl"
    assert await aiofiles.os.path.exists(script_path)
    async with aiofiles.open(script_path) as f:
        content = await f.read()
    assert "shot_description" in content

    await job.close()


def test_build_movie_messages() -> None:
    """Test that build_movie_messages includes SYSTEM_PROMPT and user description."""
    messages = StreamMovieJob.build_movie_messages("a sci-fi thriller")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "filmmaker" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "sci-fi thriller" in messages[1]["content"]
    # No shot count instruction when max_shots is not specified
    assert "EXACTLY" not in messages[1]["content"]


def test_build_movie_messages_with_max_shots() -> None:
    """build_movie_messages with max_shots includes the exact-count instruction."""
    messages = StreamMovieJob.build_movie_messages("a heist drama", max_shots=3)
    assert len(messages) == 2
    user_content = messages[1]["content"]
    assert "heist drama" in user_content
    assert "EXACTLY 3 shots" in user_content


@pytest.mark.asyncio
async def test_api_get_movie_script_jsonl(test_app: Quart) -> None:
    """GET /api/job/{job_id}/movie_script.jsonl returns the raw JSONL content as text."""
    job = _make_job("test_api_jsonl")
    await job._stream_movie_script("A noir detective story.")

    client = test_app.test_client()
    response = await client.get(f"/api/job/{job.job_id}/movie_script.jsonl")
    assert response.status_code == HTTPStatus.OK
    content = await response.get_data(as_text=True)
    assert "shot_description" in content
    assert "movie_metadata" in content

    await job.close()


@pytest.mark.asyncio
async def test_api_get_movie_script_jsonl_not_found(test_app: Quart) -> None:
    """GET /api/job/{job_id}/movie_script.jsonl returns error JSON when file is missing."""
    client = test_app.test_client()
    response = await client.get("/api/job/nonexistent_job/movie_script.jsonl")
    assert response.status_code == HTTPStatus.OK
    data = await response.get_json()
    assert data is not None
    assert data["status"] == "error"
    assert "not found" in data["error"]


@pytest.mark.asyncio
async def test_stream_movie_script_filters_noise() -> None:
    """Non-JSON prose lines from the LLM are excluded from the saved script file."""
    job = _make_noisy_job("test_noise_filter")

    shots = await job._stream_movie_script("A noisy sci-fi drama.")

    # 2 shot_description objects should still be extracted despite the noise
    assert len(shots) == 2
    assert shots[0]["shot_id"] == "S001"
    assert shots[1]["shot_id"] == "S002"

    # The saved script file must contain ONLY valid JSON lines
    script_path = f"{job.job_path}/movie_script.jsonl"
    assert await aiofiles.os.path.exists(script_path)
    async with aiofiles.open(script_path) as f:
        content = await f.read()

    lines = [line for line in content.strip().splitlines() if line]
    # Only 3 valid JSON lines: movie_metadata + 2 shot_descriptions
    assert len(lines) == 3
    for line in lines:
        parsed = json.loads(line)  # must not raise
        assert "type" in parsed

    # Noise strings must NOT appear in the file at all
    assert "Okay, here" not in content
    assert "explanatory paragraph" not in content
    assert "notes about the next act" not in content
    assert "```" not in content

    await job.close()


@pytest.mark.asyncio
async def test_stream_movie_script_max_shots_instruction() -> None:
    """When max_shots is set, the LLM receives an exact shot-count instruction."""
    job = _make_job("test_max_shots_instruction", config={"max_shots": 3})
    mock_gen = job.gen  # type: ignore[assignment]

    await job._stream_movie_script("A thriller.", max_shots=3)

    # The last messages sent to the LLM must include the exact-count instruction
    assert mock_gen.last_messages, "No messages were captured by the mock"
    user_message = mock_gen.last_messages[-1]
    assert user_message["role"] == "user"
    assert "EXACTLY 3 shots" in user_message["content"]

    await job.close()


@pytest.mark.asyncio
async def test_stream_movie_script_no_shot_instruction_when_unset() -> None:
    """When max_shots is not set, the LLM message has no exact-count instruction."""
    job = _make_job("test_no_shot_instruction")
    mock_gen = job.gen  # type: ignore[assignment]

    await job._stream_movie_script("A comedy.", max_shots=-1)

    assert mock_gen.last_messages, "No messages were captured by the mock"
    user_message = mock_gen.last_messages[-1]
    assert "EXACTLY" not in user_message["content"]

    await job.close()
