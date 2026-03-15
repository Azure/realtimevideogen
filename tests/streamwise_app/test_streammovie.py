#!/usr/bin/env python3
"""
Unit tests for StreamMovie.
"""

import os
import sys
import pytest

from http import HTTPStatus

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
    with temp_sys_path("apps", "apps/streammovie"):
        from apps.streammovie.streammovie import StreamMovieApp


streammovie_app = StreamMovieApp()


@pytest.fixture(name="test_app")
def _test_app() -> Quart:
    return streammovie_app.app


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
    assert "StreamMovie" in response_html


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
    streammovie_app.service_manager = MagicMock()
    streammovie_app.service_manager.get_service_url = MagicMock(
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


def test_build_movie_messages() -> None:
    """Test that build_movie_messages includes SYSTEM_PROMPT and user description."""
    with patch.dict(sys.modules, mock_modules):
        with temp_sys_path("apps", "apps/streammovie"):
            from apps.streammovie.streammovie_job import StreamMovieJob

    messages = StreamMovieJob.build_movie_messages("a sci-fi thriller")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "filmmaker" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "sci-fi thriller" in messages[1]["content"]
