#!/usr/bin/env python3
"""
Unit tests for StreamEdit.
"""

import os
import sys
import pytest

from http import HTTPStatus

from PIL import Image

from quart import Quart

from unittest.mock import patch
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

    # Bad video data
    response = await client.post("/api/job", json={"video_base64": "AAAA"})
    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    response_json = await response.get_json()
    assert response_json["status"] == "error"
    assert "error" in response_json
    assert "Ensure file is valid video" in response_json["error"]

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
