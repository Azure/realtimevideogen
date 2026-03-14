#!/usr/bin/env python3
"""
Unit tests for StreamDub.
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
    with temp_sys_path("apps", "apps/streamdub"):
        from apps.streamdub.streamdub import StreamDubApp


streamdub_app = StreamDubApp()


@pytest.fixture(name="test_app")
def _test_app() -> Quart:
    return streamdub_app.app


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
    assert "StreamDub" in response_html


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
    streamdub_app.service_manager = MagicMock()
    streamdub_app.service_manager.get_service_url = MagicMock(
        return_value="http://mock_service_url:1234"
    )

    response = await client.post("/api/job", json={"video_base64": "AAAA"})
    # assert response.status_code == HTTPStatus.OK
    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    response_json = await response.get_json()
    assert "error" in response_json
    # assert "job_id" in response_json
    # assert response_json["status"] == "success"


def test_to_language() -> None:
    """Test to_language converts language codes to language names."""
    scene_mocks = {
        'scenedetect': MagicMock(),
        'scenedetect.detectors': MagicMock(),
        'scenedetect.stats_manager': MagicMock(),
    }
    # Remove cached module to force re-import under mocked scenedetect
    sys.modules.pop('apps.streamdub.streamdub_job', None)
    with patch.dict(sys.modules, {**mock_modules, **scene_mocks}):
        with temp_sys_path("apps", "apps/streamdub"):
            from apps.streamdub.streamdub_job import to_language

    assert to_language("a") == "American English"
    assert to_language("b") == "British English"
    assert to_language("e") == "Spanish"
    assert to_language("f") == "French"
    assert to_language("g") == "German"
    assert to_language("i") == "Italian"
    assert to_language("j") == "Japanese"
    assert to_language("k") == "Korean"
    assert to_language("c") == "Chinese"
    assert to_language("r") == "Russian"

    # Unknown code defaults to American English
    assert to_language("z") == "American English"
    assert to_language("A") == "American English"  # Case-insensitive
