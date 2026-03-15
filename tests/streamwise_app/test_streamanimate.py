#!/usr/bin/env python3
"""
Unit tests for StreamAnimate.
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
    with temp_sys_path("apps", "apps/streamanimate"):
        from apps.streamanimate.streamanimate import StreamAnimateApp


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
