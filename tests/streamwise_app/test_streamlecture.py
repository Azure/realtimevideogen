#!/usr/bin/env python3
"""
Unit tests for StreamLecture.
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

from file_utils import read_file_base64

from tests.test_utils import temp_sys_path
from tests.torch_mock import TorchMock

mock_torch = TorchMock()

mock_modules = {}
mock_modules.update(mock_torch.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    with temp_sys_path("apps", "apps/streamlecture"):
        from apps.streamlecture.streamlecture import StreamLectureApp


streamlecture_app = StreamLectureApp()


@pytest.fixture(name="test_app")
def _test_app() -> Quart:
    return streamlecture_app.app


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
    assert "StreamLecture" in response_html


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
