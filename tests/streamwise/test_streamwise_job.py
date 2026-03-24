#!/usr/bin/env python3
"""
Unit tests for streamwise.py job submission endpoints.
"""

import sys
import pytest

from http import HTTPStatus

from unittest.mock import patch
from unittest.mock import MagicMock

from tests.test_utils import temp_sys_path
from tests.k8s_mock import K8sMock
from tests.torch_mock import TorchMock
from tests.numpy_mock import NumPyMock

mock_k8s = K8sMock()
mock_torch = TorchMock()
mock_numpy = NumPyMock()

mock_modules = {
    "scipy": MagicMock(),
    "scipy.io": MagicMock(),
    "PIL": MagicMock(),
    "PIL.Image": MagicMock(),
    "openai": MagicMock(),
    "imageio": MagicMock(),
    "imageio_ffmpeg": MagicMock(),
}
mock_modules.update(mock_k8s.get_sub_modules())
mock_modules.update(mock_torch.get_sub_modules())
mock_modules.update(mock_numpy.get_sub_modules())
with patch.dict(sys.modules, mock_modules):
    with temp_sys_path("streamwise"):
        from streamwise import streamwise

    with temp_sys_path("apps"):
        from apps.streamwise_job import StreamWiseJob


@pytest.fixture(scope="function", autouse=True)
def setup_k8s_cluster() -> None:
    # for some reason k8s_config.load_kube_config() is not async mocked
    streamwise.k8s_cluster = "unittest"


@pytest.mark.asyncio
async def test_submit_job() -> None:
    app = streamwise.app
    client = app.test_client()

    # Non existing endpoint
    response = await client.get("/job")

    # GET job submission form
    response = await client.get("/job/1.2.3.4/8080")
    assert response.status_code == HTTPStatus.OK
    assert "html" in response.content_type
    response_html = await response.get_data(as_text=True)
    assert response_html.startswith("<!DOCTYPE html>\n<html lang=\"en\">")
    assert "<title>Submit Generation</title>" in response_html


@pytest.mark.asyncio
async def test_api_submit_job() -> None:
    app = streamwise.app
    client = app.test_client()

    # Non existing GET endpoint
    await client.get("/api/job/fantasytalking/1.2.3.4/8080")

    # POST without JSON body
    response = await client.post("/api/job/fantasytalking/1.2.3.4/8080")
    assert response.status_code == HTTPStatus.BAD_REQUEST
    response_json = await response.get_json()
    assert response_json == {"error": "No job data provided"}

    # POST with JSON body
    response = await client.post(
        "/api/job/yolo/10.20.3.4/8989",
        json={"param1": "value1", "param2": 2}
    )
    assert response.status_code in (
        HTTPStatus.GATEWAY_TIMEOUT,
        HTTPStatus.INTERNAL_SERVER_ERROR
    )
    response_json = await response.get_json()
    assert response_json in (
        {"error": "Request timed out"},
        {"error": "Event loop is closed"}
    )


@pytest.mark.asyncio
async def test_get_quality() -> None:
    job_id = "test_job_quality"
    service_manager = MagicMock()
    job = StreamWiseJob("test_app", job_id, service_manager)
    assert job.get_num_steps() == 15
