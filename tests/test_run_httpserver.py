#!/usr/bin/env python3

import sys
import pytest
import os
import tempfile

from http import HTTPStatus

from unittest.mock import patch
from unittest.mock import MagicMock

from tests.test_utils import temp_sys_path
from tests.torch_mock import TorchMock

TMP_DIR = "tmp"

mock_torch = TorchMock()


with temp_sys_path("wrapper"):
    from tests.test_wrapper_model import MockModelGeneration

with patch.dict(sys.modules, {
    'nvidia_smi': MagicMock(),
    'colorlog': MagicMock(),
    'torch': mock_torch,
    'torch.distributed': MagicMock(),
    'imageio': MagicMock(),
    'cv2': MagicMock(),
    'xfuser': MagicMock(),
}):
    with temp_sys_path("wrapper"):
        import run_httpserver


@pytest.fixture(autouse=True)
def clear_models() -> None:
    run_httpserver.models = {}


@pytest.mark.asyncio
async def test_files() -> None:
    """Check the files endpoints."""
    app = run_httpserver.app
    client = app.test_client()
    response = await client.get("/files")
    assert response is not None
    response_json = await response.json
    assert isinstance(response_json, dict)

    response = await client.get("/file/filename.txt")
    assert response is not None
    assert response.status_code == HTTPStatus.NOT_FOUND
    response_json = await response.json
    assert isinstance(response_json, dict)
    assert "error" in response_json
    assert response_json["error"] == "File not found"

    # Create temp file in /tmp
    with tempfile.NamedTemporaryFile(mode="w", delete=False, dir="/tmp", suffix=".txt") as tmp_file:
        tmp_file.write("Test file content")
        tmp_filename = tmp_file.name
    filename = os.path.basename(tmp_filename)

    # Download
    response = await client.get(f"/file/{filename}")
    assert response is not None
    assert response.status_code == HTTPStatus.OK
    content = await response.get_data(as_text=True)
    assert content == "Test file content"

    # Info
    response = await client.get(f"/file_info/{filename}")
    assert response is not None
    assert response.status_code == HTTPStatus.OK
    response_json = await response.json
    assert len(response_json) > 0
    assert response_json["size"] == 17
    assert response_json["mimetype"] == "text/plain"
    assert response_json["name"].endswith(".txt")
    assert response_json["date"] > 0
    assert response_json["type"] == "text"
    # get_text_file_info()
    assert response_json["num_chars"] == 17
    assert response_json["num_words"] == 3
    assert response_json["num_lines"] == 1

    os.unlink(tmp_filename)


@pytest.mark.timeout(5)
@pytest.mark.asyncio
async def test_yolo() -> None:
    """Check the YOLO endpoint."""
    app = run_httpserver.app
    client = app.test_client()
    response = await client.post('/yolo')
    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR


@pytest.mark.timeout(5)
@pytest.mark.asyncio
async def test_transcript() -> None:
    """Check the transcript endpoint."""
    app = run_httpserver.app
    client = app.test_client()
    response = await client.post("/podcasttranscript")
    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    response_json = await response.json
    assert isinstance(response_json, dict)
    assert "error" in response_json
    assert response_json["error"] == "Podcast transcript model not initialized"

    run_httpserver.models = {"podcasttranscript": MockModelGeneration()}
    response = await client.post("/podcasttranscript")
    assert response.status_code == HTTPStatus.BAD_REQUEST
    response_json = await response.json
    assert isinstance(response_json, dict)
    assert "error" in response_json
    assert response_json["error"] == "No JSON body received"

    # Setting up the mock model
    run_httpserver.models = {"podcasttranscript": MockModelGeneration()}

    # Transcript mocked
    response = await client.post(
        "/podcasttranscript",
        json={"args": {}})
    assert response.status_code == HTTPStatus.BAD_REQUEST
    response_json = await response.json
    assert isinstance(response_json, dict)
    assert "error" in response_json
    assert response_json["error"] == "Model not initialized. Current status: initializing."

    # Transcript stream
    response = await client.post("/podcasttranscript/stream")
    assert response.status_code == HTTPStatus.BAD_REQUEST

    with pytest.raises(AssertionError):
        response = await client.post(
            "/podcasttranscript/stream",
            json={
                "args": {
                    "prompt": "Test prompt"
                }
            })
        assert response.status_code == HTTPStatus.BAD_REQUEST


@pytest.mark.asyncio
async def test_kokoro() -> None:
    """Check the kokoro endpoint."""
    app = run_httpserver.app
    client = app.test_client()
    response = await client.post("/kokoro")
    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR


@pytest.mark.asyncio
async def test_service_endpoints() -> None:
    app = run_httpserver.app
    client = app.test_client()
    service_names = [
        "flux", "fluxkontext", "fluxkrea", "fluxupscaler", "fluxupscaler/video",
        "fantasytalking", "hunyuanavatar",
        "kokoro", "xtts", "vibevoice",
        "hidream",
        "realesrgan", "realesrgan/video",
        "qwenimage", "qwenimageedit",
        "realesrgan", "imageresize",
        "bagel", "llamagen", "januspro",
        "thinksound", "dia",
        "ltx", "wan", "wan22",
        "hunyuanframepack", "hunyuanframepackf1",
    ]
    for service_name in service_names:
        response = await client.post(f"/{service_name}")
        status = response.status_code
        assert status == HTTPStatus.INTERNAL_SERVER_ERROR, f"Service {service_name} failed with {status}"


@pytest.mark.asyncio
async def test_service_health() -> None:
    """Check the service health endpoint."""
    app = run_httpserver.app
    client = app.test_client()
    response = await client.get("/wan/health")
    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR


@pytest.mark.asyncio
async def test_mock_model() -> None:
    """Check the service health endpoint."""
    app = run_httpserver.app
    client = app.test_client()

    # Setup mock model
    run_httpserver.models = {"mockmodel": MockModelGeneration()}

    # Health
    response = await client.get("/mockmodel/health")
    assert response.status_code == HTTPStatus.OK
    model_health = await response.json
    assert len(model_health) > 0
    assert "gen_timer" in model_health


@pytest.mark.asyncio
async def test_yolo_mock() -> None:
    """Check YOLO using mocks."""
    app = run_httpserver.app
    client = app.test_client()

    # Setup mock model for YOLO
    run_httpserver.models = {
        "yolo": MockModelGeneration(output_type="list_pillow")
    }

    # Health
    response = await client.get("/yolo/health")
    assert response.status_code == HTTPStatus.OK
    model_health = await response.json
    assert len(model_health) > 0
    assert "gen_timer" in model_health

    # Generate with no body
    response = await client.post("/yolo")
    assert response.status_code == HTTPStatus.BAD_REQUEST
    response_json = await response.json
    assert "error" in response_json
    error_msg = response_json["error"]
    assert "No JSON body received" in error_msg

    # Generate with wrong body
    response = await client.post(
        "/yolo",
        json={"foo": "bar"})
    assert response.status_code == HTTPStatus.BAD_REQUEST

    # Generate without arguments
    response = await client.post(
        "/yolo",
        json={
            "args": {},
            "job_id": "job0",
        },
        headers={"Content-Type": "application/json"})
    assert response.status_code == HTTPStatus.BAD_REQUEST
    # TODO need to fix this


@pytest.mark.asyncio
async def test_index() -> None:
    """Check the index endpoint."""
    app = run_httpserver.app
    client = app.test_client()
    response = await client.get("/")
    assert response.status_code == HTTPStatus.OK
    index_html = await response.get_data(as_text=True)
    assert index_html is not None
    assert "LMM Models" in index_html


@pytest.mark.asyncio
async def test_timestamps() -> None:
    """Check the timestamps endpoint."""
    app = run_httpserver.app
    client = app.test_client()
    response = await client.get("/timestamps")
    assert response.status_code == HTTPStatus.OK
    timestamps = await response.json
    assert timestamps == {}

    run_httpserver.models = {"mockmodel": MockModelGeneration()}
    response = await client.get("/timestamps")
    assert response.status_code == HTTPStatus.OK
    timestamps = await response.json
    assert timestamps is not None
    assert "mockmodel" in timestamps


@pytest.mark.asyncio
async def test_health() -> None:
    """Check the health endpoint."""
    app = run_httpserver.app
    client = app.test_client()
    response = await client.get("/health")
    assert response.status_code == HTTPStatus.OK
    health = await response.json
    assert health == {}

    run_httpserver.models = {"mockmodel": MockModelGeneration()}
    response = await client.get("/health")
    assert response.status_code == HTTPStatus.OK
    health = await response.json
    assert health is not None
    assert "mockmodel" in health
