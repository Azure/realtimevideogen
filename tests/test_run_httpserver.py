#!/usr/bin/env python3

import sys
import pytest
import os
import tempfile

from http import HTTPStatus

from unittest.mock import patch
from unittest.mock import MagicMock
from unittest.mock import AsyncMock

from tests.test_utils import temp_sys_path
from tests.torch_mock import TorchMock

TMP_DIR = "tmp"

mock_torch = TorchMock()


with temp_sys_path("wrapper"):
    from tests.test_wrapper_model import MockModelGeneration

with patch.dict(sys.modules, {
    'nvidia_smi': MagicMock(),
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


def test_setup_dist_environment_non_mig() -> None:
    """setup_dist_environment uses local_rank as device_id when multiple GPUs are visible (non-MIG)."""
    env = {
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "12355",
        "RANK": "1",
        "LOCAL_RANK": "1",
        "WORLD_SIZE": "2",
        "LOCAL_WORLD_SIZE": "2",
    }
    with patch.dict(os.environ, env, clear=False):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        run_httpserver.setup_dist_environment()
        mock_torch.cuda.set_device.assert_called_with(1)


def test_setup_dist_environment_mig() -> None:
    """setup_dist_environment uses device 0 when MIG restricts container to a single visible device."""
    env = {
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "12355",
        "RANK": "1",
        "LOCAL_RANK": "1",
        "WORLD_SIZE": "2",
        "LOCAL_WORLD_SIZE": "2",
    }
    with patch.dict(os.environ, env, clear=False):
        mock_torch.cuda.is_available.return_value = True
        # MIG: each process sees only its own MIG instance (device_count=1)
        mock_torch.cuda.device_count.return_value = 1
        run_httpserver.setup_dist_environment()
        # local_rank=1 is out of range; should fall back to device 0
        mock_torch.cuda.set_device.assert_called_with(0)


@pytest.mark.asyncio
async def test_run_httpserver_http() -> None:
    """run_httpserver() configures Hypercorn without SSL when no certfile is given."""
    with patch("run_httpserver.serve", new=AsyncMock()) as mock_serve:
        await run_httpserver.run_httpserver(host="127.0.0.1", port=9999)
        mock_serve.assert_awaited_once()
        config = mock_serve.call_args[0][1]
        assert config.bind == ["127.0.0.1:9999"]
        # No SSL attributes set
        assert getattr(config, "certfile", None) is None
        assert getattr(config, "keyfile", None) is None


@pytest.mark.asyncio
async def test_run_httpserver_https() -> None:
    """run_httpserver() configures Hypercorn with SSL when certfile and keyfile are given."""
    with patch("run_httpserver.serve", new=AsyncMock()) as mock_serve:
        await run_httpserver.run_httpserver(
            host="127.0.0.1",
            port=9999,
            certfile="/tmp/cert.pem",
            keyfile="/tmp/key.pem",
        )
        mock_serve.assert_awaited_once()
        config = mock_serve.call_args[0][1]
        assert config.bind == ["127.0.0.1:9999"]
        assert config.certfile == "/tmp/cert.pem"
        assert config.keyfile == "/tmp/key.pem"


def test_arg_parsing_https_args() -> None:
    """arg_parsing() accepts --certfile and --keyfile arguments."""
    with patch("sys.argv", ["run_httpserver", "--mock", "--certfile", "/tmp/cert.pem", "--keyfile", "/tmp/key.pem"]):
        args, _ = run_httpserver.arg_parsing()
        assert args.certfile == "/tmp/cert.pem"
        assert args.keyfile == "/tmp/key.pem"


def test_arg_parsing_https_defaults() -> None:
    """arg_parsing() defaults certfile and keyfile to None (HTTP mode)."""
    with patch("sys.argv", ["run_httpserver", "--mock"]):
        args, _ = run_httpserver.arg_parsing()
        assert args.certfile is None
        assert args.keyfile is None
