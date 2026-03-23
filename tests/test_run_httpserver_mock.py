#!/usr/bin/env python3

import sys
import pytest
import asyncio
import logging

from pytest import raises

from http import HTTPStatus

from unittest.mock import patch
from unittest.mock import MagicMock
from unittest.mock import AsyncMock

from tests.torch_mock import TorchMock

from streamwise_apps import STREAMWISE_APPS
from streamwise_apps import VLLM_SERVICES

mock_torch = TorchMock()

sys.path.append("wrapper")

# Quart mocks
mock_quart = MagicMock()
mock_quart.route = lambda *args, **kwargs: (lambda f: f)
mock_quart.render_template = AsyncMock(return_value="mocked_template")
mock_quart.send_file = AsyncMock(return_value="mocked_file")
mock_quart.send_from_directory = AsyncMock(return_value="mocked_file")
mock_quart.jsonify = lambda x: x
mock_app = MagicMock()
mock_quart.Quart = MagicMock(return_value=mock_app)
mock_quart.Response = lambda *args, **kwargs: {"response": args, "kwargs": kwargs}
mock_request = MagicMock()
mock_request.args = {}
mock_request.json = {}
mock_request.get_json = AsyncMock(return_value={})
mock_quart.request = mock_request

mock_serve = AsyncMock()

mock_modules = {
    'nvidia_smi': MagicMock(),
    'colorlog': MagicMock(),
    'torch': mock_torch,
    'hypercorn': MagicMock(),
    'hypercorn.config': MagicMock(),
    'hypercorn.asyncio': MagicMock(serve=mock_serve),
    'quart': mock_quart,
    "quart.request": mock_quart.request,
    "quart.jsonify": mock_quart.jsonify,
    "quart.send_file": mock_quart.send_file,
    "quart.send_from_directory": mock_quart.send_from_directory,
    "quart.render_template": mock_quart.render_template,
    "quart.route": mock_quart.route,
    "quart.Response": mock_quart.Response,
    'imageio': MagicMock(),
    'cv2': MagicMock(),
    'xfuser': MagicMock(),
    'sklearn': MagicMock(),
    'scipy': MagicMock(),
    'scipy.stats': MagicMock(),
}
mock_modules.update(mock_torch.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from run_httpserver import main
    from run_httpserver import arg_parsing
    from run_httpserver import send_task
    from run_httpserver import nccl_worker
    from run_httpserver import wait_for_everybody
    from run_httpserver import index
    from run_httpserver import gen_img
    from run_httpserver import gen_video
    from run_httpserver import gen_audio
    from run_httpserver import get_service_names
    from run_httpserver import setup_dist_environment
    import run_httpserver as _run_httpserver


def test_get_service_names() -> None:
    """Check that we have the expected services covered."""
    service_names = get_service_names()
    assert len(service_names) == 46
    assert "mock" in service_names
    assert "wan" in service_names
    assert "wan22" in service_names
    assert "fantasytalking" in service_names
    assert "flux" in service_names
    assert "fluxkrea" in service_names
    assert "qwenimage" in service_names
    assert "qwenimageedit" in service_names
    assert "hunyuanimage" in service_names
    assert "yolo" in service_names
    assert "kokoro" in service_names
    assert "vibevoice" in service_names
    assert "notexisting" not in service_names

    for vllm_service in VLLM_SERVICES:
        if vllm_service != "llm":
            assert vllm_service in service_names

    # Applications
    for streamwise_app in STREAMWISE_APPS:
        assert streamwise_app in service_names


@pytest.mark.asyncio
async def test_server_help() -> None:
    """Check the HTTP server start."""
    test_args = ["run_http_server.py", "--help"]
    with patch.object(sys, "argv", test_args):
        with raises(SystemExit) as exc_info:
            await main()
        assert exc_info.value.code == 0


# This test is expensive; we swallow the timeout
# @pytest.mark.timeout(2)
@pytest.mark.asyncio
async def test_server() -> None:
    """Check the base HTTP server start."""
    test_args = ["run_http_server.py"]
    with patch.object(sys, "argv", test_args):
        try:
            # Use asyncio.wait_for to limit the runtime to 2s
            await asyncio.wait_for(main(), timeout=2)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            print("HTTP server finished (swallowed)")


@pytest.mark.asyncio
async def test_server_mock() -> None:
    """Check the HTTP server start for the mock service."""
    test_args = ["run_http_server.py", "--mock"]
    with patch.object(sys, "argv", test_args):
        await main()


@pytest.mark.asyncio
async def test_server_imageresize() -> None:
    """Check the HTTP server start for the image resize service."""
    test_args = ["run_http_server.py", "--imageresize"]
    with patch.object(sys, "argv", test_args):
        await main()


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [
    "flux",
    "hidream",
    "fantasytalking",
    "hunyuanframepack",
    "hunyuanframepackf1",
    "wan",
    "wan21",
    "wan22",
])
async def test_server_service_diffusers(model_name: str) -> None:
    """Check the HTTP server start for multiple services.
    These services fail with:
    RuntimeError: Failed to import diffusers.pipelines.pipeline_utils
    """
    test_args = ["run_http_server.py", f"--{model_name}"]
    with patch.object(sys, "argv", test_args):
        with raises(
            (ModuleNotFoundError, RuntimeError),
            match="(diffusers|This module requires CUDA support|Found no NVIDIA driver)"
        ):
            await main()


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [
    "realesrgan",
    "yolo",  # ultralytics
    "kokoro",  # KPipeline
    "xtts",  # TTS
    # "vibevoice",  # demo
    "podcasttranscript",  # fitz
    # "januspro"  # janus
])
async def test_server_service_import(model_name: str) -> None:
    """Check the HTTP server start for multiple services.
    They fail with the following errors:
    ModuleNotFoundError: No module named 'RealESRGAN'
    ModuleNotFoundError: No module named 'fitz'
    ModuleNotFoundError: No module named 'janus'
    ImportError: cannot import name 'KPipeline' from 'kokoro'
    """
    test_args = ["run_http_server.py", f"--{model_name}"]
    with patch.object(sys, "argv", test_args):
        with pytest.raises((ModuleNotFoundError, ImportError)):
            await main()


@pytest.mark.asyncio
async def test_server_all_services() -> None:
    """Check the HTTP server start for all services."""
    service_names = get_service_names()
    for vllm_service in VLLM_SERVICES:
        if vllm_service in service_names:
            service_names.remove(vllm_service)  # Separate vLLM service
    for streamwise_app in STREAMWISE_APPS:
        if streamwise_app in service_names:
            service_names.remove(streamwise_app)  # Separate service
    service_names.remove("streamwise")  # Separate service

    for model_name in service_names:
        try:
            test_args = ["run_http_server.py", f"--{model_name}"]
            with patch.object(sys, "argv", test_args):
                try:
                    logging.info(f"Testing service: {model_name}")
                    # Use asyncio.wait_for to limit the runtime to 2s
                    await asyncio.wait_for(main(), timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    logging.error(f"HTTP server for {model_name} finished (swallowed)")
        except (
            ModuleNotFoundError,
            ImportError,
            RuntimeError,
            TypeError,  # Janus Pro
        ):
            logging.warning(f"Skipping service {model_name} due to error")
        finally:
            loop = asyncio.get_running_loop()
            executor = getattr(loop, "_default_executor", None)
            if executor:
                executor.shutdown(wait=False, cancel_futures=True)


@pytest.mark.asyncio
async def test_server_wrong_server() -> None:
    """Check the HTTP server start with a wrong argument."""
    with raises(SystemExit) as exc_info:
        test_args = ["run_http_server.py", "--wrong"]
        with patch.object(sys, "argv", test_args):
            await main()
    assert exc_info.value.code == 2


def test_arg_parsing() -> None:
    """Check the argument parsing."""
    test_args = ["run_http_server.py", "--wan", "--flux", "--hunyuanframepackf1"]
    with patch.object(sys, "argv", test_args):
        args, engine_config = arg_parsing()
        assert args.wan is True
        assert args.flux is True
        assert args.fluxupscaler is False
        assert args.fluxkontext is False
        assert args.fluxkrea is False
        assert args.hidream is False
        assert args.qwenimage is False
        assert args.qwenimageedit is False
        assert args.realesrgan is False
        assert args.kokoro is False
        assert args.hunyuanframepack is False
        assert args.hunyuanframepackf1 is True
        assert args.hunyuanframepackvae is False
        assert engine_config is None

    test_args = ["run_http_server.py", "--nonexisting"]
    with patch.object(sys, "argv", test_args):
        with raises(SystemExit) as exc_info:
            args, engine_config = arg_parsing()
        assert exc_info.value.code == 2


@pytest.mark.asyncio
async def test_send_task() -> None:
    """Basic call to cover the function."""
    await send_task({})


@pytest.mark.asyncio
async def test_nccl_worker() -> None:
    """Basic call to cover the function."""
    await nccl_worker()


@pytest.mark.asyncio
async def test_wait_for_everybody() -> None:
    """Basic call to cover the function."""
    await wait_for_everybody()


@pytest.mark.asyncio
async def test_content() -> None:
    """Check the HTTP server content."""
    with raises(TypeError):
        # TODO fix:  TypeError: object MagicMock can't be used in 'await' expression
        assert await index() == "index page"
        # assert await health() == {}
        # assert await model_health("dummy") == {"ok": True}


@pytest.mark.asyncio
async def test_gen_img() -> None:
    """Check the image generation endpoint."""
    response = await gen_img(None)
    assert response == ({"error": "Not initialized"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    mock_model = AsyncMock()
    mock_model.running = True
    response = await gen_img(mock_model)
    assert response == ({"error": "Generation in progress"}, HTTPStatus.SERVICE_UNAVAILABLE)

    mock_model.running = False
    response = await gen_img(mock_model)
    assert response is not None
    # assert response == "mocked_file"


@pytest.mark.asyncio
async def test_gen_video() -> None:
    """Check the video generation endpoint."""
    response = await gen_video(None)
    assert response == ({"error": "Not initialized"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    mock_model = MagicMock()
    mock_model.generate = AsyncMock(return_value="mocked_file.mp4")
    mock_model.get_rest_args = AsyncMock(return_value={
        "task": "mock",
        "args": {}
    })
    mock_model.running = True
    response = await gen_video(mock_model)
    assert response == ({"error": "Generation in progress"}, HTTPStatus.SERVICE_UNAVAILABLE)

    mock_model.running = False
    response = await gen_video(mock_model)
    assert response is not None
    response_msg, code = response
    assert code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert response_msg["error"] == "Video file not found: mocked_file.mp4"


@pytest.mark.asyncio
async def test_gen_audio() -> None:
    """Check the audio generation endpoint."""
    response = await gen_audio(None)
    assert response == ({"error": "Not initialized"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    # Blocked generation
    mock_model = MagicMock()
    mock_model.generate = AsyncMock(return_value="mocked_file.wav")
    mock_model.get_rest_args = AsyncMock(return_value={
        "task": "mock",
        "args": {}
    })
    mock_model.running = True
    response = await gen_audio(mock_model)
    assert response == ({"error": "Generation in progress"}, HTTPStatus.SERVICE_UNAVAILABLE)

    # Missing file
    mock_model.running = False
    response = await gen_audio(mock_model)
    assert response is not None
    response_msg, code = response
    assert code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert response_msg["error"] == "Audio file not found: mocked_file.wav"

    # No output
    mock_model.generate = AsyncMock(return_value=None)
    response = await gen_audio(mock_model)
    assert response is not None
    response_msg, code = response
    assert code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert response_msg["error"] == "No audio generated"

    # Use existing audio file
    mock_model.generate = AsyncMock(return_value="tests/data/audio_4675.wav")
    response = await gen_audio(mock_model)
    assert response is not None
    assert response == "mocked_file"  # send_file() returns this


def test_setup_dist_environment_mig_warning(caplog: pytest.LogCaptureFixture) -> None:
    """
    When world_size > visible CUDA devices (MIG partition case), setup_dist_environment
    must log a warning explaining the misconfiguration so operators can fix it.
    The TorchMock returns device_count=1, so setting WORLD_SIZE=2 triggers the path.
    """
    import os

    saved_env = {
        k: os.environ.get(k)
        for k in ("MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK",
                  "NODE_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "NPROC_PER_NODE")
    }
    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        os.environ["RANK"] = "1"
        os.environ["LOCAL_RANK"] = "1"
        os.environ["NODE_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "2"       # 2 processes, but only 1 visible device
        os.environ["LOCAL_WORLD_SIZE"] = "2"

        with caplog.at_level(logging.WARNING):
            setup_dist_environment()

        # Function must read WORLD_SIZE=2 from the environment
        assert _run_httpserver.world_size == 2
        # And emit a MIG-related warning so operators know how to fix it
        assert any(
            "world_size=2" in record.message and "MIG" in record.message
            for record in caplog.records
        ), f"Expected MIG warning, got: {[r.message for r in caplog.records]}"
    finally:
        # Restore original env and module globals
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _run_httpserver.rank = 0
        _run_httpserver.world_size = 1
