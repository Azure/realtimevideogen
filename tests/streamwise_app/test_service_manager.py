#!/usr/bin/env python3
"""
Unit tests for LMM Service Manager.
"""
import sys
import os
import pytest

from http import HTTPStatus

from unittest.mock import patch
from unittest.mock import MagicMock
from unittest.mock import AsyncMock

# Add current path
sys.path.append(os.getcwd())

from tests.torch_mock import TorchMock

from tests.test_utils import temp_sys_path

mock_torch = TorchMock()

mock_modules = {}
mock_modules.update(mock_torch.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    with temp_sys_path("apps"):
        from apps.client import LMMServiceManager
        from apps.client import ServiceNotFoundError


def _mock_session_http(service_name: str) -> MagicMock:
    """Mock the async aiohttp ClientSession for HTTP calls."""
    response = MagicMock()
    response.status = HTTPStatus.OK
    response.json = AsyncMock(return_value={
        service_name: {
            "status": "ok",
            "gpu": "A100",
            "world_size": 2,
        }
    })
    cm = AsyncMock()
    cm.__aenter__.return_value = response
    cm.__aexit__.return_value = None
    session = MagicMock()
    session.get.return_value = cm
    session.close = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_service_manager() -> None:
    service_manager = LMMServiceManager("streamwise")
    assert service_manager is not None

    # Mocking responses from services
    service_manager.session = _mock_session_http("flux")

    status, is_busy, gpu_model, num_gpus = await service_manager.get_container_status(
        service_name="flux",
        url="http://flux:8080/health"
    )
    assert status == "ok"
    assert is_busy is False
    assert gpu_model == "A100"
    assert num_gpus == 2

    await service_manager.start_updater()

    service_manager.print_service_status()

    with pytest.raises(ServiceNotFoundError):
        service_manager.get_service_url("fantasytalking")

    with pytest.raises(ServiceNotFoundError):
        service_manager.get_service_urls("qwenimage")

    await service_manager.stop()


def test_parse_arguments_https_args() -> None:
    """parse_arguments() accepts --certfile and --keyfile for HTTPS."""
    with patch.dict(sys.modules, mock_torch.get_sub_modules()):
        with temp_sys_path("apps"):
            from apps.streamwise_app import StreamWiseApp

    class _TestApp(StreamWiseApp):
        def setup_routes(self) -> None:
            pass

    app_instance = _TestApp("test")
    with patch("sys.argv", ["app", "--certfile", "/tmp/cert.pem", "--keyfile", "/tmp/key.pem"]):
        args = app_instance.parse_arguments("test")
        assert args.certfile == "/tmp/cert.pem"
        assert args.keyfile == "/tmp/key.pem"


def test_parse_arguments_https_defaults() -> None:
    """parse_arguments() defaults certfile and keyfile to None (HTTP mode)."""
    with patch.dict(sys.modules, mock_torch.get_sub_modules()):
        with temp_sys_path("apps"):
            from apps.streamwise_app import StreamWiseApp

    class _TestApp(StreamWiseApp):
        def setup_routes(self) -> None:
            pass

    app_instance = _TestApp("test")
    with patch("sys.argv", ["app"]):
        args = app_instance.parse_arguments("test")
        assert args.certfile is None
        assert args.keyfile is None


@pytest.mark.asyncio
async def test_run_httpserver_https() -> None:
    """run_httpserver() configures Hypercorn SSL when certfile/keyfile are given."""
    with patch.dict(sys.modules, mock_torch.get_sub_modules()):
        with temp_sys_path("apps"):
            from apps.streamwise_app import StreamWiseApp

    class _TestApp(StreamWiseApp):
        def setup_routes(self) -> None:
            pass

    app_instance = _TestApp("test")
    with patch("apps.streamwise_app.serve", new=AsyncMock()) as mock_serve:
        await app_instance.run_httpserver(
            host="127.0.0.1",
            port=9999,
            certfile="/tmp/cert.pem",
            keyfile="/tmp/key.pem",
        )
        mock_serve.assert_awaited_once()
        config = mock_serve.call_args[0][1]
        assert config.certfile == "/tmp/cert.pem"
        assert config.keyfile == "/tmp/key.pem"


@pytest.mark.asyncio
async def test_run_httpserver_http() -> None:
    """run_httpserver() configures Hypercorn without SSL when no certfile is given."""
    with patch.dict(sys.modules, mock_torch.get_sub_modules()):
        with temp_sys_path("apps"):
            from apps.streamwise_app import StreamWiseApp

    class _TestApp(StreamWiseApp):
        def setup_routes(self) -> None:
            pass

    app_instance = _TestApp("test")
    with patch("apps.streamwise_app.serve", new=AsyncMock()) as mock_serve:
        await app_instance.run_httpserver(host="127.0.0.1", port=9999)
        mock_serve.assert_awaited_once()
        config = mock_serve.call_args[0][1]
        assert getattr(config, "certfile", None) is None
        assert getattr(config, "keyfile", None) is None
