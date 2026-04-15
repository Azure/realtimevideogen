#!/usr/bin/env python3
"""
Unit tests for streamwise.py file-related endpoints.
"""

import sys
import pytest

from http import HTTPStatus

from unittest.mock import patch
from unittest.mock import MagicMock

from quart.typing import TestClientProtocol

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
}
mock_modules.update(mock_k8s.get_sub_modules())
mock_modules.update(mock_torch.get_sub_modules())
mock_modules.update(mock_numpy.get_sub_modules())
with patch.dict(sys.modules, mock_modules):
    with temp_sys_path("streamwise"):
        from streamwise import streamwise

        from file_manager import get_file_info


def _get_client() -> TestClientProtocol:
    app = streamwise.app
    return app.test_client()


@pytest.mark.asyncio
async def test_video_info() -> None:
    client = _get_client()
    response = await client.get("/video/10.1.1.1/8080/video.mp4")
    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    response_json = await response.get_json()
    assert response_json in (
        {"error": "Connection timeout to host http://10.1.1.1:8080/file/video.mp4"},
        {"error": "Event loop is closed"}
    )


@pytest.mark.asyncio
async def test_get_file_info() -> None:
    file_info = await get_file_info("127.0.0.1", 9980, "file.txt")
    assert file_info is None


@pytest.mark.asyncio
async def test_get_audio_waveform() -> None:
    client = _get_client()
    response = await client.get("/audio_waveform/1.2.3.4/8080/audio.wav")
    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    response_json = await response.get_json()
    assert response_json == {"error": "Event loop is closed"}


@pytest.mark.asyncio
async def test_list_files() -> None:
    client = _get_client()
    response = await client.get("/files")
    assert response.status_code == HTTPStatus.OK
    response_json = await response.get_json()
    assert "files" in response_json


@pytest.mark.asyncio
async def test_download_local_file() -> None:
    client = _get_client()
    response = await client.get("/file/nonexisting.mp4")
    assert response.status_code == HTTPStatus.NOT_FOUND
    response_json = await response.get_json()
    assert response_json == {"error": "File not found"}


@pytest.mark.asyncio
async def test_download_service_file() -> None:
    client = _get_client()
    response = await client.get("/file_download/10.1.1.1/8080/video.mp4")
    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    response_json = await response.get_json()
    # TODO
    assert response_json == {"error": "Event loop is closed"}


@pytest.mark.asyncio
async def test_file_stream() -> None:
    client = _get_client()
    response = await client.get("/file_stream/11.11.11.11/8080/video.mp4")
    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    response_json = await response.get_json()
    # TODO
    assert response_json == {"error": "Event loop is closed"}


@pytest.mark.asyncio
async def test_file_view() -> None:
    client = _get_client()
    response = await client.get("/file_view/flux/10.1.2.3/8080/video.mp4")
    assert response.status_code == HTTPStatus.OK
    response_data = await response.get_data(as_text=True)
    assert response_data.startswith("<!DOCTYPE html>\n<html>")
    assert "flux" in response_data
    response_json = await response.get_json()
    assert response_json is None
