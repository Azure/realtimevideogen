#!/usr/bin/env python3
"""
Unit tests for streamwise.py HTTP server routes.
"""

import sys
import pytest

from http import HTTPStatus

from unittest.mock import patch
from unittest.mock import AsyncMock

from quart.typing import TestClientProtocol

from streamwise import http_session_manager

from tests.test_utils import temp_sys_path
from tests.k8s_mock import K8sMock

mock_k8s = K8sMock()

mock_modules = {}
mock_modules.update(mock_k8s.get_sub_modules())
with patch.dict(sys.modules, mock_modules):
    with temp_sys_path("streamwise"):
        from streamwise import streamwise as sw

        from streamwise.service_account_manager import get_streamwise_service_account
        from streamwise.service_account_manager import get_streamwiseapp_service_account

        from streamwise.service_manager import get_k8s_container_logs
        from streamwise.service_manager import get_k8s_pod_events
        from streamwise.service_manager import get_services
        from streamwise.service_manager import get_services_ns
        from streamwise.service_manager import get_service_timestamps
        from streamwise.service_manager import get_service_health
        from streamwise.service_manager import get_service_files
        from streamwise.service_manager import get_health_and_files_async
        from streamwise.service_manager import parse_vllm_metrics


def _get_client() -> TestClientProtocol:
    app = sw.app
    client = app.test_client()
    return client


@pytest.fixture(scope="function", autouse=True)
def setup_k8s_cluster() -> None:
    # for some reason k8s_config.load_kube_config() is not async mocked
    sw.k8s_cluster = "unittest"


@pytest.mark.asyncio
async def test_index() -> None:
    client = _get_client()
    response = await client.get("/")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    assert "StreamWise Cluster Manager" in response_text


@pytest.mark.asyncio
async def test_index_incluster() -> None:
    sw.k8s_cluster = "incluster"

    client = _get_client()
    response = await client.get("/")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    assert "StreamWise Cluster Manager" in response_text


@pytest.mark.asyncio
async def test_health() -> None:
    sw.k8s_cluster = "clustername0"

    client = _get_client()
    response = await client.get("/health")
    assert response.status_code == HTTPStatus.OK
    response_json = await response.get_json()
    assert response_json == {
        "status": "ok",
        "k8s_cluster": "clustername0"
    }


@pytest.mark.asyncio
async def test_service_info() -> None:
    sw.k8s_cluster = "unittest"

    client = _get_client()
    response = await client.get("/service/test")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    assert response_text.startswith("<!DOCTYPE html>\n<html lang=\"en\">")
    assert "<title>test</title>" in response_text


@pytest.mark.asyncio
async def test_service_info_flux() -> None:
    client = _get_client()
    response = await client.get("/service/flux")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    assert response_text.startswith("<!DOCTYPE html>\n<html lang=\"en\">")
    assert "<title>FLUX</title>" in response_text


@pytest.mark.asyncio
async def test_service_info_fantasytalking() -> None:
    app = sw.app
    client = app.test_client()
    response = await client.get("/service/fantasytalking")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    assert response_text.startswith("<!DOCTYPE html>\n<html lang=\"en\">")
    assert "<title>Fantasy Talking</title>" in response_text


@pytest.mark.asyncio
async def test_container_info() -> None:
    app = sw.app
    client = app.test_client()
    response = await client.get("/service/test/10.1.1.1")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    assert response_text.startswith("<!DOCTYPE html>\n<html lang=\"en\">")
    assert "<title>test</title>" in response_text
    # assert "10.1.1.1" in response_text


@pytest.mark.asyncio
async def test_service_timeline() -> None:
    client = _get_client()
    response = await client.get("/service/test/timeline")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    assert response_text.startswith("<!DOCTYPE html>\n<html lang=\"en\">")
    assert "<title>test timeline</title>" in response_text


@pytest.mark.asyncio
async def test_timeline() -> None:
    client = _get_client()
    response = await client.get("/service/timeline")
    assert response.status_code == HTTPStatus.OK
    response_json = await response.get_json()
    assert response_json is None


@pytest.mark.asyncio
async def test_api_add_pod() -> None:
    client = _get_client()

    # POST without JSON should fail
    response = await client.post("/api/pod", data={
        "container_name": "flux",
    })
    assert response.status_code == HTTPStatus.BAD_REQUEST
    response_json = await response.get_json()
    assert response_json == {"error": "Missing required parameter 'container_name'"}


@pytest.mark.asyncio
async def test_nonexisting() -> None:
    client = _get_client()
    await client.get("/nonexisting")


def test_parse_vllm_metrics() -> None:
    sample_metrics = """
vllm:num_requests_running{engine="0",model_name="google/gemma-3-27b-it"} 0.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="google/gemma-3-27b-it"} 0.0
vllm:request_prompt_tokens_bucket{engine="0",le="1.0",model_name="google/gemma-3-27b-it"} 0.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="google/gemma-3-27b-it"} 0.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="google/gemma-3-27b-it"} 0.0
vllm:request_success_total{engine="0",finished_reason="abort",model_name="google/gemma-3-27b-it"} 0.0
"""
    result = parse_vllm_metrics(sample_metrics)
    assert len(result) == 3
    assert result == {
        "num_requests_running": 0.0,
        "request_success_total": 0.0,
        "request_prompt_tokens_bucket": 0.0,
    }


@pytest.mark.asyncio
async def test_get_k8s_container_logs() -> None:
    k8s_api_mock = AsyncMock()
    k8s_api_mock.read_namespaced_pod_log = AsyncMock()
    k8s_api_mock.read_namespaced_pod_log.return_value = "log line"

    logs = await get_k8s_container_logs(
        k8s_api=k8s_api_mock,
        namespace="namespace",
        pod_name="pod_name",
        container_name="container_name")
    assert logs == "log line"


@pytest.mark.asyncio
async def test_get_k8s_pod_events() -> None:
    k8s_api_mock = AsyncMock()

    events = await get_k8s_pod_events(
        k8s_api=k8s_api_mock,
        namespace="namespace",
        pod_name="pod_name")
    assert events == []


@pytest.mark.asyncio
async def test_get_services() -> None:
    services = await get_services(k8s_cluster="unittest")
    assert services == []

    services = await get_services_ns(
        namespace="namespace",
        k8s_cluster="unittest")
    assert services == []


@pytest.mark.asyncio
async def test_get_health_and_files_async() -> None:
    service_data = await get_health_and_files_async(None)  # type: ignore[arg-type]
    assert service_data is None

    service_data = await get_health_and_files_async([])
    assert service_data == []

    service_data = await get_health_and_files_async([
        {"service1": []}
    ])
    assert service_data == [
        {
            "files": [],
            "health": "N/A",
            "service1": []
        }
    ]


@pytest.mark.asyncio
async def test_get_service_account() -> None:
    sa = await get_streamwise_service_account(k8s_cluster="unittest")
    assert sa == "streamwise-service-account"

    sa = await get_streamwiseapp_service_account(k8s_cluster="unittest")
    assert sa == "streamwiseapp-service-account"


@pytest.mark.asyncio
async def test_start_stop() -> None:
    await sw.startup()
    # TODO test http_session_manager.client_session
    # assert http_session_manager.client_session is not None
    # assert sw.client_session is not None

    # session = await sw.get_global_session()
    # assert session is not None

    session = await http_session_manager.get_global_session()
    assert session is not None

    await sw.shutdown()
    # assert sw.client_session is None


@pytest.mark.asyncio
async def test_get_service_timestamps() -> None:
    timestamps = await get_service_timestamps(
        pod_name="test-pod",
        container_name="test-container",
        url="http://10.1.2.3:8080")
    assert timestamps is not None
    assert timestamps == []

    timestamps = await get_service_timestamps(
        pod_name="test-pod",
        container_name="test-container",
        url=None)  # type: ignore[arg-type]
    assert timestamps is None

    timestamps = await get_service_timestamps(
        pod_name="test-pod",
        container_name="gemma",
        url="http://20.1.2.3:8888")
    assert timestamps == []


@pytest.mark.asyncio
async def test_get_service_health() -> None:
    health = await get_service_health(
        container_name="test-container",
        url="http://10.2.2.2:8080")
    assert health is not None
    assert health == {"status": "failed"}

    health = await get_service_health(
        container_name="test-container",
        url=None)  # type: ignore[arg-type]
    assert health is None


@pytest.mark.asyncio
async def test_get_service_files() -> None:
    files = await get_service_files(
        container_name="test-container",
        url="http://10.2.2.2:8080")
    assert files == []

    files = await get_service_files(
        container_name="test-container",
        url="N/A")
    assert files is None

    files = await get_service_files(
        container_name="gemma",
        url="http://10.2.2.2:8080")
    assert files == []
