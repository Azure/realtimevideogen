#!/usr/bin/env python3
"""
Unit tests for streamwise.py HTTP server routes.
"""

import sys
import pytest
from datetime import datetime, timezone

from http import HTTPStatus

from unittest.mock import patch
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from quart.typing import TestClientProtocol

from streamwise import http_session_manager

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
    assert "Applications" in response_text
    assert "Wrappers" in response_text


@pytest.mark.asyncio
async def test_index_incluster() -> None:
    sw.k8s_cluster = "incluster"

    client = _get_client()
    response = await client.get("/")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    assert "StreamWise Cluster Manager" in response_text
    assert "Applications" in response_text
    assert "Wrappers" in response_text


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
async def test_service_info_has_submit_job_button() -> None:
    client = _get_client()
    response = await client.get("/service/test")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    assert 'href="/job/"' in response_text
    assert 'title="Submit Job"' in response_text


_MOCK_MIG_SERVICE = {
    "namespace": "rtgen",
    "pod_name": "realesrgan-pod",
    "pod_ip": "10.0.0.6",
    "container_port": 8080,
    "container_name": "realesrgan",
    "pod_status": "Running",
    "start_time": None,
    "url": "http://10.0.0.6:8080",
    "node_name": "testnode",
    "cpu": 1,
    "memory": 2147483648,
    "gpu": 1,
    "mig_profile": "1g.10gb",
    "ephemeral_storage": 0,
    "events": [],
    "image": "myacr.azurecr.io/realesrgan:latest",
    "logs": None,
    "health": None,
    "files": None,
}


@pytest.mark.asyncio
async def test_service_shows_mig_profile() -> None:
    """Service page shows MIG profile badge for MIG pods."""
    client = _get_client()
    with patch("streamwise.streamwise.get_services", new=AsyncMock(return_value=[_MOCK_MIG_SERVICE])):
        with patch("streamwise.streamwise.get_k8s_load_balancers", new=AsyncMock(return_value=[])):
            response = await client.get("/service/realesrgan")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    assert "1g.10gb" in response_text
    assert "MIG slice" in response_text


@pytest.mark.asyncio
async def test_service_shows_full_gpu() -> None:
    """Service page shows plain GPU count for full-GPU pods (no MIG)."""
    client = _get_client()
    full_gpu_svc = dict(_MOCK_MIG_SERVICE)
    full_gpu_svc["mig_profile"] = None
    full_gpu_svc["container_name"] = "flux"
    with patch("streamwise.streamwise.get_services", new=AsyncMock(return_value=[full_gpu_svc])):
        with patch("streamwise.streamwise.get_k8s_load_balancers", new=AsyncMock(return_value=[])):
            response = await client.get("/service/flux")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    assert "MIG slice" not in response_text


@pytest.mark.asyncio
async def test_service_multiple_pods_no_duplicate_http_error_class() -> None:
    """Service page with multiple pods must declare HttpError exactly once (no redeclaration error)."""
    client = _get_client()
    svc1 = dict(_MOCK_MIG_SERVICE)
    svc1["mig_profile"] = None
    svc1["pod_name"] = "realesrgan-pod-0"
    svc1["pod_ip"] = "10.0.0.6"
    svc2 = dict(_MOCK_MIG_SERVICE)
    svc2["mig_profile"] = None
    svc2["pod_name"] = "realesrgan-pod-1"
    svc2["pod_ip"] = "10.0.0.7"
    with patch("streamwise.streamwise.get_services", new=AsyncMock(return_value=[svc1, svc2])):
        with patch("streamwise.streamwise.get_k8s_load_balancers", new=AsyncMock(return_value=[])):
            response = await client.get("/service/realesrgan")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    assert response_text.count("class HttpError extends Error") == 1


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


# ---------------------------------------------------------------------------
# MIG-related UI fix tests
# ---------------------------------------------------------------------------

_MOCK_MIG_SERVICE_WITH_HEALTH = {
    **_MOCK_MIG_SERVICE,
    "health": {"gpu": "NVIDIA A100-SXM4-80GB MIG 1g.10gb", "world_size": 0},
}

_MOCK_MIG_NODE = {
    "node_name": "mig-node",
    "region": "eastus",
    "resource_group": "rg1",
    "addresses": [],
    "is_ready": True,
    "capacity_resources": {
        "cpu": 96,
        "memory": 900 * 1024 * 1024 * 1024,
        "storage": 500 * 1024 * 1024 * 1024,
        "gpu": "N/A",
    },
    "allocatable_resources": {
        "cpu": 96,
        "memory": 900 * 1024 * 1024 * 1024,
        "storage": 500 * 1024 * 1024 * 1024,
        "gpu": "N/A",
    },
    "architecture": "amd64",
    "kernel_version": "5.15.0",
    "os_image": "Ubuntu 22.04",
    "creation_timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
    "labels": {},
    "images": [],
    "gpu_model": "NVIDIA A100-SXM4-80GB",
    "mig_enabled": True,
    "mig_resources": {
        "1g.10gb": {"capacity": 7, "allocatable": 5},
    },
}

_MOCK_MIG_POD = {
    "namespace": "rtgen",
    "pod_name": "realesrgan-pod",
    "status": "Running",
    "pod_ip": "10.0.0.6",
    "container_name": "realesrgan",
    "url": "http://10.0.0.6:8080",
    "node": "mig-node",
    "cpu": 1,
    "memory": 2 * 1024 * 1024 * 1024,
    "gpu": 1,
    "mig_profile": "1g.10gb",
}

_MOCK_FULL_GPU_NODE = {
    **_MOCK_MIG_NODE,
    "node_name": "gpu-node",
    "gpu_model": "NVIDIA A100-SXM4-80GB",
    "mig_enabled": False,
    "mig_resources": {},
    "capacity_resources": {
        "cpu": 96,
        "memory": 900 * 1024 * 1024 * 1024,
        "storage": 500 * 1024 * 1024 * 1024,
        "gpu": "8",
    },
    "allocatable_resources": {
        "cpu": 96,
        "memory": 900 * 1024 * 1024 * 1024,
        "storage": 500 * 1024 * 1024 * 1024,
        "gpu": "8",
    },
}

_MOCK_FULL_GPU_POD = {
    **_MOCK_MIG_POD,
    "node": "gpu-node",
    "mig_profile": None,
    "gpu": 2,
}


@pytest.mark.asyncio
async def test_service_shows_mig_with_gpu_model() -> None:
    """Service page shows GPU model alongside MIG profile badge when health.gpu is available."""
    client = _get_client()
    with patch("streamwise.streamwise.get_services", new=AsyncMock(return_value=[_MOCK_MIG_SERVICE_WITH_HEALTH])):
        with patch("streamwise.streamwise.get_k8s_load_balancers", new=AsyncMock(return_value=[])):
            response = await client.get("/service/realesrgan")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    assert "1g.10gb" in response_text
    assert "MIG slice" in response_text
    # GPU model should appear alongside the MIG profile
    assert "A100" in response_text


@pytest.mark.asyncio
async def test_index_shows_mig_profile_in_wrappers() -> None:
    """Index page wrappers table shows MIG profile badge instead of full GPU model for MIG services."""
    client = _get_client()
    with patch("streamwise.streamwise.get_services", new=AsyncMock(return_value=[_MOCK_MIG_SERVICE])):
        with patch("streamwise.streamwise.get_k8s_nodes", new=AsyncMock(return_value=[])):
            with patch("streamwise.streamwise.get_k8s_pods", new=AsyncMock(return_value=[])):
                with patch("streamwise.streamwise.get_k8s_load_balancers", new=AsyncMock(return_value=[])):
                    response = await client.get("/")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    # MIG profile badge must appear in the wrappers table
    assert "1g.10gb" in response_text
    assert "MIG slice" in response_text


@pytest.mark.asyncio
async def test_index_shows_mig_profile_with_gpu_model_in_wrappers() -> None:
    """Index page wrappers table shows GPU model alongside MIG profile when health.gpu is available."""
    client = _get_client()
    with patch("streamwise.streamwise.get_services", new=AsyncMock(return_value=[_MOCK_MIG_SERVICE_WITH_HEALTH])):
        with patch("streamwise.streamwise.get_k8s_nodes", new=AsyncMock(return_value=[])):
            with patch("streamwise.streamwise.get_k8s_pods", new=AsyncMock(return_value=[])):
                with patch("streamwise.streamwise.get_k8s_load_balancers", new=AsyncMock(return_value=[])):
                    response = await client.get("/")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    assert "1g.10gb" in response_text
    assert "MIG slice" in response_text
    assert "A100" in response_text


@pytest.mark.asyncio
async def test_index_mig_node_shows_partitions() -> None:
    """Index page nodes table shows MIG partitions (alloc/allocatable/capacity) for MIG-enabled nodes."""
    client = _get_client()
    with patch("streamwise.streamwise.get_services", new=AsyncMock(return_value=[])):
        with patch("streamwise.streamwise.get_k8s_nodes", new=AsyncMock(return_value=[_MOCK_MIG_NODE])):
            with patch("streamwise.streamwise.get_k8s_pods", new=AsyncMock(return_value=[_MOCK_MIG_POD])):
                with patch("streamwise.streamwise.get_k8s_load_balancers", new=AsyncMock(return_value=[])):
                    response = await client.get("/")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    # MIG partition profile must appear in the nodes table
    assert "1g.10gb" in response_text
    # MIG badge class should be present
    assert "badge bg-info" in response_text


@pytest.mark.asyncio
async def test_index_mig_pods_excluded_from_full_gpu_count() -> None:
    """Index page nodes table does not count MIG pods toward the full GPU allocated total."""
    client = _get_client()
    _cpu = 96
    _memory = 900 * 1024 * 1024 * 1024
    _storage = 500 * 1024 * 1024 * 1024
    # Mixed node: has both full GPUs and MIG resources
    mixed_node = {
        **_MOCK_FULL_GPU_NODE,
        "node_name": "mixed-node",
        "mig_enabled": True,
        "mig_resources": {"1g.10gb": {"capacity": 7, "allocatable": 6}},
        "capacity_resources": {"cpu": _cpu, "memory": _memory, "storage": _storage, "gpu": "7"},
        "allocatable_resources": {"cpu": _cpu, "memory": _memory, "storage": _storage, "gpu": "7"},
    }
    # Two pods: one full-GPU pod (gpu=2, no mig_profile) and one MIG pod
    full_pod = {**_MOCK_FULL_GPU_POD, "node": "mixed-node", "gpu": 2}
    mig_pod = {**_MOCK_MIG_POD, "node": "mixed-node"}
    with patch("streamwise.streamwise.get_services", new=AsyncMock(return_value=[])):
        with patch("streamwise.streamwise.get_k8s_nodes", new=AsyncMock(return_value=[mixed_node])):
            with patch("streamwise.streamwise.get_k8s_pods", new=AsyncMock(return_value=[full_pod, mig_pod])):
                with patch("streamwise.streamwise.get_k8s_load_balancers", new=AsyncMock(return_value=[])):
                    response = await client.get("/")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    # The full GPU count should be 2 (only the non-MIG pod), not 3 (2+1 counting MIG pod)
    assert "2/7/7" in response_text
    assert "3/7/7" not in response_text
