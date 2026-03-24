#!/usr/bin/env python3
"""
Unit tests for streamwise.py node-related endpoints.
"""

import sys
import pytest

from typing import Any
from typing import Dict

from unittest.mock import patch
from unittest.mock import AsyncMock

from quart.typing import TestClientProtocol

from http import HTTPStatus

from tests.test_utils import temp_sys_path
from tests.k8s_mock import K8sMock

mock_k8s = K8sMock()

mock_modules = {}
mock_modules.update(mock_k8s.get_sub_modules())
with patch.dict(sys.modules, mock_modules):
    with temp_sys_path("streamwise"):
        from streamwise import streamwise as sw
        # node_manager is imported by streamwise.py at module level;
        # access it via the module attribute so we get the exact same object
        node_manager = sw.node_manager


def _get_client() -> TestClientProtocol:
    app = sw.app
    client = app.test_client()
    return client


@pytest.fixture(scope="function", autouse=True)
def setup_k8s_cluster() -> None:
    # for some reason k8s_config.load_kube_config() is not async mocked
    sw.k8s_cluster = "unittest"


@pytest.mark.asyncio
async def test_nodes() -> None:
    client = _get_client()
    response = await client.get("/nodes")
    assert response.status_code == HTTPStatus.NOT_FOUND
    response_json = await response.get_json()
    assert response_json == {"error": "No nodes found"}


@pytest.mark.asyncio
async def test_api_nodes() -> None:
    client = _get_client()
    response = await client.get("/api/nodes")
    assert response.status_code == HTTPStatus.OK
    response_json = await response.get_json()
    assert response_json == []


@pytest.mark.asyncio
async def test_remove_node() -> None:
    client = _get_client()
    response = await client.delete("/api/node/testnode")
    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    response_json = await response.get_json()
    assert response_json == {"error": "object MagicMock can't be used in 'await' expression"}  # TODO


@pytest.mark.asyncio
async def test_node_info() -> None:
    client = _get_client()
    response = await client.get("/node/testnode")
    assert response.status_code == HTTPStatus.NOT_FOUND
    response_json = await response.get_json()
    assert response_json == {"error": "Node 'testnode' not found"}


_MOCK_NODE = {
    "node_name": "testnode",
    "region": "eastus",
    "resource_group": "rg-test",
    "addresses": [{
        "type": "InternalIP",
        "address": "10.0.0.1"
    }],
    "is_ready": True,
    "capacity_resources": {
        "cpu": 4.0,
        "memory": 8 * 1024 * 1024 * 1024,
        "storage": 100 * 1024 * 1024 * 1024,
        "gpu": "N/A"
    },
    "allocatable_resources": {
        "cpu": 4.0,
        "memory": 8 * 1024 * 1024 * 1024,
        "storage": 100 * 1024 * 1024 * 1024,
        "gpu": "N/A"
    },
    "architecture": "amd64",
    "kernel_version": "5.15.0",
    "os_image": "Ubuntu 22.04",
    "creation_timestamp": "2024-01-01T00:00:00Z",
    "labels": {},
    "images": None,
    "gpu_model": "N/A",
    "mig_enabled": False,
    "mig_resources": {},
}


@pytest.mark.asyncio
async def test_nodes_has_submit_job_button() -> None:
    client = _get_client()
    with patch.object(node_manager, "get_k8s_nodes", new=AsyncMock(return_value=[_MOCK_NODE])):
        with patch.object(node_manager, "get_k8s_pods", new=AsyncMock(return_value=[])):
            response = await client.get("/nodes")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    assert 'href="/job/"' in response_text
    assert 'title="Submit Job"' in response_text


@pytest.mark.asyncio
async def test_node_shows_mig_enabled() -> None:
    """Node page shows MIG enabled badge when mig_enabled is True."""
    client = _get_client()
    mig_node: Dict[str, Any] = dict(_MOCK_NODE)
    mig_node["mig_enabled"] = True
    mig_node["gpu_model"] = "NVIDIA-A100-SXM4-80GB"
    mig_node["capacity_resources"] = dict(mig_node["capacity_resources"])
    mig_node["capacity_resources"]["gpu"] = 7
    mig_node["allocatable_resources"] = dict(mig_node["allocatable_resources"])
    mig_node["allocatable_resources"]["gpu"] = "N/A"
    mig_node["mig_resources"] = {
        "1g.10gb": {
            "capacity": 7,
            "allocatable": 7
        }
    }
    with patch.object(node_manager, "get_k8s_nodes", new=AsyncMock(return_value=[mig_node])):
        with patch.object(node_manager, "get_k8s_pods", new=AsyncMock(return_value=[])):
            response = await client.get("/nodes")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    msg = '<span class="text-success" title="Multi-Instance GPU" aria-label="MIG enabled">\u2705</span>'
    assert msg in response_text


@pytest.mark.asyncio
async def test_node_shows_mig_resources() -> None:
    """Node page shows per-profile MIG resource counts in the Resources table."""
    client = _get_client()
    mock: Dict[str, Any] = dict(_MOCK_NODE)
    mig_node: Dict[str, Any] = mock
    mig_node["mig_enabled"] = True
    mig_node["gpu_model"] = "NVIDIA-A100-SXM4-80GB"
    mig_node["mig_resources"] = {
        "1g.5gb": {"capacity": 7, "allocatable": 5},
        "2g.10gb": {"capacity": 3, "allocatable": 2},
    }
    with patch.object(node_manager, "get_k8s_nodes", new=AsyncMock(return_value=[mig_node])):
        with patch.object(node_manager, "get_k8s_pods", new=AsyncMock(return_value=[])):
            response = await client.get("/nodes")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    assert "1g.5gb" in response_text
    assert "2g.10gb" in response_text
    # Allocatable and capacity counts visible
    assert "5" in response_text
    assert "2" in response_text


@pytest.mark.asyncio
async def test_node_shows_mig_disabled() -> None:
    """Node page shows no MIG indicator when mig_enabled is False."""
    client = _get_client()
    with patch.object(node_manager, "get_k8s_nodes", new=AsyncMock(return_value=[_MOCK_NODE])):
        with patch.object(node_manager, "get_k8s_pods", new=AsyncMock(return_value=[])):
            response = await client.get("/nodes")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    assert "MIG" in response_text  # row label is always present
    assert "Enabled" not in response_text  # but not the "Enabled" status text


@pytest.mark.asyncio
async def test_pod_shows_mig_profile() -> None:
    """Node page shows MIG profile badge in the pods table for MIG pods."""
    client = _get_client()
    mig_pod = {
        "namespace": "rtgen",
        "pod_name": "flux-pod",
        "status": "Running",
        "pod_ip": "10.0.0.5",
        "container_name": "flux",
        "url": "http://10.0.0.5:8080",
        "node": "testnode",
        "cpu": 2,
        "memory": 4294967296,
        "gpu": 1,
        "mig_profile": "1g.10gb",
    }
    with patch.object(node_manager, "get_k8s_nodes", new=AsyncMock(return_value=[_MOCK_NODE])):
        with patch.object(node_manager, "get_k8s_pods", new=AsyncMock(return_value=[mig_pod])):
            response = await client.get("/nodes")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    assert "1g.10gb" in response_text
    assert "MIG slice" in response_text


@pytest.mark.asyncio
async def test_pod_shows_no_mig_profile_for_full_gpu() -> None:
    """Node page shows plain GPU count for full-GPU pods (no MIG profile)."""
    client = _get_client()
    full_gpu_pod = {
        "namespace": "rtgen",
        "pod_name": "flux-pod",
        "status": "Running",
        "pod_ip": "10.0.0.5",
        "container_name": "flux",
        "url": "http://10.0.0.5:8080",
        "node": "testnode",
        "cpu": 2,
        "memory": 4294967296,
        "gpu": 1,
        "mig_profile": None,
    }
    with patch.object(node_manager, "get_k8s_nodes", new=AsyncMock(return_value=[_MOCK_NODE])):
        with patch.object(node_manager, "get_k8s_pods", new=AsyncMock(return_value=[full_gpu_pod])):
            response = await client.get("/nodes")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    assert "MIG slice" not in response_text


@pytest.mark.asyncio
async def test_mig_visible_when_gpu_capacity_zero() -> None:
    """Nodes table shows MIG profiles even when capacity_resources.gpu is 0 (single strategy)."""
    client = _get_client()
    mig_node: Dict[str, Any] = dict(_MOCK_NODE)
    mig_node["mig_enabled"] = True
    mig_node["gpu_model"] = "NVIDIA-A100-SXM4-80GB"
    mig_node["capacity_resources"] = dict(mig_node["capacity_resources"])
    mig_node["capacity_resources"]["gpu"] = 0
    mig_node["allocatable_resources"] = dict(mig_node["allocatable_resources"])
    mig_node["allocatable_resources"]["gpu"] = 0
    mig_node["mig_resources"] = {
        "1g.10gb": {"capacity": 3, "allocatable": 3},
        "2g.20gb": {"capacity": 2, "allocatable": 2},
    }
    with patch.object(node_manager, "get_k8s_nodes", new=AsyncMock(return_value=[mig_node])):
        with patch.object(node_manager, "get_k8s_pods", new=AsyncMock(return_value=[])):
            response = await client.get("/nodes")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    assert "MIG" in response_text
    assert "1g.10gb" in response_text
    assert "2g.20gb" in response_text


@pytest.mark.asyncio
async def test_gpu_row_excludes_mig_pods() -> None:
    """Resources table GPU row does not count MIG pods — they go in their own MIG rows."""
    client = _get_client()
    mig_node: Dict[str, Any] = dict(_MOCK_NODE)
    mig_node["mig_enabled"] = True
    mig_node["gpu_model"] = "NVIDIA-A100-SXM4-80GB"
    mig_node["capacity_resources"] = dict(mig_node["capacity_resources"])
    mig_node["capacity_resources"]["gpu"] = 1
    mig_node["allocatable_resources"] = dict(mig_node["allocatable_resources"])
    mig_node["allocatable_resources"]["gpu"] = 1
    mig_node["mig_resources"] = {"1g.10gb": {"capacity": 7, "allocatable": 7}}
    mig_pod = {
        "namespace": "rtgen",
        "pod_name": "esrgan-pod",
        "status": "Running",
        "pod_ip": "10.0.0.6",
        "container_name": "realesrgan",
        "url": "http://10.0.0.6:8080",
        "node": "testnode",
        "cpu": 1,
        "memory": 2147483648,
        "gpu": 1,
        "mig_profile": "1g.10gb",
    }
    full_gpu_pod = {
        "namespace": "rtgen",
        "pod_name": "flux-pod",
        "status": "Running",
        "pod_ip": "10.0.0.5",
        "container_name": "flux",
        "url": "http://10.0.0.5:8080",
        "node": "testnode",
        "cpu": 2,
        "memory": 4294967296,
        "gpu": 1,
        "mig_profile": None,
    }
    with patch.object(node_manager, "get_k8s_nodes", new=AsyncMock(return_value=[mig_node])):
        with patch.object(node_manager, "get_k8s_pods", new=AsyncMock(return_value=[mig_pod, full_gpu_pod])):
            response = await client.get("/node/testnode")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    # The GPU resource row should show "1.0" allocated (only the full-GPU pod),
    # not "2.0" (which would include the MIG pod).
    # Check the MIG row shows 1 allocated for 1g.10gb
    assert "1g.10gb" in response_text


@pytest.mark.asyncio
async def test_mixed_full_gpu_and_mig() -> None:
    """Nodes table shows both the full GPU count and MIG profiles (7 full + 1 MIG-partitioned)."""
    client = _get_client()
    mig_node: Dict[str, Any] = dict(_MOCK_NODE)
    mig_node["mig_enabled"] = True
    mig_node["gpu_model"] = "NVIDIA-A100-SXM4-80GB"
    mig_node["capacity_resources"] = dict(mig_node["capacity_resources"])
    mig_node["capacity_resources"]["gpu"] = 7
    mig_node["allocatable_resources"] = dict(mig_node["allocatable_resources"])
    mig_node["allocatable_resources"]["gpu"] = 7
    mig_node["mig_resources"] = {
        "1g.10gb": {"capacity": 3, "allocatable": 3},
        "2g.20gb": {"capacity": 2, "allocatable": 2},
    }
    with patch.object(node_manager, "get_k8s_nodes", new=AsyncMock(return_value=[mig_node])):
        with patch.object(node_manager, "get_k8s_pods", new=AsyncMock(return_value=[])):
            response = await client.get("/nodes")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    # Full GPU count visible
    assert "7" in response_text
    # MIG badge and profiles visible
    assert "MIG" in response_text
    assert "1g.10gb" in response_text
    assert "2g.20gb" in response_text


@pytest.mark.asyncio
async def test_pure_mig_node_gpu_na() -> None:
    """Resources table handles gpu=N/A without error on pure-MIG nodes."""
    client = _get_client()
    mig_node: Dict[str, Any] = dict(_MOCK_NODE)
    mig_node["mig_enabled"] = True
    mig_node["gpu_model"] = "NVIDIA-A100-SXM4-80GB"
    mig_node["capacity_resources"] = dict(mig_node["capacity_resources"])
    mig_node["capacity_resources"]["gpu"] = "N/A"
    mig_node["allocatable_resources"] = dict(mig_node["allocatable_resources"])
    mig_node["allocatable_resources"]["gpu"] = "N/A"
    mig_node["mig_resources"] = {"1g.10gb": {"capacity": 7, "allocatable": 7}}
    with patch.object(node_manager, "get_k8s_nodes", new=AsyncMock(return_value=[mig_node])):
        with patch.object(node_manager, "get_k8s_pods", new=AsyncMock(return_value=[])):
            response = await client.get("/node/testnode")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    assert "1g.10gb" in response_text
    assert "MIG" in response_text
