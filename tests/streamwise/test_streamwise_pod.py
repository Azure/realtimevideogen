#!/usr/bin/env python3
"""
Unit tests for pod_manager.py functions.
"""

import sys
import pytest
import urllib.parse

from http import HTTPStatus

from unittest.mock import patch

from tests.test_utils import temp_sys_path
from tests.k8s_mock import K8sMock

mock_k8s = K8sMock()

mock_modules = {}
mock_modules.update(mock_k8s.get_sub_modules())
with patch.dict(sys.modules, mock_modules):
    with temp_sys_path("streamwise"):
        from streamwise import streamwise

        from streamwise.pod_manager import get_gpu_type_affinity
        from streamwise.pod_manager import get_container_port
        from streamwise.pod_manager import get_gemma_settings
        from streamwise.pod_manager import get_llama32_settings


@pytest.fixture(scope="function", autouse=True)
def setup_k8s_cluster() -> None:
    # for some reason k8s_config.load_kube_config() is not async mocked
    streamwise.k8s_cluster = "unittest"


def test_get_gpu_type_affinity() -> None:
    assert get_gpu_type_affinity("a100") == [
        "NVIDIA-A100-SXM4-40GB",
        "NVIDIA-A100-SXM4-80GB",
        "NVIDIA-A100-PCIe-40GB",
        "NVIDIA-A100-PCIe-80GB",
        "NVIDIA-A100-80GB-PCIe",
    ]
    assert get_gpu_type_affinity("h100") == [
        "NVIDIA-H100-SXM5-80GB",
        "NVIDIA-H100-PCIe-80GB",
        "NVIDIA-H100-NVL",
        "NVIDIA-H100-80GB-HBM3",
        "NVIDIA-H100",
    ]
    assert get_gpu_type_affinity("h200") == [
        "NVIDIA-H200-SXM5-141GB",
        "NVIDIA-H200",
    ]
    assert get_gpu_type_affinity("v100") == [
        "Tesla-V100-PCIE-16GB",
        "Tesla-V100-SXM2-16GB",
        "Tesla-V100-SXM2-32GB"
    ]
    assert get_gpu_type_affinity("unknown") == []
    assert get_gpu_type_affinity("a+") == [
        "NVIDIA-A100-SXM4-40GB",
        "NVIDIA-A100-SXM4-80GB",
        "NVIDIA-A100-PCIe-40GB",
        "NVIDIA-A100-PCIe-80GB",
        "NVIDIA-A100-80GB-PCIe",
        "NVIDIA-H100-SXM5-80GB",
        "NVIDIA-H100-PCIe-80GB",
        "NVIDIA-H100-NVL",
        "NVIDIA-H100-80GB-HBM3",
        "NVIDIA-H100",
        "NVIDIA-H200-SXM5-141GB",
        "NVIDIA-H200",
    ]


def test_get_container_port() -> None:
    assert get_container_port("fantasytalking") == 8080
    assert get_container_port("gemma") == 8000
    assert get_container_port("streamwise") == 18181
    assert get_container_port("streamcast") == 18080
    assert get_container_port("flux") == 8080


def test_get_gemma_settings() -> None:
    args, volume_mounts, volumes = get_gemma_settings(num_gpus=1)
    assert args is not None
    assert "google/gemma-3-27b-it" in args
    assert "1" in args
    assert volume_mounts is not None
    assert volumes is not None


def test_get_llama32_settings() -> None:
    args, volume_mounts, volumes = get_llama32_settings(num_gpus=2)
    assert args is not None
    assert "meta-llama/Llama-3.2-90B-Vision" in args
    assert "--tensor-parallel-size" in args
    assert "2" in args
    assert volume_mounts is not None
    assert volumes is not None


@pytest.mark.asyncio
async def test_add_pod() -> None:
    app = streamwise.app
    client = app.test_client()
    response = await client.get("/pod/qwenimage")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    assert response_text.startswith("<!DOCTYPE html>\n<html lang=\"en\">")
    assert "Add StreamWise Service" in response_text


@pytest.mark.asyncio
async def test_api_add_pod() -> None:
    app = streamwise.app
    client = app.test_client()

    response = await client.post("/api/pod")
    assert response.status_code == HTTPStatus.BAD_REQUEST
    response_json = await response.get_json()
    assert response_json == {"error": "Missing required parameter 'container_name'"}

    # Actual content
    form_data = {
        "container_name": "qwenimageedit",
    }
    response = await client.post(
        "/api/pod",
        data=urllib.parse.urlencode(form_data),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == HTTPStatus.OK
    response_json = await response.get_json()
    assert response_json["container_name"] == "qwenimageedit"
    assert response_json["message"] == "Pod creation requested"
    assert response_json["pod_name"].startswith("qwenimageedit-")
    assert "resource_request" in response_json
    assert response_json["resource_request"] == {
        "cpu": 2,
        "ephemeral-storage": "16Gi",
        "memory": "4Gi",
    }


@pytest.mark.asyncio
async def test_remove_pod() -> None:
    app = streamwise.app
    client = app.test_client()

    response = await client.delete("/api/pod/fluxkrea")
    assert response.status_code == HTTPStatus.BAD_REQUEST
    response_json = await response.get_json()
    assert response_json == {"error": "Namespace is required"}

    response = await client.delete("/api/pod/fluxkrea?namespace=default")
    assert response.status_code == HTTPStatus.OK
    response_json = await response.get_json()
    assert response_json == {"message": "Pod fluxkrea removed successfully"}
