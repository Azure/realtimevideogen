#!/usr/bin/env python3
"""
Unit tests for pod_manager.py functions.
"""

import sys
import pytest
import urllib.parse

from http import HTTPStatus

from unittest.mock import patch, AsyncMock, MagicMock

from tests.test_utils import temp_sys_path
from tests.k8s_mock import K8sMock, MockApiException

mock_k8s = K8sMock()

mock_modules = {}
mock_modules.update(mock_k8s.get_sub_modules())
mock_k8s_client = mock_modules["kubernetes_asyncio.client"]
mock_custom_api = mock_k8s_client.CustomObjectsApi.return_value
with patch.dict(sys.modules, mock_modules):
    with temp_sys_path("streamwise"):
        from streamwise import streamwise as sw

        from streamwise.pod_manager import get_gpu_type_affinity
        from streamwise.pod_manager import get_container_port
        from streamwise.pod_manager import get_gemma_settings
        from streamwise.pod_manager import get_llama32_settings
        from streamwise.pod_manager import get_mig_resource_name
        from streamwise.pod_manager import get_tls_cert_settings
        from streamwise.pod_manager import tls_cert_volume_exists
        from streamwise.pod_manager import MIG_PROFILES


@pytest.fixture(scope="function", autouse=True)
def setup_k8s_cluster() -> None:
    # for some reason k8s_config.load_kube_config() is not async mocked
    sw.k8s_cluster = "unittest"
    sw.use_https = False


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


def test_get_mig_resource_name() -> None:
    # Valid A100 40 GB profiles
    assert get_mig_resource_name("1g.5gb") == "nvidia.com/mig-1g.5gb"
    assert get_mig_resource_name("2g.10gb") == "nvidia.com/mig-2g.10gb"
    assert get_mig_resource_name("3g.20gb") == "nvidia.com/mig-3g.20gb"
    assert get_mig_resource_name("4g.20gb") == "nvidia.com/mig-4g.20gb"
    assert get_mig_resource_name("7g.40gb") == "nvidia.com/mig-7g.40gb"
    # Valid A100 80 GB / H100 80 GB profiles
    assert get_mig_resource_name("1g.10gb") == "nvidia.com/mig-1g.10gb"
    assert get_mig_resource_name("2g.20gb") == "nvidia.com/mig-2g.20gb"
    assert get_mig_resource_name("3g.40gb") == "nvidia.com/mig-3g.40gb"
    assert get_mig_resource_name("4g.40gb") == "nvidia.com/mig-4g.40gb"
    assert get_mig_resource_name("7g.80gb") == "nvidia.com/mig-7g.80gb"
    # Invalid profile
    assert get_mig_resource_name("invalid") is None
    assert get_mig_resource_name("5g.20gb") is None
    assert get_mig_resource_name("") is None


def test_mig_profiles_set() -> None:
    assert "1g.5gb" in MIG_PROFILES
    assert "1g.10gb" in MIG_PROFILES
    assert "7g.80gb" in MIG_PROFILES
    assert "invalid" not in MIG_PROFILES


def test_get_tls_cert_settings() -> None:
    # Reset the relevant mocks so calls from earlier tests (e.g. get_gemma_settings)
    # do not interfere with assert_called_once_with below.
    mock_k8s_client.V1VolumeMount.reset_mock()
    mock_k8s_client.V1Volume.reset_mock()
    mock_k8s_client.V1CSIVolumeSource.reset_mock()

    volume_mount, volume = get_tls_cert_settings()
    assert volume_mount is not None
    assert volume is not None
    # V1VolumeMount / V1Volume are MagicMocks in the test environment; verify the
    # constructors were called with the correct arguments.
    mock_k8s_client.V1VolumeMount.assert_called_once_with(name="tls-csi", mount_path="/certs", read_only=True)
    mock_k8s_client.V1CSIVolumeSource.assert_called_once_with(
        driver="secrets-store.csi.k8s.io",
        read_only=True,
        volume_attributes={"secretProviderClass": "streamwise-tls"},
    )
    mock_k8s_client.V1Volume.assert_called_once_with(name="tls-csi", csi=mock_k8s_client.V1CSIVolumeSource.return_value)


@pytest.mark.asyncio
async def test_tls_cert_volume_exists_found() -> None:
    """tls_cert_volume_exists returns True when the SecretProviderClass is found."""
    mock_custom_api.get_namespaced_custom_object = AsyncMock(return_value={"metadata": {"name": "streamwise-tls"}})
    result = await tls_cert_volume_exists("rtgen", "unittest")
    assert result is True
    mock_custom_api.get_namespaced_custom_object.assert_called_once_with(
        group="secrets-store.csi.x-k8s.io",
        version="v1",
        namespace="rtgen",
        plural="secretproviderclasses",
        name="streamwise-tls"
    )


@pytest.mark.asyncio
async def test_tls_cert_volume_exists_not_found() -> None:
    """tls_cert_volume_exists returns False when the SecretProviderClass is absent (404)."""
    mock_custom_api.get_namespaced_custom_object = AsyncMock(
        side_effect=MockApiException(status=404, reason="Not Found")
    )
    result = await tls_cert_volume_exists("rtgen", "unittest")
    assert result is False


@pytest.mark.asyncio
async def test_tls_cert_volume_exists_error() -> None:
    """tls_cert_volume_exists returns False on unexpected errors."""
    mock_custom_api.get_namespaced_custom_object = AsyncMock(
        side_effect=Exception("connection refused")
    )
    result = await tls_cert_volume_exists("rtgen", "unittest")
    assert result is False


@pytest.mark.asyncio
async def test_api_add_pod_no_tls_when_use_https_false() -> None:
    """Pod creation without use_https=True must not call get_tls_cert_settings."""
    sw.use_https = False
    with patch.object(sw.pod_manager, "get_tls_cert_settings") as mock_get_tls, \
         patch.object(sw.pod_manager, "tls_cert_volume_exists", new=AsyncMock(return_value=True)):

        app = sw.app
        client = app.test_client()
        form_data = {"container_name": "qwenimageedit"}
        response = await client.post(
            "/api/pod",
            data=urllib.parse.urlencode(form_data),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert response.status_code == HTTPStatus.OK
        mock_get_tls.assert_not_called()


@pytest.mark.asyncio
async def test_api_add_pod_tls_added_when_use_https_and_volume_exists() -> None:
    """Pod creation with use_https=True and existing SecretProviderClass must mount TLS volume."""
    sw.use_https = True
    with patch.object(sw.pod_manager, "get_tls_cert_settings") as mock_get_tls, \
         patch.object(sw.pod_manager, "tls_cert_volume_exists", new=AsyncMock(return_value=True)):
        mock_get_tls.return_value = (MagicMock(), MagicMock())

        app = sw.app
        client = app.test_client()
        form_data = {"container_name": "qwenimageedit"}
        response = await client.post(
            "/api/pod",
            data=urllib.parse.urlencode(form_data),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert response.status_code == HTTPStatus.OK
        mock_get_tls.assert_called_once()


@pytest.mark.asyncio
async def test_api_add_pod_no_tls_when_volume_missing() -> None:
    """Pod creation with use_https=True but missing SecretProviderClass must not mount TLS volume."""
    sw.use_https = True
    with patch.object(sw.pod_manager, "get_tls_cert_settings") as mock_get_tls, \
         patch.object(sw.pod_manager, "tls_cert_volume_exists", new=AsyncMock(return_value=False)):

        app = sw.app
        client = app.test_client()
        form_data = {"container_name": "qwenimageedit"}
        response = await client.post(
            "/api/pod",
            data=urllib.parse.urlencode(form_data),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert response.status_code == HTTPStatus.OK
        mock_get_tls.assert_not_called()


@pytest.mark.asyncio
async def test_add_pod() -> None:
    app = sw.app
    client = app.test_client()
    response = await client.get("/pod/qwenimage")
    assert response.status_code == HTTPStatus.OK
    response_text = await response.get_data(as_text=True)
    assert response_text.startswith("<!DOCTYPE html>\n<html lang=\"en\">")
    assert "Add StreamWise Service" in response_text


@pytest.mark.asyncio
async def test_api_add_pod() -> None:
    app = sw.app
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
async def test_api_add_pod_with_mig() -> None:
    """Pod creation with a MIG profile should use the MIG resource name."""
    app = sw.app
    client = app.test_client()

    form_data = {
        "container_name": "kokoro",
        "gpu": "1",
        "mig_profile": "1g.5gb",
        "gpu_type": "a100",
        "memory": "8",
        "cpu": "2",
    }
    response = await client.post(
        "/api/pod",
        data=urllib.parse.urlencode(form_data),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == HTTPStatus.OK
    response_json = await response.get_json()
    assert response_json["container_name"] == "kokoro"
    assert response_json["mig_profile"] == "1g.5gb"
    # Resource request must use MIG resource name, not nvidia.com/gpu
    assert "nvidia.com/mig-1g.5gb" in response_json["resource_request"]
    assert "nvidia.com/gpu" not in response_json["resource_request"]


@pytest.mark.asyncio
async def test_api_add_pod_invalid_mig() -> None:
    """Pod creation with an invalid MIG profile should be rejected."""
    app = sw.app
    client = app.test_client()

    form_data = {
        "container_name": "kokoro",
        "gpu": "1",
        "mig_profile": "bad_profile",
    }
    response = await client.post(
        "/api/pod",
        data=urllib.parse.urlencode(form_data),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST
    response_json = await response.get_json()
    assert "Invalid MIG profile" in response_json["error"]


@pytest.mark.asyncio
async def test_api_add_pod_custom_tag() -> None:
    app = sw.app
    client = app.test_client()

    # Custom tag should be reflected in the image_url
    form_data = {
        "container_name": "flux",
        "tag": "v9.9.9",
    }
    response = await client.post(
        "/api/pod",
        data=urllib.parse.urlencode(form_data),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == HTTPStatus.OK
    response_json = await response.get_json()
    assert response_json["container_name"] == "flux"
    assert response_json["image_url"].endswith(":v9.9.9")

    # Invalid container name should return 400 even with a tag
    form_data_invalid = {
        "container_name": "nonexistent-service",
        "tag": "v1.0.0",
    }
    response = await client.post(
        "/api/pod",
        data=urllib.parse.urlencode(form_data_invalid),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST
    response_json = await response.get_json()
    assert "error" in response_json
    assert "nonexistent-service" in response_json["error"]


@pytest.mark.asyncio
async def test_remove_pod() -> None:
    app = sw.app
    client = app.test_client()

    response = await client.delete("/api/pod/fluxkrea")
    assert response.status_code == HTTPStatus.BAD_REQUEST
    response_json = await response.get_json()
    assert response_json == {"error": "Namespace is required"}

    response = await client.delete("/api/pod/fluxkrea?namespace=default")
    assert response.status_code == HTTPStatus.OK
    response_json = await response.get_json()
    assert response_json == {"message": "Pod fluxkrea removed successfully"}
