"""
Tests for the auto-deploy API endpoints in streamwise.py.

Covers:
- POST /api/auto_deploy — returns optimized plan.
- POST /api/auto_deploy/confirm — deploys the plan.
- GET /api/auto_deploy/workflows — lists available options.
- Error cases (missing fields, invalid inputs).
"""

from __future__ import annotations

import sys

import pytest

from http import HTTPStatus
from unittest.mock import patch

from tests.test_utils import temp_sys_path
from tests.k8s_mock import K8sMock

mock_k8s = K8sMock()

mock_modules = {}
mock_modules.update(mock_k8s.get_sub_modules())

import streamwise.http_session_manager  # noqa: F401 — registers the streamwise package

# Permanently inject K8s mocks into sys.modules (not via context manager)
# so that simulator modules loaded alongside streamwise remain importable
# after setup completes.
_original_modules = {}
for mod_name, mock_mod in mock_modules.items():
    _original_modules[mod_name] = sys.modules.get(mod_name)
    sys.modules[mod_name] = mock_mod

with temp_sys_path("streamwise"):
    from streamwise import streamwise as sw


def _get_client():  # type: ignore[no-untyped-def]
    app = sw.app
    return app.test_client()


@pytest.fixture(scope="function", autouse=True)
def setup_k8s_cluster() -> None:
    sw.k8s_cluster = "unittest"
    sw.use_https = False


# ---------------------------------------------------------------------------
# GET /api/auto_deploy/workflows
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_auto_deploy_workflows() -> None:
    """Should return available workflows and GPU types."""
    client = _get_client()
    response = await client.get("/api/auto_deploy/workflows")
    assert response.status_code == HTTPStatus.OK
    data = await response.get_json()
    assert "workflows" in data
    assert "gpu_types" in data
    assert "streamcast" in data["workflows"]
    assert "A100" in data["gpu_types"]


# ---------------------------------------------------------------------------
# POST /api/auto_deploy
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_auto_deploy_success() -> None:
    """Valid request returns an optimized deployment plan."""
    fake_json = {
        "workflow_name": "streamcast",
        "gpu_budget": {"A100": 8},
        "metrics": {"total_time_s": 3.5, "ttff_s": 1.0, "cost": 12.0, "gpus_used": {"A100": 3}},
        "specs": [
            {"container_name": "gemma", "cpu": 4, "memory_gib": 16,
             "ephemeral_storage_gib": 10, "gpu": 1, "gpu_type": "A100", "mig_profile": None},
            {"container_name": "flux", "cpu": 4, "memory_gib": 16,
             "ephemeral_storage_gib": 10, "gpu": 2, "gpu_type": "A100", "mig_profile": None},
        ],
    }
    # Patch on the actual module object that streamwise.py holds a reference to.
    with patch.object(sw.allocator_bridge, "run_allocator") as mock_alloc, \
         patch.object(sw.allocator_bridge, "deployment_plan_to_json", return_value=fake_json):
        mock_alloc.return_value = "fake_plan"
        client = _get_client()
        response = await client.post(
            "/api/auto_deploy",
            json={
                "gpu_budget": {"A100": 8},
                "workflow": "streamcast",
            },
        )
    assert response.status_code == HTTPStatus.OK
    data = await response.get_json()
    assert "specs" in data
    assert "metrics" in data
    assert len(data["specs"]) == 2
    assert data["metrics"]["total_time_s"] == 3.5


@pytest.mark.asyncio
async def test_auto_deploy_missing_gpu_budget() -> None:
    """Missing gpu_budget field returns 400."""
    client = _get_client()
    response = await client.post(
        "/api/auto_deploy",
        json={"workflow": "streamcast"},
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST


@pytest.mark.asyncio
async def test_auto_deploy_missing_workflow() -> None:
    """Missing workflow field returns 400."""
    client = _get_client()
    response = await client.post(
        "/api/auto_deploy",
        json={"gpu_budget": {"A100": 8}},
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST


@pytest.mark.asyncio
async def test_auto_deploy_invalid_workflow() -> None:
    """Invalid workflow name returns 400."""
    client = _get_client()
    response = await client.post(
        "/api/auto_deploy",
        json={
            "gpu_budget": {"A100": 8},
            "workflow": "nonexistent",
        },
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST
    data = await response.get_json()
    assert "error" in data


@pytest.mark.asyncio
async def test_auto_deploy_insufficient_gpus() -> None:
    """Too few GPUs returns 400."""
    client = _get_client()
    response = await client.post(
        "/api/auto_deploy",
        json={
            "gpu_budget": {"A100": 2},
            "workflow": "streamcast",
        },
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST


@pytest.mark.asyncio
async def test_auto_deploy_no_json_body() -> None:
    """No JSON body returns 400."""
    client = _get_client()
    response = await client.post("/api/auto_deploy")
    assert response.status_code == HTTPStatus.BAD_REQUEST


# ---------------------------------------------------------------------------
# POST /api/auto_deploy/confirm
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_auto_deploy_confirm_success() -> None:
    """Valid confirm request deploys containers."""
    client = _get_client()
    specs = [
        {
            "container_name": "gemma",
            "cpu": 16,
            "memory_gib": 192,
            "ephemeral_storage_gib": 64,
            "gpu": 2,
            "gpu_type": "a100",
            "mig_profile": None,
        },
        {
            "container_name": "flux",
            "cpu": 12,
            "memory_gib": 128,
            "ephemeral_storage_gib": 64,
            "gpu": 2,
            "gpu_type": "a100",
            "mig_profile": None,
        },
    ]
    with patch.object(sw.pod_manager, "add_pod") as mock_add_pod:
        response = await client.post(
            "/api/auto_deploy/confirm",
            json={"specs": specs},
        )
    # Should succeed without invoking the real pod_manager.add_pod flow
    assert response.status_code in (HTTPStatus.OK, HTTPStatus.MULTI_STATUS)
    data = await response.get_json()
    assert "deployed" in data
    assert "message" in data
    assert mock_add_pod.call_count == len(specs)


@pytest.mark.asyncio
async def test_auto_deploy_confirm_missing_specs() -> None:
    """Missing specs returns 400."""
    client = _get_client()
    response = await client.post(
        "/api/auto_deploy/confirm",
        json={},
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST


@pytest.mark.asyncio
async def test_auto_deploy_confirm_tracks_add_pod_status_failures() -> None:
    """Non-2xx add_pod return statuses are surfaced as deployment errors."""
    client = _get_client()
    specs = [
        {"container_name": "gemma", "gpu": 2, "gpu_type": "a100"},
        {"container_name": "flux", "gpu": 2, "gpu_type": "a100"},
    ]
    with patch.object(
        sw.pod_manager,
        "add_pod",
        side_effect=[
            (None, HTTPStatus.OK),
            (None, HTTPStatus.BAD_REQUEST),
        ],
    ):
        response = await client.post("/api/auto_deploy/confirm", json={"specs": specs})

    assert response.status_code == HTTPStatus.MULTI_STATUS
    data = await response.get_json()
    assert data["deployed"] == ["gemma"]
    assert len(data["errors"]) == 1
    assert "flux" in data["errors"][0]
    assert "status=400" in data["errors"][0]


@pytest.mark.asyncio
async def test_auto_deploy_confirm_empty_specs() -> None:
    """Empty specs list returns 400."""
    client = _get_client()
    response = await client.post(
        "/api/auto_deploy/confirm",
        json={"specs": []},
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST
