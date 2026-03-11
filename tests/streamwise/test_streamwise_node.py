#!/usr/bin/env python3
"""
Unit tests for streamwise.py node-related endpoints.
"""

import sys
import pytest

from unittest.mock import patch

from http import HTTPStatus

from tests.test_utils import temp_sys_path
from tests.k8s_mock import K8sMock

mock_k8s = K8sMock()

mock_modules = {}
mock_modules.update(mock_k8s.get_sub_modules())
with patch.dict(sys.modules, mock_modules):
    with temp_sys_path("streamwise"):
        from streamwise import streamwise


@pytest.fixture(scope="function", autouse=True)
def setup_k8s_cluster() -> None:
    # for some reason k8s_config.load_kube_config() is not async mocked
    streamwise.k8s_cluster = "unittest"


@pytest.mark.asyncio
async def test_nodes() -> None:
    app = streamwise.app
    client = app.test_client()
    response = await client.get("/nodes")
    assert response.status_code == HTTPStatus.NOT_FOUND
    response_json = await response.get_json()
    assert response_json == {"error": "No nodes found"}


@pytest.mark.asyncio
async def test_api_nodes() -> None:
    app = streamwise.app
    client = app.test_client()
    response = await client.get("/api/nodes")
    assert response.status_code == HTTPStatus.OK
    response_json = await response.get_json()
    assert response_json == []


@pytest.mark.asyncio
async def test_remove_node() -> None:
    app = streamwise.app
    client = app.test_client()
    response = await client.delete("/api/node/testnode")
    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    response_json = await response.get_json()
    assert response_json == {"error": "object MagicMock can't be used in 'await' expression"}  # TODO


@pytest.mark.asyncio
async def test_node_info() -> None:
    app = streamwise.app
    client = app.test_client()
    response = await client.get("/node/testnode")
    assert response.status_code == HTTPStatus.NOT_FOUND
    response_json = await response.get_json()
    assert response_json == {"error": "Node 'testnode' not found"}
