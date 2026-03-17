#!/usr/bin/env python3

import os
import sys
import pytest
import asyncio

from unittest.mock import patch

# Add current path
sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path

mock_modules: dict[str, object] = {
    # "torch": mock_torch,
}

from k8s_utils import K8sService
from k8s_utils import K8sContainer

with patch.dict(sys.modules, mock_modules):
    with temp_sys_path("apps", "apps/streampersona"):
        from apps.client import ServiceRequest
        from apps.client import ServiceRequestWorker
        from apps.client import ServiceError
        from apps.lmm_service_manager import LMMServiceManager
        from k8s_utils import ServiceNotFoundError


def test_service_request() -> None:
    req = ServiceRequest(
        service_name="test_service",
        request_id="req_123",
        payload_json={"key": "value"},
        url="http://localhost:8000/test_service"
    )
    assert req.service_name == "test_service"

    assert req.is_running() is False
    assert req.client_timeout.connect is None
    assert req.get_base_request_url() == "http://localhost:8000"
    assert req.model_dump()["url"] == "http://localhost:8000/test_service"
    req_json = req.json()
    assert req_json is not None
    assert req_json.startswith('{')
    assert req_json.endswith('}')
    assert req.get_payload_len() > 0

    # req2 = ServiceRequest.parse_json(req_json)


@pytest.mark.asyncio
async def test_service_request_worker() -> None:
    service_manager = LMMServiceManager("streamwise")
    worker = ServiceRequestWorker(
        app_name="TestApp",
        service_manager=service_manager,
    )
    try:
        # Skip for unit test
        # await worker.start()

        request = ServiceRequest(
            service_name="test_service",
            request_id="req_123",
            payload_json={"key": "value"},
            url="http://localhost:8000/test_service"
        )

        with pytest.raises(ServiceNotFoundError):
            await worker.submit_request(request)

        service = K8sService("test_service")
        service_manager.services["test_service"] = service
        future = await worker.submit_request(request)
        assert future is not None
        assert isinstance(future, asyncio.Future)
        assert request.future is not None and request.future.done() is False
        assert request.status == "CREATED"
        assert request.exception is None

        await worker._http_request(request)
        assert request.future is not None and request.future.done() is True
        assert request.status == "FAILED"
        assert request.exception is not None
        assert isinstance(request.exception, ServiceError)
        assert "No active containers for test_service" in str(request.exception)
        assert request.retries == 0

        # Success request
        request_1 = ServiceRequest(
            service_name="test_service",
            request_id="req_124",
            payload_json={"key": "value"},
            url="http://localhost:8000/test_service"
        )

        container = K8sContainer("test_service_1234")
        service.add_container(container)

        await worker._http_request(request_1)
        assert request_1.future is None
        assert request_1.status == "RETRYING"
        assert request_1.exception is None
        assert request_1.retries == 1
    finally:
        await worker.stop()
