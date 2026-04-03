"""
A mock for Kubernetes modules used in testing.
"""

from unittest.mock import MagicMock
from unittest.mock import AsyncMock

from typing import Dict
from typing import Any
from typing import Tuple


class MockApiException(BaseException):
    """A mock for kubernetes_asyncio.client.exceptions.ApiException."""
    def __init__(
        self,
        status: int,
        reason: str,
        body: Any = None,
    ) -> None:
        self.status = status
        self.reason = reason
        self.body = body
        super().__init__(f"ApiException: {status} {reason}")


class MockInvalidUrlClientError(BaseException):
    """A mock for kubernetes_asyncio.client.exceptions.InvalidUrlClientError."""
    def __init__(
        self,
        message: str,
    ) -> None:
        self.message = message
        super().__init__(f"InvalidUrlClientError: {message}")


class K8sMock(MagicMock):
    """A mock for Kubernetes modules with specific mocked attributes and methods."""

    def __init__(
        self,
        *args: Tuple,
        **kwargs: Dict
    ) -> None:
        super().__init__(*args, **kwargs)

    def get_sub_modules(self) -> Dict[str, Any]:
        # Async context manager for ApiClient
        mock_kubernetes_asyncio = MagicMock()
        mock_api_client = MagicMock()
        mock_api_client_cm = AsyncMock()
        mock_api_client_cm.__aenter__.return_value = mock_api_client
        mock_api_client_cm.__aexit__.return_value = None

        mock_core_api = MagicMock()
        mock_core_api.list_namespaced_pod = AsyncMock(return_value=MagicMock(items=[]))
        mock_core_api.list_namespace = AsyncMock(return_value=MagicMock(items=[]))
        mock_core_api.list_node = AsyncMock(return_value=MagicMock(items=[]))
        mock_core_api.list_pod_for_all_namespaces = AsyncMock(return_value=MagicMock(items=[]))
        mock_core_api.list_service_for_all_namespaces = AsyncMock(return_value=MagicMock(items=[]))
        mock_core_api.delete_namespaced_pod = AsyncMock(return_value=MagicMock(status="Success"))
        mock_core_api.delete_namespaced_service = AsyncMock(return_value=MagicMock(status="Success"))
        mock_core_api.read_namespaced_service_account = AsyncMock()
        mock_core_api.read_namespaced_pod_log = AsyncMock(return_value="log line")
        mock_core_api.create_namespaced_pod = AsyncMock(return_value=[])
        mock_core_api.create_namespaced_service = AsyncMock(return_value=MagicMock(status="Success"))

        mock_custom_api = MagicMock()
        mock_custom_api.get_namespaced_custom_object = AsyncMock(return_value=MagicMock())

        mock_rbac_api = MagicMock()
        mock_rbac_api.read_cluster_role = AsyncMock()
        mock_rbac_api.read_cluster_role_binding = AsyncMock()

        mock_k8s_client = MagicMock()
        mock_k8s_client.ApiClient.return_value = mock_api_client_cm
        mock_k8s_client.CoreV1Api.return_value = mock_core_api
        mock_k8s_client.CustomObjectsApi.return_value = mock_custom_api
        mock_k8s_client.RbacAuthorizationV1Api.return_value = mock_rbac_api

        mock_k8s_config = MagicMock()
        mock_k8s_config.load_kube_config = AsyncMock()
        mock_k8s_config.load_incluster_config = MagicMock()
        mock_k8s_config.kube_config = MagicMock()

        mock_exceptions = MagicMock()
        mock_exceptions.ApiException = MockApiException
        mock_exceptions.InvalidUrlClientError = MockInvalidUrlClientError

        return {
            "kubernetes_asyncio": mock_kubernetes_asyncio,
            "kubernetes_asyncio.config": mock_k8s_config,
            "kubernetes_asyncio.client": mock_k8s_client,
            "kubernetes_asyncio.client.exceptions": mock_exceptions,
            "kubernetes_asyncio.config.config_exception": MagicMock(),
        }
