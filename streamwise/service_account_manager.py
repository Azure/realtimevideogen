"""
Service account manager for Kubernetes clusters.
"""

import sys

from http import HTTPStatus

from typing import List
from typing import Optional

from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client import ApiException
from kubernetes_asyncio.client import CoreV1Api
from kubernetes_asyncio.client import V1ObjectMeta
from kubernetes_asyncio.client import V1PolicyRule
from kubernetes_asyncio.client import V1ClusterRole
from kubernetes_asyncio.client import V1ClusterRoleBinding
from kubernetes_asyncio.client import RbacAuthorizationV1Api
from kubernetes_asyncio.client import V1ServiceAccount
from kubernetes_asyncio.client import V1RoleRef

sys.path.append("..")
from k8s_utils import load_k8s_config


async def ensure_service_account(
    core_api: CoreV1Api,
    service_account_name: str,
    namespace: str
) -> None:
    try:
        await core_api.read_namespaced_service_account(service_account_name, namespace)
    except ApiException as ex:
        if ex.status != HTTPStatus.NOT_FOUND:
            raise
        sa_body = V1ServiceAccount(
            metadata=V1ObjectMeta(name=service_account_name))
        await core_api.create_namespaced_service_account(namespace=namespace, body=sa_body)


async def ensure_cluster_role(
    rbac_api: RbacAuthorizationV1Api,
    name: str,
    rules: List[V1PolicyRule]
) -> None:
    try:
        await rbac_api.read_cluster_role(name)
    except ApiException as ex:
        if ex.status != HTTPStatus.NOT_FOUND:
            raise
        cluster_role_body = V1ClusterRole(
            metadata=V1ObjectMeta(name=name),
            rules=rules)
        await rbac_api.create_cluster_role(body=cluster_role_body)


async def ensure_cluster_role_binding(
    rbac_api: RbacAuthorizationV1Api,
    binding_name: str,
    role_name: str,
    service_account_name: str,
    namespace: str
) -> None:
    try:
        await rbac_api.read_cluster_role_binding(binding_name)
    except ApiException as ex:
        if ex.status != HTTPStatus.NOT_FOUND:
            raise
        crb_body = V1ClusterRoleBinding(
            metadata=V1ObjectMeta(name=binding_name),
            role_ref=V1RoleRef(
                kind="ClusterRole",
                name=role_name,
                api_group="rbac.authorization.k8s.io",
            ),
            subjects=[{
                "kind": "ServiceAccount",
                "name": service_account_name,
                "namespace": namespace,
            }]
        )
        await rbac_api.create_cluster_role_binding(body=crb_body)


async def get_service_account(
    k8s_cluster: Optional[str],
    service_account_name: str,
    cluster_role_name: str,
    cluster_role_binding_name: str,
    rules: List[V1PolicyRule],
    namespace: str = "default"
) -> str:
    """
    Create a service account with specified permissions if it doesn't exist.
    """
    await load_k8s_config(k8s_cluster)
    async with ApiClient() as api_client:
        core_api = CoreV1Api(api_client)
        await ensure_service_account(
            core_api, service_account_name, namespace)

        rbac_api = RbacAuthorizationV1Api(api_client)
        await ensure_cluster_role(
            rbac_api, cluster_role_name, rules)
        await ensure_cluster_role_binding(
            rbac_api, cluster_role_binding_name, cluster_role_name, service_account_name, namespace)
    return service_account_name


async def get_streamwiseapp_service_account(
    k8s_cluster: Optional[str] = None,
    namespace: str = "default"
) -> str:
    """
    Create a service account with necessary permissions for Streamcast if it doesn't exist.
    This allow listing nodes, pods, services, events, and namespaces.
    """
    rules = [
        V1PolicyRule(
            api_groups=[""],
            resources=[
                "nodes",
                "pods",
                "pods/log",
                "services",
                "events",
                "namespaces",
                "clusterroles",
            ],
            verbs=["get", "list", "watch"]
        )
    ]
    return await get_service_account(
        k8s_cluster=k8s_cluster,
        service_account_name="streamwiseapp-service-account",
        cluster_role_name="streamwiseapp-manager",
        cluster_role_binding_name="streamwiseapp-manager-binding",
        rules=rules,
        namespace=namespace)


async def get_streamwise_service_account(
    k8s_cluster: Optional[str] = None,
    namespace: str = "default"
) -> str:
    """
    Create a service account with necessary permissions for Streamwise if it doesn't exist.
    This allow listing and creating namespaces, services, pods, and nodes.
    """
    rules = [
        V1PolicyRule(
            api_groups=[""],
            resources=[
                "nodes",
                "pods",
                "pods/log",
                "services",
                "events",
                "namespaces",
                "serviceaccounts",
                "clusterroles",
                "clusterrolebindings",
            ],
            verbs=["create", "get", "list", "watch", "delete", "patch"]
        )
    ]
    return await get_service_account(
        k8s_cluster=k8s_cluster,
        service_account_name="streamwise-service-account",
        cluster_role_name="streamwise-manager",
        cluster_role_binding_name="streamwise-manager-binding",
        rules=rules,
        namespace=namespace)
