"""
Test K8s utilities.
"""

import logging
import pytest

from unittest.mock import MagicMock

from kubernetes_asyncio.config.config_exception import ConfigException
from aiohttp.client_exceptions import InvalidUrlClientError

from k8s_utils import NoRunnableContainerError
from k8s_utils import parse_k8s_resource_quantity
from k8s_utils import get_k8s_nodes
from k8s_utils import get_k8s_pods
from k8s_utils import get_k8s_services
from k8s_utils import get_svc_name_from_container_name
from k8s_utils import get_k8s_services_ns
from k8s_utils import K8sContainer
from k8s_utils import K8sService
from k8s_utils import NoActiveContainerError
from k8s_utils import is_k8s_node_ready


def test_parse_k8s_resource_quantity() -> None:
    assert parse_k8s_resource_quantity("1000m") == 1
    assert parse_k8s_resource_quantity("200m") == 0.2
    assert parse_k8s_resource_quantity("1") == 1
    assert parse_k8s_resource_quantity("2") == 2
    assert parse_k8s_resource_quantity("200Ki") == 200 * 1024
    assert parse_k8s_resource_quantity("512Mi") == 512 * (1024 ** 2)
    assert parse_k8s_resource_quantity("1Gi") == 1024 ** 3
    assert parse_k8s_resource_quantity("4Gi") == 4 * (1024 ** 3)
    assert parse_k8s_resource_quantity("0.2") == 0.2
    assert parse_k8s_resource_quantity("bogus") == 0


@pytest.mark.asyncio
async def test_get_k8s_nodes() -> None:
    try:
        nodes = await get_k8s_nodes()
        assert nodes is not None
        # assert len(nodes) == 0
    except ConfigException:
        logging.info("No K8s setup")


@pytest.mark.asyncio
async def test_get_k8s_pods() -> None:
    try:
        pods = await get_k8s_pods()
        assert pods is not None
        # assert len(pods) == 0
    except ConfigException:
        logging.info("No K8s setup")

    try:
        pods = await get_k8s_pods(context_name="unittest")
        assert pods is not None
    except InvalidUrlClientError:
        logging.info("No K8s setup for unittest context")


@pytest.mark.asyncio
async def test_get_k8s_services() -> None:
    try:
        services = await get_k8s_services()
        assert services is not None
        # assert len(services) == 0
    except ConfigException:
        logging.info("No K8s setup")

    try:
        services = await get_k8s_services(context_name="unittest")
        assert services is not None
    except InvalidUrlClientError:
        logging.info("No K8s setup for unittest context")


def test_data_model() -> None:
    # Container
    container = K8sContainer("flux_1234")
    assert container.name == "flux_1234"
    container.set_gpu_model("A100")
    assert container.get_gpu_model() == "A100"
    container.set_num_gpus(2)
    assert container.get_num_gpus() == 2
    assert container.is_active() is False
    assert container.is_busy() is False
    container.busy = True
    assert container.is_busy() is True
    container.busy = False

    container.status = "ok"
    container.ip = "1.2.3.4"
    container.port = 8080
    assert container.get_url() == "http://1.2.3.4:8080"

    assert container.is_active() is True
    assert container.is_busy() is False
    assert str(container) == "K8sContainer(name=flux_1234, status=ok, busy=False, ip=1.2.3.4, port=8080, resources={})"
    logging.info(f"Container: {container}")

    # Service
    service = K8sService("flux")
    with pytest.raises(NoActiveContainerError):
        service.get_active_containers()
    with pytest.raises(NoActiveContainerError):
        service.get_runnable_containers()

    service.add_container(container)
    active_containers = service.get_active_containers()
    assert len(active_containers) == 1

    runnable_containers = service.get_runnable_containers()
    assert len(runnable_containers) == 1
    service.containers[0].busy = True
    with pytest.raises(NoRunnableContainerError, match="flux"):
        service.get_runnable_containers()
    service.containers[0].busy = False
    runnable_containers = service.get_runnable_containers()
    assert len(runnable_containers) == 1

    best_containers = service.get_best_containers()
    assert best_containers is not None
    assert len(best_containers) == 1

    best_containers = service.get_best_containers(exclude_busy=False)
    assert best_containers is not None
    assert len(best_containers) == 1

    best_containers = service.get_best_containers(excluded_containers=[container])
    assert best_containers is not None
    assert best_containers == []

    best_container = service.get_best_container()
    assert best_container is not None

    assert str(service) == "K8sService(name=flux, containers=1)"

    container.status = None
    with pytest.raises(NoActiveContainerError):
        service.get_active_containers()


@pytest.mark.asyncio
async def test_get_k8s_services_ns() -> None:
    try:
        services = await get_k8s_services_ns()
        assert services is not None
    except ConfigException:
        logging.info("No K8s setup")

    try:
        services = await get_k8s_services_ns("unittest")
        assert services is not None
    except InvalidUrlClientError:
        logging.info("No K8s setup for unittest context")


def test_is_k8s_node_ready() -> None:
    node_mock = MagicMock()
    assert is_k8s_node_ready(node_mock) is False

    node_mock.status.conditions = [
        MagicMock(type="Ready", status="True"),
    ]
    assert is_k8s_node_ready(node_mock) is True


def test_get_svc_name_from_container_name() -> None:
    assert get_svc_name_from_container_name("flux-1234") == "flux_1234"
    assert get_svc_name_from_container_name("model-serving-abcde_5678") == "model_serving_abcde_5678"
