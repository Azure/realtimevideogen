"""
Utilities for Kubernetes.
"""
import logging

from typing import Union
from typing import List
from typing import Dict
from typing import Optional
from typing import Any

from kubernetes_asyncio import config as k8s_config
from kubernetes_asyncio import client as k8s_client
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client import CoreV1Api


class NoActiveContainerError(Exception):
    """Exception for no active containers."""
    def __init__(
        self,
        service_name: str,
        containers: Optional[List['K8sContainer']] = None,
    ) -> None:
        super().__init__(f"No active container for '{service_name}'")
        self.service_name = service_name
        self.containers = containers

    def __str__(self) -> str:
        return f"No active container for '{self.service_name}'"


class NoRunnableContainerError(Exception):
    """Exception for no runnable containers."""
    def __init__(self, service_name: str) -> None:
        super().__init__(f"No runnable containers for '{service_name}'")
        self.service_name = service_name

    def __str__(self) -> str:
        return f"No runnable containers for '{self.service_name}'"


class ServiceNotFoundError(Exception):
    """Exception for service not found."""
    def __init__(self, service_name: str) -> None:
        super().__init__(f"Service '{service_name}' not found")
        self.service_name = service_name

    def __str__(self) -> str:
        return f"Service '{self.service_name}' not found"


class K8sContainer:
    """A Kubernetes container."""
    def __init__(self, name: str) -> None:
        self.name = name
        self.ip: Optional[str] = None
        self.port: Optional[int] = None
        self.resources: Dict[str, str] = {}
        self.gpu_model: Optional[str] = None
        self.num_gpus: int = -1
        self.status: Optional[str] = None
        self.busy = False

    def set_gpu_model(self, gpu_model: Optional[str]) -> None:
        self.gpu_model = gpu_model

    def set_num_gpus(self, num_gpus: int) -> None:
        self.num_gpus = num_gpus

    def get_gpu_model(self) -> Optional[str]:
        return self.gpu_model

    def get_num_gpus(self) -> int:
        if self.resources and "nvidia.com/gpu" in self.resources:
            return int(self.resources["nvidia.com/gpu"])
        return self.num_gpus

    def get_url(self) -> Optional[str]:
        if self.ip is None or self.port is None:
            return None
        return f"http://{self.ip}:{self.port}"

    def is_active(self) -> bool:
        if self.status is None:
            return False
        if self.status != "ok":
            return False
        if self.get_url() is None:
            return False
        return True

    def is_busy(self) -> bool:
        if self.busy is None:
            return False
        return self.busy

    def __str__(self) -> str:
        return f"K8sContainer(name={self.name}, status={self.status}, busy={self.busy}, " + \
               f"ip={self.ip}, port={self.port}, resources={self.resources})"

    def __repr__(self) -> str:
        return self.__str__()


class K8sService:
    """A Kubernetes service with multiple containers."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.containers: List[K8sContainer] = []

    def add_container(self, container: K8sContainer) -> None:
        self.containers.append(container)

    def merge_service(self, other: 'K8sService') -> None:
        self.containers.extend(other.containers)

    def get_active_containers(self) -> List[K8sContainer]:
        """Get the list of active containers for this service."""
        if not self.containers:
            raise NoActiveContainerError(self.name, self.containers)
        active_containers = [c for c in self.containers if c.is_active()]
        if not active_containers:
            raise NoActiveContainerError(self.name, self.containers)
        return active_containers

    def get_runnable_containers(self) -> List[K8sContainer]:
        """Get the list of runnable (not busy) containers for this service."""
        active_containers = self.get_active_containers()
        nonbusy_containers = [c for c in active_containers if not c.is_busy()]
        if not nonbusy_containers:
            raise NoRunnableContainerError(self.name)
        return nonbusy_containers

    def get_best_containers(
        self,
        excluded_containers: Optional[List[K8sContainer]] = None,
        exclude_busy: bool = True,
    ) -> Optional[List[K8sContainer]]:
        """Get the best available containers for this service, optionally excluding some containers."""
        if exclude_busy:
            runnable_containers = self.get_runnable_containers()
        else:
            runnable_containers = self.get_active_containers()
        if not runnable_containers:
            return None
        if excluded_containers:
            runnable_containers = [
                c for c in runnable_containers
                if c not in excluded_containers
            ]

        sorted_containers = sorted(
            runnable_containers,
            key=lambda c: (
                # Prefer Active, then GPU, then CPU, then memory
                # TODO check and rank for H200, H100, A100,...
                int(c.resources.get("nvidia.com/gpu", "0")),  # GPU
                int(c.resources.get("cpu", "0")),  # CPU
                parse_k8s_resource_quantity(c.resources.get("memory", "0"))  # Memory
            ),
            reverse=True
        )
        return sorted_containers

    def get_best_container(
        self,
        excluded_containers: Optional[List[K8sContainer]] = None,
        exclude_busy: bool = True,
    ) -> Optional[K8sContainer]:
        best_containers = self.get_best_containers(
            excluded_containers=excluded_containers,
            exclude_busy=exclude_busy)
        if not best_containers:
            return None
        return best_containers[0]

    def __str__(self) -> str:
        return f"K8sService(name={self.name}, containers={len(self.containers)})"


def parse_k8s_resource_quantity(quantity: str) -> Union[int, float]:
    """Parse Kubernetes resource quantity strings into numeric values."""
    try:
        if quantity.endswith("m"):
            return float(quantity[:-1]) / 1000.0
        if quantity.endswith("Ki"):
            return float(quantity[:-2]) * 1024
        if quantity.endswith("Mi"):
            return float(quantity[:-2]) * 1024 * 1024
        if quantity.endswith("Gi"):
            return float(quantity[:-2]) * 1024 * 1024 * 1024
        if "." in quantity:
            return float(quantity)
        return int(quantity)
    except Exception as ex:
        logging.info(f"Error parsing resource quantity '{quantity}': {ex}")
        return 0


async def load_k8s_config(
    context_name: Optional[str] = None
) -> None:
    """Load Kubernetes configuration."""
    if context_name == "unittest":
        return  # Skip loading config for unit tests
    elif context_name == "incluster":
        k8s_config.load_incluster_config()  # Running in a pod
    else:
        await k8s_config.load_kube_config(context=context_name)


def is_k8s_node_ready(node: k8s_client.V1Node) -> bool:
    """Check if a Kubernetes node is ready."""
    for condition in node.status.conditions:
        if condition.type == "Ready":
            return condition.status == "True"
    return False


async def get_k8s_nodes(
    context_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get the list of Kubernetes nodes."""
    await load_k8s_config(context_name)
    async with ApiClient() as api_client:
        k8s_api = CoreV1Api(api_client)
        nodes = await k8s_api.list_node()
        ret = []
        for node in nodes.items:
            node_name = node.metadata.name
            is_ready = is_k8s_node_ready(node)
            labels = node.metadata.labels

            allocatable_resources = node.status.allocatable
            capacity_resources = node.status.capacity
            images = None
            if node.status.images is not None:
                images = [{
                    "names": image.names,
                    "size_bytes": image.size_bytes
                } for image in node.status.images]
            addresses = []
            if node.status.addresses is not None:
                for address in node.status.addresses:
                    addresses.append(address.to_dict())

            gpu_model = "N/A"
            if "nvidia.com/gpu.product" in labels:
                gpu_model = labels.get("nvidia.com/gpu.product", "N/A")
            elif "beta.kubernetes.io/instance-type" in labels:
                instance_type = labels["beta.kubernetes.io/instance-type"]
                # gpu_model = get_gpu_model_from_instance_type(instance_type)
                gpu_model = instance_type

            region = "N/A"
            if "azure/region" in labels:
                region = labels["azure/region"]
            elif "topology.kubernetes.io/region" in labels:
                region = labels["topology.kubernetes.io/region"]

            resource_group = "N/A"
            if "network-resourcegroup" in labels:
                resource_group = labels.get("network-resourcegroup", "N/A")

            mig_enabled = any(k.startswith("nvidia.com/mig-") for k in allocatable_resources)

            info = {
                "node_name": node_name,
                "region": region,
                "resource_group": resource_group,
                "addresses": addresses,
                "is_ready": is_ready,
                "capacity_resources": {
                    "cpu": parse_k8s_resource_quantity(capacity_resources.get("cpu", "N/A")),
                    "memory": parse_k8s_resource_quantity(capacity_resources.get("memory", "N/A")),
                    "storage": parse_k8s_resource_quantity(capacity_resources.get("ephemeral-storage", "N/A")),
                    "gpu": capacity_resources.get("nvidia.com/gpu", "N/A"),
                },
                "allocatable_resources": {
                    "cpu": parse_k8s_resource_quantity(allocatable_resources.get("cpu", "N/A")),
                    "memory": parse_k8s_resource_quantity(allocatable_resources.get("memory", "N/A")),
                    "storage": parse_k8s_resource_quantity(allocatable_resources.get("ephemeral-storage", "N/A")),
                    "gpu": allocatable_resources.get("nvidia.com/gpu", "N/A"),
                },
                "architecture": node.status.node_info.architecture,
                "kernel_version": node.status.node_info.kernel_version,
                "os_image": node.status.node_info.os_image,
                "creation_timestamp": node.metadata.creation_timestamp,
                "labels": labels,
                "images": images,
                "gpu_model": gpu_model,
                "mig_enabled": mig_enabled,
            }
            ret.append(info)
        return ret


async def get_k8s_pods(
    context_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get the list of Kubernetes pods."""
    ret = []
    await load_k8s_config(context_name)
    async with ApiClient() as api_client:
        k8s_api = CoreV1Api(api_client)
        pods = await k8s_api.list_pod_for_all_namespaces()
        for pod in pods.items:
            pod_name = pod.metadata.name
            namespace = pod.metadata.namespace
            pod_status = pod.status.phase
            pod_ip = pod.status.pod_ip
            node_name = pod.spec.node_name
            url = "N/A"
            for container in pod.spec.containers:
                container_name = container.name
                resources = container.resources.requests
                cpu: float = 0.0
                memory: float = 0.0
                gpu: int = 0
                mig_profile: Optional[str] = None
                if resources is not None:
                    cpu = parse_k8s_resource_quantity(resources.get("cpu", "0"))
                    memory = parse_k8s_resource_quantity(resources.get("memory", "0"))
                    gpu = int(resources.get("nvidia.com/gpu", 0))
                    # A pod using MIG requests nvidia.com/mig-<profile> instead of
                    # nvidia.com/gpu; the two resource types are mutually exclusive.
                    if gpu == 0:
                        for resource_key, resource_val in resources.items():
                            if resource_key.startswith("nvidia.com/mig-"):
                                mig_profile = resource_key[len("nvidia.com/mig-"):]
                                gpu = int(resource_val)
                                break
                if container.ports:
                    for container_port in container.ports:
                        if pod_ip and container_port and container_port.container_port:
                            # Assumes one open port per container
                            url = f"http://{pod_ip}:{container_port.container_port}"
                ret.append({
                    "namespace": namespace,
                    "pod_name": pod_name,
                    "status": pod_status,
                    "pod_ip": pod_ip,
                    "container_name": container_name,
                    "url": url,
                    "node": node_name,
                    "cpu": cpu,
                    "memory": memory,
                    "gpu": gpu,
                    "mig_profile": mig_profile,
                })
    return ret


async def get_k8s_load_balancers(
    context_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    ret: List[Dict[str, Any]] = []

    await load_k8s_config(context_name)

    async with ApiClient() as api_client:
        k8s_api = CoreV1Api(api_client)
        services = await k8s_api.list_service_for_all_namespaces()
        for service in services.items:
            service_spec = service.spec
            if service_spec.type == "LoadBalancer" and len(service_spec.ports) > 0:
                external_ip = service_spec.load_balancer_ip
                if service.status.load_balancer.ingress:
                    external_ip = service.status.load_balancer.ingress[0].ip
                spec_port = service.spec.ports[0]
                external_url = None
                if external_ip and spec_port:
                    external_url = f"http://{external_ip}:{spec_port.port}"
                lb_info = {
                    "namespace": service.metadata.namespace,
                    "svc_name": service.metadata.name,
                    "pod_name": service_spec.selector.get("app", "N/A"),
                    "cluster_ip": service_spec.cluster_ip,
                    "external_ip": external_ip,
                    "external_port": spec_port.port,
                    "external_url": external_url,
                    "cluster_port": spec_port.target_port,
                    "node_port": spec_port.node_port,
                }
                ret.append(lb_info)
    return ret


def get_svc_name_from_container_name(container_name: str) -> str:
    """Get the service name from the container name."""
    service_name = container_name.lower().replace("-", "_")
    return service_name


async def get_k8s_services_ns(
    context_name: Optional[str] = None,
    namespace: str = "rtgen"
) -> Dict[str, K8sService]:
    svc_map = {}
    await load_k8s_config(context_name)
    async with ApiClient() as api_client:
        k8s_api = CoreV1Api(api_client)
        pods = await k8s_api.list_namespaced_pod(namespace)
        for pod in pods.items:
            pod_name = pod.metadata.name
            pod_ip = pod.status.pod_ip
            logging.debug(f"Pod: {pod_name} IP: {pod_ip}.")
            for container in pod.spec.containers:
                resources = container.resources.requests

                ret_container = K8sContainer(name=pod_name)
                ret_container.ip = pod_ip
                ret_container.resources = resources

                if not container.ports:
                    continue
                for container_port in container.ports:
                    svc_name = get_svc_name_from_container_name(container.name)
                    ret_container.port = container_port.container_port

                    if svc_name not in svc_map:
                        svc_map[svc_name] = K8sService(svc_name)
                    svc_map[svc_name].add_container(ret_container)
    return svc_map


async def get_k8s_services(
    context_name: Optional[str] = None
) -> Dict[str, K8sService]:
    svc_map = {}

    await load_k8s_config(context_name)
    async with ApiClient() as api_client:
        k8s_api = CoreV1Api(api_client)
        namespaces = await k8s_api.list_namespace()
        for ns in namespaces.items:
            namespace = ns.metadata.name
            if namespace.startswith("rtgen"):
                svc_map_ns = await get_k8s_services_ns(context_name, namespace)
                if svc_map_ns:
                    for service_name, svc in svc_map_ns.items():
                        if service_name not in svc_map:
                            svc_map[service_name] = svc
                        else:
                            svc_map[service_name].merge_service(svc)
    return svc_map
