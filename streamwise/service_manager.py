"""
Kubernetes Service Manager.
"""

import sys
import re
import logging
import asyncio

from http import HTTPStatus

from aiohttp import ClientTimeout
from asyncio import TimeoutError

from typing import List
from typing import Dict
from typing import Optional
from typing import Any

from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client import CoreV1Api
from kubernetes_asyncio.client.exceptions import ApiException

import http_session_manager

sys.path.append("..")
from k8s_utils import load_k8s_config
from k8s_utils import parse_k8s_resource_quantity
from quart_utils import get_friendly_container_name
from quart_utils import get_class_emoji
from streamwise_apps import VLLM_SERVICES


async def get_k8s_pod_events(
    k8s_api: CoreV1Api,
    namespace: str,
    pod_name: str
) -> List[Dict]:
    """Fetch events related to a specific pod in a namespace."""
    pod_field_selector = f"involvedObject.name={pod_name},involvedObject.namespace={namespace}"
    try:
        events = await k8s_api.list_namespaced_event(
            namespace=namespace,
            field_selector=pod_field_selector
        )
        return [{
            "last_timestamp": event.last_timestamp,
            "reason": event.reason,
            "message": event.message,
            "type": event.type,
            "count": event.count,
        } for event in events.items]
    except ApiException as ex:
        logging.error(f"Cannot read events for pod {pod_name}: {ex.reason}.")
    except Exception as ex:
        logging.error(f"Cannot read events for pod {pod_name}: {ex}")
    return []


async def get_service_files(
    container_name: str,
    url: Optional[str] = None
) -> Optional[List[str]]:
    """Asynchronous version of get_service_files"""
    if url is None or url == "N/A":
        return None
    if container_name in VLLM_SERVICES:
        return []

    timeout = ClientTimeout(total=0.5, connect=0.5)  # Short timeout for faster responses
    try:
        session = await http_session_manager.get_global_session()
        async with session.get(f"{url}/files", timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            if response.status == HTTPStatus.OK:
                if "application/json" in content_type:
                    content_json = await response.json()
                    return content_json.get("files", [])
            logging.warning(f"Unexpected response from '{container_name}' on '{url}/files': "
                            f"status={response.status}, content-type={content_type}")
    except TimeoutError:
        logging.warning(f"Timeout fetching files for {url}.")
    except Exception as ex:
        logging.error(f"Error fetching files for {url}: {ex}.")
    return []


def parse_vllm_metrics(metrics_text: str) -> Dict[str, float]:
    """
    Parses vLLM metrics from the provided text.
    https://docs.vllm.ai/en/stable/design/v1/metrics.html#v0-metrics
    Example metric lines:
    vllm:num_requests_running{engine="0",model_name="google/gemma-3-27b-it"} 0.0
    vllm:request_success_total{engine="0",finished_reason="stop",model_name="google/gemma-3-27b-it"} 0.0
    vllm:request_prompt_tokens_bucket{engine="0",le="1.0",model_name="google/gemma-3-27b-it"} 0.0
    vllm:request_success_total{engine="0",finished_reason="stop",model_name="google/gemma-3-27b-it"} 0.0
    vllm:request_success_total{engine="0",finished_reason="length",model_name="google/gemma-3-27b-it"} 0.0
    vllm:request_success_total{engine="0",finished_reason="abort",model_name="google/gemma-3-27b-it"} 0.0
    """
    metrics: Dict[str, float] = {}
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^vllm:([a-zA-Z_][a-zA-Z0-9_:]*)(\{.*?\})?\s+([0-9.]+)$", line)
        if m:
            metric_name = m.group(1)
            # labels = m.group(2)
            value = float(m.group(3))
            metrics[metric_name] = metrics.get(metric_name, 0) + value
    return metrics


async def get_service_health(
    container_name: str,
    url: str
) -> Optional[Dict[str, Any]]:
    """Get the health status of a service asynchronously."""
    if url is None or url == "N/A":
        return None
    try:
        timeout = ClientTimeout(total=0.5, connect=0.5)  # Short timeout for faster responses
        session = await http_session_manager.get_global_session()
        # vLLM services (Gemma, Llama, Whisper) use /metrics instead of /health
        if container_name in VLLM_SERVICES:
            # /load
            # /v1/models
            # /version
            # /metrics
            async with session.get(f"{url}/metrics", timeout=timeout) as response:
                if response.status == HTTPStatus.OK:
                    text = await response.text()
                    vllm_metrics = parse_vllm_metrics(text)
                    is_running = vllm_metrics.get("num_requests_running", 0) > 0
                    return {
                        "status": "ok",
                        "running": is_running,  # It doesn't block
                        "vllm_metrics": vllm_metrics,
                    }
                return {"status": f"unhealthy ({response.status})"}
        # Rest of the services
        else:
            async with session.get(f"{url}/health", timeout=timeout) as response:
                if response.status == HTTPStatus.OK:
                    content_json = await response.json()
                    if len(content_json) == 1 and "health" not in content_json:
                        # This is a nested health response, e.g. {"service_name": { ... }}
                        for _, value in content_json.items():
                            return value
                    return content_json
                return {"status": f"Unhealthy ({response.status})"}
    except TimeoutError:
        logging.warning(f"Timeout checking health for {url}.")
        return {"status": "timeout"}
    except Exception as ex:
        logging.error(f"Error checking health for {container_name} on {url}: {ex}.")
    return {"status": "failed"}


async def get_health_and_files_async(
    services_data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Get health and files for multiple services asynchronously."""
    if not services_data:
        return services_data
    for service in services_data:
        service["health"] = "N/A"
        service["files"] = []

    async def fetch_for_service(service: Dict[str, Any]) -> Dict[str, Any]:
        url = service.get("url", "N/A")
        if not url or url == "N/A":
            return service
        container_name = service.get("container_name", "unknown")
        try:
            health_task = get_service_health(container_name, url)
            files_task = get_service_files(container_name, url)
            health, files = await asyncio.gather(
                health_task,
                files_task,
                return_exceptions=True)

            if isinstance(health, Exception):
                logging.warning(f"Health check failed for {container_name} at {url}: {health}")
            service["health"] = health

            if isinstance(files, Exception):
                logging.warning(f"Fetching files failed for {container_name} at {url}: {files}")
            else:
                service["files"] = files
        except TimeoutError:
            logging.warning(f"Timeout fetching data for {container_name} at {url}.")
            service["health"] = "Timeout"
        except Exception as ex:
            logging.error(f"Error fetching data for {container_name} at {url}: {ex}.")
            service["health"] = "Error"
        return service

    return await asyncio.gather(*(fetch_for_service(svc) for svc in services_data))


async def get_k8s_container_logs(
    k8s_api: CoreV1Api,
    namespace: str,
    pod_name: str,
    container_name: str,
    num_lines: int = 500,
    follow: bool = False
) -> Optional[str]:
    """Fetch logs for a specific container in a pod within a namespace."""
    try:
        return await k8s_api.read_namespaced_pod_log(
            name=pod_name,
            namespace=namespace,
            container=container_name,
            follow=follow,
            tail_lines=num_lines,
            _request_timeout=1.0,  # Short timeout for faster UI responses
        )
    except ApiException as ex:
        logging.error(f"Cannot read logs for pod {pod_name}/{container_name}: {ex.reason}.")
    except Exception as ex:
        logging.error(f"Cannot read logs for pod {pod_name}/{container_name} [{type(ex)}]: {ex}.")
    return None


async def get_services_ns(
    namespace: str = "default",
    container_name_filter: Optional[str] = None,
    details: bool = False,
    k8s_cluster: Optional[str] = None
) -> List[Dict]:
    """Get all services in a specific namespace."""
    ret = []

    await load_k8s_config(k8s_cluster)
    async with ApiClient() as api_client:
        k8s_api = CoreV1Api(api_client)
        pods = await k8s_api.list_namespaced_pod(namespace)
        for pod in pods.items:
            pod_name = pod.metadata.name
            namespace = pod.metadata.namespace
            pod_status = pod.status.phase
            start_time = pod.status.start_time

            if pod.status.container_statuses is not None:
                for container_status in pod.status.container_statuses:
                    container_state = container_status.state
                    if container_state.running is not None:
                        # container_state.started_at
                        pod_status = "Running"
                    if container_state.terminated is not None:
                        # container_state.terminated
                        pod_status = "Terminated"
                    if container_state.waiting is not None:
                        pod_status = container_state.waiting.reason

            # Events
            events_list = await get_k8s_pod_events(
                k8s_api,
                namespace,
                pod_name) if details else []

            pod_ip = pod.status.pod_ip
            node_name = pod.spec.node_name
            for container in pod.spec.containers:
                container_name = container.name
                if container_name_filter and container_name != container_name_filter:
                    continue

                image = container.image

                # Resources
                resources = container.resources.requests
                cpu: int | float = 0
                memory: int | float = 0
                ephemeral_storage: int | float = 0
                gpu: int = 0
                if resources is not None:
                    cpu = parse_k8s_resource_quantity(resources.get("cpu", "0"))
                    memory = parse_k8s_resource_quantity(resources.get("memory", "0"))
                    ephemeral_storage = parse_k8s_resource_quantity(resources.get("ephemeral-storage", "0"))
                    gpu = int(resources.get("nvidia.com/gpu", 0))

                # Logs
                logs = await get_k8s_container_logs(
                    k8s_api,
                    namespace,
                    pod_name,
                    container_name
                ) if details else None

                if pod_ip is None or not container.ports:
                    ret.append({
                        "namespace": namespace,
                        "pod_name": pod_name,
                        "pod_ip": pod_ip,
                        "container_port": None,
                        "container_name": container_name,
                        "pod_status": pod_status,
                        "start_time": start_time,
                        "url": "N/A",
                        "node_name": node_name,
                        "cpu": cpu,
                        "memory": memory,
                        "gpu": gpu,
                        "ephemeral_storage": ephemeral_storage,
                        "events": events_list,
                        "image": image,
                        "logs": logs,
                        "health": None,
                        "files": None,
                    })
                else:
                    for container_port in container.ports:
                        url = f"http://{pod_ip}:{container_port.container_port}"

                        ret.append({
                            "namespace": namespace,
                            "pod_name": pod_name,
                            "pod_ip": pod_ip,
                            "container_port": container_port.container_port,
                            "container_name": container_name,
                            "pod_status": pod_status,
                            "start_time": start_time,
                            "url": url,
                            "node_name": node_name,
                            "cpu": cpu,
                            "memory": memory,
                            "gpu": gpu,
                            "ephemeral_storage": ephemeral_storage,
                            "events": events_list,
                            "image": image,
                            "logs": logs,
                            "health": None,  # Populated asynchronously
                            "files": None,   # Populated asynchronously
                        })

        # Run async health and files fetching if there are services with URLs
        services_with_urls = [
            service
            for service in ret
            if service["url"] != "N/A"
        ]
        if services_with_urls:
            await get_health_and_files_async(services_with_urls)

    return ret


async def get_services(
    container_name_filter: Optional[str] = None,
    details: bool = False,
    namespace: str = "default",
    k8s_cluster: Optional[str] = None
) -> List[Dict]:
    """Get all services across all namespaces."""
    ret = []

    await load_k8s_config(k8s_cluster)
    async with ApiClient() as api_client:
        k8s_api = CoreV1Api(api_client)
        try:
            namespace_list = await k8s_api.list_namespace()
            for ns in namespace_list.items:
                namespace_ix = ns.metadata.name
                if namespace_ix.startswith(namespace):
                    services = await get_services_ns(
                        namespace,
                        container_name_filter=container_name_filter,
                        details=details,
                        k8s_cluster=k8s_cluster)
                    if services:
                        ret.extend(services)
        except ApiException as api_ex:
            print(f"Exception when listing namespace '{namespace}': {api_ex}")
    return ret


async def get_service_timestamps(
    pod_name: str,
    container_name: str,
    url: str
) -> Optional[List[Dict]]:
    """Get the timestamps from a service asynchronously."""
    if url is None or url == "N/A":
        return None
    if container_name in VLLM_SERVICES:
        return []

    try:
        session = await http_session_manager.get_global_session()
        timeout = ClientTimeout(total=0.5, connect=0.5)  # Short timeout for faster responses
        async with session.get(f"{url}/timestamps", timeout=timeout) as response:
            if response.status == HTTPStatus.OK:
                content_json = await response.json()
                if len(content_json) == 1:
                    for _, timestamps in content_json.items():
                        for timestamp in timestamps:
                            original_id = timestamp.get("id", "")
                            timestamp["id"] = f"{pod_name}_{original_id}"
                            service_name = timestamp["group"]
                            container_name = await get_friendly_container_name(service_name)
                            class_emoji = await get_class_emoji(service_name)
                            timestamp["group"] = f"{container_name} {class_emoji}"
                        return timestamps
                return content_json
    except Exception as ex:
        logging.error(f"Error checking timestamps for {url}: {ex}.")
    return []
