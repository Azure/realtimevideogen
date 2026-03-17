"""
Manage the LMM services and their URLs.
"""

import sys
import logging
import asyncio

from http import HTTPStatus

from aiohttp import TCPConnector
from aiohttp import ClientSession
from aiohttp import ClientTimeout
from aiohttp.client_exceptions import ClientConnectorError

from typing import Optional
from typing import List
from typing import Tuple
from typing import Dict

from kubernetes_asyncio.config.config_exception import ConfigException
from kubernetes_asyncio.client.exceptions import ApiException

# Local relative imports
sys.path.append("..")  # noqa: E402
sys.path.append("../..")  # noqa: E402

from console_utils import setup_logging
from k8s_utils import K8sService
from k8s_utils import K8sContainer
from k8s_utils import get_k8s_services
from k8s_utils import ServiceNotFoundError

from streamwise_apps import STREAMWISE_APPS
from streamwise_apps import VLLM_SERVICES


SERVICE_LONG_TIMEOUT = ClientTimeout(
    connect=10.0,
    sock_connect=10.0,
    sock_read=10 * 60.0,
)

STATUS_TIMEOUT = ClientTimeout(
    total=0.5
)


class LMMServiceManager:
    """
    Manage the LMM services and their URLs.
    """

    def __init__(
        self,
        app_name: str,
        k8s_cluster: Optional[str] = None
    ) -> None:
        """Initialize the service manager."""
        self.app_name = app_name
        self.k8s_cluster = k8s_cluster
        self.services: Dict[str, K8sService] = {}
        self.running = True
        self.logger = self._get_logger()

        connector = TCPConnector(
            limit=100,
            limit_per_host=10,
            use_dns_cache=True,
            force_close=True)
        self.session: Optional[ClientSession] = ClientSession(
            connector=connector,
            timeout=SERVICE_LONG_TIMEOUT)

    def _get_logger(self) -> logging.Logger:
        """Get the logger for the service manager."""
        logger = setup_logging(
            path=f"/tmp/{self.app_name}",
            file_name="service_manager.log",
            level=logging.INFO,
            use_global=True)
        return logger

    async def stop(self) -> None:
        """Stop the dependents."""
        self.running = False
        if self.session:
            await self.session.close()
            self.session = None

    async def init_k8s_services(self) -> None:
        """Initialize the K8s services."""
        await self.set_services()

    async def update_container_status(
        self,
        service_name: str,
        container: K8sContainer
    ) -> int:
        """Update the status of the container."""
        url = container.get_url()
        status, is_busy, gpu_model, num_gpus = await self.get_container_status(service_name, url)
        container.status = status
        container.busy = is_busy
        container.set_gpu_model(gpu_model)
        container.set_num_gpus(num_gpus)
        return 1  # for counting

    async def set_services(self) -> None:
        """Set the K8s services and their containers asynchronously."""
        try:
            services = await get_k8s_services(self.k8s_cluster)

            tasks = [
                self.update_container_status(service_name, container)
                for service_name, service in services.items()
                for container in service.containers
            ]
            await asyncio.gather(*tasks)

            self.services = services
        except ConfigException as config_ex:
            self.logger.error(f"Cannot load K8s services due to config error: {config_ex}")
        except Exception as ex:
            self.logger.error(f"Cannot load K8s services [{type(ex)}]: {ex}")

    async def update_service_status(self) -> None:
        """Update the status of all services and their containers asynchronously."""
        self.logger.debug("Updating service status...")

        # Update if containers/services change
        await self.update_services()

        tasks = [
            self.update_container_status(service_name, container)
            for service_name, service in self.services.items()
            for container in service.containers
        ]

        num_containers = sum(await asyncio.gather(*tasks))
        self.logger.debug(f"Services updated with {num_containers} containers.")

    async def update_services(self) -> None:
        """Update the K8s services and their containers asynchronously."""
        try:
            new_services = await get_k8s_services(self.k8s_cluster)
            for service_name, new_service in new_services.items():
                if service_name not in self.services:
                    logging.info(f"Service added: {service_name}")
                else:
                    old_service = self.services[service_name]
                    old_container_names = [c.name for c in old_service.containers]
                    new_container_names = [c.name for c in new_service.containers]
                    if set(old_container_names) != set(new_container_names):
                        logging.info(f"Containers for {service_name} changed: {new_container_names}")
                # Update in botch case for URLs, etc
                self.services[service_name] = new_service
            for service_name in list(self.services.keys()):
                if service_name not in new_services:
                    logging.info(f"Service removed: {service_name}")
                    del self.services[service_name]
        except ConfigException as config_ex:
            self.logger.error(f"Cannot update K8s services due to config error: {config_ex}")
        except ApiException as api_ex:
            self.logger.error(f"Cannot update K8s services due to API error: {api_ex}")
        except Exception as ex:
            self.logger.error(f"Cannot update K8s services [{type(ex)}]: {ex}")

    async def get_container_status(
        self,
        service_name: str,
        url: Optional[str]
    ) -> Tuple[Optional[str], bool, Optional[str], int]:
        """ Get the status of a service asynchronously from its health endpoint. """
        status = None
        is_busy = False
        gpu_model = None
        num_gpus = 0
        if not url:
            return status, is_busy, gpu_model, num_gpus
        if self.session is None:
            return status, is_busy, gpu_model, num_gpus
        try:
            timeout = STATUS_TIMEOUT
            health_url = f"{url}/health"
            async with self.session.get(health_url, timeout=timeout) as response:
                if response.status == HTTPStatus.OK:
                    if service_name in VLLM_SERVICES:
                        status = "ok"  # vLLM does not return JSON
                    else:
                        try:
                            response_data = await response.json()
                            if service_name in ("streamwise") or service_name in STREAMWISE_APPS:
                                status = "ok"  # No model reporting
                            elif service_name in response_data:
                                service_data = response_data[service_name]
                                is_busy = service_data.get("running", False)
                                # Services that can run multiple requests concurrently
                                if service_name in VLLM_SERVICES or service_name in (
                                    "podcasttranscript",
                                    "slidetranscript",
                                ):
                                    is_busy = False
                                status = service_data.get("status", None)
                                gpu_model = service_data.get("gpu", None)
                                num_gpus = int(service_data.get("world_size", -1))
                            else:
                                status = "?"
                        except Exception as ex:
                            if service_name in VLLM_SERVICES:
                                status = "ok"  # vLLM does not return JSON
                            else:
                                self.logger.error(f"Cannot parse JSON from {service_name} at {health_url}: {ex}")
                                status = "x"
                else:
                    status = f"error {response.status}"
        except TimeoutError:
            self.logger.error(f"Timeout connecting to '{service_name}' at {health_url}.")
            status = "timeout"
        except ClientConnectorError:
            self.logger.error(f"Cannot connect to '{service_name}' at {health_url}.")
        except Exception as ex:
            self.logger.error(f"Cannot connect to '{service_name}' at {health_url}: {type(ex)} {ex}")
            status = "x"
        return status, is_busy, gpu_model, num_gpus

    async def start_updater(
        self,
        interval_seconds: float = 1.0
    ) -> None:
        """Start a background task to periodically update the status of all services and their containers."""
        async def update_loop() -> None:
            while self.running:
                await self.update_service_status()
                await asyncio.sleep(interval_seconds)
        asyncio.create_task(update_loop())
        self.logger.info("Started service status update.")

    def print_service_status(self) -> list:
        """Print the current status of all services and their containers."""
        results = []
        for service_name, service in self.services.items():
            for container in service.containers:
                results.append([
                    service_name,
                    container.get_url(),
                    container.status,
                    container.get_gpu_model(),
                    container.get_num_gpus() if container.get_num_gpus() >= 0 else "-"
                ])
        return results

    async def warmup_services(self) -> None:
        """Warmup all services by sending a warmup request."""
        self.logger.info("Warming up services...")
        # TODO implement proper warmup

    def get_service_url(
        self,
        service_name: str,
        exclude_busy: bool = True,
    ) -> str:
        """Get the URL of a service, optionally excluding busy containers."""
        urls = self.get_service_urls(
            service_name,
            exclude_busy=exclude_busy)
        return urls[0]

    def get_service_urls(
        self,
        service_name: str,
        exclude_busy: bool = True,
    ) -> List[str]:
        """Get the URLs of a service, optionally excluding busy containers."""
        if service_name not in self.services:
            raise ServiceNotFoundError(service_name)
        service = self.services[service_name]
        best_containers = service.get_best_containers(exclude_busy=exclude_busy)
        if best_containers is None:
            return []
        return [url for c in best_containers if (url := c.get_url()) is not None]

    def get_num_services(self) -> int:
        return len(self.services)

    def items(self) -> List[Tuple[str, K8sService]]:
        return list(self.services.items())
