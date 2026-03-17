"""
Client for managing requests to LMM services.
"""

import sys
import time
import logging
import asyncio
import json

from urllib.parse import urlparse

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from http import HTTPStatus

from aiohttp import TCPConnector
from aiohttp import ClientTimeout
from aiohttp import ClientSession
from aiohttp import ClientOSError
from aiohttp import ClientError
from asyncio import TimeoutError

from enum import Enum

from typing import List
from typing import Optional
from typing import Dict
from typing import Tuple
from typing import Any
from typing import Union

from lmm_service_manager import LMMServiceManager

from client_timeout import SERVICE_TIMEOUT
from client_timeout import SERVICE_LONG_TIMEOUT

from client_headers import JSON_HEADERS
from client_headers import BINARY_HEADERS

sys.path.append("..")  # noqa: E402

from console_utils import setup_logging
from console_utils import bytes_to_human

from k8s_utils import NoActiveContainerError
from k8s_utils import NoRunnableContainerError
from k8s_utils import ServiceNotFoundError


class RequestStatus(str, Enum):
    """Status of a service request, JSON-serializable as strings."""
    CREATED = "CREATED"
    PENDING = "PENDING"
    RETRYING = "RETRYING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"


class ServiceRequest(BaseModel):
    """A request to a service with its associated metadata and status."""

    request_id: str
    service_name: str
    path: Optional[str] = None
    base_url: Optional[str] = None
    url: Optional[str] = None
    deadline: Optional[float] = None
    status: RequestStatus = RequestStatus.CREATED
    retries: int = 0
    times: List[Tuple[RequestStatus, float]] = Field(
        default_factory=lambda: [(RequestStatus.CREATED, time.time())])

    # Non-serializable, exclude from JSON
    payload_json: Optional[Dict] = Field(default=None, exclude=True)
    payload_bytes: Optional[bytes] = Field(default=None, exclude=True)
    timeout: Optional[ClientTimeout] = Field(default_factory=lambda: SERVICE_TIMEOUT, exclude=True)
    future: Optional[asyncio.Future] = Field(default=None, exclude=True)
    exception: Optional[Exception] = Field(default=None, exclude=True)
    tasks: List[asyncio.Task] = Field(default_factory=list, exclude=True)

    model_config = {
        "arbitrary_types_allowed": True
    }

    @model_validator(mode="before")
    def validate_payload_and_convert(
        cls: Any,
        values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ensure payload exists and do conversions before model creation."""
        if not values.get("payload_json") and not values.get("payload_bytes"):
            raise ValueError("Either payload_json or payload_bytes must be provided.")
        timeout = values.get("timeout")
        if isinstance(timeout, ClientTimeout):  # ClientTimeout object
            values["timeout"] = timeout.total
        return values

    @property
    def client_timeout(self) -> ClientTimeout:
        """Return an aiohttp ClientTimeout object for this request."""
        return ClientTimeout(total=self.timeout)

    def set_status(self, status: RequestStatus) -> None:
        """Set the status of the request and record the time."""
        self.status = status
        self.times.append((status, time.time()))

    def set_failure(self, ex: Exception) -> None:
        """Set the request status to FAILED and record the exception."""
        self.set_status(RequestStatus.FAILED)
        if self.future:
            self.future.set_exception(ex)
        self.exception = ex

    def done(self) -> bool:
        """Check if the request is in a terminal state (COMPLETED, FAILED, CANCELLED, EXPIRED)."""
        return self.status in {
            RequestStatus.COMPLETED,
            RequestStatus.FAILED,
            RequestStatus.CANCELLED,
            RequestStatus.EXPIRED,
        }

    def is_running(self) -> bool:
        """Check if the request is currently running."""
        return self.status == RequestStatus.RUNNING

    def get_base_request_url(self) -> Optional[str]:
        """
        Get the base request URL (scheme, hostname, port) for the service request.
        For example: http://10.244.0.97:8080/hunyuanframepackf1 -> http://10.244.0.97:8080
        """
        if self.url is None:
            return None
        parsed_url = urlparse(self.url)
        base_request_url = f"{parsed_url.scheme}://{parsed_url.hostname}"
        if parsed_url.port is not None:
            base_request_url += f":{parsed_url.port}"
        return base_request_url

    def set_retry(self) -> None:
        """Set the request status to RETRYING and increment the retry count."""
        self.set_status(RequestStatus.RETRYING)
        self.retries += 1

    def dict(self, **kwargs: Any) -> Dict[str, Any]:
        """Custom dict for JSON serialization."""
        d = super().dict(**kwargs)
        d["status"] = self.status.value
        d["times"] = [(s.value, t) for s, t in self.times]
        return d

    def json(self, **kwargs: Any) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.dict(**kwargs), **kwargs)

    @classmethod
    def parse_json(cls, data: str) -> "ServiceRequest":
        """Parse a JSON string back into a ServiceRequest."""
        obj = json.loads(data)
        if "status" in obj:
            obj["status"] = RequestStatus(obj["status"])
        if "times" in obj:
            obj["times"] = [(RequestStatus(s), t) for s, t in obj["times"]]
        return cls(**obj)

    def get_payload_len(self) -> int:
        """Get the length of the payload in bytes, whether it's JSON or binary data."""
        if self.payload_bytes:
            return len(self.payload_bytes)
        if self.payload_json:
            return len(json.dumps(self.payload_json).encode("utf-8"))
        return 0


class ServiceError(Exception):
    """Exception for service errors."""

    def __init__(
        self,
        service_name: Optional[str] = None,
        job_id: Optional[str] = None,
        request_id: Optional[str] = None,
        message: Optional[str] = None,
        url: Optional[str] = None,
        status: Optional[Union[int, HTTPStatus, str]] = None,
        response_body: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> None:
        """Initialize the ServiceError exception."""
        super().__init__(message)
        self.job_id = job_id
        self.request_id = request_id
        self.message = message
        self.service_name = service_name
        self.url = url
        self.status: int | HTTPStatus | str = status or HTTPStatus.INTERNAL_SERVER_ERROR
        self.response_body = response_body
        self.content_type = content_type

    def __str__(self) -> str:
        """String representation of the ServiceError."""
        ret = f"{self.message} (Service:{self.service_name}"
        if self.job_id:
            ret += f" Job:{self.job_id}"
        if self.url:
            ret += f" URL:{self.url}"
        if self.status:
            ret += f" Status:{self.status}"
        if self.response_body:
            ret += f" Response:{self.response_body}"
        if self.content_type:
            ret += f" Type:{self.content_type}"
        ret += ")"
        return ret


class ServiceRequestWorker:
    """A worker that processes service requests asynchronously and manages their execution."""

    def __init__(
        self,
        app_name: str,
        service_manager: LMMServiceManager
    ) -> None:
        """Initialize the service request worker."""
        self.app_name = app_name
        self.service_manager = service_manager
        self.running = True

        connector = TCPConnector(
            limit=100,
            limit_per_host=10,
            use_dns_cache=True,
            force_close=True)
        self.session = ClientSession(
            connector=connector,
            timeout=SERVICE_LONG_TIMEOUT)

        self.queues: Dict[str, List[ServiceRequest]] = {}
        self.requests: Dict[str, ServiceRequest] = {}

        self.logger = self._get_logger()

    def _get_logger(self) -> logging.Logger:
        """Get the logger for the service worker."""
        logger = setup_logging(
            path=f"/tmp/{self.app_name}/logs",
            file_name="service_worker.log",
            level=logging.INFO)
        return logger

    """
    def _get_logger(self) -> logging.Logger:
        log_dir = "/tmp/streamwise/logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "service_manager.log")
        logger = logging.getLogger("service_manager")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            file_handler = logging.FileHandler(log_file)
            # formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            formatter = ColoredFormatter(
                fmt="[%(asctime)s] %(log_color)s%(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "white",
                    "WARNING": "yellow",
                    "ERROR": 'red',
                    "CRITICAL": 'bold_red',
                }
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger
    """

    def get_queued_requests(self) -> List[str]:
        """Get a list of request ids that are currently queued for processing."""
        ret = []
        for queue in self.queues.values():
            for request in queue:
                ret.append(request.request_id)
        return ret

    def get_requests(self) -> Dict[str, ServiceRequest]:
        """Get a dictionary of all requests currently being managed by the worker."""
        return self.requests

    async def start(self) -> None:
        """
        Start the worker to process requests asynchronously.
        It waits if nothing available.
        """
        MAX_RETRY_SECONDS = 0.5
        INIT_RETRY_SECONDS = 0.05
        retry_seconds = INIT_RETRY_SECONDS
        while self.running:
            processed_requests = self._process_requests()
            if processed_requests == 0:
                await asyncio.sleep(retry_seconds)
                # Exponential backoff up to max
                retry_seconds = min(retry_seconds * 2, MAX_RETRY_SECONDS)
            else:
                retry_seconds = INIT_RETRY_SECONDS  # Reset backoff

    def _process_requests(self) -> int:
        """Process queued requests for all services."""
        processed_requests = 0
        for queue in self.queues.values():
            processed_requests += self._process_requests_queue(queue)
        return processed_requests

    def _process_requests_queue(
        self,
        queue: List[ServiceRequest]
    ) -> int:
        """Process requests in a specific service queue."""
        if not queue:
            return 0

        next_request = None
        for queued_request in queue:
            if not next_request:
                next_request = queued_request
            elif not next_request.deadline and queued_request.deadline:
                # Take the one with a deadline over one without
                next_request = queued_request
            elif (queued_request.deadline is not None
                  and next_request.deadline is not None
                  and queued_request.deadline < next_request.deadline):
                # Take requests with earlier deadlines first
                next_request = queued_request

        if not next_request:
            return 0

        # Take from the queue and send it
        queue.remove(next_request)
        task = asyncio.create_task(self._http_request(next_request))
        next_request.tasks.append(task)
        return 1

    async def stop(self) -> None:
        """Stop the service request worker."""
        self.running = False
        if self.session:
            await self.session.close()
            self.session = None

    async def submit_request(
        self,
        request: ServiceRequest
    ) -> asyncio.Future:
        """Submit a request to the service and return an asyncio future."""
        self.logger.debug(f"Submitting request {request.request_id} to service {request.service_name}.")
        if request.service_name not in self.service_manager.services:
            raise ServiceNotFoundError(request.service_name)

        if request.future is not None:
            if request.future.done():
                raise ValueError(f"Request {request.request_id} already has a result, cannot submit again.")
        else:
            async_loop = asyncio.get_running_loop()
            request.future = async_loop.create_future()

        if request.service_name not in self.queues:
            queue: List[ServiceRequest] = []
            self.queues[request.service_name] = queue
        else:
            queue = self.queues[request.service_name]
        queue.append(request)
        self.requests[request.request_id] = request

        return request.future

    async def _http_request(
        self,
        request: ServiceRequest
    ) -> None:
        """Perform the HTTP request to the service and handle the response."""
        t0 = time.time()
        try:
            if request.base_url is not None:
                base_url = request.base_url
            else:
                base_urls = self.service_manager.get_service_urls(request.service_name)
                base_url = base_urls[0]

            path = request.service_name
            if request.path is not None:
                path = request.path
            request.url = f"{base_url}/{path}"
        except NoRunnableContainerError:
            # Wait until a container becomes available, queue it again to retry later
            self.logger.debug(f"Service {request.service_name} has no runnable containers, retrying...")
            request.set_retry()
            task = asyncio.create_task(self.submit_request(request))
            request.tasks.append(task)
            return
        except NoActiveContainerError as nac_ex:
            # Sometimes the update of the status gets out of sync so we retry
            if not nac_ex.containers:
                request.set_status(RequestStatus.FAILED)
                ex = ServiceError(
                    message=f"No active containers for {request.service_name}: {nac_ex.containers}.",
                    request_id=request.request_id,
                    service_name=request.service_name)
                request.set_failure(ex)
                return
            else:
                # Wait until a container becomes available, queue it again to retry later
                self.logger.debug(f"Service {request.service_name} has no active containers, retrying...")
                request.set_retry()
                task = asyncio.create_task(self.submit_request(request))
                request.tasks.append(task)
                return

        try:
            request.set_status(RequestStatus.RUNNING)
            post_args: Dict[str, Any] = {
                "headers": {},
            }
            if request.payload_json is not None:
                post_args = {
                    "json": request.payload_json,
                    "headers": JSON_HEADERS,
                }
            elif request.payload_bytes is not None:
                post_args = {
                    "data": request.payload_bytes,
                    "headers": BINARY_HEADERS,
                }
            post_args["timeout"] = request.timeout

            async with self.session.post(request.url, **post_args) as response:
                content_type = response.headers.get("Content-Type", "")
                if response.status == HTTPStatus.OK:
                    response_binary = await response.read()

                    if request.future is not None:
                        request.future.set_result((
                            content_type,
                            response_binary
                        ))
                    request.set_status(RequestStatus.COMPLETED)
                    self.logger.info(
                        f"Request {request.request_id} with {bytes_to_human(request.get_payload_len())} "
                        f"to {request.service_name}@{request.url} completed with "
                        f"{bytes_to_human(len(response_binary))} and type {content_type}.")
                    return

                # Handle error response
                err_msg = ""
                if content_type == "application/json":
                    response_json = await response.json()
                    err_msg = response_json.get("error")
                else:
                    err_msg = await response.text()

                if "generation in progress" in err_msg.lower() or "no runnable containers" in err_msg.lower():
                    # Queue it again to retry later
                    request.set_retry()
                    task = asyncio.create_task(self.submit_request(request))
                    request.tasks.append(task)
                    return
                if "request entity too large" in err_msg.lower():
                    payload_size = request.get_payload_len()
                    self.logger.error(
                        f"Request too long for '{request.service_name}': {err_msg}. Size: {payload_size} bytes.")

                self.logger.error(
                    f"Request {request.request_id} to {request.service_name}@{request.url} failed: {err_msg}.")
                ex = ServiceError(
                    message="Service request failed",
                    service_name=request.service_name,
                    request_id=request.request_id,
                    url=request.url,
                    status=response.status,
                    response_body=err_msg)
                request.set_failure(ex)
        except TimeoutError:  # usually empty
            msg = f"Timeout for {request.request_id} to {request.service_name}@{request.url}: "
            msg += f"{time.time() - t0:.3f} > {request.timeout}"
            self.logger.error(f"{msg}.")
            # TODO should we retry?
            request.set_failure(TimeoutError(msg))
        except ClientOSError as client_os_err:
            payload_len = len(json.dumps(post_args["data"]).encode("utf-8"))
            self.logger.error(
                f"Client OS error for {request.request_id} "
                f"with {bytes_to_human(payload_len)} "
                f"to {request.service_name}@{request.url}: {client_os_err}.")
            if "broken pipe" in str(client_os_err).lower():
                # Usually happens when querying the same multiple times
                # [Errno 32] Broken pipe
                request.set_retry()
                task = asyncio.create_task(self.submit_request(request))
                request.tasks.append(task)
            else:
                request.set_failure(client_os_err)
        except ClientError as client_err:
            self.logger.error(
                f"Client error for {request.request_id} to {request.service_name}@{request.url}: {client_err}.")
            if "server disconnected" in str(client_err).lower():
                request.set_retry()
                task = asyncio.create_task(self.submit_request(request))
                request.tasks.append(task)
            else:
                request.set_failure(client_err)
        except AssertionError as assert_err:
            self.logger.error(
                f"Assertion error for {request.request_id} to {request.service_name}@{request.url}: {assert_err}.")
            request.set_failure(assert_err)
        except ValueError as value_err:
            self.logger.error(
                f"Value error for {request.request_id} to {request.service_name}@{request.url}: {value_err}.")
            request.set_failure(value_err)
        except Exception as ex:
            ex_name = type(ex).__name__
            self.logger.error(
                f"Error for {request.request_id} to {request.service_name}@{request.url}: {ex} [type:{ex_name}].")
            request.set_failure(ex)
