"""
StreamWise Cluster Manager.
HTTP server that forwards the requests to each sub-module.
"""

from __future__ import annotations

import sys
import os
import json
import logging
import asyncio
import traceback
import argparse
import random

from typing import List
from typing import Dict
from typing import Optional
from typing import Any
from typing import Union

from kubernetes_asyncio.client.exceptions import ApiException

from quart import Quart
from quart import request
from quart import jsonify
from quart import render_template

from http import HTTPStatus

import file_manager
import http_session_manager
import pod_manager
import node_manager
import job_manager
import allocator_bridge

from service_manager import get_services
from service_manager import get_service_timestamps

sys.path.append("..")
from console_utils import setup_logging
from streamwise_apps import STREAMWISE_APPS

from quart_utils import QuartReturn
from quart_utils import json_pretty_filter
from quart_utils import format_datetime
from quart_utils import get_friendly_container_name
from quart_utils import get_friendly_pod_name
from quart_utils import get_class_emoji
from quart_utils import get_docker_image
from quart_utils import is_rtgen_container
from quart_utils import format_bytes
from quart_utils import format_url
from quart_utils import format_gpu_model
from quart_utils import format_gpu_model_mig
from quart_utils import get_aspect_ratio
from quart_utils import get_file_type_emoji
from quart_utils import get_content_type_emoji
from quart_utils import get_friendly_region_name

from k8s_utils import get_k8s_nodes
from k8s_utils import get_k8s_pods
from k8s_utils import get_k8s_load_balancers


# Quart/Flask app configuration
HOST = "0.0.0.0"
PORT = 18181
TMP_DIR = "/tmp"
LOG_FILE_NAME = "streamwise.log"
app = Quart(__name__)
route = app.route
template_filter = app.template_filter


# Kubernetes cluster configuration
# K8S_CLUSTER = "incluster"  # If running in a pod
K8S_CLUSTER = None  # Use default context

k8s_cluster = K8S_CLUSTER

# This needs to be created using deployment/helm/deploy.sh
NAMESPACE = "rtgen"

# Set to True when the server is started with --certfile (HTTPS mode)
use_https: bool = False


# Template filters
@template_filter("get_friendly_container_name")
async def get_friendly_container_name_template(container_name: str) -> str:
    return await get_friendly_container_name(container_name)


@template_filter("get_friendly_pod_name")
async def get_friendly_pod_name_template(pod_name: str) -> str:
    return await get_friendly_pod_name(pod_name)


@template_filter("get_class_emoji")
async def get_class_emoji_template(container_name: str) -> str:
    return await get_class_emoji(container_name)


@template_filter("json_pretty")
def json_pretty_filter_template(value: str, max_len: int = 128) -> str:
    return json_pretty_filter(value, max_len=max_len)


@template_filter("format_datetime")
def format_datetime_template(value: int) -> str:
    return format_datetime(value)


@template_filter("format_bytes")
def format_bytes_template(memory: int) -> str:
    return format_bytes(memory)


@template_filter("format_gpu_model")
def format_gpu_model_template(gpu_model: str) -> Optional[str]:
    return format_gpu_model(gpu_model)


@template_filter("format_gpu_model_mig")
def format_gpu_model_mig_template(
    gpu_model: str,
    mig_profile: Optional[str] = None
) -> Optional[str]:
    return format_gpu_model_mig(gpu_model, mig_profile)


@template_filter("format_url")
def format_url_template(url: Optional[str]) -> Optional[str]:
    return format_url(url)


@template_filter("get_aspect_ratio")
def get_aspect_ratio_template(ratio: float) -> str:
    return get_aspect_ratio(ratio)


@template_filter("get_file_type_emoji")
def get_file_type_emoji_template(file_type: str) -> str:
    return get_file_type_emoji(file_type)


@template_filter("get_content_type_emoji")
def get_content_type_emoji_template(content_type: str) -> str:
    return get_content_type_emoji(content_type)


@template_filter("get_friendly_region_name")
def get_friendly_region_name_template(region: str) -> str:
    return get_friendly_region_name(region)


@template_filter("is_rtgen_container")
async def is_rtgen_container_template(container_name: str) -> bool:
    return await is_rtgen_container(container_name)


@template_filter('get_docker_image')
async def get_docker_image_template(container_name: str) -> Optional[str]:
    return await get_docker_image(container_name)


# Setup and cleanup
@app.before_serving
async def startup() -> None:
    """Initialize sessions before the server starts."""
    await http_session_manager.startup()


@app.after_serving
async def shutdown() -> None:
    """Cleanup tasks after server stops."""
    await http_session_manager.shutdown()


@app.errorhandler(HTTPStatus.INTERNAL_SERVER_ERROR)
async def internal_error(ex: Exception) -> QuartReturn:
    """Handle internal server errors and display a user-friendly error page."""
    tb = traceback.format_exc()
    error_message = getattr(ex, "description", str(ex))
    logging.error(f"Internal error: {error_message}")
    return await render_template(
        "error.html",
        error_message=str(ex),
        exception=ex,
        traceback=tb,
    ), HTTPStatus.INTERNAL_SERVER_ERROR


# HTTP routes
@route("/", methods=["GET"])
async def index() -> QuartReturn:
    """Main index page showing all services and nodes."""
    svcs = []
    nodes = []
    pods = []
    lbs = []

    try:
        svcs = await get_services(
            namespace=NAMESPACE,
            k8s_cluster=k8s_cluster)
        nodes = await get_k8s_nodes(k8s_cluster)
        pods = await get_k8s_pods(k8s_cluster)
        lbs = await get_k8s_load_balancers(k8s_cluster)
    except Exception as ex:
        logging.error(f"Error fetching index data: {ex}: {traceback.format_exc()}")

    for svc in svcs:
        pod_name = svc.get("pod_name")
        lb = await get_lb_pod(pod_name)
        if lb:
            svc["load_balancer"] = await get_lb_pod(pod_name)

    app_svcs = [svc for svc in svcs if svc.get("container_name") in STREAMWISE_APPS]
    wrapper_svcs = [svc for svc in svcs if svc.get("container_name") not in STREAMWISE_APPS]

    return await render_template(
        "index.html",
        k8s_cluster=k8s_cluster if k8s_cluster else "default",
        svcs=svcs,
        app_svcs=app_svcs,
        wrapper_svcs=wrapper_svcs,
        nodes=nodes,
        pods=pods,
        lbs=lbs)


@route("/health", methods=["GET"])
async def health() -> QuartReturn:
    """Get health status."""
    health = {
        "status": "ok",
        "k8s_cluster": k8s_cluster,
    }
    return jsonify(health), HTTPStatus.OK


async def get_lb_pod(pod_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """Get load balancer info for a pod."""
    if not pod_name:
        return None
    lbs = await get_k8s_load_balancers(k8s_cluster)
    for lb in lbs:
        if lb["pod_name"] == pod_name:
            return lb
    return None


@route("/service/<service_name>", methods=["GET"])
async def service_info(service_name: str) -> str:
    """Display information about a specific service."""
    services = await get_services(
        container_name_filter=service_name,
        details=True,
        namespace=NAMESPACE,
        k8s_cluster=k8s_cluster)

    ret_svcs = [
        svc
        for svc in services
        if svc.get("container_name") == service_name
    ]
    first_svc = ret_svcs[0] if ret_svcs else None

    # Add load balancer info if available
    lb = None
    lbs = await get_k8s_load_balancers(k8s_cluster)
    for svc in ret_svcs:
        lb = next((
            lb
            for lb in lbs
            if lb["pod_name"] == svc["pod_name"]), None)
        if lb:
            svc["load_balancer"] = lb

    return await render_template(
        "service.html",
        service_name=service_name,
        svcs=ret_svcs,
        svc=first_svc,
        lb=lb)


@route("/service/<service_name>/<container_ip>", methods=["GET"])
async def container_info(
    service_name: str,
    container_ip: str
) -> str:
    """Display information about a specific container instance of a service."""
    services = await get_services(
        container_name_filter=service_name,
        details=True,
        namespace=NAMESPACE,
        k8s_cluster=k8s_cluster)
    ret_svcs = [
        svc
        for svc in services
        if svc.get("container_name") == service_name
        and svc.get("pod_ip") == container_ip
    ]
    first_svc = ret_svcs[0] if ret_svcs else None
    return await render_template(
        "service.html",
        service_name=service_name,
        svcs=ret_svcs,
        svc=first_svc)


@route("/service/<service_name>/timeline", methods=["GET"])
async def service_timelines(service_name: str) -> QuartReturn:
    """Display timeline information for a specific service."""
    services = await get_services(
        container_name_filter=service_name,
        details=False,
        namespace=NAMESPACE,
        k8s_cluster=k8s_cluster)

    async def fetch_all_timestamps() -> List[Dict]:
        tasks = [
            get_service_timestamps(
                svc["pod_name"],
                svc["container_name"],
                svc["url"]
            )
            for svc in services if svc.get("url") and svc["url"] != "N/A"
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        timestamps = []
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"Error fetching timestamps: {result}")
            elif isinstance(result, list) and result:
                timestamps.extend(result)
        return timestamps

    try:
        # TODO doesn't seem to retrieve from all instances
        timestamps = await fetch_all_timestamps()
        return await render_template(
            "service_timeline.html",
            service_name=service_name,
            timestamps=timestamps)
    except Exception as ex:
        logging.error(f"Error fetching timestamps for {service_name}: {ex}")
        return jsonify({"error": str(ex)}), HTTPStatus.INTERNAL_SERVER_ERROR


@route("/service/timeline", methods=["GET"])
async def services_timelines() -> QuartReturn:
    """Display timeline information for all services."""
    services = await get_services(
        details=False,
        namespace=NAMESPACE,
        k8s_cluster=k8s_cluster)

    async def fetch_all_timestamps() -> List[Dict]:
        tasks = []
        for service in services:
            if service["url"] != "N/A":
                task = get_service_timestamps(
                    service["pod_name"],
                    service["container_name"],
                    service["url"]
                )
                tasks.append(task)
        ret = []
        for result in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(result, list) and result:
                ret.extend(result)
        return ret

    try:
        timestamps = await fetch_all_timestamps()
        return await render_template(
            "service_timeline.html",
            service_name="All Services",
            timestamps=timestamps)
    except Exception as ex:
        logging.error(f"Error fetching timestamp: {ex}.")
        return jsonify({"error": str(ex)}), HTTPStatus.INTERNAL_SERVER_ERROR


@route("/node/<node_name>", methods=["GET"])
async def node_info(node_name: str) -> QuartReturn:
    """Display information about a specific node."""
    return await node_manager.node_info(
        node_name,
        k8s_cluster=k8s_cluster)


@route("/nodes", methods=["GET"])
async def nodes_info() -> QuartReturn:
    """Display information about all nodes."""
    return await node_manager.nodes_info(k8s_cluster=k8s_cluster)


@route("/audio_waveform/<container_ip>/<container_port>/<file_name>", methods=["GET"])
async def get_audio_waveform(container_ip: str, container_port: int, file_name: str) -> QuartReturn:
    """Generate and return a waveform PNG image for a WAV audio file."""
    return await file_manager.get_audio_waveform(container_ip, container_port, file_name)


@route("/video/<container_ip>/<container_port>/<file_name>", methods=["GET"])
async def get_video_info(container_ip: str, container_port: int, file_name: str) -> QuartReturn:
    """Get video file information from a container."""
    return await file_manager.get_video_info(container_ip, container_port, file_name)


@route("/files", methods=["GET"])
async def list_files() -> QuartReturn:
    """List files in the TMP_DIR directory."""
    return await file_manager.list_files(TMP_DIR)


@route("/file/<file_name>", methods=["GET"])
async def download_local_file(file_name: str) -> QuartReturn:
    """Download a file."""
    return await file_manager.download_local_file(
        TMP_DIR,
        file_name)


@route("/file_download/<container_ip>/<container_port>/<file_name>", methods=["GET"])
async def download_service_file(
    container_ip: str,
    container_port: int,
    file_name: str
) -> QuartReturn:
    """Download a file from a container."""
    return await file_manager.download_service_file(
        container_ip,
        container_port,
        file_name)


@route("/file_view/<service_name>/<container_ip>/<container_port>/<file_name>", methods=["GET"])
async def file_view(
    service_name: str,
    container_ip: str,
    container_port: int,
    file_name: str
) -> QuartReturn:
    """View the contents of a file from a container."""
    return await file_manager.file_view(
        service_name,
        container_ip,
        container_port,
        file_name)


@route("/file_stream/<container_ip>/<container_port>/<file_name>")
async def file_stream(
    container_ip: str,
    container_port: int,
    file_name: str
) -> QuartReturn:
    """Stream a file from a container."""
    return await file_manager.file_stream(
        container_ip,
        container_port,
        file_name)


@route("/job/", methods=["GET"])
async def submit_job() -> str:
    """Render the job submission form."""
    svcs = []
    try:
        svcs = await get_services(
            namespace=NAMESPACE,
            k8s_cluster=k8s_cluster)
    except Exception as ex:
        logging.exception("Error fetching services for /job/: %s", ex)
    return await render_template(
        "submit_job.html",
        svcs=svcs)


@route("/job/<container_ip>/<container_port>", methods=["GET"])
async def submit_job_container(
    container_ip: str,
    container_port: int
) -> str:
    """Render the job submission form for a specific container."""
    svcs = []
    try:
        svcs = await get_services(
            namespace=NAMESPACE,
            k8s_cluster=k8s_cluster)
    except Exception as ex:
        logging.exception("Error fetching services for /job/%s/%s: %s",
                          container_ip, container_port, ex)
    return await render_template(
        "submit_job.html",
        svcs=svcs,
        container_ip=container_ip,
        container_port=container_port)


@route("/api/job/<service_name>/<container_ip>/<container_port>", methods=["POST"])
async def api_submit_job(
    service_name: str,
    container_ip: str,
    container_port: int
) -> QuartReturn:
    """API interface to submit a job to the specified service."""
    return await job_manager.submit_job(
        service_name,
        container_ip,
        container_port)


@route("/pod", methods=["GET"])
async def add_pods() -> str:
    """Render the add pod form."""
    lb_rg = os.getenv("LB_RESOURCE_GROUP", "resource_group")
    lb_ip = os.getenv("LB_IP_ADDRESS", "1.2.3.4")
    return await render_template(
        "add_pod.html",
        lb_rg=lb_rg,
        lb_ip=lb_ip,
        random_lb_port=8080 + random.randint(0, 920),
    )


@route("/pod/<service_name>", methods=["GET"])
async def add_pod(service_name: str) -> str:
    """Render the add pod form for a specific service."""
    lb_rg = os.getenv("LB_RESOURCE_GROUP", "resource_group")
    lb_ip = os.getenv("LB_IP_ADDRESS", "1.2.3.4")
    return await render_template(
        "add_pod.html",
        service_name=service_name,
        lb_rg=lb_rg,
        lb_ip=lb_ip,
        random_lb_port=8080 + random.randint(0, 920),
    )


@route("/api/pod/<pod_name>", methods=["DELETE"])
async def api_remove_pod(pod_name: str) -> QuartReturn:
    """API interface to remove a pod by name."""
    namespace = request.args.get("namespace")
    if not namespace:
        return jsonify({"error": "Namespace is required"}), HTTPStatus.BAD_REQUEST
    return await pod_manager.remove_pod(
        pod_name,
        namespace=namespace,
        k8s_cluster=k8s_cluster)


@route("/api/services", methods=["GET"])
async def api_get_services() -> QuartReturn:
    """API interface to get the list of services."""
    services = await get_services(
        namespace=NAMESPACE,
        k8s_cluster=k8s_cluster)
    return jsonify(services), HTTPStatus.OK


@route("/api/nodes", methods=["GET"])
async def api_get_nodes() -> QuartReturn:
    """API interface to get the list of nodes."""
    nodes = await get_k8s_nodes(k8s_cluster)
    return jsonify(nodes), HTTPStatus.OK


def parse_gpu_info(
    gpu_info: Optional[Union[int, str]]
) -> tuple[int, Optional[str]]:
    num_gpus = 1
    mig_profile = None
    if isinstance(gpu_info, int):
        num_gpus = gpu_info
    elif isinstance(gpu_info, str):
        num_gpus = 1
        mig_profile = gpu_info
    else:
        num_gpus = 0
    return num_gpus, mig_profile


@route("/api/service", methods=["POST"])
async def api_add_service(
    max_gpus: int = 1
) -> QuartReturn:
    """API interface to add pods for all services."""
    try:
        # CPU, memory GiB, ephemeral storage GiB, GPU count, GPU type
        # Keep in sync with the helm values
        container_dict: dict[str, tuple[int, int, int, Union[int, str]]] = {
            "podcasttranscript": (1, 4, 16, 0),
            "slidetranscript": (1, 4, 16, 0),
            "gemma": (16, 192, 64, min(2, max_gpus)),
            # "hunyuanframepackf1": (32, 192, 64, min(2, max_gpus)),
            "hunyuanframepackf1": (24, 128, 64, min(2, max_gpus)),
            "hunyuanframepackvae": (4, 32, 16, 1),
            # "flux": (16, 192, 64, min(2, max_gpus)),
            "flux": (12, 128, 64, min(2, max_gpus)),
            "fluxkontext": (12, 128, 64, 1),
            # "fantasytalking": (16, 256, 64, min(2, max_gpus)),
            "fantasytalking": (12, 192, 64, min(2, max_gpus)),
            "realesrgan": (4, 32, 16, "1g.10gb"),
            "yolo": (4, 8, 16, "1g.10gb"),
            "kokoro": (2, 8, 16, "1g.10gb"),
            "whisper": (2, 8, 16, 1),
        }
        for container_name, (cpu, mem_gib, sotrage_gib, gpu_info) in container_dict.items():
            num_gpus, mig_profile = parse_gpu_info(gpu_info)
            await pod_manager.add_pod(
                container_name,
                cpu,
                mem_gib,
                ephemeral_storage_gib=sotrage_gib,
                gpu=num_gpus,
                mig_profile=mig_profile,
                namespace=NAMESPACE,
                k8s_cluster=k8s_cluster)
        return jsonify({"message": "Services added successfully"}), HTTPStatus.OK
    except ApiException as api_ex:
        body = json.loads(api_ex.body) if api_ex.body else {}
        message = body.get("message", "No message")
        if message == "namespaces \"rtgen\" not found":
            message += ".\nRun: 'kubectl create namespace rtgen'"
        logging.error(f"K8s API error adding services: {message}.")
        return jsonify({"error": message}), HTTPStatus.INTERNAL_SERVER_ERROR
    except Exception as ex:
        logging.error(f"Error adding services: {ex}.")
        return jsonify({"error": str(ex)}), HTTPStatus.INTERNAL_SERVER_ERROR


@route("/api/apps", methods=["POST"])
async def api_add_apps() -> QuartReturn:
    """API interface to add pods for all applications."""
    try:
        lb_rg = os.getenv("LB_RESOURCE_GROUP")
        lb_ip = os.getenv("LB_IP_ADDRESS")
        # CPU, memory GiB, ephemeral storage GiB, GPU count
        container_dict: dict[str, tuple[int, int, int, int]] = {
            "streamcast": (1, 4, 4, 0),
            "streampersona": (1, 4, 4, 0),
            "streamchat": (1, 4, 4, 0),
            "streamshort": (1, 4, 4, 0),
            "streammovie": (1, 4, 4, 0),
            "streamanimate": (1, 4, 4, 0),
            "streamlecture": (1, 4, 4, 0),
            "streamdub": (1, 4, 4, 0),
            "streamedit": (1, 4, 4, 0),
        }
        lb_ports = random.sample(range(8080, 9000), len(container_dict))
        for (container_name, (cpu, mem_gib, storage_gib, gpu)), lb_port in zip(container_dict.items(), lb_ports):
            await pod_manager.add_pod(
                container_name,
                cpu,
                mem_gib,
                ephemeral_storage_gib=storage_gib,
                gpu=gpu,
                lb_rg=lb_rg,
                lb_ip=lb_ip,
                lb_port=lb_port,
                namespace=NAMESPACE,
                k8s_cluster=k8s_cluster)
        return jsonify({"message": "Applications added successfully"}), HTTPStatus.OK
    except ApiException as api_ex:
        body = json.loads(api_ex.body) if api_ex.body else {}
        message = body.get("message", "No message")
        if message == "namespaces \"rtgen\" not found":
            message += ".\nRun: 'kubectl create namespace rtgen'"
        logging.error(f"K8s API error adding applications: {message}.")
        return jsonify({"error": message}), HTTPStatus.INTERNAL_SERVER_ERROR
    except Exception as ex:
        logging.error(f"Error adding applications: {ex}.")
        return jsonify({"error": str(ex)}), HTTPStatus.INTERNAL_SERVER_ERROR


@route("/api/pod", methods=["POST"])
async def api_add_pod() -> QuartReturn:
    """API interface to add a pod for the specified container."""
    form = await request.form
    container_name = form.get("container_name")
    cpu = int(form.get("cpu", 2))
    memory_gib = int(form.get("memory", 4))
    ephemeral_storage_gib = int(form.get("ephemeralStorage", 16))
    gpu = int(form.get("gpu", 0))
    gpu_type = form.get("gpu_type")
    mig_profile = form.get("mig_profile", "").strip() or None
    tag = form.get("tag", "").strip() or None
    lb_rg = form.get("lb_rg")
    lb_ip = form.get("lb_ip")
    lb_port = form.get("lb_port")
    try:
        return await pod_manager.add_pod(
            container_name=container_name,
            cpu=cpu,
            memory_gib=memory_gib,
            ephemeral_storage_gib=ephemeral_storage_gib,
            gpu=gpu,
            gpu_type=gpu_type,
            mig_profile=mig_profile,
            tag=tag,
            lb_rg=lb_rg,
            lb_ip=lb_ip,
            lb_port=int(lb_port) if lb_port else None,
            namespace=NAMESPACE,
            k8s_cluster=k8s_cluster,
            use_https=use_https,
        )
    except ApiException as api_ex:
        body = json.loads(api_ex.body) if api_ex.body else {}
        message = body.get("message", "No message")
        if message == "namespaces \"rtgen\" not found":
            message += ".\nRun: 'kubectl create namespace rtgen'"
        logging.error(f"K8s API error adding services: {message}.")
        return jsonify({"error": message}), HTTPStatus.INTERNAL_SERVER_ERROR
    except Exception as ex:
        logging.error(f"Error adding pod for {container_name}: {ex}.")
        traceback.print_exc()
        return jsonify({"error": str(ex)}), HTTPStatus.INTERNAL_SERVER_ERROR


@route("/api/auto_deploy", methods=["POST"])
async def api_auto_deploy() -> QuartReturn:
    """Run the model allocator to produce an optimized deployment plan.

    Expects JSON body:
        {
            "gpu_budget": {"A100": 8, "H100": 0, ...},
            "workflow": "streamcast"
        }

    Returns the deployment plan with estimated metrics and per-container specs.
    """
    try:
        data = await request.get_json()
        if not data:
            return jsonify({"error": "Request body must be JSON"}), HTTPStatus.BAD_REQUEST

        gpu_budget = data.get("gpu_budget")
        workflow_name = data.get("workflow")

        if not gpu_budget or not isinstance(gpu_budget, dict):
            return jsonify({"error": "Missing or invalid 'gpu_budget' field"}), HTTPStatus.BAD_REQUEST
        for gpu_type_name, count in gpu_budget.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                return (
                    jsonify(
                        {
                            "error": (
                                "Invalid 'gpu_budget' field: each GPU type count must be a "
                                "non-negative integer"
                            )
                        }
                    ),
                    HTTPStatus.BAD_REQUEST,
                )
        if not workflow_name or not isinstance(workflow_name, str):
            return jsonify({"error": "Missing or invalid 'workflow' field"}), HTTPStatus.BAD_REQUEST

        plan = await asyncio.to_thread(
            allocator_bridge.run_allocator,
            gpu_budget=gpu_budget,
            workflow_name=workflow_name,
        )
        return jsonify(allocator_bridge.deployment_plan_to_json(plan)), HTTPStatus.OK

    except ValueError as ve:
        return jsonify({"error": str(ve)}), HTTPStatus.BAD_REQUEST
    except Exception as ex:
        logging.exception("Error in auto_deploy: %s", ex)
        return jsonify({"error": str(ex)}), HTTPStatus.INTERNAL_SERVER_ERROR


@route("/api/auto_deploy/confirm", methods=["POST"])
async def api_auto_deploy_confirm() -> QuartReturn:
    """Execute a deployment plan produced by /api/auto_deploy.

    Expects JSON body:
        {
            "specs": [
                {
                    "container_name": "gemma",
                    "cpu": 16,
                    "memory_gib": 192,
                    "ephemeral_storage_gib": 64,
                    "gpu": 2,
                    "gpu_type": "a100",
                    "mig_profile": null
                },
                ...
            ]
        }

    Deploys all containers in the plan.
    """
    try:
        data = await request.get_json()
        if not data:
            return jsonify({"error": "Request body must be JSON"}), HTTPStatus.BAD_REQUEST

        specs = data.get("specs")
        if not specs or not isinstance(specs, list):
            return jsonify({"error": "Missing or invalid 'specs' field"}), HTTPStatus.BAD_REQUEST

        deployed: List[str] = []
        errors: List[str] = []

        for spec in specs:
            container_name = spec.get("container_name")
            if not container_name:
                errors.append("Spec missing 'container_name'")
                continue

            try:
                add_pod_result = await pod_manager.add_pod(
                    container_name=container_name,
                    cpu=int(spec.get("cpu", 4)),
                    memory_gib=int(spec.get("memory_gib", 16)),
                    ephemeral_storage_gib=int(spec.get("ephemeral_storage_gib", 16)),
                    gpu=int(spec.get("gpu", 0)),
                    gpu_type=spec.get("gpu_type"),
                    mig_profile=spec.get("mig_profile"),
                    namespace=NAMESPACE,
                    k8s_cluster=k8s_cluster,
                )

                status_code = HTTPStatus.OK
                if isinstance(add_pod_result, tuple) and len(add_pod_result) >= 2:
                    status_value = add_pod_result[1]
                    if isinstance(status_value, HTTPStatus):
                        status_code = status_value
                    elif isinstance(status_value, int):
                        status_code = HTTPStatus(status_value)

                if status_code >= HTTPStatus.BAD_REQUEST:
                    msg = f"Failed to deploy '{container_name}' (status={int(status_code)})"
                    logging.error(msg)
                    errors.append(msg)
                else:
                    deployed.append(container_name)
            except Exception as pod_ex:
                msg = f"Failed to deploy '{container_name}': {pod_ex}"
                logging.error(msg)
                errors.append(msg)

        status = HTTPStatus.OK if not errors else HTTPStatus.MULTI_STATUS
        return jsonify({
            "deployed": deployed,
            "errors": errors,
            "message": f"Deployed {len(deployed)}/{len(specs)} containers.",
        }), status

    except Exception as ex:
        logging.exception("Error in auto_deploy/confirm: %s", ex)
        return jsonify({"error": str(ex)}), HTTPStatus.INTERNAL_SERVER_ERROR


@route("/api/auto_deploy/workflows", methods=["GET"])
async def api_auto_deploy_workflows() -> QuartReturn:
    """Return available workflows and GPU types for the auto-deploy UI."""
    return jsonify({
        "workflows": allocator_bridge.get_available_workflows(),
        "gpu_types": allocator_bridge.get_available_gpu_types(),
    }), HTTPStatus.OK


@route("/api/node/<node_name>", methods=["DELETE"])
async def api_remove_node(node_name: str) -> QuartReturn:
    return await node_manager.remove_node(
        node_name,
        k8s_cluster=k8s_cluster)


if __name__ == "__main__":
    setup_logging(
        path=TMP_DIR,
        file_name=LOG_FILE_NAME)

    parser = argparse.ArgumentParser(description="StreamWise Cluster Manager")
    parser.add_argument("--k8s_cluster", type=str, default=K8S_CLUSTER, help="Kubernetes cluster context name")
    parser.add_argument("--host", type=str, default=HOST, help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=PORT, help="Port to bind the server to")
    parser.add_argument("--certfile", type=str, default=None, help="Path to SSL certificate file for HTTPS")
    parser.add_argument("--keyfile", type=str, default=None, help="Path to SSL private key file for HTTPS")
    parser.add_argument("--use-https", action="store_true", default=False,
                        help="Use HTTPS for outbound service connections (health checks, file fetches, jobs)")
    args = parser.parse_args()

    k8s_cluster = args.k8s_cluster
    host = args.host
    port = args.port
    use_https = bool(args.certfile)

    if args.use_https:
        http_session_manager.set_service_scheme("https")
        http_session_manager.set_verify_ssl(False)

    try:
        scheme = "https" if args.certfile else "http"
        logging.info(f"Starting on {scheme}://{host}:{port} for K8S cluster '{k8s_cluster}'.")
        app.run(
            host=host,
            port=port,
            certfile=args.certfile,
            keyfile=args.keyfile,
            # threaded=True,
            # debug=True,
        )
    except OSError as os_err:
        logging.error(f"OS error starting: {os_err}")
    except Exception as ex:
        logging.error(f"Error starting: {ex}")
