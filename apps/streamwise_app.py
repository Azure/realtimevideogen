"""
StreamWise HTTP application.
"""

import sys
import os
import json
import errno
import argparse
import logging
import traceback
import asyncio
import aiofiles
import aiofiles.os
import mimetypes

from io import BytesIO

from abc import ABC
from abc import abstractmethod

from typing import Optional
from typing import Dict
from typing import Any
from typing import List
from typing import Tuple
from typing import Type

from http import HTTPStatus

from datetime import datetime

from quart import Quart
from quart import request
from quart import jsonify
from quart import render_template
from quart import send_file
from quart import send_from_directory

from jinja2 import ChoiceLoader
from jinja2 import FileSystemLoader

from hypercorn.config import Config
from hypercorn.asyncio import serve

from streamwise_job import StreamWiseJob
from streamwise_job import JobStatus
from streamwise_job import get_job_id
from streamwise_job import is_job_id
from streamwise_job import is_status_terminal
from streamwise_job import is_status_expired
from lmm_service_manager import LMMServiceManager

from resolutions import ASPECT_RATIO
from resolutions import RESOLUTIONS

# Local relative imports
sys.path.append("..")  # noqa: E402
sys.path.append("../..")  # noqa: E402

from console_utils import setup_logging

from media_utils import get_video_file_info
from media_utils import get_audio_file_info
from media_utils import get_tensor_file_info
from media_utils import get_image_file_info
from media_utils import get_text_file_info

import quart_utils

from quart_utils import QuartReturn

from quart_utils import format_string
from quart_utils import format_duration
from quart_utils import format_duration_short
from quart_utils import parse_request_id
from quart_utils import get_content_type_emoji
from quart_utils import get_file_type_emoji
from quart_utils import json_pretty_filter
from quart_utils import get_file_type
from quart_utils import get_mime_type
from quart_utils import get_aspect_ratio
from quart_utils import format_bytes
from quart_utils import get_friendly_container_name
from quart_utils import get_class_emoji

from tts_utils import generate_waveform_plt

HOST = "0.0.0.0"
PORT = 18080

K8S_CLUSTER = "incluster"


def status_history_to_times(status_history: Dict[float, JobStatus]) -> Dict[str, datetime]:
    """Convert status history timestamps to datetime objects."""
    times: Dict[str, datetime] = {}
    for timestamp_float, status_str in status_history.items():
        if status_str == "COMPLETED" or status_str == "EXPIRED":
            times[status_str] = datetime.fromtimestamp(timestamp_float)
        elif status_str not in times:
            times[status_str] = datetime.fromtimestamp(timestamp_float)
    return times


class StreamWiseApp(ABC):
    """Generic StreamWise Quart application."""

    def __init__(
        self,
        app_name: str = "streamwise"
    ) -> None:
        self.app_name = app_name

        self.tmp_dir = f"/tmp/{self.app_name}"
        self.log_file_name = f"{self.app_name}.log"

        self.file_manager = StreamWiseAppFileManager(self.tmp_dir)

        self.app = Quart(__name__)
        self.register_templates()
        self.register_filters()
        self.register_routes()

        self.args: Optional[argparse.Namespace] = None
        self.jobs: Dict[str, StreamWiseJob] = {}
        self.service_manager: Optional[LMMServiceManager] = None

    @abstractmethod
    def create_job(
        self,
        job_id: str,
        job_config: Dict[str, Any]
    ) -> StreamWiseJob:
        raise NotImplementedError("Subclasses must implement create_job.")

    async def submit_job_handler(self) -> QuartReturn:
        """Handle job submission."""
        try:
            job_id, job_dir, job_config = await self.prepare_submit_job()

            # Create the job
            job = self.create_job(job_id, job_config)
            self.jobs[job_id] = job

            # Create an async task to handle the job processing in the background
            task = asyncio.create_task(job.generate(job_config))
            job.task = task

            # Wait a bit to catch immediate exceptions
            await asyncio.sleep(0.1)
            if task.done() and task.exception():
                ex = task.exception()
                return {
                    "status": "error",
                    "error": str(ex),
                }, self.get_http_status_from_exception(ex)

            return {
                "status": "success",
                "job_id": job_id,
            }
        except ValueError as value_err:
            logging.error(f"Value error: {value_err}")
            return jsonify({
                "status": "error",
                "error": str(value_err),
                "traceback": traceback.format_exc()
            }), HTTPStatus.BAD_REQUEST
        except Exception as ex:
            logging.error(f"Error: {ex}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({
                "status": "error",
                "error": str(ex),
                "traceback": traceback.format_exc()
            }), self.get_http_status_from_exception(ex)

    def register_templates(self) -> None:
        self.app.jinja_env.loader = ChoiceLoader([
            FileSystemLoader("templates"),
            FileSystemLoader("apps/templates"),
            FileSystemLoader(f"{self.app_name}/templates"),
            FileSystemLoader(f"apps/{self.app_name}/templates"),
            FileSystemLoader("../templates"),
        ])

    def parse_arguments(
        self,
        description: str = "StreamWise"
    ) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("--k8s_cluster", type=str, default=K8S_CLUSTER, help="Kubernetes cluster context name")
        parser.add_argument("--host", type=str, default=HOST, help="Host to bind the server to")
        parser.add_argument("--port", type=int, default=PORT, help="Port to bind the server to")
        return parser.parse_args()

    def get_http_status_from_exception(
        self,
        ex: Exception
    ) -> HTTPStatus:
        """Map exceptions to HTTP status codes."""
        if isinstance(ex, ValueError):
            return HTTPStatus.BAD_REQUEST
        return HTTPStatus.INTERNAL_SERVER_ERROR

    async def main(
        self,
        args: Any
    ) -> None:
        """Main entry point for StreamWise application."""
        logging.info(f"Starting {self.app_name} app on {args.host}:{args.port} with K8S cluster '{args.k8s_cluster}'")

        self.service_manager = LMMServiceManager(
            app_name=self.app_name,
            k8s_cluster=args.k8s_cluster
        )
        await self.service_manager.init_k8s_services()
        service_manager_task = asyncio.create_task(
            self.service_manager.start_updater())

        await self.service_manager.warmup_services()

        try:
            http_task = asyncio.create_task(self.run_httpserver(
                host=args.host,
                port=args.port
            ))
            await http_task
        except OSError as os_err:
            if os_err.errno == errno.EADDRINUSE:
                logging.error(f"{args.host}:{args.port} already in use.")
            else:
                logging.error(f"OS error: {os_err}")
        except Exception as ex:
            logging.error(f"Error: {ex}")
            logging.error(traceback.format_exc())
        finally:
            await self.service_manager.stop()
            if service_manager_task and not service_manager_task.done():
                service_manager_task.cancel()
                await service_manager_task

    async def run_httpserver(
        self,
        host: str = HOST,
        port: int = PORT
    ) -> None:
        """HTTP server runs in the main process."""
        config = Config()
        config.bind = [f"{host}:{port}"]

        # Display the access logs for debugging
        config.accesslog = "-"

        # Increase max request body size to 128 MB (default is 16 MB)
        config.limit_max_request_size = 128 * 1024 * 1024
        self.app.config["MAX_CONTENT_LENGTH"] = 128 * 1024 * 1024

        await serve(self.app, config)

    def register_filters(self):
        """Register template filters."""
        app = self.app
        app.template_filter("format_string")(lambda s: format_string(s))
        app.template_filter("format_duration")(lambda s: format_duration(s))
        app.template_filter("format_duration_short")(lambda s: format_duration_short(s))
        app.template_filter("parse_request_id")(parse_request_id)
        app.template_filter("get_content_type_emoji")(get_content_type_emoji)
        app.template_filter("get_file_type_emoji")(lambda f: get_file_type_emoji(get_file_type(f)))
        app.template_filter("json_pretty")(json_pretty_filter)
        app.template_filter("get_file_type")(get_file_type)
        app.template_filter("get_mime_type")(get_mime_type)
        app.template_filter("get_aspect_ratio")(get_aspect_ratio)
        app.template_filter("format_bytes")(format_bytes)

        app.template_filter("get_friendly_container_name")(get_friendly_container_name)
        app.template_filter("get_class_emoji")(get_class_emoji)

    async def prepare_submit_job(self) -> Tuple[str, str, Dict[str, Any]]:
        """Prepare for job submission."""
        request_json = await request.get_json()
        if not request_json:
            raise ValueError("No JSON body received")

        job_id = request_json.get("job_id") or get_job_id()
        logging.info(f"Received job_id {job_id}.")

        job_dir = f"{self.tmp_dir}/{job_id}"
        await aiofiles.os.makedirs(job_dir, exist_ok=True)

        async with aiofiles.open(f"{job_dir}/request.json", "w") as file:
            await file.write(json.dumps(request_json, indent=4))

        job_config = self.get_job_config_from_request(job_id, request_json)

        async with aiofiles.open(f"{job_dir}/config.json", "w") as file:
            await file.write(json.dumps(job_config, indent=4))

        if self.service_manager is None:
            raise ValueError("Service manager not initialized")

        return job_id, job_dir, job_config

    def get_job_config_from_request(
        self,
        job_id: str,
        request_json: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process and return the job configuration from the request JSON."""
        ret = request_json.copy()
        ret["job_id"] = job_id

        if "resolution" in ret:
            resolution_str = ret["resolution"]
            resolution_str = resolution_str.lower()
            aspect_ratio = ASPECT_RATIO
            width, height = RESOLUTIONS[aspect_ratio][resolution_str]
            ret["width"] = width
            ret["height"] = height

        return ret

    async def get_logs(
        self,
        path: Optional[str] = None,
        file_name: Optional[str] = None
    ) -> str:
        """Get the application logs."""
        if path is None:
            path = self.tmp_dir
        if file_name is None:
            file_name = self.log_file_name
        logs = ""
        if not await aiofiles.os.path.exists(f"{path}/{file_name}"):
            logging.warning(f"Log file {path}/{file_name} does not exist.")
            return logs
        async with aiofiles.open(f"{path}/{file_name}", "r") as file:
            logs = await file.read()
        return logs

    async def get_job_config(self, job_id: str) -> Dict[str, Any]:
        """Get the job configuration from the job directory."""
        job_dir = f"{self.tmp_dir}/{job_id}"
        config_file = f"{job_dir}/config.json"
        if not await aiofiles.os.path.exists(config_file):
            logging.warning(f"Job config file {config_file} does not exist.")
            return {}
        try:
            async with aiofiles.open(config_file, "r") as file:
                content = await file.read()
                config_json = json.loads(content)
                return config_json
        except Exception as ex:
            logging.error(f"Error reading job config: {ex}")
            return {}

    async def get_job_status(
        self,
        job_id: str
    ) -> Dict[str, JobStatus]:
        """Get the status of a job asynchronously."""
        job_dir = f"{self.tmp_dir}/{job_id}"
        status_file = f"{job_dir}/status.txt"
        if not await aiofiles.os.path.exists(status_file):
            return {"status": JobStatus.UNKNOWN.name}

        # Get last modified time of the status file
        last_modified_time = await aiofiles.os.path.getmtime(status_file)
        async with aiofiles.open(status_file, "r") as file:
            content = await file.read()
            contet_strip = content.strip()
            if not contet_strip:
                return {"status": JobStatus.UNKNOWN.name}
            line = contet_strip.splitlines()[-1].strip()
            line_split = line.split(",")
            if len(line_split) == 2:
                timestamp_str, status_str = line_split
            else:
                status_str = line_split[0]
            status_str = status_str.strip()
            if not status_str.isdigit():
                return {"status": JobStatus.UNKNOWN.name}
            status_val = int(status_str)
            status = JobStatus(status_val)
            if not is_status_terminal(status) and is_status_expired(last_modified_time):
                return {"status": JobStatus.EXPIRED.name}
            return {"status": status.name}
        return {"status": JobStatus.UNKNOWN.name}

    async def get_services(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get the list of available services."""
        ret: Dict[str, List[Dict[str, Any]]] = {"services": []}
        if self.service_manager is None:
            logging.error("Service manager not initialized")
            return ret
        for service_name, service in self.service_manager.services.items():
            for container in service.containers:
                container_status = {
                    "service_name": service_name,
                    "pod_name": service_name,
                    "container_name": container.name,
                    "ip": container.ip,
                    "port": container.port,
                    "status": container.status,
                    "busy": container.busy,
                }
                ret["services"].append(container_status)
        return ret

    async def get_jobs(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get the list of jobs."""
        ret: Dict[str, List[Dict[str, Any]]] = {"jobs": []}

        if not await aiofiles.os.path.exists(self.tmp_dir):
            logging.warning(f"{self.tmp_dir} directory does not exist")
            return ret

        file_names = await aiofiles.os.listdir(self.tmp_dir)
        job_ids = []
        for file_name in file_names:
            if is_job_id(file_name):
                job_ids.append(file_name)

        async def get_job_details(job_id: str) -> Optional[Dict[str, Any]]:
            status = await self.get_job_status(job_id)
            status_str = status.get("status", "unknown")
            job_details = {
                "job_id": job_id,
                "status": status_str,
            }

            job_config = await self.get_job_config(job_id)
            if job_config:
                job_details.update(job_config)

            file_path = f"{self.tmp_dir}/{job_id}/{job_id}.mp4"
            if await aiofiles.os.path.exists(file_path):
                loop = asyncio.get_running_loop()
                file_video_info = await loop.run_in_executor(None, get_video_file_info, file_path)
                job_details.update(file_video_info)
            return job_details

        jobs = await asyncio.gather(*(get_job_details(job_id) for job_id in job_ids))
        ret["jobs"] = [job for job in jobs if job is not None]
        return ret

    def register_routes(self):
        """Register HTTP routes."""
        route = self.app.route

        @route("/", methods=["GET"])
        async def index() -> str:
            """Render the index HTML page."""
            services = await self.get_services()
            jobs = await self.get_jobs()
            logs = {
                "app": await self.get_logs(),
                "service_manager": await self.get_logs(file_name="service_manager.log"),
            }
            return await render_template(
                "index.html",
                services=services["services"],
                jobs=jobs["jobs"],
                logs=logs,
            )

        @route("/health", methods=["GET"])
        async def health() -> QuartReturn:
            """Get health status."""
            health: Dict[str, Any] = {
                "status": "ok",
                "host": self.args.host if self.args else None,
                "port": self.args.port if self.args else None,
                "k8s_cluster": self.args.k8s_cluster if self.args else None,
                "jobs": {},
                "services": {}
            }
            if self.jobs:
                for job_id, job in self.jobs.items():
                    job_status = await job.get_status()
                    requests = job.get_requests()
                    health["jobs"][job_id] = {
                        "job_id": job.job_id,
                        "status": job_status.name,
                        "num_requests": len(requests),
                    }
            if self.service_manager:
                for service_name, service in self.service_manager.services.items():
                    health["services"][service_name] = {
                        "name": service.name,
                        "num_containers": len(service.containers),
                    }
            return jsonify(health), HTTPStatus.OK

        @route("/api/services", methods=["GET"])
        async def api_get_services() -> Dict[str, List[Dict[str, Any]]]:
            """Get the list of available services."""
            return await self.get_services()

        @route("/files", methods=["GET"])
        async def list_files() -> QuartReturn:
            return await self.file_manager.list_files()

        @route("/file/<file_name>", methods=["GET"])
        async def download_file(file_name: str) -> QuartReturn:
            return await self.file_manager.download_file(file_name)

        @route("/file_stream/<job_id>/<file_name>", methods=["GET"])
        async def file_stream(job_id: str, file_name: str) -> QuartReturn:
            return await self.file_manager.stream(job_id, file_name)

        @route("/file_view/<job_id>/<file_name>", methods=["GET"])
        async def file_view(job_id: str, file_name: str) -> QuartReturn:
            return await self.file_manager.view(job_id, file_name)

        @route("/audio_waveform/<job_id>/<file_name>", methods=["GET"])
        async def get_audio_waveform(job_id: str, file_name: str) -> QuartReturn:
            """Generate and return a waveform PNG image for a WAV audio file."""
            if not job_id or not file_name:
                return jsonify({"error": "Job id and file name are required"}), HTTPStatus.BAD_REQUEST
            if not file_name or not file_name.endswith((".wav")):
                return jsonify({"error": f"Invalid file name: {file_name}"}), HTTPStatus.BAD_REQUEST

            # Download the WAV file into a local temp file
            job_dir = f"{self.tmp_dir}/{job_id}"
            wav_path = f"{job_dir}/{file_name}"
            waveform_png_path = generate_waveform_plt(wav_path)

            return await send_file(
                waveform_png_path,
                as_attachment=True,
                attachment_filename=f"waveform_{file_name}.png",
                mimetype="image/png")

        @route("/job", methods=["GET"])
        async def submit_job() -> str:
            """Render the submit job HTML page."""
            return await render_template("submit_job.html")

        @route("/api/jobs", methods=["GET"])
        async def api_get_jobs() -> Dict[str, List[Dict[str, Any]]]:
            """Get the list of jobs."""
            return await self.get_jobs()

        @route("/api/job/<job_id>/status", methods=["GET"])
        async def api_get_job_status(job_id: str) -> Dict[str, JobStatus]:
            """Get the status of a job."""
            return await self.get_job_status(job_id)

        @route("/api/job", methods=["POST"])
        async def api_submit_job() -> QuartReturn:
            """Submit a new job."""
            return await self.submit_job_handler()

        @route("/api/job/<job_id>/<json_file_name>", methods=["GET"])
        async def api_get_job_json_file(job_id: str, json_file_name: str) -> QuartReturn:
            """Get a JSON file from the job directory."""
            job_dir = f"{self.tmp_dir}/{job_id}"
            request_file = f"{job_dir}/{json_file_name}.json"
            if not await aiofiles.os.path.exists(request_file):
                return {
                    "status": "error",
                    "error": f"Job {job_id} JSON file {json_file_name}.json not found"
                }
            try:
                async with aiofiles.open(request_file, "r") as file:
                    content = await file.read()
                    request_json = json.loads(content)
                    return request_json
            except Exception as ex:
                logging.error(f"Error reading job config: {ex}")
                return {
                    "status": "error",
                    "error": str(ex),
                    "traceback": traceback.format_exc()
                }

        @route("/api/job/<job_id>/config", methods=["GET"])
        async def api_get_job_config(job_id: str) -> QuartReturn:
            """Get the job configuration."""
            return await api_get_job_json_file(job_id, "config")

        @route("/api/job/<job_id>/request", methods=["GET"])
        async def api_get_job_request(job_id: str) -> QuartReturn:
            """Get the job request JSON."""
            return await api_get_job_json_file(job_id, "request")

        @route("/job/<job_id>", methods=["GET"])
        async def job_status(job_id: str) -> str:
            """Render the job status HTML page for a specific job ID."""
            files = await self.file_manager.list_job_files(job_id)
            job_request = await api_get_job_request(job_id)
            job_config = await api_get_job_config(job_id)

            status = await self.get_job_status(job_id)
            status = status.get("status", "unknown")
            status_history = await get_job_status_history(job_id)
            times = status_history_to_times(status_history)

            job = self.jobs.get(job_id, None)
            requests = {}
            if job is not None:
                requests = job.get_requests().copy()
                # Convert deadline from seconds to milliseconds
                for req in requests.values():
                    if "deadline" in req:
                        req["deadline"] = req["deadline"] * 1000  # ms
            return await render_template(
                "job.html",
                job_id=job_id,
                job=job,
                job_request=job_request,
                job_config=job_config,
                requests=requests,
                status=status,
                times=times,
                files=files)

        @route("/api/job/<job_id>/status/history", methods=["GET"])
        async def get_job_status_history(job_id: str) -> Dict[float, JobStatus]:
            """Get the status of a job asynchronously."""
            job_dir = f"{self.tmp_dir}/{job_id}"
            status_file = f"{job_dir}/status.txt"
            ret = {}
            if not await aiofiles.os.path.exists(status_file):
                return ret

            async with aiofiles.open(status_file, "r") as file:
                content = await file.read()
                contet_strip = content.strip()

                for line in contet_strip.splitlines():
                    line_split = line.split(",")
                    if len(line_split) == 2:
                        timestamp_str, status_str = line_split
                    else:
                        timestamp_str = "-1.0"
                        status_str = line_split[0]
                    status_str = status_str.strip()
                    if not status_str.isdigit():
                        continue
                    status_val = int(status_str)
                    status = JobStatus(status_val)
                    timestamp_float = float(timestamp_str)
                    ret[timestamp_float] = status.name
            return ret

        @route("/api/job/<job_id>/requests", methods=["GET"])
        async def api_get_job_requests(job_id: str) -> Dict[str, Any]:
            job = self.jobs.get(job_id, None)
            if not job:
                return {}
            requests = job.get_requests()
            ret = {}
            for request_id, req in requests.items():
                ret[request_id] = req.json()
            return ret


class StreamWiseAppFileManager:
    """Manage files for the StreamWise application."""

    def __init__(self, tmp_dir: str):
        self.tmp_dir = tmp_dir

    async def list_job_files(
        self,
        job_id: str
    ) -> List[Dict[str, Any]]:
        """Asynchronously list files in the job directory."""
        job_dir = f"{self.tmp_dir}/{job_id}"
        if not await aiofiles.os.path.exists(job_dir):
            return []
        files = []
        file_names = await aiofiles.os.listdir(job_dir)
        for file_name in file_names:
            file_path = os.path.join(job_dir, file_name)
            file_date = await aiofiles.os.path.getmtime(file_path)
            file_size = await aiofiles.os.path.getsize(file_path)
            # file_type = get_file_type(file_name)
            mime_type, _ = mimetypes.guess_type(file_name)
            files.append({
                "name": file_name,
                "size": file_size,
                "date": datetime.fromtimestamp(file_date),
                "mimetype": mime_type,
            })
        return files

    async def list_files(self) -> QuartReturn:
        """List files in the TMP_DIR directory."""
        try:
            files = await quart_utils.list_files(self.tmp_dir)
            return jsonify({
                "files": files
            })
        except Exception as ex:
            logging.error(f"Error listing files in {self.tmp_dir}: {ex}.")
            return jsonify({"error": str(ex)}), HTTPStatus.INTERNAL_SERVER_ERROR

    async def download_file(self, file_name: str) -> QuartReturn:
        """Download a file."""
        try:
            filepath = f"{self.tmp_dir}/{file_name}"
            if not await aiofiles.os.path.exists(filepath):
                return jsonify({"error": f"File '{filepath}' not found"}), HTTPStatus.NOT_FOUND

            if await aiofiles.os.path.isdir(filepath):
                files = await aiofiles.os.listdir(filepath)
                return jsonify({
                    "files": files
                })

            mimetype = get_mime_type(file_name)
            return await send_from_directory(
                self.tmp_dir,
                file_name,
                mimetype=mimetype,
                as_attachment=True)
        except Exception as ex:
            return jsonify({"error": str(ex)}), HTTPStatus.INTERNAL_SERVER_ERROR

    async def stream(
        self,
        job_id: str,
        file_name: str
    ) -> QuartReturn:
        """Stream a file."""
        try:
            file_path = f"{self.tmp_dir}/{job_id}/{file_name}"
            if not await aiofiles.os.path.exists(file_path):
                return jsonify({"error": "File not found"}), HTTPStatus.NOT_FOUND

            async with aiofiles.open(file_path, mode="rb") as file:
                data = await file.read()

            mimetype = get_mime_type(file_name)
            return await send_file(
                BytesIO(data),
                mimetype=mimetype,
                attachment_filename=file_name)
        except Exception as ex:
            return jsonify({"error": str(ex)}), HTTPStatus.INTERNAL_SERVER_ERROR

    async def view(
        self,
        job_id: str,
        file_name: str
    ) -> QuartReturn:
        """View a file in the browser."""
        if not file_name:
            return jsonify({"error": "File name required"}), HTTPStatus.BAD_REQUEST

        file_path = f"{self.tmp_dir}/{job_id}/{file_name}"
        if not await aiofiles.os.path.exists(file_path):
            return await render_template(
                "file_view.html",
                job_id=job_id,
                file_name=file_name,
                content=None,
                content_type=None,
                content_length=None,
                error="File not found")

        try:
            async with aiofiles.open(file_path, "rb") as file:
                content_bytes = await file.read()
            content_length = len(content_bytes)

            file_type = get_file_type(file_name)
            mimetype = get_mime_type(file_name)
            file_info = {
                "name": file_name,
                "size": await aiofiles.os.path.getsize(file_path),
                "date": await aiofiles.os.path.getmtime(file_path),
                "type": file_type,
                "mimetype": mimetype
            }
            if file_type == "audio":
                file_audio_info = get_audio_file_info(file_path)
                file_info.update(file_audio_info)
            elif file_type == "video":
                file_video_info = get_video_file_info(file_path)
                video_info = file_video_info["video"]
                file_info.update(video_info)
                # audio_info = file_video_info["audio"]
                # file_info_ret.update(audio)
            elif file_type == "image":
                file_image_info = get_image_file_info(file_path)
                file_info.update(file_image_info)
            elif file_type == "text":
                file_text_info = get_text_file_info(file_path)
                file_info.update(file_text_info)
            elif file_type == "tensor":
                file_tensor_info = get_tensor_file_info(file_path)
                file_info.update(file_tensor_info)

            content_type = mimetype
            if content_type.startswith("text/") or content_type in ("application/json", "application/x-ndjson"):
                content_str = content_bytes.decode("utf-8", errors="replace")
            else:
                content_str = content_bytes

            return await render_template(
                "file_view.html",
                job_id=job_id,
                file_name=file_name,
                content=content_str,
                content_type=content_type,
                content_length=content_length,
                file_info=file_info,
                error=None)
        except Exception as ex:
            return await render_template(
                "file_view.html",
                job_id=job_id,
                file_name=file_name,
                content=None,
                content_type=None,
                content_length=None,
                error=str(ex))


def run_app(
    app_cls: Type[StreamWiseApp],
    *,
    tmp_dir: str,
    log_files: List[str],
    app_name: str,
) -> None:
    setup_logging(
        path=tmp_dir,
        file_name=log_files,
    )

    app: StreamWiseApp = app_cls()
    args = app.parse_arguments(app_name)
    asyncio.run(app.main(args))
