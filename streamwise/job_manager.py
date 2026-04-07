"""
Submit jobs to services in the cluster.
"""

import sys
import logging

from aiohttp import ClientTimeout
from asyncio import TimeoutError

from http import HTTPStatus

from quart import request
from quart import jsonify
from quart import Response

import http_session_manager

sys.path.append("..")
from quart_utils import QuartReturn


# HTTP clients
JSON_HEADER = {
    "Content-Type": "application/json"
}
GENERATION_TIMEOUT = ClientTimeout(
    total=60.0,
    connect=5.0)


async def submit_job(
    service_name: str,
    container_ip: str,
    container_port: int
) -> QuartReturn:
    """API interface to submit a job to the specified service."""
    if not service_name or not container_ip or not container_port:
        return jsonify({"error": "Service name, container IP and port are required"}), HTTPStatus.BAD_REQUEST
    try:
        payload_json = await request.get_json()
        if not payload_json:
            return jsonify({"error": "No job data provided"}), HTTPStatus.BAD_REQUEST

        url = f"{http_session_manager.SERVICE_SCHEME}://{container_ip}:{container_port}/{service_name}"
        session = await http_session_manager.get_global_session()
        async with session.post(url, json=payload_json, headers=JSON_HEADER, timeout=GENERATION_TIMEOUT) as response:
            content_type = response.headers.get("Content-Type", "application/octet-stream")
            data = await response.read()
            headers = {}
            content_disposition = response.headers.get("Content-Disposition")
            if content_disposition:
                headers["Content-Disposition"] = content_disposition
            return Response(
                response=data,
                status=response.status,
                headers=headers,
                content_type=content_type)
    except TimeoutError:
        logging.error(f"Timeout submitting job to {service_name} at {container_ip}:{container_port}.")
        return jsonify({"error": "Request timed out"}), HTTPStatus.GATEWAY_TIMEOUT
    except Exception as ex:
        logging.error(f"Error submitting job to {service_name} at {container_ip}:{container_port}: {ex}")
        return jsonify({"error": str(ex)}), HTTPStatus.INTERNAL_SERVER_ERROR
