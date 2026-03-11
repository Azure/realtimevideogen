"""
File manager functions for the Cluster Manager.
"""

import sys
import logging
import aiofiles
import aiofiles.os

from aiohttp import ClientTimeout
from aiohttp import ClientError

from typing import Optional
from typing import Dict
from typing import Any
from typing import Union

from io import BytesIO

from quart import jsonify
from quart import send_file
from quart import send_from_directory
from quart import render_template

from http import HTTPStatus

from http_session_manager import get_global_session

sys.path.append("..")
import quart_utils
from quart_utils import QuartReturn
from quart_utils import get_mime_type

from tts_utils import generate_waveform_plt


# HTTP clients
CLIENT_TIMEOUT = ClientTimeout(total=5.0, connect=1.0)


async def get_audio_waveform(container_ip: str, container_port: int, file_name: str) -> QuartReturn:
    """Generate and return a waveform PNG image for a WAV audio file."""
    if not container_ip or not container_port or not file_name:
        return jsonify({"error": "Container IP, port and file name are required"}), HTTPStatus.BAD_REQUEST
    if not file_name or not file_name.endswith((".wav")):
        return jsonify({"error": f"Invalid file name: {file_name}"}), HTTPStatus.BAD_REQUEST

    # Download the WAV file into a local temp file
    temp_wav_path = f"/tmp/waveform_{file_name}"
    url = f"http://{container_ip}:{container_port}/file/{file_name}"
    try:
        session = await get_global_session()
        async with session.get(url, timeout=CLIENT_TIMEOUT) as response:
            if response.status == HTTPStatus.OK:
                content_type = "application/octet-stream"
                if "Content-Type" in response.headers:
                    content_type = response.headers["Content-Type"]
                if content_type not in ("audio/wav", "audio/x-wav"):
                    return jsonify({
                        "error": f"File {file_name} is not a WAV audio file: {content_type}"
                    }), HTTPStatus.BAD_REQUEST
                audio_binary = await response.read()
                async with aiofiles.open(temp_wav_path, mode="wb") as wav_file:
                    await wav_file.write(audio_binary)
    except Exception as ex:
        logging.error(f"Error downloading file from {url}: {ex}.")
        return jsonify({"error": str(ex)}), HTTPStatus.INTERNAL_SERVER_ERROR

    waveform_png_path = generate_waveform_plt(temp_wav_path)

    return await send_file(
        waveform_png_path,
        as_attachment=True,
        attachment_filename=f"waveform_{file_name}.png",
        mimetype="image/png")


async def get_video_info(container_ip: str, container_port: int, file_name: str) -> QuartReturn:
    """Get video file information from a container."""
    if not container_ip or not container_port or not file_name:
        return jsonify({"error": "Container IP, port and file name are required"}), HTTPStatus.BAD_REQUEST
    if not file_name or not file_name.endswith((".mp4", ".avi", ".mkv", ".webm")):
        return jsonify({"error": f"Invalid file name: {file_name}"}), HTTPStatus.BAD_REQUEST

    url = f"http://{container_ip}:{container_port}/file/{file_name}"
    try:
        session = await get_global_session()
        async with session.get(url, timeout=CLIENT_TIMEOUT) as response:
            if response.status != HTTPStatus.OK:
                return jsonify({"error": "Failed to get video info"}), HTTPStatus.INTERNAL_SERVER_ERROR

            content_type = response.headers.get("Content-Type", "video/mp4")
            content = await response.read()
            content_length = len(content)
            return jsonify({
                "file_name": file_name,
                "content_type": content_type,
                "content_length": content_length
            }), HTTPStatus.OK
    except Exception as ex:
        logging.error(f"Error getting video info from {url}: {ex}.")
        return jsonify({"error": str(ex)}), HTTPStatus.INTERNAL_SERVER_ERROR


async def list_files(path: str) -> QuartReturn:
    """List files in a directory."""
    try:
        files = await quart_utils.list_files(path)
        return jsonify({"files": files})
    except Exception as ex:
        logging.error(f"Error listing files in {path}: {ex}.")
        return jsonify({"error": str(ex)}), HTTPStatus.INTERNAL_SERVER_ERROR


async def download_local_file(path: str, file_name: str) -> QuartReturn:
    """Download a file."""
    filepath = f"{path}/{file_name}"
    if not await aiofiles.os.path.exists(filepath):
        return jsonify({"error": "File not found"}), HTTPStatus.NOT_FOUND

    if await aiofiles.os.path.isdir(filepath):
        files = await aiofiles.os.listdir(filepath)
        return jsonify({
            "files": files
        })

    mimetype = get_mime_type(file_name)
    return await send_from_directory(
        path,
        file_name,
        mimetype=mimetype,
        as_attachment=True)


async def download_service_file(
    container_ip: str,
    container_port: int,
    file_name: str
) -> QuartReturn:
    """Download a file from a container."""
    if not container_ip or not container_port or not file_name:
        return jsonify({"error": "Container IP, port and file name are required"}), HTTPStatus.BAD_REQUEST

    url = f"http://{container_ip}:{container_port}/file/{file_name}"
    try:
        session = await get_global_session()
        timeout = ClientTimeout(total=1.0, connect=1.0)
        async with session.get(url, timeout=timeout) as response:
            if response.status == HTTPStatus.OK:
                content_type = "application/octet-stream"
                if "Content-Type" in response.headers:
                    content_type = response.headers["Content-Type"]
                data = await response.read()
                return await send_file(
                    BytesIO(data),
                    as_attachment=True,
                    attachment_filename=file_name,
                    mimetype=content_type)
        return jsonify({"error": "Failed to download"}), HTTPStatus.INTERNAL_SERVER_ERROR
    except ClientError as client_err:
        logging.error(f"Client error downloading file from {url}: {client_err}.")
        return jsonify({"error": f"Client error: {client_err}"}), HTTPStatus.SERVICE_UNAVAILABLE
    except Exception as ex:
        logging.error(f"Error downloading file from {url} [{type(ex)}]: {ex}.")
        return jsonify({"error": str(ex)}), HTTPStatus.INTERNAL_SERVER_ERROR


async def get_file_info(
    container_ip: str,
    container_port: int,
    file_name: str
) -> Optional[Dict[str, Any]]:
    url = f"http://{container_ip}:{container_port}/file_info/{file_name}"
    try:
        session = await get_global_session()
        async with session.get(url, timeout=CLIENT_TIMEOUT) as response:
            content_type = response.headers.get("Content-Type")
            if response.status == HTTPStatus.OK and content_type == "application/json":
                return await response.json()
    except ClientError as client_err:
        logging.error(f"Client error getting file info from {url}: {client_err}.")
    except Exception as ex:
        logging.error(f"Error getting file info from {url} [{type(ex)}]: {ex}.")
    return None


async def file_view(
    service_name: str,
    container_ip: str,
    container_port: int,
    file_name: str
) -> QuartReturn:
    """
    View the contents of a file from a container.
    Returns HTML rendered template with file content.
    """
    if not container_ip or not container_port or not file_name:
        return jsonify({"error": "Container IP, port and file name are required"}), HTTPStatus.BAD_REQUEST

    content: Optional[Union[str, bytes]] = None
    content_type: Optional[str] = None
    content_length: Optional[int] = None
    file_info = None
    error: Optional[str] = None
    try:
        url_data = f"http://{container_ip}:{container_port}/file/{file_name}"
        session = await get_global_session()
        timeout = ClientTimeout(total=1.0, connect=1.0)
        async with session.get(url_data, timeout=timeout) as file_response:
            if file_response.status != HTTPStatus.OK:
                return jsonify({
                    "error": f"Failed to get file data: {file_response.status}"
                }), HTTPStatus.INTERNAL_SERVER_ERROR
            content_type = file_response.headers.get("Content-Type", "text/plain")
            raw_content = await file_response.read()
            content_length = len(raw_content)
            if content_type.startswith("text/") or content_type in ("application/json", "application/x-ndjson"):
                text_content: str = raw_content.decode("utf-8", errors="replace")
                content = text_content
            else:
                binary_content: bytes = raw_content
                content = binary_content

            file_info = await get_file_info(container_ip, container_port, file_name)
    except ClientError as client_err:
        error = f"Client error getting file from {url_data}: {client_err}"
    except Exception as ex:
        error = str(ex)

    return await render_template(
        "file_view.html",
        service_name=service_name,
        file_name=file_name,
        container_ip=container_ip,
        container_port=container_port,
        content=content,
        content_type=content_type,
        content_length=content_length,
        file_info=file_info,
        error=error)


async def file_stream(
    container_ip: str,
    container_port: int,
    file_name: str
) -> QuartReturn:
    """Stream a file from a container."""
    url = f"http://{container_ip}:{container_port}/file/{file_name}"
    try:
        session = await get_global_session()
        async with session.get(url, timeout=CLIENT_TIMEOUT) as response:
            if response.status == HTTPStatus.OK:
                content_type = response.headers.get("Content-Type", "application/octet-stream")
                data = await response.read()
                return await send_file(
                    BytesIO(data),
                    mimetype=content_type,
                    attachment_filename=file_name)
            return jsonify({
                "error": f"Failed to stream file {file_name} from {url}: {response.status}"
            }), response.status
    except ClientError as client_err:
        return jsonify({
            "error": f"Client error while trying to reach {url}: {client_err}"
        }), HTTPStatus.SERVICE_UNAVAILABLE
    except TimeoutError as timeout_err:
        return jsonify({
            "error": f"Timeout error while trying to reach {url}: {timeout_err}"
        }), HTTPStatus.GATEWAY_TIMEOUT
    except Exception as ex:
        return jsonify({"error": str(ex)}), HTTPStatus.INTERNAL_SERVER_ERROR
