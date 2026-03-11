"""
Utils for Quart/Flask applications, including filters and helper functions.
"""
import os
import re
import logging
import json
import aiofiles
import aiofiles.os
import mimetypes

from http import HTTPStatus

from datetime import datetime
from datetime import timedelta

from typing import Union
from typing import Tuple
from typing import Any
from typing import List
from typing import Dict
from typing import Optional

from quart import Response


QuartReturn = Union[
    str,
    Response,
    Tuple[Response, int],
    Dict[str, Any],
    Tuple[str, HTTPStatus],
]


def json_pretty_filter(
    value: str,
    max_len: int = 128
) -> str:
    def truncate(obj: Any) -> Union[str, List, Dict]:
        if isinstance(obj, str) and len(obj) > max_len:
            return obj[:max_len] + f"... [truncated, {format_bytes(len(obj))}]"
        if isinstance(obj, list):
            return [truncate(item) for item in obj]
        if isinstance(obj, dict):
            return {k: truncate(v) for k, v in obj.items()}
        return obj

    try:
        obj = json.loads(value) if isinstance(value, str) else value
        truncated = truncate(obj)
        return json.dumps(truncated, indent=2, ensure_ascii=False)
    except Exception:
        return value


def format_datetime(value: int) -> str:
    try:
        dt = datetime.fromtimestamp(value)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, OSError, TypeError):
        return "Invalid date"


def format_bytes(memory: int) -> str:
    if memory == 0:
        return '<span class="text-muted">-</span>'
    if memory < 1024:
        return f"{memory} B"
    if memory < 1024 ** 2:
        return f"{memory / 1024:.1f} KiB"
    if memory < 1024 ** 3:
        return f"{memory / 1024 / 1024:.1f} MiB"
    if memory < 1024 ** 4:
        return f"{memory / 1024 / 1024 / 1024:.1f} GiB"
    return f"{memory / 1024 / 1024 / 1024 / 1024:.1f} TiB"


def format_string(in_string: Optional[str]) -> Optional[str]:
    """Format a string to be more readable."""
    if not isinstance(in_string, str):
        return in_string
    ret = in_string.replace("_", " ").title()

    # Make some words uppercase
    UPPER_WORDS = ["gpu", "cpu", "api", "sm", "http"]
    # Fix the capitalization of some words
    REPLACE_WORDS = {
        "gib": "GiB",
        "mib": "MiB",
        "kib": "KiB",
        "mbps": "Mbps",
        "gbps": "Gbps",
        "tbps": "Tbps",
        "vcpu": "vCPU",
    }

    words = ret.split(" ")
    for i in range(len(words)):
        word_lower = words[i].lower()
        if word_lower in UPPER_WORDS:
            words[i] = word_lower.upper()
        elif word_lower in REPLACE_WORDS:
            words[i] = REPLACE_WORDS[word_lower]
    ret = " ".join(words)

    return ret


def format_duration(input_date: Optional[Union[timedelta, float]]) -> str:
    """Format a duration (timedelta) into a human-readable string."""
    if not input_date:
        return "0"
    if isinstance(input_date, float):
        total_seconds = int(input_date)
    elif isinstance(input_date, timedelta):
        total_seconds = int(input_date.total_seconds())
    else:
        raise TypeError("input_date must be a timedelta or float")
    if total_seconds < 0:
        return "?"

    hours, remainder = divmod(total_seconds, 60 * 60)
    minutes, seconds = divmod(remainder, 60)

    if hours > 1:
        return f"{hours} hours {minutes} minutes {seconds} seconds"
    if hours > 0:
        return f"{hours} hour {minutes} minutes {seconds} seconds"
    if minutes > 1:
        return f"{minutes} minutes {seconds} seconds"
    if minutes > 0:
        return f"{minutes} minute {seconds} seconds"
    return f"{seconds} seconds"


def format_duration_short(input_date: Optional[Union[timedelta, float]]) -> str:
    """Format a duration (timedelta) into a human-readable string."""
    if not input_date:
        return "0"
    if isinstance(input_date, float):
        total_seconds = int(input_date)
    elif isinstance(input_date, timedelta):
        total_seconds = int(input_date.total_seconds())
    else:
        raise TypeError("input_date must be a timedelta or float")
    hours, remainder = divmod(total_seconds, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_url(url: Optional[str]) -> Optional[str]:
    """Format a URL to remove the scheme (http/https)."""
    if not url:
        return url
    url = url.strip()
    if url.startswith("http://"):
        return url[7:]
    if url.startswith("https://"):
        return url[8:]
    return url


def format_gpu_model(gpu_model: Optional[str]) -> Optional[str]:
    """Format GPU model names to be more user-friendly."""
    if not gpu_model or gpu_model == "N/A":
        return gpu_model
    if not isinstance(gpu_model, str):
        return gpu_model
    gpu_model = gpu_model.strip()
    if gpu_model == "NVIDIA A100-SXM4-80GB" or gpu_model == "NVIDIA-A100-SXM4-80GB":
        return "A100 80GB"
    if gpu_model == "NVIDIA-A100-80GB-PCIe" or gpu_model == "NVIDIA A100 80GB PCIe":
        return "A100 80GB"
    if gpu_model == "Standard_NC96ads_A100_v4":
        return "A100 80GB"
    if gpu_model == "Standard_ND96isrf_H100_v5":
        return "H100"
    if gpu_model == "NVIDIA-H200" or gpu_model == "NVIDIA H200":
        return "H200"
    if gpu_model == "NVIDIA-H100" or gpu_model == "NVIDIA H100":
        return "H100"
    if gpu_model == "NVIDIA-H100-NVL" or gpu_model == "NVIDIA H100 NVL":
        return "H100 NVL"
    if gpu_model == "NVIDIA-H100-80GB-HBM3" or gpu_model == "NVIDIA H100 80GB HBM3":
        return "H100"
    if gpu_model == "Tesla-V100-PCIE-16GB" or gpu_model == "Tesla V100-PCIE-16GB":
        return "V100 16GB"
    if gpu_model == "Tesla-V100-SXM2-32GB" or gpu_model == "Tesla V100-SXM2-32GB":
        return "V100 32GB"
    return gpu_model


def get_aspect_ratio(ratio: float) -> str:
    if abs(ratio - 1) < 0.01:
        return "1:1"
    if abs(ratio - (16. / 9.0)) < 0.01:
        return "16:9"
    if abs(ratio - (16. / 10.0)) < 0.01:
        return "16:10"
    if abs(ratio - (4. / 3.0)) < 0.01:
        return "4:3"
    if abs(ratio - (5. / 4.0)) < 0.01:
        return "5:4"
    if abs(ratio - (3. / 2.0)) < 0.01:
        return "3:2"
    if abs(ratio - (2. / 1.0)) < 0.01:
        return "2:1"
    return f"{ratio:.2f}:1"


async def get_service_json_filename() -> Optional[str]:
    if await aiofiles.os.path.exists("../services.json"):
        return "../services.json"
    if await aiofiles.os.path.exists("services.json"):
        return "services.json"
    logging.warning("services.json file not found")
    return None


def get_k8s_service_emoji(container_name: Optional[str]) -> str:
    if container_name is None:
        return ""
    if container_name.startswith("nvidia-device-plugin-"):
        return "⚙️"
    if container_name.startswith("node-driver-"):
        return "🧩"
    if container_name.startswith("azure"):
        return "☁️"
    if container_name.startswith("metrics-server"):
        return "📊"
    if container_name.startswith("liveness-"):
        return "💓"
    if container_name.startswith("kube-proxy"):
        return "🔀"
    if container_name.startswith("gatekeeper"):
        return "🛡️"
    if container_name.startswith("cns-container"):
        return "🧩"
    if container_name.startswith("coredns"):
        return "🌐"
    if container_name.startswith("autoscaler"):
        return "📈"
    if container_name.startswith("cloud-node-manager"):
        return "☁️"
    if "security" in container_name:
        return "🛡️"
    if "azsec" in container_name:
        return "🛡️"
    if "konnect" in container_name:
        return "🌐"
    if "log" in container_name:
        return "📄"
    if "auoms" in container_name:
        return "📄"
    if "mdsdmgr" in container_name:
        return "📊"
    return f"<span class='text-muted' title='{container_name}'>❓</span>"


async def get_class_emoji(container_name: Optional[str]) -> str:
    if container_name is None:
        return ""
    services_file_name = await get_service_json_filename()
    if services_file_name is None:
        return ""
    async with aiofiles.open(services_file_name, mode="r") as file:
        data_str = await file.read()
        data_json = json.loads(data_str)

    if container_name not in data_json:
        return get_k8s_service_emoji(container_name)

    data_service_json = data_json[container_name]
    if "class" not in data_service_json:
        return f"<span class='text-muted' title='{container_name}'>❓</span>"
    service_class = data_service_json["class"]

    if service_class == "text2audio" or service_class == "text2speech":
        return "📄→🔉"
    if service_class == "video2audio":
        return "🎬→🔉"
    if service_class == "text2image":
        return "📄→🖼️"
    if service_class == "text2video":
        return "📄→🎬"
    if service_class == "image2video":
        return "📄→🎬"
    if service_class == "textimageaudio2video":
        return "📄🖼️🔉→🎬"
    if service_class == "image2image":
        return "🖼️→🖼️"
    if service_class == "text2text":
        return "📄→📄"
    if service_class == "textimage2video":
        return "📄🖼️→🎬"
    if service_class == "manager":
        return "⚒️"
    if service_class == "doc2video":
        return "📄→🎬"
    if service_class == "video2video":
        return "🎬→🎬"
    if service_class == "audio2audio":
        return "🔉→🔉"
    if service_class == "audio2text":
        return "🔉→📄"

    # VAE
    if service_class == "latent2image":
        return "🔢→🖼️"
    if service_class == "latent2video":
        return "🔢→🎬"
    if service_class == "image2latent":
        return "🖼️→🔢"
    if service_class == "video2latent":
        return "🎬→🔢"

    # Unknown class
    return f"<span class='text-muted' title='{service_class}'>❓</span>"


def get_file_type_emoji(file_type: Optional[str]) -> str:
    if file_type is None:
        return "❓"
    if file_type == "directory":
        return "📁"
    if file_type == "image":
        return "🖼️"
    if file_type == "audio":
        return "🎵"
    if file_type == "video":
        return "🎥"
    if file_type == "archive":
        return "📦"
    if file_type in ("file", "text", "json", "jsonl", "x-ndjson", "pdf"):
        return "📄"
    if file_type == "tensor":
        return "📊"
    if file_type == "kernel":
        return "📊"
    if file_type == "presentation":
        return "📊"
    if file_type == "base64":
        return "📄"
    return f"❓ {file_type}"


def get_file_type(file_name: str) -> str:
    """Determine the file type based on its extension."""
    file_name = file_name.lower()
    if file_name.endswith((".mp4", ".avi", ".mkv")):
        return "video"
    if file_name.endswith((".wav", ".mp3", ".aac", ".flac", ".ogg", ".m4a")):
        return "audio"
    if file_name.endswith((".png", ".jpg", ".jpeg")):
        return "image"
    if file_name.endswith((".log", ".txt")):
        return "text"
    if file_name.endswith(".pt"):
        return "tensor"
    if file_name.endswith(".ptx"):
        return "kernel"
    if file_name.endswith(".json"):
        return "json"
    if file_name.endswith(".jsonl"):
        return "jsonl"
    if file_name.endswith(".pdf"):
        return "pdf"
    if file_name.endswith((".pptx", ".ppt")):
        return "presentation"
    if file_name.endswith(".base64"):
        return "base64"
    if file_name.endswith((".zip", ".tar", ".gz", ".bz2", ".7z")):
        return "archive"
    return "unknown"


def get_mime_type(file_name: str) -> str:
    mimetype, _ = mimetypes.guess_type(file_name)
    if mimetype is None:
        mimetype = "application/octet-stream"
    if file_name.endswith(".log"):
        mimetype = "text/plain"
    if file_name.endswith(".jsonl"):
        mimetype = "application/x-ndjson"
    return mimetype


def get_content_type_emoji(content_type: str) -> str:
    if content_type is None:
        return "❓"
    if content_type.startswith("text/"):
        return "📄"
    if content_type.startswith("image/"):
        return "🖼️"
    if content_type.startswith("audio/"):
        return "🎵"
    if content_type.startswith("video/"):
        return "🎥"
    if content_type == "application/json":
        return "📄"
    if content_type == "application/x-ndjson":
        return "📄"
    if content_type == "application/pdf":
        return "📄"
    if content_type == "application/octet-stream":
        return "📦"
    return f"❓ {content_type}"


def get_friendly_region_name(region: Optional[str]) -> str:
    if not region or region == "N/A":
        return "N/A"
    region_lower = region.lower()
    if region_lower.startswith("eastus2"):
        return "East US 2"
    if region_lower.startswith("eastus"):
        return "East US"
    if region_lower.startswith("westus3"):
        return "West US 3"
    if region_lower.startswith("westus2"):
        return "West US 2"
    if region_lower.startswith("westus"):
        return "West US"
    if region_lower.startswith("centralus"):
        return "Central US"
    if region_lower.startswith("northcentralus"):
        return "North Central US"
    if region_lower.startswith("southcentralus"):
        return "South Central US"
    if region_lower.startswith("westeurope"):
        return "West Europe"
    if region_lower.startswith("eastasia"):
        return "East Asia"
    if region_lower.startswith("southeastasia"):
        return "Southeast Asia"
    if region_lower.startswith("swedencentral"):
        return "Sweden Central"
    return region.capitalize()


async def get_friendly_container_name(container_name: str) -> str:
    """
    Get a friendly name for a container from services.json.
    For example,
    "imageresize" -> "Image Resize"
    "fantasytalking" -> "Fantasy Talking"
    """
    services_file_name = await get_service_json_filename()
    if services_file_name is None:
        return container_name
    async with aiofiles.open(services_file_name) as file:
        data_str = await file.read()
        data_json = json.loads(data_str)
    if container_name not in data_json:
        return container_name
    data_service_json = data_json[container_name]
    friendly_name = data_service_json["friendlyName"]
    return friendly_name


async def get_friendly_pod_name(pod_name: str) -> str:
    """Get a friendly name for a pod from services.json."""
    container_name = pod_name.split("-")[0]
    return await get_friendly_container_name(container_name)


async def get_friendly_model_name(model_name: str) -> str:
    """Get a friendly name for a service from services.json."""
    return await get_friendly_container_name(model_name)


async def is_rtgen_container(container_name: str) -> bool:
    services_file_name = await get_service_json_filename()
    if services_file_name is None:
        return False
    async with aiofiles.open(services_file_name) as file:
        data_str = await file.read()
        data_json = json.loads(data_str)
    return container_name in data_json.keys()


async def get_docker_image(
    container_name: str
) -> Optional[str]:
    """
    Get the docker image for a container from services.json.
    """
    services_file_name = await get_service_json_filename()
    if services_file_name is None:
        return ""
    async with aiofiles.open(services_file_name, "r") as file:
        data_str = await file.read()
        data_json = json.loads(data_str)
    if container_name not in data_json:
        return None
    data_service_json = data_json[container_name]
    docker_image = data_service_json["dockerImage"]
    docker_repo = docker_image["repository"]
    docker_name = docker_image["name"]
    docker_tag = docker_image["tag"]
    return f"{docker_repo}/{docker_name}:{docker_tag}"


def parse_request_id(request_id: str) -> Dict[str, str]:
    """
    Parse request ids like:
    20250904T010335296_flux
    20250904T010335296_007_kokoro
    20250904T010335296_006_001_fantasytalking
    20260105T194652416_main_image_flux
    """
    match = re.match(
        r"^(\d{8}T\d{9})"      # job_id
        r"(?:_(\d+))?"          # scene_id (optional)
        r"(?:_(\d+))?"          # sub_scene_id (optional)
        r"(?:_(.*?))?"         # task_id (optional, lazy)
        r"_([^_]+)$",          # service_name (last part)
        request_id
    )
    if not match:
        return {}
    job_id = match.group(1) or ""
    scene_id = match.group(2) or ""
    sub_scene_id = match.group(3) or ""
    task_id = match.group(4) or ""
    service_name = match.group(5) or ""
    return {
        "job_id": job_id,
        "scene_id": scene_id,
        "sub_scene_id": sub_scene_id,
        "task_id": task_id,
        "service_name": service_name,
    }


async def list_files(
    folder_path: str
) -> List[Dict[str, Any]]:
    """List files in a folder with their metadata."""
    files = []
    file_names = await aiofiles.os.listdir(folder_path)
    for file_name in file_names:
        try:
            file_path = os.path.join(folder_path, file_name)
            file_date = await aiofiles.os.path.getmtime(file_path)
            mimetype, _ = mimetypes.guess_type(file_name)
            if await aiofiles.os.path.isfile(file_path):
                file_size = await aiofiles.os.path.getsize(file_path)
                file_type = get_file_type(file_name)
                files.append({
                    "name": file_name,
                    "size": file_size,
                    "date": file_date,
                    "type": file_type,
                    "mimetype": mimetype,
                })
            elif await aiofiles.os.path.isdir(file_path):
                files.append({
                    "name": file_name,
                    "size": 0,
                    "date": file_date,
                    "type": "directory",
                    "mimetype": "inode/directory",
                })
        except PermissionError:
            files.append({
                "name": file_name,
                "size": 0,
                "date": 0,
                "type": "unknown",
                "mimetype": "unknown",
                "error": "Permission denied"
            })
        except Exception as ex:
            logging.error(f"Error accessing file {file_name}: {ex}.")
            files.append({
                "name": file_name,
                "size": 0,
                "date": 0,
                "type": "unknown",
                "mimetype": "unknown",
                "error": str(ex)
            })
    return files
