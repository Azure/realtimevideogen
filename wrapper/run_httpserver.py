"""
REST API server for LMM generation.

Linux example:
IMG_BASE64=$(base64 -w 0 benchmark/samples/sample.png)
cat > payload.json <<EOF
{
  "img": "$IMG_BASE64",
  "prompt": "The person in the right is speaking to the person in the left",
  "num_frames": 17,
  "sampling_steps": 5
}
EOF
curl -X POST http://localhost:8080/wan -H "Content-Type: application/json" -d @payload.json
"""
import argparse
import asyncio
import gzip
import io
import json
import logging
import mimetypes
import os
import pickle
import sys
import time
import errno
import traceback
import aiofiles
import aiofiles.os

from PIL import Image

from datetime import datetime
from datetime import timedelta
from functools import wraps
from http import HTTPStatus
from pickle import UnpicklingError
from json import JSONDecodeError

from typing import Tuple
from typing import Optional
from typing import Callable
from typing import Any
from typing import Awaitable
from typing import List
from typing import Dict
from typing import AsyncIterable

import torch
import torch.distributed as dist
from torch.distributed import DistBackendError

from hypercorn.config import Config
from hypercorn.asyncio import serve

from quart import Quart
from quart import request
from quart import jsonify
from quart import send_file
from quart import send_from_directory
from quart import render_template
from quart import Response

from wrapper_model import ModelGeneration

from console_utils import setup_logging

from media_utils import base64_to_tensor
from image_utils import img_to_bytesio
from image_utils import img_to_base64
from media_utils import get_audio_file_info
from media_utils import get_image_file_info
from media_utils import get_text_file_info
from media_utils import get_video_file_info
from media_utils import get_tensor_file_info

import quart_utils
from quart_utils import QuartReturn
from quart_utils import get_mime_type
from quart_utils import get_file_type
from quart_utils import get_friendly_model_name
from quart_utils import get_friendly_container_name
from quart_utils import get_class_emoji
from quart_utils import format_string


GPU_SETUP = True
try:
    from xfuser import xFuserArgs
    from xfuser.config import FlexibleArgumentParser
    from xfuser.config import EngineConfig
    from xfuser.core.distributed import init_distributed_environment
except Exception as ex:
    if "No CUDA GPUs are available" in str(ex):
        logging.error("No GPUs available, running without xfuser.")
    elif "Found no NVIDIA driver on your system." in str(ex):
        logging.error("No NVIDIA driver found, running without xfuser.")
    elif "cannot import name 'SanaAttnProcessor2_0' from 'diffusers.models.transformers.sana_transformer'" in str(ex):
        logging.error("Old diffusers version (no SanaAttnProcessor2_0), running without xfuser.")
    else:
        logging.error(f"Likely no GPU setup, running without xfuser: {ex}.")
        logging.error(traceback.format_exc())
    EngineConfig = None
    GPU_SETUP = False

    def init_distributed_environment(rank: int, world_size: int) -> None:
        """Dummy function when no GPU setup."""
        logging.warning("No distributed environment setup.")


# Quart/Flask app configuration
HOST = "0.0.0.0"
PORT = 8080
TMP_DIR = "/tmp"
app = Quart(__name__)
route = app.route
template_filter = app.template_filter
route_locks: Dict[str, asyncio.Lock] = {}
last_ping_time = time.time()

# Models
models: Dict[str, ModelGeneration] = {}

EXCLUDED_NCCL_MODELS: List[str] = [
    "hunyuanimage"
]


def get_job_id() -> str:
    """Generate a unique job ID based on the current timestamp: 20240605T153000123."""
    return datetime.now().strftime("%Y%m%dT%H%M%S%f")[:-3]


def get_service_names() -> List[str]:
    """Get the list of available service names from services.json."""
    services = {}
    with open("services.json", "r", encoding="utf-8") as file:
        services = json.load(file)
    # Only include entries that represent runnable model services (have a class field)
    return [name for name, config in services.items() if "class" in config]


def get_model(
    service_name: str,
    sub_module: Optional[str] = None
) -> Optional[ModelGeneration]:
    """Get the model instance for the given service name."""
    if service_name in models:
        return models[service_name]
    if sub_module is not None:
        full_service_name = f"{service_name}{sub_module}"
        if full_service_name in models:
            return models[full_service_name]
    if "mock" in models:
        return models["mock"]
    return None


def exclusive_route(key: str) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Decorator to ensure exclusive access to a route based on a key."""
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        """Decorator to ensure exclusive access to a route based on a key."""

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            lock = route_locks.setdefault(key, asyncio.Lock())
            if lock.locked():
                request_json = await request.get_json()
                job_id = request_json.get("job_id")
                if job_id is not None:
                    logging.warning(f"{job_id} trying to use exclusive route '{key}' already running.")
                else:
                    logging.warning(f"Exclusive route '{key}' already running.")
                return jsonify({"error": "Generation in progress"}), HTTPStatus.SERVICE_UNAVAILABLE
            await lock.acquire()
            try:
                return await func(*args, **kwargs)
            finally:
                lock.release()
        return wrapper

    return decorator


@template_filter("get_friendly_container_name")
async def get_friendly_container_name_template(container_name: str) -> str:
    return await get_friendly_container_name(container_name)


@template_filter("format_string")
def format_string_template(value: Any) -> Optional[str]:
    return format_string(value)


@template_filter("get_class_emoji")
async def get_class_emoji_template(container_name: str) -> str:
    return await get_class_emoji(container_name)


# HTTP routes
@route("/", methods=["GET"])
async def index() -> str:
    """Render the index HTML page."""
    models_health = {}
    for model_name, model in models.items():
        if model is not None:
            models_health[model_name] = model.get_health()
    return await render_template(
        "index.html",
        models_health=models_health)


@route("/health", methods=["GET"])
async def health() -> Response:
    """Get health status of all models."""
    ret = {}
    for model_name, model in models.items():
        if model is not None:
            ret[model_name] = model.get_health()
    return jsonify(ret)


@route("/<service_name>/health", methods=["GET"])
async def model_health(service_name: str) -> QuartReturn:
    """Get health status of a specific model."""
    model = get_model(service_name)
    if model is None:
        return (
            {"error": f"{service_name} not initialized"},
            HTTPStatus.INTERNAL_SERVER_ERROR
        )
    return model.get_health()


@route("/timestamps", methods=["GET"])
async def timestamps() -> Response:
    """Get timing timestamps for all models."""
    ret = {}
    for model_name, model in models.items():
        if model is not None:
            ret[model_name] = model.get_timestamps()
    return jsonify(ret)


@route("/files", methods=["GET"])
async def list_files() -> QuartReturn:
    """List files in the TMP_DIR directory."""
    try:
        files = await quart_utils.list_files(TMP_DIR)
        return jsonify({"files": files})
    except Exception as ex:
        return jsonify({"error": str(ex)}), HTTPStatus.INTERNAL_SERVER_ERROR


@route("/file/<file_name>", methods=["GET"])
async def download_file(file_name: str) -> QuartReturn:
    """Download a file."""
    filepath = f"{TMP_DIR}/{file_name}"
    if not await aiofiles.os.path.exists(filepath):
        return jsonify({"error": "File not found"}), HTTPStatus.NOT_FOUND

    # Directory listing
    if await aiofiles.os.path.isdir(filepath):
        files = await aiofiles.os.listdir(filepath)
        return jsonify({
            "files": files
        })

    mimetype = get_mime_type(file_name)
    # Fixes length error when sending large files with send_file, use send_from_directory instead
    # TODO conditional=True?
    return await send_from_directory(
        TMP_DIR,
        file_name,
        mimetype=mimetype,
        as_attachment=True)


@route("/file_info/<file_name>", methods=["GET"])
async def file_info(file_name: str) -> QuartReturn:
    """Get detailed information about a file."""
    file_path = f"{TMP_DIR}/{file_name}"
    if not await aiofiles.os.path.exists(file_path):
        return jsonify({"error": "File not found"}), HTTPStatus.NOT_FOUND

    file_type = get_file_type(file_name)
    mimetype, _ = mimetypes.guess_type(file_name)
    if mimetype is None:
        mimetype = "application/octet-stream"

    file_info_ret = {
        "name": file_name,
        "size": await aiofiles.os.path.getsize(file_path),
        "date": await aiofiles.os.path.getmtime(file_path),
        "type": file_type,
        "mimetype": mimetype
    }

    if file_type == "audio":
        file_audio_info = get_audio_file_info(file_path)
        file_info_ret.update(file_audio_info)
    elif file_type == "video":
        file_video_info = get_video_file_info(file_path)
        video_info = file_video_info["video"]
        file_info_ret.update(video_info)
        # audio_info = file_video_info["audio"]
        # file_info_ret.update(audio)
    elif file_type == "image":
        file_image_info = get_image_file_info(file_path)
        file_info_ret.update(file_image_info)
    elif file_type == "text":
        file_text_info = get_text_file_info(file_path)
        file_info_ret.update(file_text_info)
    elif file_type == "tensor":
        file_tensor_info = get_tensor_file_info(file_path)
        file_info_ret.update(file_tensor_info)

    return jsonify(file_info_ret)


@route("/yolo", methods=["POST"])
@exclusive_route("yolo")
async def yolo_e2e() -> QuartReturn:
    """YOLO object detection endpoint."""
    model = get_model("yolo")
    if not model:
        return jsonify({"error": "YOLO not initialized"}), HTTPStatus.INTERNAL_SERVER_ERROR
    try:
        request_json = await request.get_json()
        if request_json is None:
            return jsonify({"error": "No JSON body received"}), HTTPStatus.BAD_REQUEST

        job_id = request_json.get("job_id") or get_job_id()
        async with aiofiles.open(f"{TMP_DIR}/{job_id}.json", "w", encoding="utf-8") as file:
            data = json.dumps(request_json, indent=4)
            await file.write(data)
            await file.flush()

        gen_args = await model.get_rest_args(request_json)
        args = gen_args["args"]
        args["job_id"] = job_id

        if "img" in args:
            img = args["img"]
            img.save(f"{TMP_DIR}/{job_id}.png")

        # Run the one chunk in GPU 0 (blocks for a while)
        logging.info(f"Generating YOLO with args: {args}")
        extracted_images = await model.generate(**args)

        if len(extracted_images) > 0:
            debug_img = extracted_images[0]
            debug_img.save(f"{TMP_DIR}/{job_id}_debug.png")

        images_json = {}
        for img_id, extracted_image in enumerate(extracted_images[1:]):
            if not extracted_image:
                logging.warning(f"No character extracted for image {img_id} in job {job_id}, skipping.")
            else:
                extracted_image.save(f"{TMP_DIR}/{job_id}_{img_id:03d}.png")
                extracted_img_base64 = img_to_base64(extracted_image)
                images_json[f"image{img_id:03d}"] = extracted_img_base64
        return jsonify(images_json)
    except ValueError as value_err:
        logging.error(f"Error generating YOLO: {value_err}")
        return jsonify({
            "error": str(value_err),
            "traceback": traceback.format_exc()
        }), HTTPStatus.BAD_REQUEST
    except Exception as ex:
        logging.error(f"Error generating YOLO: {ex} {traceback.format_exc()}")
        return jsonify({
            "error": str(ex),
            "traceback": traceback.format_exc()
        }), HTTPStatus.INTERNAL_SERVER_ERROR


@route("/podcasttranscript", methods=["POST"])
async def podcasttranscript_e2e() -> QuartReturn:
    """podcasttranscript transcript generation endpoint."""
    model = get_model("podcasttranscript")
    if not model:
        return jsonify({"error": "Podcast transcript model not initialized"}), HTTPStatus.INTERNAL_SERVER_ERROR
    try:
        request_json = await request.get_json()
        if request_json is None:
            return jsonify({"error": "No JSON body received"}), HTTPStatus.BAD_REQUEST

        job_id = request_json.get("job_id") or get_job_id()
        async with aiofiles.open(f"{TMP_DIR}/{job_id}.json", "w") as file:
            data = json.dumps(request_json, indent=4)
            await file.write(data)

        gen_args = await model.get_rest_args(request_json)
        args = gen_args["args"]
        args["job_id"] = job_id

        # Run locally in a single GPU
        logging.info(f"Generating podcast transcript with args: {args}")
        podcast = await model.generate(**args)

        if podcast is None:
            return jsonify({"error": "No podcast generated"}), HTTPStatus.INTERNAL_SERVER_ERROR

        async with aiofiles.open(f"{TMP_DIR}/{job_id}_podcast.json", "w") as file:
            data = json.dumps(podcast.model_dump(), indent=4)
            await file.write(data)

        return jsonify(podcast.model_dump())
    except ValueError as value_err:
        logging.error(f"Error generating podcast transcript: {value_err}")
        return jsonify({"error": str(value_err)}), HTTPStatus.BAD_REQUEST
    except Exception as ex:
        logging.error(f"Error generating podcast transcript: {ex}")
        return jsonify({
            "error": str(ex),
            "traceback": traceback.format_exc()
        }), HTTPStatus.INTERNAL_SERVER_ERROR


@route("/slidetranscript/stream", methods=["POST"])
async def slidetranscript_stream() -> QuartReturn:
    """Stream slide transcript generation endpoint."""
    model = get_model("slidetranscript")
    if not model:
        return jsonify({"error": "Slide transcript model not initialized"}), HTTPStatus.INTERNAL_SERVER_ERROR
    try:
        request_json = await request.get_json()
        if request_json is None:
            return jsonify({"error": "No JSON body received"}), HTTPStatus.BAD_REQUEST

        job_id = request_json.get("job_id") or get_job_id()
        request_json["job_id"] = job_id
        async with aiofiles.open(f"{TMP_DIR}/{job_id}.json", "w") as file:
            data = json.dumps(request_json, indent=4)
            await file.write(data)

        gen_args = await model.get_rest_args(request_json)
        args = gen_args["args"]
        args["job_id"] = job_id

        async def slide_transcript_generate() -> AsyncIterable[str]:
            try:
                async with aiofiles.open(f"{TMP_DIR}/{job_id}_slides.jsonl", "w") as f_out:
                    async for scene in model.generate_stream(**args):
                        line = json.dumps(scene) + "\n"
                        await f_out.write(line)
                        await f_out.flush()
                        yield line
            except Exception as ex:
                logging.exception(f"Exception streaming (job_id={job_id}): {ex}")
                yield json.dumps({
                    "error": "Stream generation failed",
                    "details": str(ex)
                }) + "\n"

        return Response(
            slide_transcript_generate(),
            mimetype="application/x-ndjson")
    except Exception as ex:
        logging.error(f"Error generating slide transcript: {ex}")
        return jsonify({
            "error": str(ex),
            "traceback": traceback.format_exc()
        }), HTTPStatus.INTERNAL_SERVER_ERROR


@route("/podcasttranscript/stream", methods=["POST"])
async def podcasttranscript_stream() -> QuartReturn:
    """Stream podcast transcript generation endpoint."""
    model = get_model("podcasttranscript")
    if not model:
        return jsonify({"error": "Podcast transcript model not initialized"}), HTTPStatus.INTERNAL_SERVER_ERROR
    try:
        request_json = await request.get_json()
        if request_json is None:
            return jsonify({"error": "No JSON body received"}), HTTPStatus.BAD_REQUEST

        job_id = request_json.get("job_id") or get_job_id()
        request_json["job_id"] = job_id
        async with aiofiles.open(f"{TMP_DIR}/{job_id}.json", "w") as file:
            data = json.dumps(request_json, indent=4)
            await file.write(data)

        gen_args = await model.get_rest_args(request_json)
        args = gen_args["args"]
        args["job_id"] = job_id

        async def podcast_transcript_generate() -> AsyncIterable[str]:
            try:
                async with aiofiles.open(f"{TMP_DIR}/{job_id}_podcast.jsonl", "w") as f_out:
                    async for scene in model.generate_stream(**args):
                        line = json.dumps(scene) + "\n"
                        await f_out.write(line)
                        await f_out.flush()
                        yield line
            except Exception as ex:
                logging.exception(f"Exception streaming (job_id={job_id}): {ex}")
                yield json.dumps({
                    "error": "Stream generation failed",
                    "details": str(ex)
                }) + "\n"

        return Response(
            podcast_transcript_generate(),
            mimetype="application/x-ndjson")
    except ValueError as value_err:
        logging.error(f"Error generating podcast transcript: {value_err}")
        return jsonify({"error": str(value_err)}), HTTPStatus.BAD_REQUEST
    except Exception as ex:
        logging.error(f"Error generating podcast transcript: {ex}")
        return jsonify({
            "error": str(ex),
            "traceback": traceback.format_exc()
        }), HTTPStatus.INTERNAL_SERVER_ERROR


async def gen_video(model: Optional[ModelGeneration]) -> QuartReturn:
    """Generic video generation endpoint."""
    if not model:
        return jsonify({"error": "Not initialized"}), HTTPStatus.INTERNAL_SERVER_ERROR
    # TODO not all models are constrained to running a request at a time
    if model.running:
        logging.warning("Generation in progress.")
        return jsonify({"error": "Generation in progress"}), HTTPStatus.SERVICE_UNAVAILABLE
    try:
        request_json = await request.get_json()
        if request_json is None:
            return jsonify({"error": "No JSON body received"}), HTTPStatus.BAD_REQUEST

        job_id = request_json.get("job_id") or get_job_id()
        request_json["job_id"] = job_id
        async with aiofiles.open(f"{TMP_DIR}/{job_id}.json", "w") as file:
            data = json.dumps(request_json, indent=4)
            await file.write(data)

        gen_args = await model.get_rest_args(request_json)
        args = gen_args["args"]
        args["job_id"] = job_id
        args["output_type"] = "video_path"

        # Trigger video generation in all GPUs
        await send_task(gen_args)

        # Run the one chunk in GPU 0 (blocks for a while)
        logging.info(f"Generating video with args: {args}")
        video_path = await model.generate(**args)

        if not video_path:
            return jsonify({"error": "No video generated"}), HTTPStatus.INTERNAL_SERVER_ERROR
        if not await aiofiles.os.path.exists(video_path):
            return jsonify({"error": f"Video file not found: {video_path}"}), HTTPStatus.INTERNAL_SERVER_ERROR

        return await send_file(
            video_path,
            mimetype="video/mp4",
            as_attachment=True,
            attachment_filename=f"{job_id}.mp4")
    except JSONDecodeError as json_err:
        logging.error(f"Error processing JSON request: {json_err}")
        return jsonify({"error": "Invalid JSON format"}), HTTPStatus.BAD_REQUEST
    except ValueError as value_err:
        logging.error(f"Error generating video: {value_err}")
        return jsonify({"error": str(value_err)}), HTTPStatus.BAD_REQUEST
    except Exception as ex:
        logging.error(f"Error generating video: {ex}")
        return jsonify({
            "error": str(ex),
            "traceback": traceback.format_exc()
        }), HTTPStatus.INTERNAL_SERVER_ERROR


async def gen_audio(model: Optional[ModelGeneration]) -> QuartReturn:
    """Generic audio generation endpoint."""
    if not model:
        return jsonify({"error": "Not initialized"}), HTTPStatus.INTERNAL_SERVER_ERROR
    # TODO not all models are constrained to running a request at a time
    if model.running:
        logging.warning("Generation in progress.")
        return jsonify({"error": "Generation in progress"}), HTTPStatus.SERVICE_UNAVAILABLE
    try:
        request_json = await request.get_json()
        job_id = request_json.get("job_id") or get_job_id()
        request_json["job_id"] = job_id
        async with aiofiles.open(f"{TMP_DIR}/{job_id}.json", "w") as file:
            data = json.dumps(request_json, indent=4)
            await file.write(data)

        gen_args = await model.get_rest_args(request_json)
        args = gen_args["args"]
        args["job_id"] = job_id

        # Trigger image generation in all GPUs
        await send_task(gen_args)

        # Run the one chunk in GPU 0 (blocks for a while)
        logging.info(f"Generating audio with args: {args}")
        audio_path = await model.generate(**args)

        if not audio_path:
            return jsonify({"error": "No audio generated"}), HTTPStatus.INTERNAL_SERVER_ERROR
        if not await aiofiles.os.path.exists(audio_path):
            return jsonify({"error": f"Audio file not found: {audio_path}"}), HTTPStatus.INTERNAL_SERVER_ERROR

        return await send_file(
            audio_path,
            mimetype="audio/wav",
            as_attachment=True,
            attachment_filename=f"{job_id}.wav")
    except JSONDecodeError as json_err:
        logging.error(f"Error processing JSON request: {json_err}")
        return jsonify({"error": "Invalid JSON format"}), HTTPStatus.BAD_REQUEST
    except ValueError as value_err:
        logging.error(f"Error generating audio: {value_err}")
        return jsonify({"error": str(value_err)}), HTTPStatus.BAD_REQUEST
    except Exception as ex:
        logging.error(f"Error generating audio: {ex}")
        return jsonify({
            "error": str(ex),
            "traceback": traceback.format_exc()
        }), HTTPStatus.INTERNAL_SERVER_ERROR


async def gen_img(model: Optional[ModelGeneration]) -> QuartReturn:
    """Generic image generation endpoint."""
    if not model:
        return jsonify({"error": "Not initialized"}), HTTPStatus.INTERNAL_SERVER_ERROR
    # TODO not all models are constrained to running a request at a time
    if model.running:
        logging.warning("Generation in progress.")
        return jsonify({"error": "Generation in progress"}), HTTPStatus.SERVICE_UNAVAILABLE
    try:
        request_json = await request.get_json()
        job_id = request_json.get("job_id") or get_job_id()
        async with aiofiles.open(f"{TMP_DIR}/{job_id}.json", mode="w") as file:
            data = json.dumps(request_json, indent=4)
            await file.write(data)

        gen_args = await model.get_rest_args(request_json)
        args = gen_args["args"]
        args["job_id"] = job_id

        # Trigger image generation in all GPUs
        await send_task(gen_args)

        # Run the one chunk in GPU 0 (blocks for a while)
        img_wrapper = await model.generate(**args)

        if isinstance(img_wrapper, list) and len(img_wrapper) > 0:
            img = img_wrapper[0]
        else:
            img = img_wrapper

        if not img:
            return jsonify({"error": "No image generated"}), HTTPStatus.INTERNAL_SERVER_ERROR
        if not isinstance(img, Image.Image):
            return jsonify({"error": f"No image generated: {type(img)}"}), HTTPStatus.INTERNAL_SERVER_ERROR

        img.save(f"{TMP_DIR}/{job_id}.png")
        img_io = img_to_bytesio(img)
        if img_io is None:
            return jsonify({"error": "Failed to convert image to bytes"}), HTTPStatus.INTERNAL_SERVER_ERROR
        return await send_file(
            img_io,
            mimetype="image/png",
            as_attachment=True,
            attachment_filename=f"{job_id}.png")
    except JSONDecodeError as json_err:
        logging.error(f"Error processing JSON request: {json_err}")
        return jsonify({"error": "Invalid JSON format"}), HTTPStatus.BAD_REQUEST
    except ValueError as value_err:
        logging.error(f"Error generating image: {value_err}")
        return jsonify({"error": str(value_err)}), HTTPStatus.BAD_REQUEST
    except Exception as ex:
        logging.error(f"Error generating image: {ex}")
        return jsonify({
            "error": str(ex),
            "traceback": traceback.format_exc()
        }), HTTPStatus.INTERNAL_SERVER_ERROR


@route("/fantasytalking", methods=["POST"])
@exclusive_route("fantasytalking")
async def fantasytalking_e2e() -> QuartReturn:
    """FantasyTalking video generation endpoint."""
    model = get_model("fantasytalking")
    return await gen_video(model)


@route("/kokoro", methods=["POST"])
@exclusive_route("kokoro")
async def kokoro_e2e() -> QuartReturn:
    """Kokoro audio generation endpoint."""
    model = get_model("kokoro")
    return await gen_audio(model)


@route("/dia", methods=["POST"])
@exclusive_route("dia")
async def dia_e2e() -> QuartReturn:
    """DIA audio generation endpoint."""
    model = get_model("dia")
    return await gen_audio(model)


@route("/xtts", methods=["POST"])
@exclusive_route("xtts")
async def xtts_e2e() -> QuartReturn:
    """XTTS audio generation endpoint."""
    model = get_model("xtts")
    return await gen_audio(model)


@route("/thinksound", methods=["POST"])
@exclusive_route("thinksound")
async def thinksound_e2e() -> QuartReturn:
    """ThinkSound audio generation endpoint."""
    model = get_model("thinksound")
    return await gen_audio(model)


@route("/vibevoice", methods=["POST"])
@exclusive_route("vibevoice")
async def vibevoice_e2e() -> QuartReturn:
    """VibeVoice audio generation endpoint."""
    model = get_model("vibevoice")
    return await gen_audio(model)


@route("/flux", methods=["POST"])
@exclusive_route("flux")
async def flux_e2e() -> QuartReturn:
    """Flux image generation endpoint."""
    model = get_model("flux")
    return await gen_img(model)


@route("/fluxupscaler", methods=["POST"])
@exclusive_route("fluxupscaler")
async def fluxupscaler_e2e() -> QuartReturn:
    """Flux image upscaling endpoint."""
    model = get_model("fluxupscaler")
    return await gen_img(model)


@route("/fluxupscaler/video", methods=["POST"])
@exclusive_route("fluxupscaler")
async def fluxupscaler_video_e2e() -> QuartReturn:
    """Flux video upscaling endpoint."""
    model = get_model("fluxupscaler")
    return await gen_video(model)


@route("/fluxkontext", methods=["POST"])
@exclusive_route("fluxkontext")
async def fluxkontext_e2e() -> QuartReturn:
    """Flux Kontext image generation endpoint."""
    model = get_model("fluxkontext")
    return await gen_img(model)


@route("/fluxkrea", methods=["POST"])
@exclusive_route("fluxkrea")
async def fluxkrea_e2e() -> QuartReturn:
    """Flux Krea image generation endpoint."""
    model = get_model("fluxkrea")
    return await gen_img(model)


@route("/flux2", methods=["POST"])
@exclusive_route("flux2")
async def flux2_e2e() -> QuartReturn:
    """FLUX.2-dev image generation endpoint."""
    model = get_model("flux2")
    return await gen_img(model)


@route("/flux2klein", methods=["POST"])
@exclusive_route("flux2klein")
async def flux2klein_e2e() -> QuartReturn:
    """FLUX.2-klein-9B image generation endpoint."""
    model = get_model("flux2klein")
    return await gen_img(model)


@route("/hidream", methods=["POST"])
@exclusive_route("hidream")
async def hidream_e2e() -> QuartReturn:
    """HiDream image generation endpoint."""
    model = get_model("hidream")
    return await gen_img(model)


@route("/qwenimage", methods=["POST"])
@exclusive_route("qwenimage")
async def qwenimage_e2e() -> QuartReturn:
    """QwenImage image generation endpoint."""
    model = get_model("qwenimage")
    return await gen_img(model)


@route("/qwenimageedit", methods=["POST"])
@exclusive_route("qwenimageedit")
async def qwenimageedit_e2e() -> QuartReturn:
    """QwenImageEdit image generation endpoint."""
    model = get_model("qwenimageedit")
    return await gen_img(model)


@route("/hunyuanimage", methods=["POST"])
@exclusive_route("hunyuanimage")
async def hunyuanimage_e2e() -> QuartReturn:
    """Hunyuan Image generation endpoint."""
    model = get_model("hunyuanimage")
    return await gen_img(model)


@route("/januspro", methods=["POST"])
@exclusive_route("januspro")
async def januspro_e2e() -> QuartReturn:
    """JanusPro image generation endpoint."""
    model = get_model("januspro")
    return await gen_img(model)


@route("/llamagen", methods=["POST"])
@exclusive_route("llamagen")
async def llamagen_e2e() -> QuartReturn:
    """LlamaGen image generation endpoint."""
    model = get_model("llamagen")
    return await gen_img(model)


@route("/cogview", methods=["POST"])
@exclusive_route("cogview")
async def cogview_e2e() -> QuartReturn:
    """CogView4 image generation endpoint."""
    model = get_model("cogview")
    return await gen_img(model)


@route("/bagel", methods=["POST"])
@exclusive_route("bagel")
async def bagel_e2e() -> QuartReturn:
    """Bagel image generation endpoint."""
    model = get_model("bagel")
    return await gen_img(model)


@route("/imageresize", methods=["POST"])
async def imageresize_e2e() -> QuartReturn:
    """Basic pillow image resize endpoint."""
    model = get_model("imageresize")
    return await gen_img(model)


@route("/realesrgan", methods=["POST"])
@exclusive_route("realesrgan")
async def realesrgan_e2e() -> QuartReturn:
    """Real-ESRGAN image upscaling endpoint."""
    model = get_model("realesrgan")
    return await gen_img(model)


@route("/realesrgan/video", methods=["POST"])
@exclusive_route("realesrgan")
async def realesrgan_video_e2e() -> QuartReturn:
    model = get_model("realesrgan")
    return await gen_video(model)


"""
# TODO add a multi form data endpoint for realesrgan
@route("/realesrgan/video", methods=["POST"])
@exclusive_route("realesrgan")
async def realesrgan_video_e2e():
    form = await request.form
    file = (await request.files)["video"]

    job_id = form.get("job_id")
    width = int(form.get("width", 768))
    height = int(form.get("height", 576))

    print(f"Received job_id={job_id}, width={width}, height={height}, filename={file.filename}")

    # Simulate reading video file
    video_bytes = await file.read()
    print(f"Received video size: {len(video_bytes)} bytes")

    # Simulate processing and return dummy video
    dummy_output = io.BytesIO(video_bytes)  # Just echo input for demo
    dummy_output.seek(0)
    return await send_file(
        dummy_output,
        mimetype="video/mp4",
        download_name="upscaled.mp4",
    )
"""


@route("/hunyuanframepack", methods=["POST"])
@exclusive_route("hunyuanframepack")
async def hunyuanframepack_e2e() -> QuartReturn:
    """Hunyuan FramePack video generation endpoint."""
    model = get_model("hunyuanframepack")
    return await gen_video(model)


@route("/hunyuanframepackf1", methods=["POST"])
@exclusive_route("hunyuanframepack")
async def hunyuanframepackf1_e2e() -> QuartReturn:
    """Hunyuan FramePack F1 video generation endpoint."""
    model = get_model("hunyuanframepackf1")
    return await gen_video(model)


@route("/hunyuanframepack/vae", methods=["POST"])
@exclusive_route("hunyuanframepack")
async def hunyuanframepack_vae() -> QuartReturn:
    """Hunyuan FramePack VAE decode endpoint."""
    model = get_model("hunyuanframepack", "vae")
    if model is None or model.vae is None:
        return jsonify({"error": "Hunyuan FramePack VAE not available"}), HTTPStatus.INTERNAL_SERVER_ERROR
    return await gen_video(model)


@route("/hunyuanframepackvae", methods=["POST"])
@exclusive_route("hunyuanframepack")
async def hunyuanframepackvae() -> QuartReturn:
    """Hunyuan FramePack VAE decode endpoint."""
    return await hunyuanframepack_vae()


@route("/hunyuanframepack/vae/<job_id>", methods=["POST"])
@exclusive_route("hunyuanframepack")
async def hunyuanframepack_vae_binary(job_id: str) -> QuartReturn:
    """Hunyuan FramePack VAE decode binary endpoint."""
    model = get_model("hunyuanframepack", "vae")
    if model is None or model.vae is None:
        return jsonify({"error": "Hunyuan FramePack VAE not available"}), HTTPStatus.INTERNAL_SERVER_ERROR
    if job_id is None:
        return jsonify({"error": "Missing 'job_id' parameter"}), HTTPStatus.BAD_REQUEST
    try:
        encoding = request.headers.get('Content-Encoding', '').lower()
        data = await request.get_data()
        if not isinstance(data, (bytes, bytearray)):
            return jsonify({"error": f"Invalid data: {type(data)}"}), HTTPStatus.BAD_REQUEST
        data_bytes = io.BytesIO(data)
        if encoding == 'gzip':
            with gzip.GzipFile(fileobj=data_bytes, mode='rb') as file_gzip:
                data_decompressed = file_gzip.read()
            data_bytes = io.BytesIO(data_decompressed)
            decompressed_len = len(data_decompressed)
        else:
            decompressed_len = len(data)
        latents = torch.load(data_bytes, weights_only=True)

        logging.info(
            f"Process latent for '{job_id}' with {decompressed_len} bytes (HTTP:{len(data)}) "
            f"shape {latents.shape} and {latents.dtype}.")
        torch.save(latents, f"{TMP_DIR}/{job_id}_latents.pt")
        request_json = {
            "job_id": job_id,
            "size": decompressed_len,
            "http_size": len(data),
            "shape": str(latents.shape),
            "dtype": str(latents.dtype),
            "filename": f"{TMP_DIR}/{job_id}_latents.pt",
        }
        async with aiofiles.open(f"{TMP_DIR}/{job_id}.json", mode="w") as file:
            data = json.dumps(request_json, indent=4)
            await file.write(data)

        args = {
            "latents": latents,
            "job_id": job_id,
            "output_type": "video_path"
        }

        # Run the one chunk in GPU 0 (blocks for a while)
        logging.info(f"Decoding latents with args: {args}")
        video_path = await model.generate(**args)

        return await send_file(
            video_path,
            mimetype="video/mp4",
            as_attachment=True,
            attachment_filename=f"{job_id}.mp4")
    except ValueError as value_err:
        logging.error(f"Error processing VAE latents: {value_err}")
        return jsonify({
            "error": str(value_err),
            "traceback": traceback.format_exc()
        }), HTTPStatus.BAD_REQUEST
    except Exception as ex:
        logging.error(f"Error processing VAE latents: {ex}")
        return jsonify({
            "error": str(ex),
            "traceback": traceback.format_exc()
        }), HTTPStatus.INTERNAL_SERVER_ERROR


@route("/wan", methods=["POST"])
@exclusive_route("wan")
async def wan_e2e() -> QuartReturn:
    """Wan video generation endpoint."""
    model = get_model("wan")
    return await gen_video(model)


@route("/wan/vae", methods=["POST"])
@exclusive_route("wan")
async def wan_vae() -> QuartReturn:
    """Wan VAE decode endpoint."""
    # TODO this is not tested
    model = get_model("wanvae")
    if model is None or model.vae is None:
        return jsonify({"error": "Wan VAE not available"}), HTTPStatus.INTERNAL_SERVER_ERROR

    try:
        request_json = await request.get_json()
        job_id = request_json.get("job_id") or get_job_id()
        async with aiofiles.open(f"{TMP_DIR}/{job_id}.json", mode="w") as file:
            data = json.dumps(request_json, indent=4)
            await file.write(data)

        latents_base64 = request_json.get("latents", None)
        if latents_base64 is None:
            return jsonify({"error": "Missing 'latents' parameter"}), HTTPStatus.BAD_REQUEST
        latents = base64_to_tensor(latents_base64)

        # Run locally in a single GPU
        pixels = await asyncio.to_thread(model.vae_decode, latents)

        # Save the pixels to a file
        file_path = f"{TMP_DIR}/{job_id}_pixels.pt"
        torch.save(pixels, file_path)

        return await send_file(
            file_path,
            mimetype="application/octet-stream",
            as_attachment=True,
            attachment_filename=f"{job_id}_pixels.pt"
        )
    except JSONDecodeError as json_err:
        logging.error(f"Error processing JSON request: {json_err}")
        return jsonify({"error": "Invalid JSON format"}), HTTPStatus.BAD_REQUEST
    except ValueError as value_err:
        logging.error(f"Error processing VAE latents: {value_err}")
        return jsonify({
            "error": str(value_err),
            "traceback": traceback.format_exc()
        }), HTTPStatus.BAD_REQUEST
    except Exception as ex:
        logging.error(f"Error processing VAE latents: {ex}")
        return jsonify({
            "error": str(ex),
            "traceback": traceback.format_exc()
        }), HTTPStatus.INTERNAL_SERVER_ERROR


@route("/wanvae", methods=["POST"])
@exclusive_route("wan")
async def wanvae() -> QuartReturn:
    """Wan VAE decode endpoint."""
    return await wan_vae()


@route("/wan22", methods=["POST"])
@exclusive_route("wan22")
async def wan22_e2e() -> QuartReturn:
    """Wan 2.2 video generation endpoint."""
    model = get_model("wan22")
    return await gen_video(model)


@route("/hunyuanavatar", methods=["POST"])
@exclusive_route("hunyuanavatar")
async def hunyuanavatar_e2e() -> QuartReturn:
    """HunyuanAvatar video generation endpoint."""
    model = get_model("hunyuanavatar")
    return await gen_video(model)


@route("/ltx", methods=["POST"])
@exclusive_route("ltx")
async def ltx_e2e() -> QuartReturn:
    """LTX video generation endpoint."""
    model = get_model("ltx")
    return await gen_video(model)


@route("/longcatvideo", methods=["POST"])
@exclusive_route("longcatvideo")
async def longcatvideo_e2e() -> QuartReturn:
    """LongCat-Video generation endpoint."""
    model = get_model("longcatvideo")
    return await gen_video(model)


@route("/interrupt", methods=["POST"])
async def interrupt_gen() -> QuartReturn:
    """Interrupt any ongoing generation."""
    for model in models.values():
        if model:
            model.interrupt()
    return jsonify({"status": "interrupted"}), HTTPStatus.OK


# Distributed environment setup
rank = 0
local_rank = 0
device = torch.device(f"cuda:{rank}")
node_rank = 0
world_size = 1
local_world_size = 1

MAX_PAYLOAD_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB max payload size

# Signal file path for distributed tasks, this only works in single node multi-GPU mode
SIGNAL_0_PATH = f"{TMP_DIR}/streamwise_signal.txt"


def setup_dist_environment() -> None:
    """
    Initialize the distributed environment if running in multi-GPU mode.
    If `param_only` is True, only set the parameters without initializing the process group.
    """
    if "MASTER_ADDR" not in os.environ:
        logging.info("MASTER_ADDR not set, setting local.")
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

    if not torch.cuda.is_available():
        logging.warning("CUDA not available, skipping distributed initialization.")
        return

    global rank
    global local_rank
    global node_rank
    global world_size
    global local_world_size
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    node_rank = int(os.getenv("NODE_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get(
        "NPROC_PER_NODE") or torch.cuda.device_count())

    # With MIG, the device plugin restricts CUDA_VISIBLE_DEVICES to only the allocated
    # MIG instance(s) for this process, so the valid device indices start at 0.
    # Use the number of visible devices to clamp local_rank to a valid index.
    num_visible_devices = torch.cuda.device_count()
    device_id = local_rank if local_rank < num_visible_devices else 0
    torch.cuda.set_device(device_id)

    if world_size > num_visible_devices:
        logging.warning(
            f"world_size={world_size} but only {num_visible_devices} visible CUDA device(s). "
            "This usually means the container is running on a MIG partition and "
            "torchrun was invoked with too many processes. "
            "Clamping world_size to the number of visible devices."
        )
        world_size = num_visible_devices
        local_world_size = num_visible_devices

    logging.info(f"[{rank}] Initializing distributed: "
                 f"rank={rank}, local_rank={local_rank}, node_rank={node_rank}, "
                 f"world_size={world_size}, local_world_size={local_world_size}, "
                 f"device={device_id}")


def init_dist_environment() -> None:
    """Initialize the distributed process group."""
    if not torch.cuda.is_available():
        logging.debug(f"[{rank}] CUDA not available, skipping distributed initialization.")
        return
    if dist.is_initialized():
        logging.info(f"[{rank}] Distributed process group already initialized.")
        return

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(hours=24),  # Prevent NCCL timeout
    )

    if dist.get_world_size() > 1:
        init_distributed_environment(
            rank=dist.get_rank(),
            world_size=dist.get_world_size()
        )

    if not dist.is_initialized():
        raise RuntimeError("Distributed process group not initialized.")


async def wait_for_everybody() -> None:
    """
    Wait for all processes to reach this point.
    This is useful to ensure that all workers are ready before sending tasks.
    This only works in single node multi-GPU mode.
    """
    if not dist.is_initialized() or world_size <= 1:
        return

    logging.info(f"[{rank}] Waiting for all {local_world_size} workers to be ready...")

    # Specify that we are ready
    signal_worker_path = f"{TMP_DIR}/streamwise_signal_worker_{local_rank:03d}.txt"
    async with aiofiles.open(signal_worker_path, mode="w") as file:
        await file.write(f"ready at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    # Wait for the signal file from all other ranks
    for other_rank in range(local_world_size):
        other_signal_worker_path = f"{TMP_DIR}/streamwise_signal_worker_{other_rank:03d}.txt"
        while not await aiofiles.os.path.exists(other_signal_worker_path):
            await asyncio.sleep(0.1)  # Wait until the file exists

        async with aiofiles.open(other_signal_worker_path, mode="r") as file:
            content = await file.read()
            logging.info(f"[{rank}] Worker {other_rank}: {content}.")

    logging.info(f"[{rank}] All {local_world_size} workers are ready.")


async def send_task(gen_task: dict) -> None:
    """
    Send a generation task to all workers in the distributed environment (through NCCL).
    """
    if not dist.is_initialized():
        logging.warning(f"[{rank}] Torch distributed not initialized.")
        return
    if rank != 0:
        logging.error(f"[{rank}] Task can only be sent from rank 0.")
        return
    if world_size <= 1:
        logging.debug(f"[{rank}] Single GPU mode, skipping task broadcast.")
        return
    task_id = gen_task.get("task", "")
    if task_id in EXCLUDED_NCCL_MODELS:
        logging.debug(f"[{rank}] No NCCL-based parallelism for {task_id}.")
        return

    global last_ping_time
    last_ping_time = time.time()

    try:
        payload_bytes = await asyncio.to_thread(pickle.dumps, gen_task)
        payload_bytes = bytearray(payload_bytes)
        payload_tensor = torch.frombuffer(payload_bytes, dtype=torch.uint8).to("cuda")
        payload_size = torch.tensor([payload_tensor.numel()], dtype=torch.int64, device="cuda")

        if payload_size.item() > MAX_PAYLOAD_BYTES:
            logging.error(f"[{rank}] Payload too large: {payload_size.item()} bytes.")
            return

        # Notify that rank 0 is ready
        async with aiofiles.open(SIGNAL_0_PATH, mode="w") as file:
            await file.write(f"ready at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

        # Broadcast size and data
        logging.info(f"[{rank}] Broadcasting payload with {payload_size.item()} bytes.")
        dist.broadcast(payload_size, src=0)
        torch.cuda.synchronize()
        dist.barrier()

        # Clean up the rank 0 signal file
        await asyncio.to_thread(os.remove, SIGNAL_0_PATH)

        dist.broadcast(payload_tensor, src=0)
        torch.cuda.synchronize()

        logging.debug(f"[{rank}] Broadcast complete.")
    except DistBackendError as dist_err:
        logging.error(f"[{rank}] Cannot send task NCCL error: {dist_err}")
    except Exception as ex:
        logging.exception(f"[{rank}] Cannot send task {type(ex)}: {ex}")


async def nccl_worker() -> None:
    """Torch requests (through NCCL) from rank 0."""
    if rank == 0:
        logging.error(f"[{rank}] Worker should not be started on rank 0.")
        return

    while dist.is_initialized():
        logging.info(f"[{rank}] Waiting for tasks from rank 0...")
        try:
            # Wait until signal file exists
            while not await aiofiles.os.path.exists(SIGNAL_0_PATH):
                await asyncio.sleep(0.1)

            # Receive payload size
            payload_size = torch.tensor([0], dtype=torch.int64, device="cuda")
            dist.broadcast(payload_size, src=0)
            torch.cuda.synchronize()
            dist.barrier()

            payload_size_val = int(payload_size.item())
            if payload_size_val <= 0 or payload_size_val > MAX_PAYLOAD_BYTES:
                logging.error(f"[{rank}] Invalid payload size: {payload_size_val}.")
                continue

            # Receive payload data
            payload_tensor = torch.empty(payload_size_val, dtype=torch.uint8, device="cuda")
            dist.broadcast(payload_tensor, src=0)
            torch.cuda.synchronize()

            payload_bytes = payload_tensor.cpu().numpy().tobytes()
            payload = pickle.loads(payload_bytes)  # nosec B301 - internal IPC from rank 0

            if not isinstance(payload, dict):
                logging.error(f"[{rank}] Invalid payload received: {payload}.")
                continue

            if not payload:
                logging.error(f"[{rank}] Empty payload received.")
                continue

            if "task" not in payload:
                logging.error(f"[{rank}] No 'task' in payload: {payload}.")
                continue

            gen_task = payload

            task_id = gen_task["task"]
            if task_id == "ping":
                logging.debug(f"[{rank}] Received ping task to keep NCCL alive.")
            elif models.get(task_id) is None:
                logging.error(f"[{rank}] Model '{task_id}' not initialized.")
            else:
                model = models[task_id]
                args = gen_task.get("args", {})
                if rank > 1:
                    logging.info(f"[{rank}] Work to do for {task_id}: {len(args)} arguments.")
                else:
                    logging.info(f"[{rank}] Work to do for {task_id}:")
                    for key, value in args.items():
                        if isinstance(value, list) and len(value) > 5:
                            logging.info(f"[{rank}]   {key}: {value[0:5]}... ({len(value)} items)")
                        else:
                            logging.info(f"[{rank}]   {key}: {value}")
                # This can block a little
                await model.generate(**args)
        except UnpicklingError as pickle_err:
            logging.error(f"[{rank}] Parsing message: {pickle_err}.")
        except ValueError as value_err:
            logging.error(f"[{rank}] Processing task: {value_err}.")
        except DistBackendError as dist_err:
            logging.error(f"[{rank}] Processing task NCCL error: {dist_err}.")
        except Exception as ex:
            logging.error(f"[{rank}] Processing task: {ex}.", exc_info=True)

        await asyncio.sleep(0.1)  # Breathing time between requests just in case
    logging.info(f"[{rank}] Exiting worker thread.")


def is_model_running() -> bool:
    """Check if any model is currently running a generation task."""
    for model in models.values():
        if model.running:
            return True
    return False


def get_model_names() -> List[str]:
    """Get the list of available model names."""
    return list(models.keys())


def is_nccl_excluded_model() -> bool:
    """Check if any of the loaded models are in the NCCL excluded list."""
    model_names = get_model_names()
    for model_name in EXCLUDED_NCCL_MODELS:
        if model_name in model_names:
            return True
    return False


def arg_parsing() -> Tuple[argparse.Namespace, Optional[EngineConfig]]:
    """Parse command line arguments and return the parsed args and engine config."""
    if GPU_SETUP:
        parser = FlexibleArgumentParser(description="REST API for LMM generation")
    else:
        parser = argparse.ArgumentParser(description="REST API for LMM generation")

    parser.add_argument("--host", type=str, default=HOST, help="Host to bind the server")
    parser.add_argument("--port", type=int, default=PORT, help="Port to bind the server")
    parser.add_argument("--certfile", type=str, default=None, help="Path to SSL certificate file for HTTPS")
    parser.add_argument("--keyfile", type=str, default=None, help="Path to SSL private key file for HTTPS")

    parser.add_argument("--wan", action="store_true", help="Wan 2.1 model")
    parser.add_argument("--wan21", action="store_true", help="Wan 2.1 model")
    parser.add_argument("--wanvae", action="store_true", help="Wan 2.1 VAE model")
    parser.add_argument("--wan_variation", choices=["480p", "720p"],
                        default="480p", help="Wan 2.1 model resolution (480p or 720p)")
    parser.add_argument("--wan22", action="store_true", help="Wan 2.2 model")
    parser.add_argument("--hunyuanframepack", action="store_true", help="Hunyuan FramePack model")
    parser.add_argument("--hunyuanframepackf1", action="store_true", help="Hunyuan FramePack F1 model")
    parser.add_argument("--hunyuanframepackvae", action="store_true", help="Hunyuan FramePack VAE model")
    parser.add_argument("--flux", action="store_true", help="Flux model")
    parser.add_argument("--fluxupscaler", action="store_true", help="Flux Upscaler model")
    parser.add_argument("--fluxkontext", action="store_true", help="Flux Kontext model")
    parser.add_argument("--fluxkrea", action="store_true", help="Flux Krea model")
    parser.add_argument("--flux2", action="store_true", help="FLUX.2-dev model")
    parser.add_argument("--flux2klein", action="store_true", help="FLUX.2-klein-9B model")
    parser.add_argument("--cogview", action="store_true", help="CogView4 model")
    parser.add_argument("--hidream", action="store_true", help="HiDream model")
    parser.add_argument("--qwenimage", action="store_true", help="Qwen Image model")
    parser.add_argument("--qwenimageedit", action="store_true", help="Qwen Image Edit model")
    parser.add_argument("--hunyuanimage", action="store_true", help="Hunyuan Image model")
    parser.add_argument("--januspro", action="store_true", help="Janus Pro model")
    parser.add_argument("--llamagen", action="store_true", help="LlamaGen model")
    parser.add_argument("--kokoro", action="store_true", help="Kokoro model")
    parser.add_argument("--dia", action="store_true", help="Dia model")
    parser.add_argument("--xtts", action="store_true", help="XTTS model")
    parser.add_argument("--vibevoice", action="store_true", help="VibeVoice model")
    parser.add_argument("--thinksound", action="store_true", help="ThinkSound model")
    parser.add_argument("--fantasytalking", action="store_true", help="Fantasy Talking model")
    parser.add_argument("--podcasttranscript", action="store_true", help="Podcast transcript wrapper")
    parser.add_argument("--slidetranscript", action="store_true", help="Slide transcript wrapper")
    parser.add_argument("--yolo", action="store_true", help="YOLO model")
    parser.add_argument("--bagel", action="store_true", help="Bagel model")
    parser.add_argument("--imageresize", action="store_true", help="Image Resize")
    parser.add_argument("--realesrgan", action="store_true", help="Real-ESRGAN model")
    parser.add_argument("--4kagent", action="store_true", help="4K Agent")
    parser.add_argument("--hunyuanavatar", action="store_true", help="Hunyuan-Avatar model")
    parser.add_argument("--ltx", action="store_true", help="LTX-Video model")
    parser.add_argument("--longcatvideo", action="store_true", help="LongCat-Video model")
    parser.add_argument("--mock", action="store_true", help="Mock model")

    if GPU_SETUP and world_size > 1:
        args = xFuserArgs.add_cli_args(parser).parse_args()
        engine_args = xFuserArgs.from_cli_args(args)
        engine_config, _ = engine_args.create_config()
    else:
        # Add others for compatibility with xFuserArgs
        parser.add_argument("--ulysses_degree", type=int, default=1, help="Ulysses degree")
        parser.add_argument("--ring_degree", type=int, default=16, help="Ring degree")
        parser.add_argument("--use_torch_compile", action="store_true", help="Use torch.compile if available")
        args = parser.parse_args()
        engine_config = None

    return args, engine_config


async def load_model_wrapper_file(rank: int, model_name: str) -> None:
    """
    Set the model wrapper file path based on the model name.
    For example: /wan/wrapper_wan21.py
    """
    friendly_name = await get_friendly_model_name(model_name)
    if rank == 0:
        logging.info(f"[{rank}] Loading {friendly_name} model.")

    if await aiofiles.os.path.exists(f"/wrapper/{model_name}/wrapper_{model_name}.py"):
        sys.path.append(f"/wrapper/{model_name}")
        return
    if await aiofiles.os.path.exists(f"/{model_name}/wrapper_{model_name}.py"):
        sys.path.append(f"/{model_name}")
        return
    if await aiofiles.os.path.exists(f"{model_name}/wrapper_{model_name}.py"):
        sys.path.append(f"{model_name}")
        return
    if await aiofiles.os.path.exists(f"wrapper/{model_name}/wrapper_{model_name}.py"):
        sys.path.append(f"wrapper/{model_name}")
        return
    if await aiofiles.os.path.exists(f"wrapper_{model_name}.py"):
        sys.path.append(".")
        return

    if rank == 0:
        files = await aiofiles.os.listdir("/")
        logging.info(f"[{rank}] Model wrapper not found. Files available: {files}.")
    raise FileNotFoundError(f"Model {friendly_name} not found")


async def init_model(
    args: argparse.Namespace,
    engine_config: Optional[EngineConfig],
) -> None:
    """Initialize models based on parsed arguments."""
    if args.wan or args.wan21:
        model_name = "wan"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_wan21 import Wan21VideoGeneration
        wan_ckpt_dir = "/wan/Wan2.1-I2V-14B-480P"
        if args.wan_variation == "720p":
            wan_ckpt_dir = "/wan/Wan2.1-I2V-14B-720P"
        models[model_name] = Wan21VideoGeneration(
            ckpt_dir=wan_ckpt_dir,
            engine_config=engine_config,
        )

    if args.wanvae:
        model_name = "wanvae"
        await load_model_wrapper_file(rank, model_name)
        # TODO implement WanVideoVAEGeneration

    if args.wan22:
        model_name = "wan22"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_wan22 import Wan22VideoGeneration
        wan_ckpt_dir = "/wan/Wan2.2-I2V-A14B"
        models[model_name] = Wan22VideoGeneration(
            model_name=model_name,
            ckpt_dir=wan_ckpt_dir,
            engine_config=engine_config)

    if args.hunyuanframepack:
        model_name = "hunyuanframepack"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_hunyuanframepack import HunyuanFramepackGeneration
        models[model_name] = HunyuanFramepackGeneration(engine_config=engine_config)

    if args.hunyuanframepackf1:
        model_name = "hunyuanframepackf1"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_hunyuanframepackf1 import HunyuanFramepackF1Generation
        models[model_name] = HunyuanFramepackF1Generation(engine_config=engine_config)

    if args.hunyuanframepackvae:
        model_name = "hunyuanframepackvae"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_hunyuanframepackvae import HunyuanFramepackVAEGeneration
        models[model_name] = HunyuanFramepackVAEGeneration()

    if args.flux:
        model_name = "flux"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_flux import FluxGeneration
        models[model_name] = FluxGeneration(engine_config=engine_config)

    if args.fluxupscaler:
        model_name = "fluxupscaler"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_fluxupscaler import FluxUpscalerGeneration
        models[model_name] = FluxUpscalerGeneration(engine_config=engine_config)

    if args.fluxkontext:
        model_name = "fluxkontext"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_fluxkontext import FluxKontextGeneration
        models[model_name] = FluxKontextGeneration(engine_config=engine_config)

    if args.fluxkrea:
        model_name = "fluxkrea"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_fluxkrea import FluxKreaGeneration
        models[model_name] = FluxKreaGeneration(engine_config=engine_config)

    if args.flux2:
        model_name = "flux2"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_flux2 import Flux2Generation
        models[model_name] = Flux2Generation(engine_config=engine_config)

    if args.flux2klein:
        model_name = "flux2klein"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_flux2klein import Flux2KleinGeneration
        models[model_name] = Flux2KleinGeneration(engine_config=engine_config)

    if args.cogview:
        model_name = "cogview"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_cogview import CogViewGeneration
        models[model_name] = CogViewGeneration(engine_config=engine_config)

    if args.hidream:
        model_name = "hidream"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_hidream import HiDreamGeneration
        models[model_name] = HiDreamGeneration(engine_config=engine_config)

    if args.qwenimage:
        model_name = "qwenimage"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_qwenimage import QwenImageGeneration
        models[model_name] = QwenImageGeneration(engine_config=engine_config)

    if args.qwenimageedit:
        model_name = "qwenimageedit"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_qwenimageedit import QwenImageEditGeneration
        models[model_name] = QwenImageEditGeneration(engine_config=engine_config)

    if args.hunyuanimage:
        model_name = "hunyuanimage"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_hunyuanimage import HunyuanImageGeneration
        models[model_name] = HunyuanImageGeneration(engine_config=engine_config)

    if args.januspro:
        model_name = "januspro"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_januspro import JanusProGeneration
        models[model_name] = JanusProGeneration(engine_config=engine_config)

    if args.llamagen:
        model_name = "llamagen"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_llamagen import LlamaGenGeneration
        models[model_name] = LlamaGenGeneration(engine_config=engine_config)

    if args.kokoro:
        model_name = "kokoro"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_kokoro import KokoroGeneration
        models[model_name] = KokoroGeneration()

    if args.dia:
        model_name = "dia"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_dia import DiaGeneration
        models[model_name] = DiaGeneration()

    if args.xtts:
        model_name = "xtts"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_xtts import XTTSGeneration
        models[model_name] = XTTSGeneration()

    if args.vibevoice:
        model_name = "vibevoice"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_vibevoice import VibeVoiceGeneration
        models[model_name] = VibeVoiceGeneration()

    if args.thinksound:
        model_name = "thinksound"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_thinksound import ThinkSoundGeneration
        models[model_name] = ThinkSoundGeneration()

    if args.fantasytalking:
        model_name = "fantasytalking"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_fantasytalking import FantasyTalking
        models[model_name] = FantasyTalking(engine_config=engine_config)

    if args.hunyuanavatar:
        model_name = "hunyuanavatar"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_hunyuanavatar import HunyuanAvatarGeneration
        models[model_name] = HunyuanAvatarGeneration(engine_config=engine_config)

    if args.podcasttranscript:
        model_name = "podcasttranscript"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_podcasttranscript import PodcastTranscriptGenerator
        models[model_name] = PodcastTranscriptGenerator()

    if args.slidetranscript:
        model_name = "slidetranscript"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_slidetranscript import SlideTranscriptGenerator
        models[model_name] = SlideTranscriptGenerator()

    if args.yolo:
        model_name = "yolo"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_yolo import ImageCharacterExtractor
        models[model_name] = ImageCharacterExtractor()

    if args.bagel:
        model_name = "bagel"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_bagel import BagelGeneration
        models[model_name] = BagelGeneration()

    if args.imageresize:
        model_name = "imageresize"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_imageresize import ImageResize
        models[model_name] = ImageResize()

    if args.realesrgan:
        model_name = "realesrgan"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_realesrgan import RealESRGANGeneration
        models[model_name] = RealESRGANGeneration()

    if getattr(args, "4kagent"):  # Starting with number has issues
        model_name = "4kagent"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_4kagent import Upscale4KAgent
        models[model_name] = Upscale4KAgent()

    if args.ltx:
        model_name = "ltx"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_ltx import LTXVideoGeneration
        models[model_name] = LTXVideoGeneration()

    if args.longcatvideo:
        model_name = "longcatvideo"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_longcatvideo import LongCatVideoGeneration
        models[model_name] = LongCatVideoGeneration()

    if args.mock:
        model_name = "mock"
        await load_model_wrapper_file(rank, model_name)
        from wrapper_mock import MockGeneration
        models[model_name] = MockGeneration()

    for model_name, model in models.items():
        try:
            logging.info(f"[{rank}] Initializing model '{model_name}'...")
            await asyncio.to_thread(model.init)
        except Exception as ex:
            logging.error(f"[{rank}] Error during '{model_name}' initialization: {ex}.")
        try:
            # Run async warmup in separate process with proper event loop handling
            logging.info(f"[{rank}] Warming up model '{model_name}'...")
            await model.warmup()
        except torch.OutOfMemoryError as oom_err:
            logging.error(f"[{rank}] OOM during '{model_name}' warmup: {oom_err}.")
        except Exception as ex:
            err_msg = str(ex)
            logging.error(f"[{rank}] Error during '{model_name}' warmup: {err_msg}.")
            if "Model not initialized." not in err_msg:
                logging.error(traceback.format_exc())
        if rank == 0:
            logging.info(f"[{rank}] Model '{model_name}' loaded.")


async def run_httpserver(
    host: str = HOST,
    port: int = PORT,
    certfile: Optional[str] = None,
    keyfile: Optional[str] = None,
) -> None:
    """HTTP/HTTPS server runs in the main process (rank 0)."""
    config = Config()
    config.bind = [f"{host}:{port}"]

    config.accesslog = "-"

    # Increase max request body size to 128 MB (default is 16 MB)
    config.limit_max_request_size = 128 * 1024 * 1024  # type: ignore[attr-defined]
    app.config["MAX_CONTENT_LENGTH"] = 128 * 1024 * 1024

    # Configure for better concurrency (allow more concurrent connections)
    config.worker_connections = 64  # type: ignore[attr-defined]
    config.keep_alive_timeout = 5 * 60  # Keep connections alive for 1 minute
    config.graceful_timeout = 30  # Graceful shutdown timeout

    if certfile:
        config.certfile = certfile
    if keyfile:
        config.keyfile = keyfile

    scheme = "https" if certfile else "http"
    logging.info(f"[{rank}] Starting HTTP server on {scheme}://{host}:{port} with routes:")
    for rule in app.url_map.iter_rules():
        logging.info(f"[{rank}] - {rule.rule}")

    logging.debug(f"[{rank}] HTTP server config:")
    for key, value in app.config.items():
        logging.debug(f"{key}: {value}")

    try:
        await serve(app, config)
    except OSError as os_err:
        if os_err.errno == errno.EADDRINUSE:
            logging.error(f"{host}:{port} already in use.")
        raise


async def main() -> None:
    """
    Main entry point for the application.
    It starts:
    - The HTTP server in the main process (rank 0).
    - The NCCL workers in other processes (rank > 0).
    """
    try:
        setup_dist_environment()

        args, engine_config = arg_parsing()

        if not args.hunyuanavatar:
            init_dist_environment()  # Hunyuan-Avatar handles its own distributed environment

        if rank == 0:
            # HTTP server runs in the main process (rank 0)
            http_task = asyncio.create_task(run_httpserver(
                host=args.host,
                port=args.port,
                certfile=getattr(args, "certfile", None),
                keyfile=getattr(args, "keyfile", None),
            ))

            await init_model(args, engine_config)

            await wait_for_everybody()

            # Keep alive ping for NCCL workers
            try:
                SLEEP_TIME_PING_SECONDS = 60.0
                while http_task is not None and not http_task.done():
                    await asyncio.sleep(SLEEP_TIME_PING_SECONDS)
                    if world_size > 1 and not is_model_running() and not is_nccl_excluded_model():
                        logging.info(f"[{rank}] Sending ping to workers.")
                        args_ping = {
                            "task": "ping",
                            "args": {}
                        }
                        await send_task(args_ping)
            finally:
                if http_task is not None and not http_task.done():
                    http_task.cancel()
                    await http_task
        else:
            await init_model(args, engine_config)

            await wait_for_everybody()

            # Start the workers in the other GPUs
            await nccl_worker()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        logging.info(f"[{rank}] Exiting main process.")


if __name__ == "__main__":
    setup_logging(
        path=TMP_DIR,
        file_name="streamwise.log",
        level=logging.INFO)

    try:
        asyncio.run(main())
    except OSError as os_err:
        if os_err.errno == errno.EADDRINUSE:
            logging.error(f"Port already in use: {os_err}")
        else:
            raise
    except Exception as ex:
        logging.error(f"Fatal error in main: {ex}", exc_info=True)
