"""
LMM Generator Client.
"""

import sys
import os
import re
import time
import json
import logging
import asyncio
import torch
import traceback

from PIL import Image
from io import BytesIO

from http import HTTPStatus

from aiohttp import TCPConnector
from aiohttp import ClientSession
from aiohttp import ClientTimeout

from json import JSONDecodeError

from openai import AsyncOpenAI

from enum import Enum

from typing import List
from typing import Optional
from typing import Tuple
from typing import Dict
from typing import Union
from typing import AsyncGenerator

from lmm_service_manager import LMMServiceManager

from resolutions import ASPECT_RATIO
from resolutions import RESOLUTIONS

from video import VideoQuality

from client import ServiceRequest
from client import ServiceRequestWorker
from client import ServiceError

from client_headers import JSON_HEADERS

from client_timeout import SERVICE_TIMEOUT
from client_timeout import SERVICE_MEDIUM_TIMEOUT
from client_timeout import SERVICE_LONG_TIMEOUT
from client_timeout import SERVICE_WARMUP_TIMEOUT

sys.path.append("..")  # noqa: E402

from console_utils import setup_logging
from console_utils import bytes_to_human

from file_utils import binary_to_base64

from image_utils import img_to_base64
from image_utils import base64_to_img
from media_utils import bytes_to_tensor
from media_utils import get_video_frames
from media_utils import video_frames_to_base64
from media_utils import chunk_audio_base64

from tts_utils import is_audio_base64_silence

from k8s_utils import NoActiveContainerError
from k8s_utils import NoRunnableContainerError


class TaskClass(str, Enum):
    """Class of tasks that can be performed by the services."""
    TXT2IMG = "txt2img"
    TXTIMG2IMG = "txtimg2img"
    IMG2IMG = "img2img"
    TXTIMG2VIDEO = "txtimg2video"
    TTS = "tts"
    VIDEOAUDIO2VIDEO = "videoaudio2video"
    UPSCALE = "upscale"


def get_service_name(
    task_class: TaskClass,
    quality: VideoQuality = VideoQuality.MEDIUM
) -> str:
    """
    Get the name of the service to run the task
    TODO get this from services.json.
    """
    if task_class == TaskClass.TXT2IMG:
        return "flux"
        # return "qwenimage"
        # return "hunyuanimage"
    if task_class == TaskClass.TXTIMG2IMG:
        return "fluxkontext"
        # return "qwenimageedit"
    if task_class == TaskClass.IMG2IMG:
        return "yolo"
    if task_class == TaskClass.TXTIMG2VIDEO:
        return "hunyuanframepackf1"
    if task_class == TaskClass.TTS:
        return "kokoro"
        # return "vibevoice"
    if task_class == TaskClass.VIDEOAUDIO2VIDEO:
        return "fantasytalking"
    if task_class == TaskClass.UPSCALE:
        return "realesrgan"
    raise ValueError(f"Unknown class '{task_class}'")


class LMMGenerator:
    """A client for managing requests to the LMM services and generating images, videos, and audio."""

    def __init__(
        self,
        app_name: str,
        job_id: str,
        service_manager: LMMServiceManager,
    ) -> None:
        """Initialize the LMMGenerator with a service manager and an aiohttp client session."""
        self.app_name = app_name
        self.job_id = job_id
        self.service_manager = service_manager

        connector = TCPConnector(
            limit=100,
            limit_per_host=10,
            use_dns_cache=True,
            force_close=True)
        self.session = ClientSession(
            connector=connector,
            timeout=SERVICE_LONG_TIMEOUT)

        self.requests: List[ServiceRequest] = []
        self.request_executor = ServiceRequestWorker(
            self.app_name,
            self.service_manager)
        self.request_executor_task = asyncio.create_task(self.request_executor.start())

        self.logger = self._get_logger()

    def _get_logger(self) -> logging.Logger:
        """Get the logger for the LMMGenerator."""
        logger = setup_logging(
            path=f"/tmp/{self.app_name}/{self.job_id}",
            file_name=f"{self.job_id}_lmm_generator.log",
            level=logging.INFO,
            use_global=False)
        return logger

    async def _submit_request(self, request: ServiceRequest) -> asyncio.Future:
        """Submit a service request to the request executor and return the future."""
        self.requests.append(request)
        future = await self.request_executor.submit_request(request)
        return future

    async def stop(self) -> None:
        """Stop the LMMGenerator and clean up resources."""
        if self.request_executor:
            await self.request_executor.stop()
            self.request_executor = None
        if self.session:
            await self.session.close()
            self.session = None

    def get_queued_requests(self) -> List[str]:
        """Get the list of queued request IDs from the request executor, sorted in ascending order."""
        request_ids = self.request_executor.get_queued_requests()
        request_ids.sort()
        return request_ids

    def get_requests(self) -> Dict[str, ServiceRequest]:
        """Get the dictionary of all requests currently being managed by the request executor."""
        return self.request_executor.get_requests()

    def get_service_url(
        self,
        service_name: str,
        exclude_busy: bool = True,
    ) -> str:
        """Get the URL of a service, optionally excluding busy containers."""
        return self.service_manager.get_service_url(
            service_name,
            exclude_busy=exclude_busy)

    def get_service_urls(
        self,
        service_name: str,
        exclude_busy: bool = True,
    ) -> List[str]:
        """Get the URLs of a service, optionally excluding busy containers."""
        return self.service_manager.get_service_urls(
            service_name,
            exclude_busy=exclude_busy)

    async def get_files(
        self,
        base_url: str,
        timeout: ClientTimeout = SERVICE_TIMEOUT,
    ) -> Optional[List[str]]:
        """Get the list of files available from the service at the given base URL."""
        t0 = time.time()
        url = f"{base_url}/files"
        try:
            # TODO use the request executor?
            async with self.session.get(url, timeout=timeout) as response:
                if response.status == HTTPStatus.OK and "application/json" in response.headers.get("Content-Type", ""):
                    response_json = await response.json()
                    if "files" in response_json:
                        return response_json["files"]
        except TimeoutError as timeout_err:
            total_time = time.time() - t0
            self.logger.error(f"Timeout ({total_time:.3f} > {timeout}) getting files from {url}: {timeout_err}")
            return None
        except Exception as ex:
            self.logger.error(f"Error getting files from {url}: {ex}")
            return None

        err_msg = await response.text() if response else "No response"
        status = response.status if response else "N/A"
        self.logger.error(f"Cannot get files from {url}: HTTP status {status} Message:{err_msg}")
        return None

    async def get_file(
        self,
        base_url: str,
        file_name: str,
        timeout: ClientTimeout = SERVICE_TIMEOUT
    ) -> Optional[bytes]:
        """Get a specific file from the service at the given base URL."""
        url = f"{base_url}/file/{file_name}"
        try:
            # TODO use the request executor?
            async with self.session.get(url, timeout=timeout) as response:
                if response.status == HTTPStatus.OK:
                    # return await response.read()
                    # return await response.content.read()
                    # return await response.content.read(-1)
                    chunks = []
                    MAX_CHUNK_SIZE = 8 * 1024
                    async for chunk in response.content.iter_chunked(MAX_CHUNK_SIZE):
                        chunks.append(chunk)
                    return b"".join(chunks)
                self.logger.error(f"Cannot get file {file_name} from {url}: HTTP status {response.status}")
        except Exception as ex:
            self.logger.error(f"Error getting file {file_name} from {url}: {ex}")
        return None

    def _assert_content_type(
        self,
        expected_content_type: str,
        content_type: str,
        msg: str,
        request: ServiceRequest
    ) -> None:
        """Assert that the content type of a response matches the expected content type."""
        if content_type != expected_content_type:
            err_msg = f"{msg} {content_type}"
            self.logger.error(f"{err_msg} for request {request.request_id}.")
            raise ValueError(err_msg)

    async def gen_image(
        self,
        prompt: str,
        neg_prompt: str = "",
        width: int = 1280,
        height: int = 720,
        steps: int = 25,
        task_id: Optional[str] = None,
        timeout: ClientTimeout = SERVICE_TIMEOUT,
        deadline: Optional[float] = None,
    ) -> Image.Image:
        """Generate an image from a text prompt using a txt2img Flux service."""
        service_name = get_service_name(TaskClass.TXT2IMG)
        payload_json = {
            "job_id": f"{self.job_id}_{task_id}",
            "prompt": prompt,
            "neg_prompt": neg_prompt,
            "width": width,
            "height": height,
            "sampling_steps": steps,
        }
        request = ServiceRequest(
            request_id=f"{self.job_id}_{task_id}_{service_name}",
            service_name=service_name,
            payload_json=payload_json,
            timeout=timeout,
            deadline=deadline,
        )
        try:
            self.logger.info(f"Submitting request {request.request_id} to generate image.")
            future = await self._submit_request(request)
            content_type, image_binary = await future
            self.logger.info(
                f"Received response for request {request.request_id} "
                f"with {bytes_to_human(len(image_binary))} and type {content_type}.")
            self._assert_content_type(
                "image/png", content_type,
                "Unexpected image type", request)
            image = await asyncio.to_thread(Image.open, BytesIO(image_binary))
            return image
        except Exception as ex:
            err_msg = "Error generating image"
            self.logger.error(f"{err_msg} for request {request.request_id}: {ex}")
            raise ServiceError(
                service_name=request.service_name,
                job_id=f"{self.job_id}_{task_id}",
                request_id=request.request_id,
                message=err_msg,
                url=self.get_service_url(request.service_name),
                status=None,  # Status is not available in this case
                response_body=str(ex))

    async def gen_edit_image(
        self,
        image: Image.Image,
        prompt: str,
        neg_prompt: str = "",
        width: int = 1280,
        height: int = 800,
        steps: int = 25,
        task_id: Optional[str] = None,
        timeout: ClientTimeout = SERVICE_TIMEOUT,
        deadline: Optional[float] = None,
    ) -> Image.Image:
        """Generate an edited image from an input image and a text prompt using the FluxKontext service."""
        service_name = get_service_name(TaskClass.TXTIMG2IMG)
        img_base64 = img_to_base64(image)
        payload_json = {
            "job_id": f"{self.job_id}_{task_id}",
            "img": img_base64,
            "prompt": prompt,
            "neg_prompt": neg_prompt,
            "width": width,
            "height": height,
            "sampling_steps": steps,
        }
        request = ServiceRequest(
            request_id=f"{self.job_id}_{task_id}_{service_name}",
            service_name=service_name,
            payload_json=payload_json,
            timeout=timeout,
            deadline=deadline,
        )
        try:
            self.logger.info(f"Submitting request {request.request_id} to edit image.")
            future = await self._submit_request(request)
            content_type, image_binary = await future
            self.logger.info(
                f"Received response for request {request.request_id} "
                f"with {bytes_to_human(len(image_binary))} and type {content_type}.")
            self._assert_content_type(
                "image/png", content_type,
                "Unexpected image type", request)
            image = await asyncio.to_thread(Image.open, BytesIO(image_binary))
            return image
        except Exception as ex:
            err_msg = "Error generating edited image"
            self.logger.error(f"{err_msg} for request {request.request_id}: {ex}")
            raise ServiceError(
                service_name=request.service_name,
                job_id=f"{self.job_id}_{task_id}",
                request_id=request.request_id,
                message=err_msg,
                url=self.get_service_url(request.service_name),
                status=None,  # Status is not available in this case
                response_body=str(ex))

    async def gen_extract_characters(
        self,
        image: Image.Image,
        num_characters: int = 2,
        zoom_factor: float = 1.5,
        task_id: Optional[str] = None,
        timeout: ClientTimeout = SERVICE_TIMEOUT,
        deadline: Optional[float] = None,
    ) -> List[Image.Image]:
        """Generate character images extracted from an input image using the YOLO service."""
        service_name = get_service_name(TaskClass.IMG2IMG)
        img_base64 = img_to_base64(image)
        payload_json = {
            "job_id": f"{self.job_id}_{task_id}",
            "img": img_base64,
            "num_characters": num_characters,
            "zoom_factor": zoom_factor,
        }
        request = ServiceRequest(
            request_id=f"{self.job_id}_{task_id}_{service_name}",
            service_name=service_name,
            payload_json=payload_json,
            timeout=timeout,
            deadline=deadline,
        )
        try:
            self.logger.info(f"Submitting request {request.request_id} to generate images from image.")
            future = await self._submit_request(request)
            content_type, response_binary = await future
            self.logger.info(
                f"Received response for request {request.request_id} "
                f"with {bytes_to_human(len(response_binary))} and type {content_type}.")
            self._assert_content_type(
                "application/json", content_type,
                "Unexpected response type", request)
            response_str = response_binary.decode("utf-8")
            response_json = json.loads(response_str)
            ret = []
            for img_base64 in response_json.values():
                img_character = base64_to_img(img_base64)
                ret.append(img_character)
            return ret
        except Exception as ex:
            err_msg = "Error generating images from image"
            self.logger.error(f"{err_msg} for request {request.request_id}: {ex}")
            raise ServiceError(
                service_name=request.service_name,
                job_id=f"{self.job_id}_{task_id}",
                request_id=request.request_id,
                message=err_msg,
                url=self.get_service_url(request.service_name),
                status=None,  # Status is not available in this case
                response_body=str(ex))

    async def get_video_last_latents(
        self,
        base_url: Optional[str] = None,
        task_id: Optional[str] = None
    ) -> Tuple[int, Optional[torch.Tensor]]:
        """Get the last latents for a video generation job from the service."""
        if not task_id:
            return -1, None
        if base_url is None:
            base_url = self.get_service_url("hunyuanframepackf1")
        file_descrs = await self.get_files(base_url)
        if not file_descrs:
            return -1, None

        last_file_name = None
        last_index = -1

        for file_desc in file_descrs:
            file_name = file_desc["name"]
            # File names of the type: 20250706T191739_latents_024.pt
            match = re.match(rf"^{self.job_id}_{task_id}_latents_(\d+)\.pt$", file_name)
            if match:
                iteration_index = int(match.group(1))
                if iteration_index > last_index:
                    last_file_name = file_name
                    last_index = iteration_index

        if last_file_name:
            file_content = await self.get_file(base_url, last_file_name)
            if file_content is None:
                self.logger.error(f"Cannot get file content for {last_file_name} from {base_url}.")
                return last_index, None
            # ix, [B, C, T, H, W]
            return last_index, bytes_to_tensor(file_content)

        return last_index, None

    async def gen_intermediate_video_frames(
        self,
        base_url: str,
        task_id: str,
        video_gen_request: ServiceRequest,
        poll_secs: float = 0.1
    ) -> AsyncGenerator[Image.Image, None]:
        """Generate intermediate video frames from the latents of a video generation job."""
        last_decoded_frame = 0

        async def decode_and_yield(
            current_index: int,
            current_latents: torch.Tensor
        ) -> AsyncGenerator[Image.Image, None]:
            nonlocal last_decoded_frame
            total_latents = current_latents.shape[2]
            if last_decoded_frame >= total_latents:
                return  # Nothing new to decode
            ix = 0 if last_decoded_frame == 0 else last_decoded_frame - 1  # Add an extra frame for VAE 1+4n
            new_latents = current_latents[:, :, ix:, :, :]  # [B, C, T, H, W]

            try:
                temp_video_binary = await self.gen_video_from_latents(
                    new_latents,
                    f"{self.job_id}_{task_id}_{current_index:03d}")
                if temp_video_binary is not None:
                    video_frames = await get_video_frames(temp_video_binary)
                    ix = 0 if last_decoded_frame == 0 else 1  # Skip the first frame for VAE 1+4n
                    for frame in video_frames[ix:]:
                        yield frame
            except NoActiveContainerError:
                self.logger.error("Cannot decode latents: No active container for VAE service.")
            except NoRunnableContainerError:
                self.logger.error("Cannot decode latents: No runnable container for VAE service.")
            except ServiceError as service_err:
                self.logger.error(f"Cannot decode latents: {service_err}.")
            except Exception as ex:
                self.logger.error(f"Cannot generate video from intermediate latents: {ex}.")
                traceback.print_exc()
            finally:
                last_decoded_frame = total_latents  # Mark as decoded

        # Keep polling until the video generation request is done
        while not video_gen_request.done():
            try:
                # current_latents: [B, C, T, H, W]
                current_index, current_latents = await self.get_video_last_latents(
                    base_url,
                    f"{self.job_id}_{task_id}")
                if current_latents is not None:
                    total_latents = current_latents.shape[2]
                    if last_decoded_frame < total_latents:
                        async for frame in decode_and_yield(current_index, current_latents):
                            yield frame
            except Exception as ex:
                self.logger.warning(
                    f"Cannot poll intermediate latents for {self.job_id}_{task_id}: {ex}.")

            await asyncio.sleep(poll_secs)

        # After the request is done, yield the remaining frames (if any)
        try:
            current_index, current_latents = await self.get_video_last_latents(
                base_url,
                f"{self.job_id}_{task_id}")
            if current_latents is None:
                return
            async for frame in decode_and_yield(current_index, current_latents):
                yield frame
        except Exception as ex:
            self.logger.warning(f"Final decode failed: {ex}.")

    async def gen_video(
        self,
        img: Image.Image,
        prompt: str,
        neg_prompt: str = "",
        width: int = 640,
        height: int = 400,
        num_frames: int = -1,
        video_seconds: float = -1,
        steps: int = 10,
        base_url: Optional[str] = None,
        task_id: Optional[str] = None,
        deadline: Optional[float] = None,
        timeout: ClientTimeout = SERVICE_LONG_TIMEOUT,
        wait_request: bool = True
    ) -> Union[bytes, ServiceRequest]:
        """Generate a video from an input image and a text prompt using the HunyuanFramePackF1 service."""
        service_name = get_service_name(TaskClass.TXTIMG2VIDEO)
        img_base64 = img_to_base64(img)
        payload_json = {
            "job_id": f"{self.job_id}_{task_id}",
            "img": img_base64,
            "prompt": prompt,
            "neg_prompt": neg_prompt,
            "width": width,
            "height": height,
            "sampling_steps": steps,
            "save_intermediate": f"{self.job_id}_{task_id}",  # Save the intermediate latents
            "output_type": "pil",
        }
        if num_frames > 0:
            payload_json["num_frames"] = num_frames
        if video_seconds > 0:
            payload_json["video_seconds"] = video_seconds
        request = ServiceRequest(
            request_id=f"{self.job_id}_{task_id}_{service_name}",
            service_name=service_name,
            payload_json=payload_json,
            base_url=base_url,
            timeout=timeout,
            deadline=deadline,
        )
        try:
            self.logger.info(f"Submitting request {request.request_id} to generate video.")
            future = await self._submit_request(request)
            if not wait_request:
                return request

            content_type, video_binary = await future
            self.logger.info(
                f"Received response for request {request.request_id} "
                f"with {bytes_to_human(len(video_binary))} and type {content_type}.")
            self._assert_content_type(
                "video/mp4", content_type,
                "Unexpected response type", request)
            return video_binary
        except Exception as ex:
            err_msg = "Error generating video"
            self.logger.error(f"{err_msg} for request {request.request_id}: {ex}")
            raise ServiceError(
                service_name=request.service_name,
                job_id=f"{self.job_id}_{task_id}",
                request_id=request.request_id,
                message=err_msg,
                url=self.get_service_url(request.service_name),
                status=None,  # Status is not available in this case
                response_body=str(ex))

    async def gen_image_upscale(
        self,
        image: Image.Image,
        width: int = 1280,
        height: int = 800,
        task_id: Optional[str] = None,
        timeout: ClientTimeout = SERVICE_LONG_TIMEOUT,
        deadline: Optional[float] = None,
    ) -> Image.Image:
        """Generate an upscaled image from an input image using RealESRGAN."""
        service_name = get_service_name(TaskClass.UPSCALE)
        payload_json = {
            "job_id": f"{self.job_id}_{task_id}",
            "img": img_to_base64(image),
            "width": width,
            "height": height,
        }
        request = ServiceRequest(
            request_id=f"{self.job_id}_{task_id}_{service_name}",
            service_name=service_name,
            payload_json=payload_json,
            timeout=timeout,
            deadline=deadline,
        )
        try:
            self.logger.info(f"Submitting request {request.request_id} to generate upscaled image.")
            future = await self._submit_request(request)
            content_type, image_binary = await future
            self.logger.info(
                f"Received response for request {request.request_id} "
                f"with {bytes_to_human(len(image_binary))} and type {content_type}.")
            self._assert_content_type(
                "image/png", content_type,
                "Unexpected image type", request)
            image = await asyncio.to_thread(Image.open, BytesIO(image_binary))
            return image
        except Exception as ex:
            err_msg = "Error generating video image"
            self.logger.error(f"{err_msg} for request {request.request_id}: {ex}")
            raise ServiceError(
                service_name=request.service_name,
                job_id=f"{self.job_id}_{task_id}",
                request_id=request.request_id,
                message=err_msg,
                url=self.get_service_url(request.service_name),
                status=None,  # Status is not available in this case
                response_body=str(ex))

    async def gen_video_upscale(
        self,
        video_binary: bytes,
        width: int = 1280,
        height: int = 800,
        task_id: Optional[str] = None,
        timeout: ClientTimeout = SERVICE_LONG_TIMEOUT,
        deadline: Optional[float] = None,
    ) -> bytes:
        """
        Generate an upscaled video from a base64 encoded video using RealESRGAN.
        TODO We could stream frames out.
        Returns video/mp4 video binary.
        """
        service_name = get_service_name(TaskClass.UPSCALE)
        video_base64 = binary_to_base64(video_binary)
        payload_json = {
            "job_id": f"{self.job_id}_{task_id}",
            "video": video_base64,
            "width": width,
            "height": height,
        }
        payload_len = len(json.dumps(payload_json).encode("utf-8"))
        request = ServiceRequest(
            request_id=f"{self.job_id}_{task_id}_{service_name}",
            service_name=service_name,
            payload_json=payload_json,
            path="realesrgan/video",
            timeout=timeout,
            deadline=deadline,
        )
        try:
            self.logger.info(
                f"Submitting request {request.request_id} with "
                f"{bytes_to_human(payload_len)} to generate upscaled video.")
            future = await self._submit_request(request)
            content_type, video_binary = await future
            self.logger.info(
                f"Received response for request {request.request_id} "
                f"with {bytes_to_human(len(video_binary))} and type {content_type}.")
            self._assert_content_type(
                "video/mp4", content_type,
                "Unexpected video type", request)
            return video_binary
        except Exception as ex:
            err_msg = "Error generating video upscaled"
            self.logger.error(f"{err_msg} for request {request.request_id}: {ex}")
            raise ServiceError(
                service_name=request.service_name,
                job_id=f"{self.job_id}_{task_id}",
                request_id=request.request_id,
                message=err_msg,
                url=self.get_service_url(request.service_name),
                status=None,  # Status is not available in this case
                response_body=str(ex))

    async def gen_video_from_latents(
        self,
        latents: torch.Tensor,
        task_id: Optional[str] = None,
        timeout: ClientTimeout = SERVICE_TIMEOUT,
        deadline: Optional[float] = None,
    ) -> bytes:
        """
        Generate a video from latents using the Hunyuan FramePack VAE service.
        VAE: latents to video.
        Returns video/mp4 video binary.
        """
        payload_buf = BytesIO()
        torch.save(latents, payload_buf)
        payload_bytes = payload_buf.getvalue()

        service_name = "hunyuanframepackvae"
        request = ServiceRequest(
            request_id=f"{self.job_id}_{task_id}_{service_name}",
            service_name=service_name,
            payload_bytes=payload_bytes,
            path=f"hunyuanframepack/vae/{self.job_id}_{task_id}",
            timeout=timeout,
            deadline=deadline,
        )
        try:
            self.logger.info(f"Submitting request {request.request_id} to generate video from latent.")
            future = await self._submit_request(request)
            content_type, video_binary = await future
            self.logger.info(
                f"Received response for request {request.request_id} "
                f"with {bytes_to_human(len(video_binary))} and type {content_type}.")
            self._assert_content_type(
                "video/mp4", content_type,
                "Unexpected video type", request)
            return video_binary
        except Exception as ex:
            err_msg = "Error generating video from latents"
            self.logger.error(f"{err_msg} for request {request.request_id}: {ex}")
            raise ServiceError(
                service_name=request.service_name,
                job_id=f"{self.job_id}_{task_id}",
                request_id=request.request_id,
                message=err_msg,
                url=self.get_service_url(request.service_name),
                status=None,  # Status is not available in this case
                response_body=str(ex))

    async def gen_video_audio_from_img(
        self,
        img: Image.Image,
        audio_base64: str,
        prompt: str,
        neg_prompt: str = "",
        width: int = 640,
        height: int = 400,
        steps: int = 25,
        end_percent: float = 0.9,
        task_id: Optional[str] = None,
        timeout: ClientTimeout = SERVICE_MEDIUM_TIMEOUT,
        deadline: Optional[float] = None,
    ) -> bytes:
        """Generate a video from an input image and audio using the FantasyTalking service."""
        assert isinstance(img, Image.Image), f"Image should be a PIL Image: {type(img)}"
        img_base64 = img_to_base64(img)
        assert type(img_base64) is str, f"Image should be a string: {type(img_base64)}"
        assert type(audio_base64) is str, f"Audio should be a string: {type(audio_base64)}"
        assert type(prompt) is str, f"Prompt should be a string: {type(prompt)}"
        assert type(neg_prompt) is str, f"Negative prompt should be a string: {type(neg_prompt)}"

        if is_audio_base64_silence(audio_base64):
            # TODO do not use fantasy talking then
            self.logger.warning("Audio is silence, generating video without audio.")

        service_name = get_service_name(TaskClass.VIDEOAUDIO2VIDEO)
        payload_json = {
            "job_id": f"{self.job_id}_{task_id}",
            "img": img_base64,
            "prompt": prompt,
            "neg_prompt": neg_prompt,
            "audio": audio_base64,
            "width": width,
            "height": height,
            "sampling_steps": steps,
            "end_percent": end_percent,
        }
        request = ServiceRequest(
            request_id=f"{self.job_id}_{task_id}_{service_name}",
            service_name=service_name,
            payload_json=payload_json,
            timeout=timeout,
            deadline=deadline,
        )
        try:
            self.logger.info(f"Submitting request {request.request_id} to generate video+audio.")
            future = await self._submit_request(request)
            content_type, video_binary = await future
            self.logger.info(
                f"Received response for request {request.request_id} "
                f"with {bytes_to_human(len(video_binary))} and type {content_type}.")
            self._assert_content_type(
                "video/mp4", content_type,
                "Unexpected video type", request)
            return video_binary
        except Exception as ex:
            err_msg = "Error generating video from image and audio"
            self.logger.error(f"{err_msg} for request {request.request_id}: {ex}")
            raise ServiceError(
                service_name=request.service_name,
                job_id=f"{self.job_id}_{task_id}",
                request_id=request.request_id,
                message=err_msg,
                url=self.get_service_url(request.service_name),
                status=None,  # Status is not available in this case
                response_body=str(ex))

    async def gen_video_audio_from_video(
        self,
        video: List[Image.Image],
        audio_base64: str,
        prompt: str,
        neg_prompt: str = "",
        width: int = 640,
        height: int = 400,
        cfg_scale: float = 7.0,  # for lipsync, to preserve input video
        audio_cfg_scale: float = 7.0,  # for lipsync, to be consistent with the audio
        steps: int = 10,
        end_percent: float = 0.9,
        task_id: Optional[str] = None,
        timeout: ClientTimeout = SERVICE_MEDIUM_TIMEOUT,
        deadline: Optional[float] = None,
    ) -> bytes:
        """Generate a video from an input video and audio using the FantasyTalking service."""
        assert isinstance(video, list), f"Video should be a list of PIL Images: {type(video)}"
        assert len(video) > 0, f"Video list should not be empty: {len(video)}"
        assert all(isinstance(frame, Image.Image)
                   for frame in video), f"All video frames should be PIL Images: {[type(frame) for frame in video]}"
        video_base64 = video_frames_to_base64(video, fps=30)
        assert type(video_base64) is str, f"Image should be a string: {type(video_base64)}"
        assert type(audio_base64) is str, f"Audio should be a string: {type(audio_base64)}"
        assert type(prompt) is str, f"Prompt should be a string: {type(prompt)}"
        assert type(neg_prompt) is str, f"Negative prompt should be a string: {type(neg_prompt)}"

        if is_audio_base64_silence(audio_base64):
            # TODO do not use fantasy talking then
            self.logger.warning("Audio is silence, generating video without audio.")

        service_name = get_service_name(TaskClass.VIDEOAUDIO2VIDEO)
        payload_json = {
            "job_id": f"{self.job_id}_{task_id}",
            "video": video_base64,
            "prompt": prompt,
            "neg_prompt": neg_prompt,
            "audio": audio_base64,
            "width": width,
            "height": height,
            "cfg_scale": cfg_scale,
            "audio_cfg_scale": audio_cfg_scale,
            "sampling_steps": steps,
            "end_percent": end_percent,
        }
        request = ServiceRequest(
            request_id=f"{self.job_id}_{task_id}_{service_name}",
            service_name=service_name,
            payload_json=payload_json,
            timeout=timeout,
            deadline=deadline,
        )
        try:
            self.logger.info(f"Submitting request {request.request_id} to generate video+audio from video.")
            future = await self._submit_request(request)
            content_type, video_binary = await future
            self.logger.info(
                f"Received response for request {request.request_id} "
                f"with {bytes_to_human(len(video_binary))} and type {content_type}.")
            self._assert_content_type(
                "video/mp4", content_type,
                "Unexpected video type", request)
            return video_binary
        except Exception as ex:
            err_msg = "Error generating video with audio from video"
            self.logger.error(f"{err_msg} for request {request.request_id}: {ex}")
            raise ServiceError(
                service_name=request.service_name,
                job_id=f"{self.job_id}_{task_id}",
                request_id=request.request_id,
                message=err_msg,
                url=self.get_service_url(request.service_name),
                status=None,  # Status is not available in this case
                response_body=str(ex))

    async def _get_service_url(self, service_name: str) -> str:
        """
        Get the service URL, retrying if no active container is found.
        """
        # TODO Implement this in client.py and use the request executor?
        retry = 0
        MAX_RETRIES = 3
        base_url = None
        while base_url is None:
            try:
                base_url = self.get_service_url(service_name)
            except NoActiveContainerError:
                if retry > MAX_RETRIES:
                    raise
                self.logger.error(f"No container for {service_name}, retry {retry}/{MAX_RETRIES}...")
                await asyncio.sleep(0.1 * (2 ** retry))
            retry += 1
        return base_url

    async def gen_text(
        self,
        messages: List[Dict],
        service_name: str = "gemma",
        llm_model: str = "google/gemma-3-27b-it",
        api_key: str = "n/a",
        max_tokens: int = 1024,
        extra_body: Optional[Dict] = None,
        task_id: Optional[str] = None,
    ) -> str:
        base_url = await self._get_service_url(service_name)
        url = f"{base_url}/v1"

        # vLLM OpenAI-compatible client
        async with AsyncOpenAI(base_url=url, api_key=api_key,) as llm_client:
            response = await llm_client.chat.completions.create(
                model=llm_model,
                messages=messages,
                max_tokens=max_tokens,
                extra_body=extra_body,
                # timeout=10.0,
                extra_headers={"X-Request-ID": f"{self.job_id}_{task_id}"},
                stream=False,
            )

            # Process LLM response
            self.logger.debug("LLM tokens:")
            self.logger.debug(f"  Prompt: {response.usage.prompt_tokens}")
            self.logger.debug(f"  Completion: {response.usage.completion_tokens}")
            self.logger.debug(f"  Total: {response.usage.total_tokens}")
            if response.usage.completion_tokens == max_tokens:
                self.logger.error(f"Completion hit max tokens limit ({response.usage.completion_tokens}/{max_tokens}).")

            if not response.choices:
                raise ValueError("No LLM response.")
            response_choice = response.choices[0]
            response_message = response_choice.message
            response_message_content = response_message.content
            if response_message_content:
                response_message_content = response_message_content.strip()
            return response_message_content

    async def gen_text_stream(
        self,
        messages: List[Dict],
        service_name: str = "gemma",
        llm_model: str = "google/gemma-3-27b-it",
        api_key: str = "n/a",
        max_tokens: int = 1024,
        extra_body: Optional[Dict] = None,
        task_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        base_url = await self._get_service_url(service_name)
        url = f"{base_url}/v1"

        # vLLM OpenAI-compatible client
        async with AsyncOpenAI(base_url=url, api_key=api_key,) as llm_client:
            response = await llm_client.chat.completions.create(
                model=llm_model,
                messages=messages,
                max_tokens=max_tokens,
                extra_body=extra_body,
                # timeout=10.0,
                extra_headers={"X-Request-ID": f"{self.job_id}_{task_id}"},
                stream=True,
            )
            async for chunk in response:
                choice = chunk.choices[0]
                delta = choice.delta.content
                yield delta

    async def gen_audio_transcript(
        self,
        audio_path: str,
        service_name: str = "whisper",
        whisper_model: str = "openai/whisper-large-v3",
        language: str = "en",
        api_key: str = "n/a",
        task_id: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Generate a transcript from an audio file.
        This may take noise and music and try to transcribe it; this need to be accounted for.
        Returns a tuple of (transcript_text, language_code).
        """
        if not os.path.isfile(audio_path):
            raise ValueError(f"Audio path is not a file: {audio_path}")

        base_url = await self._get_service_url(service_name)
        url = f"{base_url}/v1"

        # Whisper audio client
        async with AsyncOpenAI(base_url=url, api_key=api_key) as whisper_client:
            with open(audio_path, "rb") as file_audio:
                response = await whisper_client.audio.transcriptions.create(
                    model=whisper_model,
                    file=file_audio,
                    response_format="verbose_json",
                    language=language,
                    # timeout=10.0,
                    extra_headers={"X-Request-ID": f"{self.job_id}_{task_id}"},
                )
            transcript = response.text
            language_code = response.language or language
            return transcript, language_code

    async def gen_podcast_transcript(
        self,
        pdf_base64: Optional[str] = None,
        pdf_url: Optional[str] = None,
        max_tokens: int = 5 * 1024,
        num_characters: int = 2,
        style_prompt: Optional[str] = None,
        scene_prompt: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        max_dialogues: int = 10,
        max_words_per_dialogue: int = 50,
        streaming: bool = True,
        task_id: Optional[str] = None,
        timeout: ClientTimeout = SERVICE_LONG_TIMEOUT
    ) -> AsyncGenerator[dict, None]:
        """Generate a podcast transcript from a PDF document using the Gemma service."""
        # TODO move it to the podcast subclass
        # TODO add to list of requests self.requests.append(request)
        llm_model = "google/gemma-3-27b-it"
        base_url = self.get_service_url("gemma")
        llm_url = f"{base_url}/v1"
        multi_modal = True

        if not isinstance(pdf_base64, str):
            raise ValueError(f"pdf_base64 must be a string but is {type(pdf_base64)}")

        payload_json = {
            "pdf_url": pdf_url,
            "doc": pdf_base64,
            "llm_model": llm_model,
            "llm_url": llm_url,
            "multi_modal": multi_modal,
            "max_tokens": max_tokens,
            "num_characters": num_characters,
            "max_dialogues": max_dialogues,
            "max_words_per_dialogue": max_words_per_dialogue,
        }
        if style_prompt:
            payload_json["style_prompt"] = style_prompt
        if scene_prompt:
            payload_json["scene_prompt"] = scene_prompt
        if custom_prompt:
            payload_json["custom_prompt"] = custom_prompt

        service_name = "podcasttranscript"
        base_url = self.get_service_url(service_name)
        url = f"{base_url}/{service_name}"
        if streaming:
            url = f"{base_url}/{service_name}/stream"

        try:
            # TODO use request executor?
            async with self.session.post(
                url,
                json=payload_json,
                headers=JSON_HEADERS,
                timeout=timeout
            ) as response:
                if response.ok:
                    if streaming:
                        async for line in response.content:
                            try:
                                line_strip = line.decode("utf-8").strip()
                                if line_strip:
                                    line_json = json.loads(line_strip)
                                    yield line_json
                            except JSONDecodeError as json_error:
                                self.logger.error(f"JSON decode error: {json_error}. Line: {line_strip}")
                        return
                    else:
                        data_json = await response.json()
                        yield data_json
                        return

                # Handle error responses
                if response.headers.get("Content-Type") == "application/json":
                    data_json = await response.json()
                    error_message = data_json.get("error", "Unknown error")
                else:
                    error_message = await response.text()

                err_msg = "Error generating podcast transcript"
                self.logger.error(
                    f"{err_msg} for job_id={self.job_id} task_id={task_id} at {url}: "
                    f"HTTP status {response.status} Message: {error_message}")
                raise ServiceError(
                    service_name=service_name,
                    job_id=f"{self.job_id}_{task_id}",
                    message=err_msg,
                    url=url,
                    status=response.status,
                    response_body=error_message)
        except TimeoutError:
            err_msg = "Timeout generating podcast transcript"
            self.logger.error(f"{err_msg} for job_id={self.job_id} task_id={task_id} at {url}.")
            raise ServiceError(
                service_name=service_name,
                job_id=f"{self.job_id}_{task_id}",
                message=err_msg,
                url=url,
                status=HTTPStatus.REQUEST_TIMEOUT,
                response_body="Timeout generating podcast transcript.")
        except Exception as ex:
            err_msg = "Error generating podcast transcript"
            self.logger.error(f"{err_msg} for job_id={self.job_id} task_id={task_id} at {url} [{type(ex)}]: {ex}")
            self.logger.error(f"Trace: {traceback.format_exc()}")
            raise ServiceError(
                service_name=service_name,
                job_id=f"{self.job_id}_{task_id}",
                message=err_msg,
                url=url,
                response_body=str(ex))

    async def gen_slides_transcript(
        self,
        pptx_base64: str,
        task_id: Optional[str] = None,
        max_words_per_slide: Optional[int] = None,
        timeout: ClientTimeout = SERVICE_LONG_TIMEOUT,
        # deadline: Optional[float] = None,
    ) -> AsyncGenerator[dict, None]:
        llm_model = "google/gemma-3-27b-it"
        base_url = self.get_service_url("gemma")
        llm_url = f"{base_url}/v1"
        multi_modal = True

        if not isinstance(pptx_base64, str):
            raise ValueError(f"pptx_base64 must be a string but is {type(pptx_base64)}")

        payload_json = {
            "pptx": pptx_base64,
            "llm_model": llm_model,
            "llm_url": llm_url,
            "multi_modal": multi_modal,
            "max_words_per_slide": max_words_per_slide,
        }

        service_name = "slidetranscript"
        base_url = self.get_service_url(service_name)
        url = f"{base_url}/{service_name}/stream"

        try:
            # TODO use service request executor? with deadline
            async with self.session.post(url, json=payload_json, headers=JSON_HEADERS, timeout=timeout) as response:
                if response.ok:
                    async for line in response.content:
                        try:
                            line_strip = line.decode("utf-8").strip()
                            if line_strip:
                                line_json = json.loads(line_strip)
                                yield line_json
                        except JSONDecodeError as json_error:
                            self.logger.error(f"JSON decode error: {json_error}. Line: {line_strip}")
                    return

                # Handle error responses
                if response.headers.get("Content-Type") == "application/json":
                    data_json = await response.json()
                    error_message = data_json.get("error", "Unknown error")
                else:
                    error_message = await response.text()

                err_msg = "Error generating slides transcript"
                self.logger.error(
                    f"{err_msg} for job_id={self.job_id} task_id={task_id} at {url}: "
                    f"HTTP status {response.status} Message: {error_message}")
                raise ServiceError(
                    service_name=service_name,
                    job_id=f"{self.job_id}_{task_id}",
                    message=err_msg,
                    url=url,
                    status=response.status,
                    response_body=error_message)
        except TimeoutError:
            err_msg = "Timeout generating slides transcript"
            self.logger.error(f"{err_msg} for job_id={self.job_id} task_id={task_id} at {url}.")
            raise ServiceError(
                service_name=service_name,
                job_id=f"{self.job_id}_{task_id}",
                message=err_msg,
                url=url,
                status=HTTPStatus.REQUEST_TIMEOUT,
                response_body="Timeout generating slides transcript.")
        except Exception as ex:
            err_msg = "Error generating slides transcript"
            self.logger.error(f"{err_msg} for job_id={self.job_id} task_id={task_id} at {url} [{type(ex)}]: {ex}")
            raise ServiceError(
                service_name=service_name,
                job_id=f"{self.job_id}_{task_id}",
                message=err_msg,
                url=url,
                response_body=str(ex))

    async def gen_audio(
        self,
        text: str,
        voice: str = "af_heart",  # Default voice
        speed: float = 1.0,
        lang_code: str = "a",  # American English
        task_id: Optional[str] = None,
        timeout: ClientTimeout = SERVICE_TIMEOUT,
        deadline: Optional[float] = None,
    ) -> str:
        """
        Generate audio from text using the TTS service.
        Returns base64 encoded audio string.
        """
        service_name = get_service_name(TaskClass.TTS)
        payload_json = {
            "job_id": f"{self.job_id}_{task_id}",
            "text": text,
            "voice": voice,
            "speed": speed,
            "lang_code": lang_code,
        }
        request = ServiceRequest(
            request_id=f"{self.job_id}_{task_id}_{service_name}",
            service_name=service_name,
            payload_json=payload_json,
            timeout=timeout,
            deadline=deadline,
        )
        try:
            self.logger.info(f"Submitting request {request.request_id} to generate audio.")
            future = await self._submit_request(request)
            content_type, audio_binary = await future
            self.logger.info(
                f"Received response for request {request.request_id} "
                f"with {bytes_to_human(len(audio_binary))} and type {content_type}.")
            self._assert_content_type(
                "audio/wav", content_type,
                "Unexpected audio type", request)
            audio_base64 = binary_to_base64(audio_binary)
            return audio_base64
        except Exception as ex:
            err_msg = "Error generating audio"
            self.logger.error(f"{err_msg} for request {request.request_id}: {ex}")
            raise ServiceError(
                service_name=request.service_name,
                job_id=f"{self.job_id}_{task_id}",
                request_id=request.request_id,
                message=err_msg,
                url=self.get_service_url(request.service_name),
                response_body=str(ex))

    async def gen_clone_audio(
        self,
        text: str,
        voice_sample: str,
        lang_code: str = "a",  # American English
        task_id: Optional[str] = None,
        timeout: ClientTimeout = SERVICE_TIMEOUT,
        deadline: Optional[float] = None,
    ) -> str:
        """
        Generate audio from text using voice cloning via the VibeVoice service.
        Returns base64 encoded audio string.

        Args:
            text: The text to synthesise.
            voice_sample: Base64-encoded WAV audio used as the reference speaker for cloning.
            lang_code: Target language code.
            task_id: Optional task identifier for logging and request tracking.
            timeout: aiohttp client timeout for the service request.
            deadline: Optional absolute time (epoch seconds) by which the result must arrive.
        """
        service_name = "vibevoice"
        payload_json: Dict[str, Union[str, float]] = {
            "job_id": f"{self.job_id}_{task_id}",
            "text": text,
            "lang_code": lang_code,
            "voice_sample": voice_sample,
        }
        request = ServiceRequest(
            request_id=f"{self.job_id}_{task_id}_{service_name}",
            service_name=service_name,
            payload_json=payload_json,
            timeout=timeout,
            deadline=deadline,
        )
        try:
            self.logger.info(f"Submitting request {request.request_id} to clone audio.")
            future = await self._submit_request(request)
            content_type, audio_binary = await future
            self.logger.info(
                f"Received response for request {request.request_id} "
                f"with {bytes_to_human(len(audio_binary))} and type {content_type}.")
            self._assert_content_type(
                "audio/wav", content_type,
                "Unexpected audio type", request)
            audio_base64 = binary_to_base64(audio_binary)
            return audio_base64
        except Exception as ex:
            err_msg = "Error cloning audio"
            self.logger.error(f"{err_msg} for request {request.request_id}: {ex}")
            raise ServiceError(
                service_name=request.service_name,
                job_id=f"{self.job_id}_{task_id}",
                request_id=request.request_id,
                message=err_msg,
                url=self.get_service_url(request.service_name),
                response_body=str(ex))

    async def warmup_services(
        self,
        job_id: str = "StreamWiseWarmup",
    ) -> None:
        """Warmup the services used by the LMMGenerator to reduce latency for the first requests."""
        try:
            # TODO warmup also Hunyuan FramePack VAE
            # TODO warmup all replicas

            # Skipping for now as it is too expensive and doesn't add much.
            async def consume_podcast_transcript() -> None:
                async for line in self.gen_podcast_transcript(
                    pdf_url="https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
                    num_characters=2,
                    max_dialogues=3,
                    max_words_per_dialogue=5,
                    streaming=True,
                    task_id="warmup",
                    timeout=SERVICE_WARMUP_TIMEOUT
                ):
                    logging.debug(f"Podcast transcript line: {line}")

            podcast_transcript_task = asyncio.create_task(consume_podcast_transcript())
            width, height = RESOLUTIONS[ASPECT_RATIO]["High"]
            image_task = asyncio.create_task(
                self.gen_image(
                    "warmup image",
                    neg_prompt="blue",
                    steps=5,
                    width=width,
                    height=height,
                    task_id="warmup",
                    timeout=SERVICE_WARMUP_TIMEOUT))
            audio_task = asyncio.create_task(
                self.gen_audio(
                    "warmup text saying something",
                    task_id="warmup",
                    timeout=SERVICE_WARMUP_TIMEOUT))

            warmup_image = await image_task

            extract_character_task = asyncio.create_task(
                self.gen_extract_characters(
                    warmup_image,
                    task_id="warmup"))

            RESOLUTION_LOW = RESOLUTIONS[ASPECT_RATIO]["Low"]
            video_task = asyncio.create_task(
                self.gen_video(
                    warmup_image,
                    "warmup video",
                    width=RESOLUTION_LOW[0],
                    height=RESOLUTION_LOW[1],
                    video_seconds=0.5,
                    steps=2,
                    task_id="warmup",
                    timeout=SERVICE_WARMUP_TIMEOUT))

            RESOLUTION_HIGH = RESOLUTIONS[ASPECT_RATIO]["High"]
            upscale_image_task = asyncio.create_task(
                self.gen_image_upscale(
                    warmup_image,
                    width=RESOLUTION_HIGH[0],
                    height=RESOLUTION_HIGH[1],
                    task_id="warmup",
                    timeout=SERVICE_WARMUP_TIMEOUT))

            warmup_audio = await audio_task
            short_audio = chunk_audio_base64(warmup_audio, 0.0, 0.5)
            RESOLUTION_MEDIUM = RESOLUTIONS[ASPECT_RATIO]["Medium"]
            video_audio_task = asyncio.create_task(
                self.gen_video_audio_from_img(
                    warmup_image,
                    short_audio,
                    "warmup video",
                    width=RESOLUTION_MEDIUM[0],
                    height=RESOLUTION_MEDIUM[1],
                    steps=2,
                    task_id="warmup",
                    timeout=SERVICE_WARMUP_TIMEOUT))

            results = await asyncio.gather(
                podcast_transcript_task,  # TODO move to separate class
                extract_character_task,
                upscale_image_task,
                video_task,
                video_audio_task,
                return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logging.error(f"Error during warmup: {result}")
        except ServiceError as service_error:
            logging.error(f"Error during warmup: {service_error}")
        except Exception as ex:
            logging.error(f"Error during warmup: {ex}.")
            # traceback.print_exc()
