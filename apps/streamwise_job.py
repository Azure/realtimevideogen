"""
StreamWise job to generate a video.
"""
import os
import sys
import time
import logging
import aiofiles
import asyncio
import traceback

from contextlib import asynccontextmanager

from enum import Enum
from enum import StrEnum
from enum import auto

from typing import List
from typing import Dict
from typing import Optional
from typing import Any
from typing import Union
from typing import AsyncIterator
from typing import Mapping
from typing import Type
from typing import Callable

from datetime import datetime

from client import ServiceRequest

from video import VideoQuality
from video import QUALITY_TO_NUM_STEPS

from resolutions import ASPECT_RATIO
from resolutions import RESOLUTIONS


# Local relative imports
sys.path.append("..")  # noqa: E402
sys.path.append("../..")  # noqa: E402

from console_utils import setup_logging

from lmm_generator import LMMGenerator
from lmm_service_manager import LMMServiceManager

from client import ServiceError

from character import Characters

from console_utils import bytes_to_human

from media_utils import get_video_file_info

from k8s_utils import NoActiveContainerError
from k8s_utils import NoRunnableContainerError
from k8s_utils import ServiceNotFoundError


STATUS_EXPIRE_TIME_SECONDS = 10 * 60  # 10 minutes
MAX_LOG_TEXT = 100  # Max text length to log
SCENE_DEADLINE_INCREMENT_SECS = 5.0  # Seconds added per scene for deadline estimation


ExceptionHandler = Callable[[Exception], None]


class JobStatus(Enum):
    """Status of a StreamWise job."""
    CREATED = auto()
    PENDING = auto()
    STARTED = auto()
    RUNNING = auto()
    RETRYING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    EXPIRED = auto()
    UNKNOWN = auto()


class OutputMode(StrEnum):
    """Output type of the video."""
    AUDIO_ONLY = "audio_only"
    VIDEO_AUDIO_SYNCED = "video_audio_synced"
    VIDEO_AUDIO_UNSYNCED = "video_audio_unsynced"
    UNKNOWN = "unknown"


def get_job_id() -> str:
    """Generate a unique job ID based on the current timestamp: 20240605T153000123."""
    return datetime.now().strftime("%Y%m%dT%H%M%S%f")[:-3]


def is_job_id(job_id: str) -> bool:
    """Check if the given string is a valid job ID."""
    try:
        datetime.strptime(job_id, "%Y%m%dT%H%M%S%f")
        return True
    except ValueError:
        return False


def is_status_terminal(status: JobStatus) -> bool:
    """Check if the job status is a terminal status."""
    return status in (
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    )


def is_status_expired(last_modified_time: float) -> bool:
    """Check if the job status has expired based on the last modified time."""
    return time.time() - last_modified_time > STATUS_EXPIRE_TIME_SECONDS


class StreamWiseJob:
    """A generic StreamWise job to generate a video with images, audio, and text."""

    def __init__(
        self,
        app_name: str,
        job_id: str,
        service_manager: LMMServiceManager,
        config: Dict[str, Any] = {},
    ) -> None:
        self.app_name = app_name
        self.job_id = job_id
        self.job_path = f"/tmp/{self.app_name}/{self.job_id}"

        self.logger = self._get_logger()

        self.service_manager = service_manager
        self.gen = LMMGenerator(
            self.app_name,
            self.job_id,
            self.service_manager)

        """
        max_tokens: int = 5 * 1024,
        num_characters: int = 2,
        max_dialogues: int = 10,
        max_words_per_dialogue: int = 50,
        edit_image: bool = True,
        debug_image: bool = True,
        upscaling: bool = True,
        output_mode: str = "video_audio_synced",
        resolution: str = "high",
        speech_speed: float = 1.1,
        """
        self.config = config

        # Useful to setup deadlines for each request
        self.submission_time = time.time()

        # TODO self.get_config_str("aspect_ratio", ASPECT_RATIO)
        self.aspect_ratio = ASPECT_RATIO
        resolution_str = self.get_config_str("resolution", "high")
        if resolution_str == "adaptive":
            resolution_str = "medium"
        self.width, self.height = RESOLUTIONS[self.aspect_ratio][resolution_str]

        self.characters = Characters()
        self.transcript_scenes: List[Any] = []

        # Async task for the job processing, initialized to None until the job is started
        self.task: Optional[asyncio.Task] = None

    async def generate(
        self,
        job_config: Dict[str, Any],  # type: ignore
    ) -> None:
        """Generate the video based on the job configuration."""
        raise NotImplementedError("Subclasses must implement generate.")

    def _handle_service_not_found(self, ex: Exception) -> None:
        self.logger.error(f"Service not found for {self.job_id}: {ex}")

    def _handle_service_error(self, ex: Exception) -> None:
        self.logger.error(f"Service error for {self.job_id}: {ex}")

    def _handle_container_error(self, ex: Exception) -> None:
        self.logger.error(str(ex))

    def _handle_value_error(self, ex: Exception) -> None:
        self.logger.error(f"Value error for {self.job_id}: {ex}")

    def _handle_file_not_found(
        self,
        fnfe: FileNotFoundError
    ) -> None:
        self.logger.error(f"File not found for {self.job_id}: {fnfe}")

    def _handle_unknown_error(self, ex: Exception) -> None:
        self.logger.error(f"Error for {self.job_id} [{type(ex).__name__}]: {ex}")
        self.logger.error(traceback.format_exc())

    def _default_exception_handlers(
        self,
    ) -> dict[Type[Exception], ExceptionHandler]:
        """
        Base exception -> handler mapping.
        Subclasses may extend or override this.
        """
        return {
            ServiceNotFoundError: self._handle_service_not_found,
            ServiceError: self._handle_service_error,
            NoActiveContainerError: self._handle_container_error,
            NoRunnableContainerError: self._handle_container_error,
            ValueError: self._handle_value_error,
            FileNotFoundError: self._handle_file_not_found,
        }

    @asynccontextmanager
    async def job_status_handler(
        self,
        extra_handlers: Mapping[Type[Exception], ExceptionHandler] | None = None,
    ) -> AsyncIterator[None]:
        """
        Async context manager that:
        * Logs wall-clock execution time.
        * Saves job status transitions.
        * Catches and handles exceptions with appropriate handlers.
        """
        await self.save_status(JobStatus.STARTED)
        self.logger.info(f"Starting job {self.job_id}.")

        t0 = time.time()
        await self.save_status(JobStatus.RUNNING)
        try:
            yield

            # Successful case
            await self.save_status(JobStatus.COMPLETED)
            self.logger.info(f"Job {self.job_id} completed successfully.")
        except Exception as ex:
            await self.save_status(JobStatus.FAILED)

            handlers = self._default_exception_handlers()
            if extra_handlers:
                handlers = {
                    **handlers,
                    **extra_handlers
                }
            for exc_type, handler in handlers.items():
                if isinstance(ex, exc_type):
                    handler(ex)
                    raise
            self._handle_unknown_error(ex)
            raise
        finally:
            elapsed = time.time() - t0
            self.logger.info(f"Job {self.job_id} finished in {elapsed:.3f} seconds.")

    def get_submission_time(self) -> float:
        """Get the submission time of the job."""
        return self.submission_time

    def get_config_bool(self, config_name: str) -> bool:
        """Get a boolean configuration value."""
        if not self.config:
            return False
        return bool(self.config.get(config_name, False))

    def get_config_int(
        self,
        config_name: str,
        default_value: int = -1
    ) -> int:
        """Get an int configuration value."""
        if not self.config:
            return default_value
        return int(self.config.get(config_name, default_value))

    def get_config_float(
        self,
        config_name: str,
        default_value: float = -1.0
    ) -> float:
        """Get a float configuration value."""
        if not self.config:
            return default_value
        return float(self.config.get(config_name, default_value))

    def get_config_str(
        self,
        config_name: str,
        default_ret: str = ""
    ) -> str:
        """Get a string configuration value."""
        if not self.config:
            return default_ret
        return str(self.config.get(config_name, default_ret))

    def get_config_output_mode(self) -> OutputMode:
        output_mode_str = self.get_config_str("output_mode")
        if not output_mode_str:
            return OutputMode.UNKNOWN
        output_mode = OutputMode(output_mode_str)
        return output_mode

    def get_num_steps(self) -> int:
        """Get the number of steps for video generation."""
        quality_str = self.get_config_str("quality", VideoQuality.MEDIUM.value)
        num_steps = QUALITY_TO_NUM_STEPS[VideoQuality.MEDIUM.value]
        num_steps = QUALITY_TO_NUM_STEPS.get(quality_str, num_steps)  # Default to medium
        return num_steps

    async def close(self) -> None:
        """Close all clients."""
        await self.gen.stop()

    def _get_logger(self) -> logging.Logger:
        logger = setup_logging(
            path=self.job_path,
            file_name=f"job_{self.job_id}.log",
            level=logging.DEBUG,
            use_global=False)
        return logger

    def _log_video_info(
        self,
        prefix: str,
        video_content: Union[bytes, str],
    ) -> None:
        video_file_info = get_video_file_info(video_content)
        video_num_bytes = video_file_info["overall"]["num_bytes"]

        video_info = video_file_info["video"]
        video_fps = video_info["fps"]
        video_duration = video_info["duration_seconds"]
        video_num_frames = video_info["num_frames"]
        width, height = video_info["width"], video_info["height"]

        self.logger.info(
            f"{prefix} with "
            f"{video_duration:.3f} seconds, "
            f"{video_num_frames} frames, "
            f"{video_fps} FPS, "
            f"{bytes_to_human(video_num_bytes)}, and "
            f"{width}x{height} pixels.")

    def get_queued_requests(self) -> List[str]:
        return self.gen.get_queued_requests()

    def get_requests(self) -> Dict[str, ServiceRequest]:
        return self.gen.get_requests()

    async def save_status(
        self,
        status: JobStatus,
    ) -> None:
        """Save the status of a job asynchronously."""
        if not isinstance(status, JobStatus):
            raise ValueError(f"Invalid status type: {status}. Must be a JobStatus.")
        await aiofiles.os.makedirs(self.job_path, exist_ok=True)
        status_file = os.path.join(self.job_path, "status.txt")
        async with aiofiles.open(status_file, mode="a") as file:
            await file.write(f"{time.time()},{status.value}\n")

    async def get_status(self) -> JobStatus:
        """Get the latest status of a job asynchronously."""
        status_file = os.path.join(self.job_path, "status.txt")
        if not os.path.exists(status_file):
            return JobStatus.UNKNOWN
        async with aiofiles.open(status_file, mode="r") as file:
            lines = await file.readlines()
            if not lines:
                return JobStatus.UNKNOWN
            last_line = lines[-1].strip()
            try:
                _, status_value = last_line.split(",", 1)
                status = JobStatus(int(status_value))
                return status
            except (ValueError, IndexError) as e:
                self.logger.error(f"Error parsing status file: {e}")
                return JobStatus.UNKNOWN

    def _handle_scene_exception(self, scene_id: int, ex: Exception) -> None:
        """
        Handle exceptions during scene generation.
        Covers ServiceError, container errors (NoRunnableContainerError,
        NoActiveContainerError, ServiceNotFoundError), and any other exception.
        """
        if isinstance(ex, ServiceError):
            self.logger.error(f"[{scene_id}] Service error: {ex}")
        elif isinstance(ex, (NoRunnableContainerError, NoActiveContainerError, ServiceNotFoundError)):
            self.logger.error(f"[{scene_id}] {ex}")
        else:
            self.logger.error(f"[{scene_id}] Error ({type(ex).__name__}): {ex}")

    def get_scene_deadline(
        self,
        scene_index: int,
    ) -> float:
        """Get the deadline for a given scene."""
        return self.get_submission_time() + (scene_index * SCENE_DEADLINE_INCREMENT_SECS)
