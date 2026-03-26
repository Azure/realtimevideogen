"""
Generate a video in chunks of a maximum length.
It uses Hunyuan FramePack to generate a long sketch video at low resolution,
then splits the audio into subvideos based on silences, and for each subvideo,
it generates video+audio at medium resolution using Fantasy Talking.

Pipeline:
1. Split audio into silence-aligned subvideos
2. Generate a long low-res sketch video (Hunyuan FramePack)
3. Stream sketch frames and schedule subvideo generation
4. Generate video+audio chunks (Fantasy Talking)
5. Concatenate results
"""
import sys
import os
import asyncio
import aiofiles
import logging

from typing import List
from typing import Dict
from typing import Optional
from typing import Union
from typing import Tuple
from typing import cast
import math

from PIL import Image

# Local relative imports
sys.path.append("..")  # noqa: E402

from lmm_generator import LMMGenerator

from client import ServiceRequest

from video import get_num_video_frames_from_duration
from video import MAX_FT_DURATION_SECS
from video import FANTASYTALKING_FPS
from video import HUNYUANFRAMEPACK_FPS
from video import HUNYUANFRAMEPACK_VAE_T
from video import VAE_T

from tts_utils import get_audio_chunks_by_silences

from console_utils import bytes_to_human

from file_utils import read_file_base64
from file_utils import save_base64_as_binary

from media_utils import get_video_file_info
from media_utils import get_audio_duration
from media_utils import get_video_with_text
from media_utils import get_video_frames_at_fps
from media_utils import get_font_size
from media_utils import split_text_lines
from media_utils import add_text_to_frame
from media_utils import chunk_audio_base64
from media_utils import concatenate_videos

MAX_IMG_LINE_CHARS = 50


class SubVideoInfo:
    """Subvideo information."""

    def __init__(
        self,
        start_seconds: float,
        end_seconds: float,
    ) -> None:
        self.start_seconds = start_seconds
        self.end_seconds = end_seconds

    def get_seconds(self) -> float:
        """Get subvideo duration in seconds."""
        return self.end_seconds - self.start_seconds

    def get_start_frame(
        self,
        fps: float,
    ) -> int:
        """Get start frame."""
        return int(math.ceil(self.start_seconds * fps))

    def get_end_frame(
        self,
        fps: float,
    ) -> int:
        """Get end frame."""
        return int(math.ceil(self.end_seconds * fps))

    def get_frames(self, fps: float) -> Tuple[int, int]:
        """Get start and end frames."""
        start_frames = self.get_start_frame(fps)
        end_frames = self.get_end_frame(fps)
        return start_frames, end_frames

    def __str__(self) -> str:
        return f"SubVideoInfo({self.start_seconds:.3f}-{self.end_seconds:.3f}s)"


class GenVideoChunked:
    """
    Generate a video in chunks of a maximum length.
    """

    def __init__(
        self,
        video_id: int,
        gen: LMMGenerator,
        job_path: str,
        logger: logging.Logger,
    ) -> None:
        """Initialize the video generator."""
        self.video_id = video_id
        self.gen = gen
        self.job_id = gen.job_id
        self.job_path = job_path
        self.logger = logger

    async def _prepare_audio(
        self,
        audio_path: str,
        max_duration: int = MAX_FT_DURATION_SECS,
    ) -> Tuple[str, float, List[SubVideoInfo]]:
        """Prepare audio: load, get duration, split into subvideos. """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio_b64 = await read_file_base64(audio_path)
        duration = get_audio_duration(audio_b64)

        if duration < max_duration:
            raise ValueError(f"Audio too short for chunked generation ({duration}/{max_duration}).")

        audio_splits = get_audio_chunks_by_silences(
            audio_path,
            max_duration,
            chunk_alignment_seconds=1.0 / FANTASYTALKING_FPS)

        if len(audio_splits) > math.ceil(duration / max_duration):
            self.logger.warning(
                f"[{self.video_id}] Too many subvideos ({len(audio_splits)}) for audio with "
                f"{duration:.3f} seconds and max {max_duration:.3f}.")

        subvideos = [
            SubVideoInfo(start_secs, end_secs)
            for start_secs, end_secs in audio_splits
        ]

        for subvideo_id, subvideo in enumerate(subvideos):
            self.logger.info(
                f"[{self.video_id}.{subvideo_id}] "
                f"{subvideo.start_seconds:.3f}-{subvideo.end_seconds:.3f} "
                f"({subvideo.get_seconds():.3f}s)")
            if subvideo.get_seconds() > max_duration:
                self.logger.error(
                    f"[{self.video_id}.{subvideo_id}] Subvideo too long: "
                    f"{subvideo.get_seconds():.3f}s > {max_duration:.3f}s.")
            elif subvideo.get_seconds() < 0.5:
                self.logger.warning(
                    f"[{self.video_id}.{subvideo_id}] Subvideo too short: "
                    f"{subvideo.get_seconds():.3f}s < 0.5s.")

        return audio_b64, duration, subvideos

    async def gen_video_chunked(
        self,
        audio_path: str,
        image: Image.Image,
        prompt: str,
        neg_prompt: str,
        width: int,
        height: int,
        num_steps: int,
        subvideo_duration_max: int = MAX_FT_DURATION_SECS,
        upscaling: bool = False,
        debug: bool = False,
        deadline: Optional[float] = None,
    ) -> bytes:
        """
        Generate a video with audio and video starting with image at high resolution.
        1. Split audio into subvideos based on silences.
        2. Generate sketch at low resolution with Hunyuan FramePack.
        3. Generate video+audio at medium resolution with Fantasy Talking.
        Returns video synced with the audio in binary.
        """
        audio_base64, audio_duration, subvideos = await self._prepare_audio(
            audio_path,
            subvideo_duration_max
        )

        sketch_request, sketch_num_frames = await self._gen_sketch(
            duration=audio_duration,
            image=image,
            prompt=prompt,
            neg_prompt=neg_prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            deadline=deadline,
        )

        subvideo_tasks = await self._schedule_subvideos(
            sketch_request=sketch_request,
            sketch_num_frames=sketch_num_frames,
            subvideos=subvideos,
            audio_base64=audio_base64,
            width=width,
            height=height,
            num_steps=num_steps,
            prompt=prompt,
            neg_prompt=neg_prompt,
            upscaling=upscaling,
            debug=debug,
            deadline=deadline,
        )

        await self._save_sketch(sketch_request)

        subvideo_binaries = await self._collect_subvideos(
            subvideo_tasks,
            subvideos,
            width,
            height,
        )

        if upscaling:
            subvideo_binaries = await self._upscale_subvideos(subvideo_binaries)

        video_binary = await self._concatenate_subvideos(subvideo_binaries)

        return video_binary

    async def _gen_sketch(
        self,
        duration: float,
        image: Image.Image,
        prompt: str,
        neg_prompt: str,
        width: int,
        height: int,
        num_steps: int,
        deadline: Optional[float] = None,
    ) -> Tuple[ServiceRequest, int]:
        """Start long sketch video (no audio) generation in the background."""
        sketch_num_frames = get_num_video_frames_from_duration(
            duration,
            HUNYUANFRAMEPACK_FPS,
            HUNYUANFRAMEPACK_VAE_T)

        sketch_request = await self.gen.gen_video(
            image,
            prompt,
            neg_prompt,
            width=width,
            height=height,
            num_frames=sketch_num_frames,
            steps=num_steps // 2,  # TODO Less steps for sketch video
            task_id=f"{self.video_id:03d}_sketch",
            wait_request=False,
            deadline=deadline,
        )

        self.logger.info(
            f"[{self.video_id}] Generating long sketch video with {sketch_num_frames} frames, "
            f"{duration:.3f} seconds, "
            f"{HUNYUANFRAMEPACK_FPS} FPS, and "
            f"{width}x{height} pixels.")

        return sketch_request, sketch_num_frames

    async def _save_sketch(
        self,
        sketch_request: ServiceRequest,
    ) -> str:
        """Save full sketch video (no audio)."""
        content_type, video_binary = await sketch_request.future
        if not video_binary:
            raise ValueError("Cannot generate sketch video.")
        if content_type != "video/mp4":
            raise ValueError(f"Invalid content type for video: {content_type}.")

        self._log_video_info(f"[{self.video_id}] Video sketch", video_binary)
        video_path = f"{self.job_path}/{self.video_id:03d}_chunks_sketch.mp4"
        async with aiofiles.open(video_path, "wb") as file:
            await file.write(video_binary)
        return video_path

    async def _schedule_subvideos(
        self,
        sketch_request: ServiceRequest,
        sketch_num_frames: int,
        subvideos: List[SubVideoInfo],
        audio_base64: str,
        width: int,
        height: int,
        num_steps: int,
        prompt: str,
        neg_prompt: str,
        upscaling: bool,
        debug: bool,
        deadline: Optional[float],
    ) -> List[Optional[asyncio.Task]]:
        """Schedule subvideo generation while sketch video is being generated."""
        tasks: List[Optional[asyncio.Task]] = [None] * len(subvideos)

        sketch_frames: List[Image.Image] = []
        base_url = sketch_request.get_base_request_url()
        subvideo_id = 0

        # Get frames while the sketch request is running
        video_gen_request_done = False
        while len(sketch_frames) < sketch_num_frames and not video_gen_request_done:
            if sketch_request.done():
                video_gen_request_done = True
            elif not sketch_request.is_running() or not sketch_request.url:
                RETRY_SLEEP_SECONDS = 1.0
                await asyncio.sleep(RETRY_SLEEP_SECONDS)  # Wait for it to be running before checking again
                continue

            # Get intermediate frames (while long sketch video generation is running)
            async for frame in self.gen.gen_intermediate_video_frames(
                base_url,
                task_id=f"{self.video_id:03d}_sketch",
                video_gen_request=sketch_request,
            ):
                sketch_frames.append(frame)

                # Schedule as many subvideos as possible with available frames
                while self._can_schedule_subvideo(subvideos, subvideo_id, sketch_frames):
                    # Enough frames for the subvideo, generate video+audio
                    subvideo = subvideos[subvideo_id]
                    task = await self._gen_subvideo(
                        subvideo_id=subvideo_id,
                        subvideo_info=subvideo,
                        width=width,
                        height=height,
                        num_steps=num_steps,
                        video_frames=sketch_frames,
                        audio_base64=audio_base64,
                        video_prompt=prompt,
                        video_neg_prompt=neg_prompt,
                        upscaling=upscaling,
                        debug=debug,
                        deadline=deadline,
                    )
                    if task is None:
                        self.logger.error(f"[{self.video_id}.{subvideo_id}] Cannot generate subvideo task.")
                    else:
                        tasks[subvideo_id] = task
                    subvideo_id += 1

                if len(sketch_frames) >= sketch_num_frames:
                    break  # All frames received

        # Schedule the remaining subvideos if any
        while subvideo_id < len(subvideos):
            subvideo = subvideos[subvideo_id]
            task = await self._gen_subvideo(
                subvideo_id=subvideo_id,
                subvideo_info=subvideo,
                width=width,
                height=height,
                num_steps=num_steps,
                video_frames=sketch_frames,
                audio_base64=audio_base64,
                video_prompt=prompt,
                video_neg_prompt=neg_prompt,
                upscaling=upscaling,
                debug=debug,
                deadline=deadline,
            )
            if task is None:
                self.logger.error(f"[{self.video_id}.{subvideo_id}] Cannot generate subvideo task.")
            else:
                tasks[subvideo_id] = task
            subvideo_id += 1

        # Final checks
        self.logger.info(
            f"[{self.video_id}] Got {len(sketch_frames)}/{sketch_num_frames} streamed video frames.")

        if len(sketch_frames) < sketch_num_frames:
            self.logger.warning(
                f"[{self.video_id}] Not enough frames ({len(sketch_frames)} < {sketch_num_frames}).")

        return tasks

    def _can_schedule_subvideo(
        self,
        subvideos: List[SubVideoInfo],
        idx: int,
        frames: List[Image.Image],
    ) -> bool:
        """Check if we can schedule the next subvideo given available frames."""
        if idx >= len(subvideos):
            return False
        subvideo = subvideos[idx]
        needed = subvideo.get_end_frame(HUNYUANFRAMEPACK_FPS) + VAE_T
        return len(frames) >= needed

    async def _collect_subvideos(
        self,
        subvideo_tasks: List[Optional[asyncio.Task]],
        subvideos: List[SubVideoInfo],
        width: int,
        height: int,
    ) -> List[bytes]:
        """Collecting video+audio subvideos."""
        self.logger.info(f"[{self.video_id}] Generating {len(subvideo_tasks)} video+audio subvideos...")

        subvideo_binaries: List[bytes] = [b""] * len(subvideos)

        subvideo_ids = list(range(len(subvideo_tasks)))
        gather_tasks: List[asyncio.Task] = []
        subvideo_to_task: Dict[int, int] = {}
        for subvideo_id, subvideo_task in enumerate(subvideo_tasks):
            if subvideo_task is not None:
                gather_task_id = len(gather_tasks)
                subvideo_to_task[subvideo_id] = gather_task_id
                gather_tasks.append(subvideo_task)

        gather_results = await asyncio.gather(
            *gather_tasks,
            return_exceptions=True
        )
        self.logger.info(f"[{self.video_id}] Generated {len(gather_tasks)} video+audio subvideos.")

        if not gather_results:
            raise ValueError(f"No subvideos generated for video {self.video_id}.")

        for subvideo_id in subvideo_ids:
            subvideo = subvideos[subvideo_id]
            duration_seconds = subvideo.get_seconds()
            task_id = subvideo_to_task.get(subvideo_id)

            subvideo_binary = None
            if task_id is not None:
                subvideo_binary = gather_results[task_id]

            if isinstance(subvideo_binary, bytes):
                self._log_video_info(f"[{self.video_id}.{subvideo_id}] Generated video", subvideo_binary)
            else:
                # Error case -> replace with static error video
                # TODO add audio
                err_msg = "No video generated"
                if subvideo_binary is not None:
                    err_msg = str(subvideo_binary)  # This was an exception
                self.logger.error(
                    f"[{self.video_id}.{subvideo_id}] {err_msg}. "
                    f"Adding error video with {duration_seconds:.3f} seconds and {width}x{height} pixels.")
                subvideo_binary = await self._gen_error_subvideo(
                    subvideo_id=subvideo_id,
                    width=width,
                    height=height,
                    duration_seconds=duration_seconds,
                    fps=FANTASYTALKING_FPS,
                    err_msg=err_msg)

            subvideo_binaries[subvideo_id] = subvideo_binary

            video_path = f"{self.job_path}/{self.video_id:03d}_{subvideo_id:03d}_chunks.mp4"
            async with aiofiles.open(video_path, "wb") as file:
                await file.write(subvideo_binary)
        return subvideo_binaries

    async def _gen_error_subvideo(
        self,
        subvideo_id: int,
        width: int,
        height: int,
        duration_seconds: float,
        fps: float,
        err_msg: str,
    ) -> bytes:
        frame_text = "Cannot generate video\n"
        frame_text += f"Subvideo {self.video_id:03d}.{subvideo_id:03d}\n"
        frame_text += "\n".join(split_text_lines(err_msg, MAX_IMG_LINE_CHARS))
        font_size = get_font_size(width, height)
        subvideo_binary = await get_video_with_text(
            width=width,
            height=height,
            text=frame_text,
            font_size=font_size,
            fps=fps,
            duration_seconds=duration_seconds)
        return subvideo_binary

    async def _upscale_subvideos(
        self,
        subvideo_binaries: List[bytes],
    ) -> List[bytes]:
        """Upscale subvideos."""
        # TODO
        # await self.gen.gen_video_upscale()
        return subvideo_binaries

    async def _concatenate_subvideos(
        self,
        subvideo_binaries: List[bytes],
    ) -> bytes:
        """Concatenate subvideos into final video."""
        self.logger.info(f"[{self.video_id}] Concatenating {len(subvideo_binaries)} subvideos...")
        video_binary = await concatenate_videos(
            subvideo_binaries,
            fast_copy=False)  # move to True once we fix the durations
        if not video_binary:
            raise ValueError(f"Cannot concatenate subvideos for video {self.video_id}")
        self._log_video_info(
            f"[{self.video_id}] Concatenated {len(subvideo_binaries)} videos into video",
            video_binary)
        return video_binary

    async def _gen_subvideo(
        self,
        subvideo_id: int,
        subvideo_info: SubVideoInfo,
        width: int,
        height: int,
        num_steps: int,
        video_frames: List[Image.Image],  # video_hy_video_frames
        audio_base64: str,
        video_prompt: str = "",
        video_neg_prompt: str = "",
        upscaling: bool = False,
        debug: bool = False,
        deadline: Optional[float] = None,
    ) -> Optional[asyncio.Task]:
        """
        Generate video+audio chunk with Fantasy Talking from sketch frames (Hunyuan FramePack) and audio_base64.
        """
        start_frame = subvideo_info.get_start_frame(HUNYUANFRAMEPACK_FPS)
        end_frame = subvideo_info.get_end_frame(HUNYUANFRAMEPACK_FPS)

        if start_frame > len(video_frames):
            self.logger.error(
                f"[{self.video_id}.{subvideo_id}] Not enough sketch frames"
                f"[{start_frame}..{end_frame} ] > {len(video_frames)}.")
            return None

        # Add some frames to account for the 1+4n alignment
        end_frame += VAE_T
        if end_frame > len(video_frames):
            self.logger.warning(
                f"[{self.video_id}.{subvideo_id}] Sketch frames truncated "
                f"{end_frame} > {len(video_frames)}.")
            end_frame = len(video_frames)  # Don't go beyond available frames

        # TODO align audio lengths
        # TODO check if this is just silence

        # Audio
        subaudio_b64 = chunk_audio_base64(
            audio_base64,
            subvideo_info.start_seconds,
            subvideo_info.end_seconds)
        subvideo_duration = subvideo_info.end_seconds - subvideo_info.start_seconds
        subvideo_audio_num_frames = get_num_video_frames_from_duration(subvideo_duration)

        subvideo_audio_path = f"{self.job_path}/{self.video_id:03d}_{subvideo_id:03d}.wav"
        await save_base64_as_binary(
            subvideo_audio_path,
            subaudio_b64)

        # Video
        # Hunyuan FramePack -> Fantasy Talking intermediate frames
        hy_frames = video_frames[start_frame:end_frame]
        ft_frames = get_video_frames_at_fps(
            hy_frames,
            src_fps=HUNYUANFRAMEPACK_FPS,
            dst_fps=FANTASYTALKING_FPS)

        # Adjust video frames portion to match audio if needed
        len_video = len(ft_frames)
        if len_video == 0:
            self.logger.error(f"[{self.video_id}.{subvideo_id}] No video frames for subvideo.")
            return None

        if len_video < subvideo_audio_num_frames:
            msg = f"[{self.video_id}.{subvideo_id}] Video < Audio ({len_video}<{subvideo_audio_num_frames}). Extend."
            self.logger.warning(msg)
            ft_frames += [ft_frames[-1]] * (subvideo_audio_num_frames - len_video)
        elif len_video > subvideo_audio_num_frames:
            msg = f"[{self.video_id}.{subvideo_id}] Video > Audio ({len_video}>{subvideo_audio_num_frames}). Trim."
            if len_video > subvideo_audio_num_frames + VAE_T:
                self.logger.warning(msg)
            else:
                self.logger.debug(msg)  # If it is only a few frames, it is just VAE 1+4n rounding
            ft_frames = ft_frames[:subvideo_audio_num_frames]
        if debug:
            ft_frames = self._add_debug(subvideo_id, ft_frames)

        # TODO
        # width, height = self.width, self.height
        if upscaling:
            # width, height = RESOLUTIONS[self.aspect_ratio]["medium"]
            width = width // 2
            height = height // 2
        # TODO run the upscaling after Fantasy Talking

        # num_steps = self.get_num_steps()
        task = asyncio.create_task(
            self.gen.gen_video_audio_from_video(
                ft_frames,
                subaudio_b64,
                prompt=video_prompt,
                neg_prompt=video_neg_prompt,
                width=width,
                height=height,
                steps=num_steps,
                task_id=f"{self.video_id:03d}_{subvideo_id:03d}",
                deadline=deadline,
            ))

        self.logger.info(
            f"[{self.video_id}.{subvideo_id}] Generating video+audio using "
            f"{len(hy_frames)}@{HUNYUANFRAMEPACK_FPS}FPS->"
            f"{len(ft_frames)}@{FANTASYTALKING_FPS}FPS frames...")

        return task

    def _add_debug(
        self,
        subvideo_id: int,
        ft_frames: List[Image.Image],
    ) -> List[Image.Image]:
        """
        Add debug text to frames.
        We are adding this pre Fantasy Talking which could make it worse.
        """
        # Id
        frame_text = f"{self.video_id:03d}.{subvideo_id:03d}"
        ft_frames = [
            cast(Image.Image, add_text_to_frame(frame, text=frame_text, position="top-left"))
            for frame in ft_frames
        ]

        # Size and number of frames
        width, height = ft_frames[0].size
        frame_text = f"{width}x{height} {len(ft_frames)} frames"
        ft_frames = [
            cast(Image.Image, add_text_to_frame(frame, text=frame_text, position="top-right"))
            for frame in ft_frames
        ]
        return ft_frames

    def _log_video_info(
        self,
        prefix: str,
        video_content: Union[bytes, str],
    ) -> None:
        """Log video information."""
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


# Backup code before the refactor
"""
async def gen_scene_chunks(
    self,
    scene_id: int,
    audio_path: str,
    image: Image.Image,
    video_prompt: str = VIDEO_PROMPT,
    video_neg_prompt: str = VIDEO_NEG_PROMPT,
    sub_scene_duration_max: int = MAX_FT_DURATION_SECS,
) -> bytes:
    " ""
    Generate a scene with audio and video starting with image at high resolution.
    1. Split audio into sub-scenes based on silences.
    2. Generate sketch at low resolution with Hunyuan FramePack.
    3. Generate video+audio at medium resolution with Fantasy Talking.
    Returns video synced with the audio in binary.
    " ""

    # Split into sub-scenes based on silences (aligned with Fantasy Talking frames)
    audio_base64 = await read_file_base64(audio_path)
    audio_duration = get_audio_duration(audio_base64)
    audio_splits = get_audio_chunks_by_silences(
        audio_path,
        sub_scene_duration_max,
        chunk_alignment_seconds=1.0 / FANTASYTALKING_FPS)

    if len(audio_splits) > audio_duration / sub_scene_duration_max + 1:
        self.logger.warning(
            f"[{scene_id}] Too many sub-scenes ({len(audio_splits)}) for audio with "
            f"{audio_duration:.3f} seconds and max {sub_scene_duration_max:.3f}.")

    # Calculate #frames for each sub-scene
    scene_info = SceneInfo()
    for sub_scene_id, (start_secs, end_secs) in enumerate(audio_splits):
        sub_scene = SubSceneInfo(start_secs, end_secs)
        scene_info.append(sub_scene)
        num_audio_frames = sub_scene.get_num_audio_frames()
        duration_secs = end_secs - start_secs
        self.logger.info(
            f"[{scene_id}.{sub_scene_id}] Sub-scene: {start_secs:6.3f}-{end_secs:6.3f} ({duration_secs:6.3f}). "
            f"{num_audio_frames:4d} frames. FP:"
            f"{sub_scene.get_start_frame(HUNYUANFRAMEPACK_FPS):4d}-"
            f"{sub_scene.get_end_frame(HUNYUANFRAMEPACK_FPS):4d} frames.")
        if duration_secs > sub_scene_duration_max:
            self.logger.error(
                f"[{scene_id}.{sub_scene_id}] Sub-scene too long: {start_secs:.3f}-{end_secs:.3f} "
                f"({duration_secs:.3f} seconds) > {sub_scene_duration_max:.3f}.")
        elif duration_secs < 0.5:
            self.logger.warning(
                f"[{scene_id}.{sub_scene_id}] Sub-scene too short: {start_secs:.3f}-{end_secs:.3f} "
                f"({duration_secs:.3f} seconds).")

    # Start long sketch video (no audio) generation in the background
    width, height = RESOLUTIONS[self.aspect_ratio]["low"]
    video_num_frames = get_num_video_frames_from_duration(
        audio_duration,
        HUNYUANFRAMEPACK_FPS,
        HUNYUANFRAMEPACK_VAE_T)

    num_steps = self.get_num_steps()
    video_gen_request = await self.gen.gen_video(
        image,
        video_prompt,
        video_neg_prompt,
        width=width,
        height=height,
        num_frames=video_num_frames,
        steps=num_steps // 2,  # Less steps for sketch video
        task_id=f"{scene_id:03d}_sketch",
        deadline=self.get_scene_deadline(scene_id),
        wait_request=False,
    )
    self.logger.info(
        f"[{scene_id}] Generating long sketch video with {video_num_frames} frames, "
        f"{audio_duration:.3f} seconds, "
        f"{HUNYUANFRAMEPACK_FPS} FPS, and "
        f"{width}x{height} pixels.")

    # Get frames while the request is running
    scene_hy_video_frames: List[Image.Image] = []
    sub_scene_tasks: Dict[int, asyncio.Task] = {}
    video_gen_request_done = False
    while len(scene_hy_video_frames) < video_num_frames and not video_gen_request_done:
        if video_gen_request.done():
            video_gen_request_done = True
        elif not video_gen_request.is_running() or not video_gen_request.url:
            RETRY_SLEEP_SECONDS = 1.0
            await asyncio.sleep(RETRY_SLEEP_SECONDS)  # Wait for it to be running before checking again
            continue

        # Get intermediate frames (while long sketch video generation is running)
        sub_scene_id = 0
        base_url = video_gen_request.get_base_request_url()
        async for frame in self.gen.gen_intermediate_video_frames(
            base_url,
            task_id=f"{scene_id:03d}_sketch",
            video_gen_request=video_gen_request,
        ):
            scene_hy_video_frames.append(frame)

            # Process as many sub-scenes as possible with available frames
            while sub_scene_id < len(scene_info.sub_scenes):
                sub_scene_info = scene_info[sub_scene_id]
                # Add 4 more frames to account for the 1+4n VAE alignment
                if len(scene_hy_video_frames) < sub_scene_info.get_end_frame(HUNYUANFRAMEPACK_FPS) + VAE_T:
                    break  # Wait for more frames

                # Enough frames for the sub-scene, generate video+audio
                sub_scene_task = await self.gen_sub_scene(
                    scene_id,
                    sub_scene_id,
                    sub_scene_info,
                    scene_hy_video_frames,
                    audio_base64,
                    video_prompt=video_prompt,
                    video_neg_prompt=video_neg_prompt)
                if sub_scene_task is None:
                    self.logger.error(f"[{scene_id}.{sub_scene_id}] Cannot generate sub-scene task.")
                elif sub_scene_id in sub_scene_tasks:
                    self.logger.error(
                        f"[{scene_id}.{sub_scene_id}] Sub-scene task already exists "
                        f"({sub_scene_tasks.keys()}).")
                else:
                    sub_scene_tasks[sub_scene_id] = sub_scene_task
                sub_scene_id += 1

    self.logger.info(f"[{scene_id}] Got {len(scene_hy_video_frames)}/{video_num_frames} streamed video frames.")
    if len(scene_hy_video_frames) < video_num_frames:
        self.logger.warning(f"[{scene_id}] Not enough frames ({len(scene_hy_video_frames)} < {video_num_frames}).")

    # Get the full video (without audio)
    content_type, video_binary = await video_gen_request.future
    if video_binary is None:
        raise ValueError(f"Cannot generate video for scene {scene_id}.")

    self._log_video_info(f"[{scene_id}] Video sketch", video_binary)

    video_path = f"{self.job_path}/{scene_id:03d}_chunks_sketch.mp4"
    async with aiofiles.open(video_path, "wb") as file:
        await file.write(video_binary)

    # Collecting video+audio sub-scenes
    self.logger.info(f"[{scene_id}] Generating {len(sub_scene_tasks)} video+audio sub-scenes...")
    sub_scene_binaries = await asyncio.gather(*sub_scene_tasks.values(), return_exceptions=True)
    self.logger.info(f"[{scene_id}] Generated {len(sub_scene_tasks)} video+audio sub-scenes.")

    if not sub_scene_binaries:
        raise ValueError(f"No sub-scenes generated for scene {scene_id}.")

    for sub_scene_id, sub_scene_binary in enumerate(sub_scene_binaries):
        sub_scene_binary = sub_scene_binaries[sub_scene_id]
        if isinstance(sub_scene_binary, bytes):
            # Success case
            self._log_video_info(
                f"[{scene_id}.{sub_scene_id}] Generated video",
                sub_scene_binary)
        else:
            # Error case -> replace with static error video
            err_msg = str(sub_scene_binary)
            duration_seconds = scene_info[sub_scene_id].get_seconds()
            width, height = RESOLUTIONS[self.aspect_ratio]["medium"]
            self.logger.error(
                f"[{scene_id}.{sub_scene_id}] Failed to get video+audio: {err_msg}. "
                f"Adding error video with {duration_seconds:.3f} seconds and {width}x{height} pixels.")

            frame_text = "Cannot generate video+audio\n"
            frame_text += f"Sub-scene {scene_id:03d}.{sub_scene_id:03d}\n"
            frame_text += "\n".join(split_text_lines(err_msg, MAX_IMG_LINE_CHARS))
            font_size = get_font_size(width, height)
            sub_scene_binary = await get_video_with_text(
                width=width,
                height=height,
                text=frame_text,
                font_size=font_size,
                fps=FANTASYTALKING_FPS,
                duration_seconds=duration_seconds)
            sub_scene_binaries[sub_scene_id] = sub_scene_binary

        video_path = f"{self.job_path}/{scene_id:03d}_{sub_scene_id:03d}_chunks.mp4"
        async with aiofiles.open(video_path, "wb") as file:
            await file.write(sub_scene_binary)

    # Concatenate sub-scenes into final scene video
    self.logger.info(f"[{scene_id}] Concatenating {len(sub_scene_binaries)} sub-scenes...")
    scene_binary = await concatenate_videos(
        sub_scene_binaries,
        fast_copy=False)  # move to True once we fix the durations
    if not scene_binary:
        raise ValueError(f"Cannot concatenate sub-scenes for scene {scene_id}")
    self._log_video_info(
        f"[{scene_id}] Concatenated {len(sub_scene_binaries)} videos into video",
        scene_binary)
    return scene_binary
"""

"""
async def gen_sub_scene(
    self,
    scene_id: int,
    sub_scene_id: int,
    sub_scene_info: SubSceneInfo,
    scene_frames: List[Image.Image],  # scene_hy_video_frames
    audio_base64: str,
    video_prompt: str = VIDEO_PROMPT,
    video_neg_prompt: str = VIDEO_NEG_PROMPT,
) -> Optional[asyncio.Task]:
    " ""
    Generate a sub-scene with video+audio from Hunyuan FramePack frames and audio_base64.
    " ""
    start_frame = sub_scene_info.get_start_frame(HUNYUANFRAMEPACK_FPS)
    end_frame = sub_scene_info.get_end_frame(HUNYUANFRAMEPACK_FPS)
    # Add some frames to account for the 1+4n alignment
    end_frame += VAE_T
    if end_frame > len(scene_frames):
        end_frame = len(scene_frames)  # Don't go beyond available frames

    # TODO align audio lengths
    # TODO check if this is just silence

    # Audio
    sub_scene_audio_base64 = chunk_audio_base64(
        audio_base64,
        sub_scene_info.start_seconds,
        sub_scene_info.end_seconds)
    sub_scene_duration = sub_scene_info.end_seconds - sub_scene_info.start_seconds
    sub_scene_audio_num_frames = get_num_video_frames_from_duration(sub_scene_duration)

    sub_scene_audio_path = f"{self.job_path}/{scene_id:03d}_{sub_scene_id:03d}.wav"
    await save_base64_as_binary(sub_scene_audio_path, sub_scene_audio_base64)

    # Video
    # Hunyuan FramePack -> Fantasy Talking intermediate frames
    sub_scene_hy_frames = scene_frames[start_frame:end_frame]
    sub_scene_ft_frames = get_video_frames_at_fps(
        sub_scene_hy_frames,
        src_fps=HUNYUANFRAMEPACK_FPS,
        dst_fps=FANTASYTALKING_FPS)

    # Adjust video frames portion to match audio if needed
    len_video = len(sub_scene_ft_frames)
    if len_video == 0:
        self.logger.error(f"[{scene_id}.{sub_scene_id}] No video frames for sub-scene.")
        return None
    elif len_video < sub_scene_audio_num_frames:
        msg = f"[{scene_id}.{sub_scene_id}] Video < Audio ({len_video}<{sub_scene_audio_num_frames}). Extending."
        self.logger.warning(msg)
        sub_scene_ft_frames += [sub_scene_ft_frames[-1]] * (sub_scene_audio_num_frames - len_video)
    elif len_video > sub_scene_audio_num_frames:
        msg = f"[{scene_id}.{sub_scene_id}] Video > Audio ({len_video}>{sub_scene_audio_num_frames}). Trimming."
        if len_video > sub_scene_audio_num_frames + VAE_T:
            self.logger.warning(msg)
        else:
            self.logger.debug(msg)  # If it is only a few frames, it is just VAE 1+4n rounding
        sub_scene_ft_frames = sub_scene_ft_frames[:sub_scene_audio_num_frames]

    if self.get_config_bool("debug_image"):
        # We are adding this pre Fantasy Talking which could make it worse
        frame_text = f"{scene_id:03d}.{sub_scene_id:03d}"
        sub_scene_ft_frames = [
            add_text_to_frame(frame, text=frame_text, position="top-left")
            for frame in sub_scene_ft_frames
        ]
        width, height = sub_scene_ft_frames[0].size
        frame_text = f"{width}x{height} {len(sub_scene_ft_frames)} frames"
        sub_scene_ft_frames = [
            add_text_to_frame(frame, text=frame_text, position="top-right")
            for frame in sub_scene_ft_frames
        ]

    width, height = self.width, self.height
    if self.get_config_bool("upscaling"):
        # width, height = RESOLUTIONS[self.aspect_ratio]["medium"]
        width = self.width // 2
        height = self.height // 2

    num_steps = self.get_num_steps()
    task = asyncio.create_task(
        self.gen.gen_video_audio_from_video(
            sub_scene_ft_frames,
            sub_scene_audio_base64,
            prompt=video_prompt,
            neg_prompt=video_neg_prompt,
            width=width,
            height=height,
            steps=num_steps,
            deadline=self.get_scene_deadline(scene_id),
            task_id=f"{scene_id:03d}_{sub_scene_id:03d}",
        ))

    self.logger.info(
        f"[{scene_id}.{sub_scene_id}] Generating video+audio using "
        f"{len(sub_scene_hy_frames)}@{HUNYUANFRAMEPACK_FPS}FPS->"
        f"{len(sub_scene_ft_frames)}@{FANTASYTALKING_FPS}FPS frames...")

    return task
"""
