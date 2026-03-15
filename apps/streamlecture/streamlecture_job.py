"""
StreamLecture job to generate a lecture video.
"""
import asyncio
import json
import sys
import aiofiles

from PIL import Image

from typing import override
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from lecture_prompts import IMG_PROMPT
from lecture_prompts import IMG_NEG_PROMPT
from lecture_prompts import VIDEO_PROMPT
from lecture_prompts import VIDEO_NEG_PROMPT
from lecture_prompts import LECTURE_STYLE_PROMPT
from lecture_prompts import LECTURE_CUSTOM_PROMPT

# Local relative imports
sys.path.append("..")  # noqa: E402
sys.path.append("../..")  # noqa: E402

from streamwise_job import StreamWiseJob
from streamwise_job import JobStatus
from streamwise_job import OutputMode
from streamwise_job import MAX_LOG_TEXT

from lmm_service_manager import LMMServiceManager

from gen_video_chunked import GenVideoChunked

from console_utils import bytes_to_human

from file_utils import base64_to_binary
from file_utils import save_base64_as_binary

from pdf_utils import parse_pdf

from media_utils import get_audio_duration
from media_utils import get_video_frames
from media_utils import get_video_file_info
from media_utils import save_video_audio
from media_utils import concatenate_videos

from video import MAX_FT_DURATION_SECS
from video import FANTASYTALKING_FPS


class StreamLectureJob(StreamWiseJob):
    """A job to generate a lecture video."""

    def __init__(
        self,
        job_id: str,
        service_manager: LMMServiceManager,
        config: Dict[str, Any] = {},
    ) -> None:
        super().__init__(
            "streamlecture",
            job_id,
            service_manager,
            config)
        self.image: Optional[Image.Image] = None

    @override
    async def generate(
        self,
        job_config: Dict[str, Any],
    ) -> None:
        pdf_base64 = job_config.get("pdf_base64", None)
        await self.gen_lecture(pdf_base64)

    async def gen_scene(
        self,
        scene_id: int,
        text: str,
    ) -> str:
        """
        Generate a scene with TTS audio and video from the classroom image.
        Returns the path to the saved scene video.
        """
        log_text = text[:MAX_LOG_TEXT] + "..." if len(text) > MAX_LOG_TEXT else text
        self.logger.info(f"[{scene_id}] Generating scene with text '{log_text}'.")

        # Generate TTS audio
        voice = "af_heart"
        speed = self.get_config_float("speech_speed", 1.1)
        audio_base64 = await self.gen.gen_audio(
            text,
            voice=voice,
            speed=speed,
            task_id=f"{scene_id:03d}",
            deadline=self.get_scene_deadline(scene_id),
        )
        if not audio_base64:
            raise ValueError(f"Cannot generate audio for scene {scene_id}")
        audio_duration = get_audio_duration(audio_base64)
        audio_path = f"{self.job_path}/{scene_id:03d}.wav"
        await save_base64_as_binary(audio_path, audio_base64)
        self.logger.info(
            f"[{scene_id}] Generated audio with {bytes_to_human(len(audio_base64))} "
            f"and {audio_duration:.3f} seconds.")

        if self.image is None:
            raise ValueError(f"[{scene_id}] Classroom image not available for video generation.")

        output_mode = self.get_config_output_mode()
        num_steps = self.get_num_steps()
        width, height = self.image.size

        # Generate video from the classroom image + audio
        if output_mode is OutputMode.AUDIO_ONLY:
            # Audio-only: no video generation needed, handled at concatenation
            return audio_path

        if audio_duration < MAX_FT_DURATION_SECS and output_mode == OutputMode.VIDEO_AUDIO_SYNCED:
            scene_video_binary = await self.gen.gen_video_audio_from_img(
                img=self.image,
                audio_base64=audio_base64,
                prompt=VIDEO_PROMPT,
                neg_prompt=VIDEO_NEG_PROMPT,
                width=width,
                height=height,
                steps=num_steps,
                task_id=f"{scene_id:03d}",
                deadline=self.get_scene_deadline(scene_id),
            )
        elif audio_duration >= MAX_FT_DURATION_SECS and output_mode == OutputMode.VIDEO_AUDIO_SYNCED:
            gen_video_chunked = GenVideoChunked(
                video_id=scene_id,
                gen=self.gen,
                job_path=self.job_path,
                logger=self.logger)
            scene_video_binary = await gen_video_chunked.gen_video_chunked(
                audio_path=audio_path,
                image=self.image,
                prompt=VIDEO_PROMPT,
                neg_prompt=VIDEO_NEG_PROMPT,
                width=width,
                height=height,
                num_steps=num_steps,
                upscaling=self.get_config_bool("upscaling"),
                debug=self.get_config_bool("debug_image"),
                deadline=self.get_scene_deadline(scene_id),
            )
        else:
            # VIDEO_AUDIO_UNSYNCED: generate video and merge audio separately
            scene_video_binary = await self.gen.gen_video(
                img=self.image,
                prompt=VIDEO_PROMPT,
                neg_prompt=VIDEO_NEG_PROMPT,
                width=width,
                height=height,
                video_seconds=audio_duration,
                steps=num_steps,
                task_id=f"{scene_id:03d}",
                wait_request=True,
                deadline=self.get_scene_deadline(scene_id),
            )

        video_frames = await get_video_frames(scene_video_binary)
        video_file_info = get_video_file_info(scene_video_binary)
        video_info = video_file_info.get("video", {})
        video_fps = video_info.get("fps", FANTASYTALKING_FPS)

        video_audio_path = f"{self.job_path}/{scene_id:03d}.mp4"
        video_audio_path = await save_video_audio(
            video_content=video_frames,
            audio_path=audio_path,
            out_video_path=video_audio_path,
            fps=video_fps)

        self.logger.info(f"[{scene_id}] Scene saved to '{video_audio_path}'.")
        return video_audio_path

    async def gen_lecture(
        self,
        pdf_base64: str,
    ) -> None:
        """
        Generate a lecture video from a PDF.
        """
        async with self.job_status_handler():
            if not pdf_base64:
                self.logger.error("Document is required.")
                await self.save_status(JobStatus.FAILED)
                raise ValueError("Missing 'pdf_base64' in request")
            self.logger.info(f"Generating lecture video for document with {bytes_to_human(len(pdf_base64))}.")

            # Save PDF for processing and debugging
            self.logger.info(f"Document base64 with {bytes_to_human(len(pdf_base64))}.")
            pdf_path = f"{self.job_path}/document.pdf"
            async with aiofiles.open(pdf_path, "wb") as file:
                pdf_binary = base64_to_binary(pdf_base64)
                await file.write(pdf_binary)

            # Extract the materials from the PDF
            pdf_text, pdf_images = parse_pdf(pdf_path)
            num_pages = len(pdf_text)
            self.logger.info(f"Extracted {num_pages} pages and {len(pdf_images)} images from PDF.")

            # Save extracted text for debugging
            txt_path = f"{self.job_path}/document.txt"
            async with aiofiles.open(txt_path, "w", encoding="utf-8") as txt_file:
                for page_ix, page_text in enumerate(pdf_text):
                    await txt_file.write(f"--- Page {page_ix + 1} ---\n{page_text}\n")

            # Generate the main classroom image in parallel with transcript generation
            image_task: asyncio.Task[Optional[Image.Image]] = asyncio.create_task(
                self.gen.gen_image(
                    prompt=IMG_PROMPT,
                    neg_prompt=IMG_NEG_PROMPT,
                    width=self.width,
                    height=self.height,
                    task_id="main_image",
                    deadline=self.get_submission_time(),
                )
            )

            await self.save_status(JobStatus.RUNNING)

            # Generate the lecture transcript using the podcast transcript service (1 professor character)
            output_mode = self.get_config_output_mode()
            scene_texts: List[Dict] = []
            async with aiofiles.open(f"{self.job_path}/lecture_transcript.jsonl", "wb") as file:
                async for line_json in self.gen.gen_podcast_transcript(
                    task_id="transcript",
                    pdf_base64=pdf_base64,
                    num_characters=1,
                    style_prompt=LECTURE_STYLE_PROMPT,
                    custom_prompt=LECTURE_CUSTOM_PROMPT,
                    max_tokens=self.get_config_int("max_tokens"),
                    max_dialogues=self.get_config_int("max_dialogues"),
                    max_words_per_dialogue=self.get_config_int("max_words_per_dialogue"),
                ):
                    line = (json.dumps(line_json) + "\n").encode("utf-8")
                    await file.write(line)
                    await file.flush()

                    line_type = line_json.get("type", "")
                    if line_type == "dialogue":
                        log_text = line_json.get("content", "")[:MAX_LOG_TEXT]
                        self.logger.info(f"Scene {len(scene_texts)}: {log_text}")
                        scene_texts.append(line_json)
            self.logger.info(f"Lecture transcript generated with {len(scene_texts)} scenes.")

            await self.save_status(JobStatus.RUNNING)

            # Wait for classroom image
            self.image = await image_task
            if self.image is not None:
                width, height = self.image.size
                image_path = f"{self.job_path}/main_image.png"
                self.image.save(image_path)
                self.logger.info(f"Classroom image {width}x{height} saved to '{image_path}'.")
            elif output_mode is not OutputMode.AUDIO_ONLY:
                raise ValueError("Cannot generate classroom image for lecture video.")

            # Generate each scene concurrently
            scene_video_tasks: Dict[int, asyncio.Task] = {}
            for scene_id, scene_json in enumerate(scene_texts):
                scene_text = scene_json.get("content", "")
                if not scene_text:
                    self.logger.warning(f"[{scene_id}] Empty scene text. Skipping.")
                    continue
                scene_task = asyncio.create_task(
                    self.gen_scene(scene_id=scene_id, text=scene_text)
                )
                scene_video_tasks[scene_id] = scene_task

            await self.save_status(JobStatus.RUNNING)

            # Collect scene results
            scene_video_paths: List[str] = []
            results = await asyncio.gather(*scene_video_tasks.values(), return_exceptions=True)
            for scene_id, result in enumerate(results):
                if isinstance(result, Exception):
                    self._handle_scene_exception(scene_id, result)
                elif not result:
                    self.logger.warning(f"[{scene_id}] No video generated. Skipping.")
                else:
                    scene_video_paths.append(result)
                    self.logger.info(f"[{scene_id}] Scene saved to '{result}'.")

            if not scene_video_paths:
                raise ValueError("No scene videos generated. Cannot create final lecture video.")

            await self.save_status(JobStatus.RUNNING)

            # Concatenate all scene videos into a final lecture video
            self.logger.info(f"Concatenating {len(scene_video_paths)} scenes into final lecture video...")
            scene_binaries: List[bytes] = []
            for scene_video_path in scene_video_paths:
                async with aiofiles.open(scene_video_path, "rb") as vfile:
                    scene_binaries.append(await vfile.read())

            video_binary = await concatenate_videos(scene_binaries, fast_copy=False)
            if not video_binary:
                raise ValueError("Cannot concatenate scenes into final lecture video.")

            video_path = f"{self.job_path}/{self.job_id}.mp4"
            async with aiofiles.open(video_path, "wb") as vfile:
                await vfile.write(video_binary)

            self.logger.info(
                f"Generated lecture video with {bytes_to_human(len(video_binary))} at '{video_path}'.")
