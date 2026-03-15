"""
StreamMovie job to generate a movie.
It coordinates the execution of the different models.
"""
import sys
import json
import asyncio
import aiofiles

from typing import override
from typing import Dict
from typing import Any
from typing import List
from typing import Optional

from movie_prompts import SYSTEM_PROMPT

# Local relative imports
sys.path.append("..")  # noqa: E402
sys.path.append("../..")  # noqa: E402

from streamwise_job import StreamWiseJob
from streamwise_job import JobStatus
from streamwise_job import OutputMode

from lmm_service_manager import LMMServiceManager

from console_utils import bytes_to_human

from media_utils import concatenate_videos
from media_utils import save_video_audio
from media_utils import get_audio_duration

from file_utils import save_base64_as_binary


DEFAULT_SHOT_DURATION_SECS = 4.0
DEFAULT_MAX_TOKENS = 8192
DEFAULT_SPEECH_SPEED = 1.1
MAX_LOG_TEXT = 80
SHOT_DEADLINE_BUFFER_SECS = 120.0  # Extra buffer per shot on top of its timeline offset


class StreamMovieJob(StreamWiseJob):
    """Job class for StreamMovie movie generation."""

    def __init__(
        self,
        job_id: str,
        config: Dict[str, Any],
        service_manager: LMMServiceManager
    ) -> None:
        super().__init__(
            "streammovie",
            job_id=job_id,
            config=config,
            service_manager=service_manager
        )

    @override
    async def generate(
        self,
        job_config: Dict[str, Any],
    ) -> None:
        movie_description = job_config.get("movie_description", None)
        await self.gen_movie(movie_description)

    @staticmethod
    def build_movie_messages(movie_description: str) -> list:
        """
        Build LLM messages for movie planning using the system prompt.
        Returns messages suitable for gen_text to generate a movie structure.
        """
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Create a movie: {movie_description}"},
        ]

    async def gen_movie(
        self,
        movie_description: str
    ) -> None:
        """
        Generate a movie based on the provided description.

        Steps:
        1. Stream a structured movie script (JSONL) from the LLM.
        2. Collect shot_description objects from the script.
        3. For each shot, generate an image and a video (with optional lip-synced audio).
        4. Concatenate all shot videos into the final movie.
        """
        async with self.job_status_handler():
            if not movie_description:
                self.logger.error("Movie description is required.")
                await self.save_status(JobStatus.FAILED)
                raise ValueError("Missing 'movie_description' in request")

            log_desc = movie_description[:MAX_LOG_TEXT]
            self.logger.info(f"Generating movie for description: '{log_desc}...'")

            max_tokens = self.get_config_int("max_tokens", DEFAULT_MAX_TOKENS)

            # Stream the movie script from the LLM
            shot_descriptions = await self._stream_movie_script(
                movie_description=movie_description,
                max_tokens=max_tokens,
            )

            await self.save_status(JobStatus.RUNNING)

            self.logger.info(f"Parsed {len(shot_descriptions)} shots from the movie script.")
            if not shot_descriptions:
                raise ValueError("No shots generated from the movie script.")

            # Limit number of shots if configured
            max_shots = self.get_config_int("max_shots", -1)
            if max_shots > 0 and len(shot_descriptions) > max_shots:
                self.logger.info(f"Limiting to {max_shots} shots (out of {len(shot_descriptions)}).")
                shot_descriptions = shot_descriptions[:max_shots]

            # Generate each shot: image -> video (+ audio if dialogue present)
            shot_tasks: Dict[int, asyncio.Task] = {}
            for idx, shot in enumerate(shot_descriptions):
                task = asyncio.create_task(self._gen_shot(idx, shot))
                shot_tasks[idx] = task

            await self.save_status(JobStatus.RUNNING)

            shot_video_paths: List[str] = []
            results = await asyncio.gather(*shot_tasks.values(), return_exceptions=True)
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"[{idx}] Shot generation failed: {result}")
                elif result:
                    shot_video_paths.append(result)
                    self.logger.info(f"[{idx}] Shot video saved to '{result}'.")
                else:
                    self.logger.warning(f"[{idx}] No video generated for shot.")

            if not shot_video_paths:
                raise ValueError("No shot videos generated. Cannot create final movie.")

            await self.save_status(JobStatus.RUNNING)

            # Concatenate all shot videos into the final movie
            self.logger.info(f"Concatenating {len(shot_video_paths)} shots into final movie...")
            scene_binaries: List[bytes] = []
            for video_path in shot_video_paths:
                async with aiofiles.open(video_path, "rb") as file:
                    scene_binary = await file.read()
                    scene_binaries.append(scene_binary)

            video_binary = await concatenate_videos(scene_binaries, fast_copy=False)
            if not video_binary:
                raise ValueError("Cannot concatenate shots into final movie.")

            video_path = f"{self.job_path}/{self.job_id}.mp4"
            async with aiofiles.open(video_path, "wb") as file:
                await file.write(video_binary)

            self.logger.info(
                f"Generated movie with {bytes_to_human(len(video_binary))} at '{video_path}'.")

    async def _stream_movie_script(
        self,
        movie_description: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> List[Dict[str, Any]]:
        """
        Stream a structured movie script from the LLM.
        Returns a list of shot_description dicts.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Generate a movie based on the following description:\n\n{movie_description}"
                ),
            },
        ]

        script_path = f"{self.job_path}/movie_script.jsonl"
        shot_descriptions: List[Dict[str, Any]] = []
        buffer = ""
        line_count = 0

        async with aiofiles.open(script_path, "w") as script_file:
            async for chunk in self.gen.gen_text_stream(
                messages=messages,
                max_tokens=max_tokens,
                task_id="movie_script",
            ):
                if not chunk:
                    continue
                buffer += chunk
                # Parse complete JSONL lines as they arrive
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    # Skip markdown code fences
                    if line.startswith("```"):
                        continue
                    await script_file.write(line + "\n")
                    await script_file.flush()
                    line_count += 1
                    parsed = self._try_parse_json(line, line_count)
                    if parsed is None:
                        continue
                    line_type = parsed.get("type", "")
                    if line_type == "shot_description":
                        shot_descriptions.append(parsed)
                        self.logger.info(
                            f"Shot {parsed.get('shot_id', line_count)}: "
                            f"{str(parsed.get('visual_prompt', ''))[:MAX_LOG_TEXT]}...")

            # Flush any remaining buffer content
            remainder = buffer.strip()
            if remainder and not remainder.startswith("```"):
                await script_file.write(remainder + "\n")
                parsed = self._try_parse_json(remainder, line_count)
                if parsed and parsed.get("type") == "shot_description":
                    shot_descriptions.append(parsed)

        self.logger.info(f"Movie script streamed: {line_count} JSONL lines, {len(shot_descriptions)} shots.")
        return shot_descriptions

    def _try_parse_json(
        self,
        line: str,
        line_num: int,
    ) -> Optional[Dict[str, Any]]:
        """Try to parse a JSON line; return None on failure."""
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            self.logger.debug(f"Skipping non-JSON line {line_num}: {line[:80]}")
            return None

    def _get_shot_deadline(
        self,
        shot_idx: int,
        shot_duration: float,
    ) -> float:
        """
        Compute the scheduling deadline for a shot.
        Uses the shot's time offset in the final movie plus a generation buffer,
        so earlier shots get higher priority while all shots have enough time.
        """
        return (
            self.get_submission_time()
            + (shot_idx * shot_duration)
            + SHOT_DEADLINE_BUFFER_SECS
        )

    async def _gen_shot(
        self,
        shot_idx: int,
        shot: Dict[str, Any],
    ) -> Optional[str]:
        """
        Generate a single movie shot: image -> video (with optional lip-synced audio).
        Returns the path to the saved shot video, or None on failure.
        """
        visual_prompt = shot.get("visual_prompt", "")
        neg_prompt = shot.get("negative_prompt", "")
        dialogue = shot.get("dialogue", None)
        tech_specs = shot.get("technical_specs", {})
        shot_duration = float(tech_specs.get("duration_seconds", DEFAULT_SHOT_DURATION_SECS))

        deadline = self._get_shot_deadline(shot_idx, shot_duration)

        self.logger.info(
            f"[{shot_idx}] Generating shot with prompt: '{visual_prompt[:MAX_LOG_TEXT]}...'")

        # Generate image for this shot
        image = await self.gen.gen_image(
            prompt=visual_prompt,
            neg_prompt=neg_prompt,
            width=self.width,
            height=self.height,
            task_id=f"shot_{shot_idx:03d}_img",
            deadline=deadline,
        )

        image_path = f"{self.job_path}/shot_{shot_idx:03d}.png"
        image.save(image_path)
        self.logger.info(f"[{shot_idx}] Image saved to '{image_path}'.")

        output_mode = self.get_config_output_mode()
        video_binary: Optional[bytes] = None

        if dialogue and dialogue.strip() and output_mode is not OutputMode.AUDIO_ONLY:
            # Generate audio for the dialogue
            speech_speed = self.get_config_float("speech_speed", DEFAULT_SPEECH_SPEED)
            audio_base64 = await self.gen.gen_audio(
                text=dialogue.strip(),
                speed=speech_speed,
                task_id=f"shot_{shot_idx:03d}_audio",
                deadline=deadline,
            )
            if audio_base64:
                audio_path = f"{self.job_path}/shot_{shot_idx:03d}.wav"
                await save_base64_as_binary(audio_path, audio_base64)
                audio_duration = get_audio_duration(audio_base64)
                self.logger.info(
                    f"[{shot_idx}] Generated audio with {bytes_to_human(len(audio_base64))} "
                    f"and {audio_duration:.2f} seconds.")

                if output_mode is OutputMode.VIDEO_AUDIO_SYNCED:
                    # Lip-synced video
                    video_binary = await self.gen.gen_video_audio_from_img(
                        img=image,
                        audio_base64=audio_base64,
                        prompt=visual_prompt,
                        neg_prompt=neg_prompt,
                        width=self.width,
                        height=self.height,
                        steps=self.get_num_steps(),
                        task_id=f"shot_{shot_idx:03d}_video",
                        deadline=deadline,
                    )
                else:
                    # Unsynced: generate video then merge audio track
                    raw_video_binary = await self.gen.gen_video(
                        img=image,
                        prompt=visual_prompt,
                        neg_prompt=neg_prompt,
                        width=self.width,
                        height=self.height,
                        video_seconds=audio_duration,
                        steps=self.get_num_steps(),
                        task_id=f"shot_{shot_idx:03d}_video",
                        deadline=deadline,
                    )
                    merged_path = f"{self.job_path}/shot_{shot_idx:03d}_merged.mp4"
                    await save_video_audio(
                        raw_video_binary,
                        audio_path,
                        out_video_path=merged_path,
                    )
                    async with aiofiles.open(merged_path, "rb") as fh:
                        video_binary = await fh.read()

        if video_binary is None:
            # No dialogue, or audio generation failed: plain video
            video_binary = await self.gen.gen_video(
                img=image,
                prompt=visual_prompt,
                neg_prompt=neg_prompt,
                width=self.width,
                height=self.height,
                video_seconds=shot_duration,
                steps=self.get_num_steps(),
                task_id=f"shot_{shot_idx:03d}_video",
                deadline=deadline,
            )

        if not video_binary:
            self.logger.error(f"[{shot_idx}] No video binary produced for shot.")
            return None

        shot_path = f"{self.job_path}/shot_{shot_idx:03d}.mp4"
        async with aiofiles.open(shot_path, "wb") as file:
            await file.write(video_binary)

        self._log_video_info(f"[{shot_idx}] Shot video", video_binary)
        return shot_path
