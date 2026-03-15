"""
StreamShort job to generate a video short.
It coordinates the execution of the different models.
"""

import sys
import json
import cv2
import aiofiles
import asyncio

from dataclasses import asdict

from typing import override
from typing import Dict
from typing import Any
from typing import List
from typing import Optional

from scenedetect import open_video
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.stats_manager import StatsManager

from short_prompts import DESCRIPTION_PROMPT
from short_prompts import HIGHLIGHT_PROMPT

# Local relative imports
sys.path.append("..")  # noqa: E402
sys.path.append("../..")  # noqa: E402

from streamwise_job import StreamWiseJob
from streamwise_job import JobStatus

from scene import SceneSegment

from lmm_service_manager import LMMServiceManager

from console_utils import bytes_to_human

from file_utils import save_base64_as_binary
from file_utils import read_file_base64
from file_utils import read_file_bytes
from media_utils import chunk_video_binary
from media_utils import concatenate_videos
from media_utils import extract_audio_from_video
from media_utils import chunk_audio_base64


MAX_KEY_FRAMES = 64
DESCRIPTION_BATCH_SIZE = 16


class StreamShortJob(StreamWiseJob):
    """A job to generate a short video summary."""

    def __init__(
        self,
        job_id: str,
        service_manager: LMMServiceManager,
        config: Dict[str, Any] = {},
    ) -> None:
        super().__init__(
            "streamshort",
            job_id,
            service_manager,
            config)
        self.scenes: List[SceneSegment] = []

    @override
    async def generate(
        self,
        job_config: Dict[str, Any],
    ) -> None:
        video_base64 = job_config.get("video_base64", None)
        await self.gen_short(video_base64)

    def find_scene_for_frame(
        self,
        frame_num: int
    ) -> Optional[SceneSegment]:
        """Find the scene segment that contains the given frame number."""
        scenes = self.scenes
        left = 0
        right = len(scenes) - 1
        while left <= right:
            mid = (left + right) // 2
            scene = scenes[mid]
            if frame_num < scene.start_frame:
                right = mid - 1
            elif frame_num >= scene.end_frame:
                left = mid + 1
            else:
                return scene
        return None

    async def gen_short(
        self,
        video_base64: str,
    ) -> None:
        """
        Generate the video short.
        """
        async with self.job_status_handler():
            if not video_base64:
                self.logger.error("Video is required.")
                await self.save_status(JobStatus.FAILED)
                raise ValueError("Missing 'video_base64' in request")
            self.logger.info(f"Generating short for video with {bytes_to_human(len(video_base64))}.")

            # Save as video for debugging
            self.logger.info(f"Saving input video with {bytes_to_human(len(video_base64))}.")
            video_path = f"{self.job_path}/video.mp4"
            await save_base64_as_binary(video_path, video_base64)

            await self.save_status(JobStatus.RUNNING)

            # Detect scenes
            self.scenes = await self.detect_scenes()
            self.logger.info(f"Detected {len(self.scenes)} scenes.")

            await self.save_status(JobStatus.RUNNING)

            # Extract key frames
            key_frames = self.extract_key_frames(video_path)
            self.logger.info(f"Extracted {len(key_frames)} key frames: {', '.join(map(str, key_frames))}.")

            # Transcribe audio
            await self.chunk_audio_into_scenes()

            transcript_task = asyncio.create_task(self.transcribe_audio())

            # Describe key frames
            description_task = asyncio.create_task(self.describe_frames(key_frames))

            await self.save_status(JobStatus.RUNNING)

            # Wait for async tasks to complete
            await transcript_task
            await description_task

            # Output some debug files
            self.logger.info("Scenes:")
            for idx, scene in enumerate(self.scenes):
                self.logger.info(f"  Scene {idx}: {scene}")
            scenes_path = f"{self.job_path}/scenes.json"
            async with aiofiles.open(scenes_path, "w") as scene_file:
                scenes_dict_list = [asdict(scene) for scene in self.scenes]
                scenes_json = json.dumps(scenes_dict_list, indent=2)
                await scene_file.write(scenes_json)

            await self.save_status(JobStatus.RUNNING)

            # Select highlight scenes
            avg_scene_seconds = sum(scene.duration_sec for scene in self.scenes) / max(len(self.scenes), 1)
            video_duration_seconds = self.get_config_float("video_duration_seconds", 10.0)
            max_scenes = 1
            if avg_scene_seconds > 0:
                max_scenes = max(1, int(video_duration_seconds / avg_scene_seconds))
            self.logger.info(f"Generating a short of {video_duration_seconds} seconds and max {max_scenes} scenes.")
            chosen_scenes = await self.choose_scenes_for_highlight(
                total_length=video_duration_seconds,
                max_scenes=max_scenes,
            )
            short_duration_seconds = sum(self.scenes[scene_id].duration_sec for scene_id in chosen_scenes)
            self.logger.info(
                f"Chosen {len(chosen_scenes)} scenes for a short of {short_duration_seconds:.1f} seconds: "
                f"{', '.join(map(str, chosen_scenes))}.")
            if not chosen_scenes:
                raise RuntimeError("Failed to choose scenes for highlight short")

            # Generate some video using diffusion
            # TODO

            # Build short
            short_video_path = await self.save_highlight_short(chosen_scenes)
            self.logger.info(f"Saved short video to {short_video_path}.")

    def extract_key_frames(
        self,
        video_path: str
    ) -> List[int]:
        """Extract key frames from the video and associate them with scenes."""
        cap = cv2.VideoCapture(video_path)
        key_frames = self.pick_key_frames()

        for frame_num in key_frames:
            frame_file_name = f"frame_{frame_num:04d}.jpg"
            frame_path = f"{self.job_path}/{frame_file_name}"
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Failed to read frame {frame_num}")
            cv2.imwrite(frame_path, frame)

            scene = self.find_scene_for_frame(frame_num)
            if scene:
                scene.add_image_path(frame_file_name)
        cap.release()

        return key_frames

    async def detect_scenes(
        self,
        threshold: float = 27.0,
        min_scene_len: int = 15,
    ) -> List[SceneSegment]:
        """Return list of (start_frame, end_frame, start_sec, end_sec)."""
        video_path = f"{self.job_path}/video.mp4"
        if not await aiofiles.os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        video = open_video(video_path)

        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        content_detector = ContentDetector(
            threshold=threshold,
            min_scene_len=min_scene_len)
        scene_manager.add_detector(content_detector)

        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        scenes = []
        for scene_id, (start_tc, end_tc) in enumerate(scene_list):
            scene = SceneSegment(
                scene_id,
                start_tc.get_frames(), end_tc.get_frames(),
                start_tc.get_seconds(), end_tc.get_seconds()
            )
            scenes.append(scene)
        return scenes

    def pick_key_frames(
        self,
        max_frames: int = MAX_KEY_FRAMES,
    ) -> List[int]:
        """Pick up to max_frames evenly spaced frame numbers within a scene."""
        if not self.scenes:
            return []

        start_frame = 0
        end_frame = self.scenes[-1].end_frame

        if max_frames <= 1 or end_frame <= start_frame:
            return [max(start_frame, 0)]
        total_frames = max(end_frame - start_frame, 1)
        count = min(max_frames, total_frames)
        frames = []
        for i in range(count):
            pos = start_frame + int((i + 0.5) * total_frames / count)
            pos = min(pos, end_frame - 1)
            frames.append(pos)

        return sorted(set(frames))

    async def describe_frames(
        self,
        key_frames: List[int]
    ):
        description_tasks = []

        key_frame_batch = []
        batch_id = 0
        for frame_num in key_frames:
            key_frame_batch.append(frame_num)
            if len(key_frame_batch) >= DESCRIPTION_BATCH_SIZE:
                description_task = asyncio.create_task(self.describe_frames_batch(
                    key_frame_batch, batch_id=batch_id))
                description_tasks.append(description_task)
                batch_id += 1
                key_frame_batch = []
        if key_frame_batch:
            description_task = asyncio.create_task(self.describe_frames_batch(
                key_frame_batch,
                batch_id=batch_id))
            description_tasks.append(description_task)

        for description_task in description_tasks:
            await description_task

    async def describe_frames_batch(
        self,
        frame_nums: List[int],
        max_tokens: int = 8192,
        batch_id: int = 0,
    ) -> str:
        """Describe a list of frames using the LLM."""
        MAX_LOG_TEXT = 80

        self.logger.info(f"Describing {len(frame_nums)} key frames: {', '.join(map(str, frame_nums))}.")
        image_base64s = []
        for frame_num in frame_nums:
            frame_file_name = f"frame_{frame_num:04d}.jpg"
            frame_path = f"{self.job_path}/{frame_file_name}"
            frame_base64 = await read_file_base64(frame_path)
            image_base64s.append(frame_base64)

        # Send the frames to the LLM
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": DESCRIPTION_PROMPT}
            ],
        }
        for image_base64 in image_base64s:
            message["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            })
        message["content"].append({"type": "text", "text": "Generate the JSON descriptions now."})
        messages = [message]

        prompt_path = f"{self.job_path}/description_prompt_{batch_id}.json"
        async with aiofiles.open(prompt_path, "w") as prompt_file:
            await prompt_file.write(json.dumps(messages, indent=2))

        # Query the LLM
        response_message = await self.gen.gen_text(
            messages,
            max_tokens=max_tokens,
            task_id=f"describe{batch_id:03d}",
        )
        """
        response = await self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            max_tokens=max_tokens,
            extra_body=None,
            # stream=True,
        )
        response_message = response.choices[0].message
        response_message = response_message.content.strip()
        """

        prompt_path = f"{self.job_path}/description_response_{batch_id}.txt"
        async with aiofiles.open(prompt_path, "w") as prompt_file:
            await prompt_file.write(json.dumps(response_message, indent=2))

        # Parse multiple descriptions from response
        self.logger.info("Frames:")
        num_described_frames = 0
        for response_line in response_message.splitlines():
            response_line_strip = response_line.strip()
            response_line_strip = response_line_strip.strip(",")
            if not response_line_strip:
                pass
            elif "```" in response_line_strip or response_line_strip == "[" or response_line_strip == "]":
                pass
            elif not response_line_strip.startswith("{"):
                self.logger.debug(f"Skipping: {response_line_strip}")
            else:
                try:
                    response_json = json.loads(response_line_strip)
                    frame_num_response = response_json.get("frame_num", None)
                    description = response_json.get("description", None)
                    if not description:
                        self.logger.warning(f"No description for frame {frame_num_response}.")
                    elif frame_num_response < 0 or frame_num_response >= len(frame_nums):
                        self.logger.warning(
                            f"Invalid frame_num {frame_num_response} in response: {response_line_strip}.")
                    else:
                        frame_num = frame_nums[frame_num_response]
                        frame_description_path = f"{self.job_path}/frame_{frame_num:04d}.txt"
                        async with aiofiles.open(frame_description_path, "w") as f:
                            await f.write(description)
                        num_described_frames += 1

                        scene = self.find_scene_for_frame(frame_num)
                        if scene:
                            scene.add_description(description)
                            self.logger.info(
                                f"  Frame {frame_num} in scene {scene.scene_id}: "
                                f"{description[0:MAX_LOG_TEXT]}...")
                        else:
                            self.logger.warning(f"No scene found for frame {frame_num}.")
                except Exception as ex:
                    self.logger.error(f"Error parsing description line '{response_line_strip}': {ex}")

        if num_described_frames < len(frame_nums):
            self.logger.warning(
                f"Only described {num_described_frames} out of {len(frame_nums)} frames in batch {batch_id}.")

    async def choose_scenes_for_highlight(
        self,
        total_length: int = 30,
        max_scenes: int = 5,
        max_tokens: int = 256,
    ) -> List[int]:
        """Select the scenes for highlight short."""
        prompt = HIGHLIGHT_PROMPT.format(
            total_length=total_length,
            max_scenes=max_scenes,
        )
        for scene in self.scenes:
            prompt += f"- Scene {scene.scene_id}:\n"
            prompt += f"Duration: {scene.duration_sec:.1f} seconds.\n"
            if scene.descriptions:
                prompt += f"Description: {' '.join(scene.descriptions)}\n"
            if scene.transcript:
                prompt += f"Transcript: {scene.transcript}\n"
            prompt += "\n"

        messages = [
            {"role": "user", "content": prompt}
        ]

        prompt_json_path = f"{self.job_path}/highlight_prompt.json"
        async with aiofiles.open(prompt_json_path, "w") as prompt_file:
            await prompt_file.write(json.dumps(messages, indent=2))

        prompt_path = f"{self.job_path}/highlight_prompt.txt"
        async with aiofiles.open(prompt_path, "w") as prompt_file:
            messages_json = json.dumps(messages, indent=2)
            await prompt_file.write(messages_json)

        """
        response = await self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            max_tokens=max_tokens,
            extra_body=None,
            # stream=True,
        )
        response_message = response.choices[0].message
        response_message_strip = response_message.content.strip()
        """
        response_message_strip = await self.gen.gen_text(
            messages=messages,
            max_tokens=max_tokens,
            task_id="highlight",
        )

        response_json = []
        try:
            response_json = json.loads(response_message_strip)
        except Exception as ex:
            self.logger.error(f"Error parsing {response_message_strip}: {ex}")
        return response_json

    async def save_highlight_short(
        self,
        chosen_scenes: List[int]
    ) -> str:
        """Save the highlight short video."""
        input_video_path = f"{self.job_path}/video.mp4"
        input_video_binary = await read_file_bytes(input_video_path)

        highlight_scene_binaries = []
        for chosen_scene_id in chosen_scenes:
            scene = self.scenes[chosen_scene_id]
            scene_binary = chunk_video_binary(
                video_binary=input_video_binary,
                start_seconds=scene.start_sec,
                end_seconds=scene.end_sec,
                # Resize to the requested size if needed
                width=self.width,
                height=self.height,
            )
            highlight_scene_binaries.append(scene_binary)

        out_video_binary = await concatenate_videos(highlight_scene_binaries)
        out_video_path = f"{self.job_path}/{self.job_id}.mp4"
        async with aiofiles.open(out_video_path, "wb") as file:
            await file.write(out_video_binary)
        return out_video_path

    async def chunk_audio_into_scenes(self) -> List[str]:
        """
        Chunk the audio of the video into scenes.
        """
        chunks = []
        try:
            video_path = f"{self.job_path}/video.mp4"
            audio_path = f"{self.job_path}/audio.wav"
            audio_path = await extract_audio_from_video(video_path, audio_path)
            audio_base64 = await read_file_base64(audio_path)
            self.logger.info(f"Extracted audio with {bytes_to_human(len(audio_base64))}.")

            for scene in self.scenes:
                scene_audio_base64 = chunk_audio_base64(
                    audio_base64=audio_base64,
                    start_seconds=scene.start_sec,
                    end_seconds=scene.end_sec)
                scene_audio_path = f"{self.job_path}/scene_{scene.scene_id:03d}.wav"
                await save_base64_as_binary(scene_audio_path, scene_audio_base64)
                scene.audio_path = f"scene_{scene.scene_id:03d}.wav"
                chunks.append(scene_audio_base64)
        except Exception as ex:
            self.logger.error(f"Error during audio chunking: {ex} [{type(ex)}]")
        return chunks

    async def transcribe_audio(
        self
    ) -> str:
        """
        Transcribe the audio of the video.
        """
        ret = ""
        try:
            for scene in self.scenes:
                if not scene.audio_path:
                    continue
                audio_path = f"{self.job_path}/{scene.audio_path}"
                audio_transcript = await self.gen.gen_audio_transcript(
                    audio_path,
                    task_id=f"{scene.scene_id:03d}",
                )
                if not audio_transcript:
                    continue
                ret += audio_transcript + "\n"
                scene.transcript = audio_transcript
                self.logger.info(f"Scene {scene.scene_id} transcript: {audio_transcript[0:80]}...")
                transcript_path = f"{self.job_path}/scene_{scene.scene_id:03d}.txt"
                async with aiofiles.open(transcript_path, "w") as file:
                    await file.write(audio_transcript)
        except Exception as ex:
            self.logger.error(f"Error during transcription: {ex} [{type(ex)}]")
        return ret
