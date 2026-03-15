"""
StreamEdit job to generate an edited video.
"""
import sys
import json
import aiofiles
import aiofiles.os

from typing import override
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from dataclasses import asdict

from scenedetect import open_video
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.stats_manager import StatsManager

# Local relative imports
sys.path.append("..")  # noqa: E402
sys.path.append("../..")  # noqa: E402

from streamwise_job import StreamWiseJob
from streamwise_job import JobStatus

from lmm_service_manager import LMMServiceManager

from scene import SceneSegment

from console_utils import bytes_to_human

from file_utils import save_base64_as_binary
from file_utils import read_file_bytes
from file_utils import read_file_base64
from media_utils import chunk_video_binary
from media_utils import chunk_audio_base64
from media_utils import get_video_frames
from media_utils import extract_audio_from_video
from media_utils import concatenate_videos

from edit_prompts import build_edit_prompt
from edit_prompts import EDIT_PROMPT


class StreamEditJob(StreamWiseJob):
    """A job to generate an edited video."""

    def __init__(
        self,
        job_id: str,
        service_manager: LMMServiceManager,
        config: Dict[str, Any] = {},
    ) -> None:
        super().__init__(
            "streamedit",
            job_id,
            service_manager,
            config)
        self.scenes: List[SceneSegment] = []  # Populated by detect_scenes() during gen_edit()

    @override
    async def generate(
        self,
        job_config: Dict[str, Any],
    ) -> None:
        video_base64 = job_config.get("video_base64", None)
        assert video_base64 is not None
        await self.gen_edit(video_base64)

    async def _save_video_as_output(self, video_path: str) -> None:
        """Copy a video file to the job output path."""
        out_path = f"{self.job_path}/{self.job_id}.mp4"
        video_binary = await read_file_bytes(video_path)
        async with aiofiles.open(out_path, "wb") as file:
            await file.write(video_binary)
        self.logger.info(
            f"Video ({bytes_to_human(len(video_binary))}) saved to '{out_path}'.")

    async def gen_edit(
        self,
        video_base64: str,
    ) -> None:
        """
        Generate an edited video given an input video.
        """
        async with self.job_status_handler():
            if not video_base64:
                self.logger.error("Video is required.")
                await self.save_status(JobStatus.FAILED)
                raise ValueError("Missing 'video_base64' in request")
            self.logger.info(f"Generating edit for video with {bytes_to_human(len(video_base64))}.")

            # Save video locally for processing and debugging
            self.logger.info(f"Saving input video with {bytes_to_human(len(video_base64))}.")
            video_path = f"{self.job_path}/video.mp4"
            await save_base64_as_binary(video_path, video_base64)

            await self.save_status(JobStatus.RUNNING)

            # Detect scenes first (this validates the video and may take time)
            self.scenes = await self.detect_scenes(video_path)
            self.logger.info(f"Detected {len(self.scenes)} scenes.")

            # Write scenes debug file
            scenes_path = f"{self.job_path}/scenes.json"
            async with aiofiles.open(scenes_path, "w") as scene_file:
                scenes_dict_list = [asdict(scene) for scene in self.scenes]
                await scene_file.write(json.dumps(scenes_dict_list, indent=2))

            await self.save_status(JobStatus.RUNNING)

            # Build edit prompt from user instructions
            edit_instructions = self.get_config_str("edit_instructions")
            edit_prompt = build_edit_prompt(edit_instructions)

            # If no scenes detected, fall back to returning the original video unchanged
            if not self.scenes:
                self.logger.warning("No scenes detected. Saving original video as output.")
                await self._save_video_as_output(video_path)
                return

            # Extract full audio from input video for re-use across scenes
            audio_path = f"{self.job_path}/audio.wav"
            audio_path = await extract_audio_from_video(video_path, audio_path)
            audio_base64 = await read_file_base64(audio_path)
            self.logger.info(f"Extracted audio with {bytes_to_human(len(audio_base64))}.")

            # Edit each scene
            scene_video_paths: List[Optional[str]] = []
            for scene in self.scenes:
                try:
                    scene_path = await self.gen_edit_scene(scene, audio_base64, edit_prompt)
                    scene_video_paths.append(scene_path)
                    self.logger.info(f"[{scene.scene_id}] Edited scene saved to '{scene_path}'.")
                except Exception as ex:
                    self.logger.error(f"[{scene.scene_id}] Error editing scene: {ex}")
                    scene_video_paths.append(None)

            await self.save_status(JobStatus.RUNNING)

            # Combine the edited scenes into a final video
            valid_paths = [p for p in scene_video_paths if p]
            if not valid_paths:
                # No scenes could be edited – fall back to the original video
                self.logger.warning("No edited scenes produced. Saving original video as output.")
                await self._save_video_as_output(video_path)
                return

            self.logger.info(f"Combining {len(valid_paths)} edited scenes into final video...")
            scene_binaries: List[bytes] = []
            for path in valid_paths:
                scene_binary = await read_file_bytes(path)
                scene_binaries.append(scene_binary)

            video_binary = await concatenate_videos(scene_binaries)
            if not video_binary:
                raise ValueError("Cannot concatenate edited scenes into final video.")

            out_path = f"{self.job_path}/{self.job_id}.mp4"
            async with aiofiles.open(out_path, "wb") as file:
                await file.write(video_binary)

            self.logger.info(
                f"Generated edited video with {bytes_to_human(len(video_binary))} at '{out_path}'.")

    async def detect_scenes(
        self,
        video_path: str,
        threshold: float = 27.0,
        min_scene_len: int = 15,
    ) -> List[SceneSegment]:
        """
        Return list of scenes for a video.
        TODO Move to a library later.
        """
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

    async def gen_edit_scene(
        self,
        scene: SceneSegment,
        audio_base64: str,
        edit_prompt: str = EDIT_PROMPT,
    ) -> str:
        """
        Generate edited version of a scene and save it to disk.
        Returns the path to the saved file.
        """
        input_video_path = f"{self.job_path}/video.mp4"
        input_video_binary = await read_file_bytes(input_video_path)

        scene_binary = chunk_video_binary(
            video_binary=input_video_binary,
            start_seconds=scene.start_sec,
            end_seconds=scene.end_sec,
        )
        scene_video_frames = await get_video_frames(scene_binary)

        # Chunk audio for this scene's time range
        scene_audio_base64 = chunk_audio_base64(
            audio_base64=audio_base64,
            start_seconds=scene.start_sec,
            end_seconds=scene.end_sec,
        )

        scene_edit_binary = await self.gen.gen_video_audio_from_video(
            video=scene_video_frames,
            audio_base64=scene_audio_base64,
            prompt=edit_prompt,
            task_id=f"{scene.scene_id:03d}",
            deadline=self.get_submission_time() + scene.start_sec,
        )

        scene_path = f"{self.job_path}/{scene.scene_id:03d}_edit.mp4"
        async with aiofiles.open(scene_path, "wb") as file:
            await file.write(scene_edit_binary)
        return scene_path
