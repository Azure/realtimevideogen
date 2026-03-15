"""
StreamEdit job to generate an edited video.
"""
import sys
import aiofiles
import aiofiles.os

from typing import override
from typing import Any
from typing import Dict
from typing import List

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
from media_utils import chunk_video_binary
from file_utils import read_file_bytes
from media_utils import get_video_frames

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

    @override
    async def generate(
        self,
        job_config: Dict[str, Any],
    ) -> None:
        video_base64 = job_config.get("video_base64", None)
        assert video_base64 is not None
        await self.gen_edit(video_base64)

    async def gen_edit(
        self,
        video_base64: str,
    ) -> None:
        """
        Generate an edited video given an input video.
        """
        async with self.job_status_handler():
            if not video_base64:
                self.logger.info("Video is required.")
                await self.save_status(JobStatus.FAILED)
                raise ValueError("Missing 'video_base64' in request")
            self.logger.info(f"Generating edit for video with {bytes_to_human(len(video_base64))}.")

            # Save video locally for processing and debugging
            self.logger.info(f"Saving input video with {bytes_to_human(len(video_base64))}.")
            video_path = f"{self.job_path}/video.mp4"
            await save_base64_as_binary(video_path, video_base64)

            await self.save_status(JobStatus.RUNNING)

            # Detect scenes
            self.scenes = await self.detect_scenes(video_path)
            self.logger.info(f"Detected {len(self.scenes)} scenes.")

            await self.save_status(JobStatus.RUNNING)

            # Edit the video
            for scene in self.scenes:
                await self.gen_edit_scene(scene)

            # Combine the edited scenes into a final video
            # TODO

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
    ) -> bytes:
        """
        Generate edited version of a scene.
        """
        input_video_path = f"{self.job_path}/video.mp4"
        input_video_binary = await read_file_bytes(input_video_path)

        scene_binary = chunk_video_binary(
            video_binary=input_video_binary,
            start_seconds=scene.start_sec,
            end_seconds=scene.end_sec,
        )
        scene_video_frames = await get_video_frames(scene_binary)

        # TODO create method for editing
        scene_edit_binary = await self.gen.gen_video_audio_from_video(
            video=scene_video_frames,
            audio_base64="TODO",
            prompt=EDIT_PROMPT,
            task_id=f"{scene.scene_id:03d}",
            deadline=self.get_submission_time() + scene.start_sec,
        )
        return scene_edit_binary
