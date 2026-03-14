"""
StreamPersona job to generate a video podcast.
It coordinates the execution of the different models.
"""
import sys

from typing import override
from typing import Dict
from typing import Any


# Local relative imports
sys.path.append("..")  # noqa: E402

from streamwise_job import StreamWiseJob

from lmm_service_manager import LMMServiceManager

from movie_prompts import SYSTEM_PROMPT


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
        """
        async with self.job_status_handler():
            # TODO: call gen_text with build_movie_messages to generate movie structure
            # movie_structure = await self.gen.gen_text(
            #     self.build_movie_messages(movie_description), task_id="movie_plan"
            # )
            # Generate images for each scene
            img_prompt = f"Movie scene: {movie_description}"
            image = await self.gen.gen_image(
                prompt=img_prompt,
                task_id="main_image",
                deadline=self.get_submission_time(),
            )
            # Generate video from images
            # video_frames =
            video_prompt = f"Create a movie scene based on: {movie_description}"
            await self.gen.gen_video(
                img=image,
                prompt=video_prompt,
                task_id="main_video",
                deadline=self.get_submission_time(),
            )
            # Generate audio narration
            audio_prompt = f"Create an audio narration based on: {movie_description}"
            # audio =
            await self.gen.gen_audio(
                text=audio_prompt,
                task_id="main_audio",
                deadline=self.get_submission_time(),
            )
            # Generate conversation/dialogues if any
            # Combine video and audio into final movie file
            # TODO
            return
