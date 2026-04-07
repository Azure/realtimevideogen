"""
StreamAnimate job to generate an animated video.
"""
import sys
import aiofiles

from PIL import Image

from io import BytesIO
from base64 import b64decode

from typing import override
from typing import Any
from typing import Dict
from typing import Optional

# Local relative imports
sys.path.append("..")  # noqa: E402
sys.path.append("../..")  # noqa: E402

from streamwise_job import StreamWiseJob
from streamwise_job import JobStatus
from streamwise_job import OutputMode

from lmm_service_manager import LMMServiceManager

from animate_prompts import IMG_PROMPT
from animate_prompts import IMG_NEG_PROMPT
from animate_prompts import VIDEO_PROMPT
from animate_prompts import VIDEO_NEG_PROMPT

from console_utils import bytes_to_human

from file_utils import save_base64_as_binary

from media_utils import get_audio_duration
from media_utils import get_video_frames
from media_utils import get_video_file_info
from media_utils import save_video_audio

from video import MAX_FT_DURATION_SECS
from video import FANTASYTALKING_FPS


def _append_base_prompt(user_text: str, base_prompt: str) -> str:
    """Combine an optional user text prefix with a base prompt string."""
    if user_text:
        return f"{user_text.rstrip('.!?,;')}. {base_prompt}"
    return base_prompt


class StreamAnimateJob(StreamWiseJob):
    """A job to generate an animated video."""

    def __init__(
        self,
        job_id: str,
        service_manager: LMMServiceManager,
        config: Dict[str, Any] = {},
    ) -> None:
        super().__init__(
            "streamanimate",
            job_id,
            service_manager,
            config)

    @override
    async def generate(
        self,
        job_config: Dict[str, Any],
    ) -> None:
        image_base64: Optional[str] = job_config.get("image_base64", None)
        text_prompt: str = job_config.get("text_prompt", "")
        narration_text: str = job_config.get("narration_text", "")
        await self.gen_animate(
            image_base64=image_base64,
            text_prompt=text_prompt,
            narration_text=narration_text,
        )

    async def gen_animate(
        self,
        image_base64: Optional[str],
        text_prompt: str = "",
        narration_text: str = "",
    ) -> None:
        """
        Generate an animated video from an image (or a text-to-image prompt).

        Pipeline:
          1. If no image is provided, generate one from text_prompt.
          2. Generate a video animation from the image.
          3. If narration_text is provided, generate TTS audio and merge it with the video.
          4. Save the final video.
        """
        async with self.job_status_handler():
            output_mode = self.get_config_output_mode()
            num_steps = self.get_num_steps()

            # --- Step 1: Obtain source image ---
            image: Optional[Image.Image] = None
            if image_base64:
                self.logger.info(f"Using uploaded image with {bytes_to_human(len(image_base64))}.")
                image_bytes = b64decode(image_base64)
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                image_path = f"{self.job_path}/input_image.png"
                image.save(image_path)
                self.logger.info(f"Input image {image.width}x{image.height} saved to '{image_path}'.")
            else:
                if not text_prompt:
                    raise ValueError("Either 'image_base64' or 'text_prompt' must be provided.")
                img_prompt = _append_base_prompt(text_prompt, IMG_PROMPT)
                self.logger.info(f"Generating image for prompt: '{img_prompt[:80]}'.")
                image = await self.gen.gen_image(
                    prompt=img_prompt,
                    neg_prompt=IMG_NEG_PROMPT,
                    width=self.width,
                    height=self.height,
                    task_id="input_image",
                    deadline=self.get_submission_time(),
                )
                if image is None:
                    raise ValueError("Failed to generate image from text prompt.")
                image_path = f"{self.job_path}/input_image.png"
                image.save(image_path)
                self.logger.info(f"Generated image {image.width}x{image.height} saved to '{image_path}'.")

            await self.save_status(JobStatus.RUNNING)

            width, height = image.size
            video_prompt = _append_base_prompt(text_prompt, VIDEO_PROMPT)
            video_duration_seconds = self.get_config_float("video_duration_seconds", 5.0)

            # --- Step 2: Generate TTS narration (if requested) ---
            audio_base64: Optional[str] = None
            audio_path: Optional[str] = None
            if narration_text and output_mode is not OutputMode.AUDIO_ONLY:
                voice = "af_heart"
                speed = self.get_config_float("speech_speed", 1.1)
                self.logger.info(f"Generating narration audio for: '{narration_text[:80]}'.")
                audio_base64 = await self.gen.gen_audio(
                    narration_text,
                    voice=voice,
                    speed=speed,
                    task_id="narration",
                    deadline=self.get_submission_time(),
                )
                if audio_base64:
                    audio_duration = get_audio_duration(audio_base64)
                    audio_path = f"{self.job_path}/narration.wav"
                    await save_base64_as_binary(audio_path, audio_base64)
                    self.logger.info(
                        f"Narration audio: {bytes_to_human(len(audio_base64))}, "
                        f"{audio_duration:.3f} seconds.")
                    video_duration_seconds = audio_duration  # Match video length to narration

            await self.save_status(JobStatus.RUNNING)

            # --- Step 3: Generate animation video ---
            scene_video_binary: Optional[bytes] = None
            if output_mode is OutputMode.AUDIO_ONLY:
                # Audio-only mode: narration over a static image slide
                self.logger.info("Audio-only mode: skipping video generation.")
            elif (
                audio_base64
                and audio_path
                and output_mode == OutputMode.VIDEO_AUDIO_SYNCED
                and video_duration_seconds <= MAX_FT_DURATION_SECS
            ):
                # Lip-synced animation (Fantasy Talking)
                self.logger.info(
                    f"Generating lip-synced animation {width}x{height} "
                    f"for {video_duration_seconds:.1f} seconds.")
                scene_video_binary = await self.gen.gen_video_audio_from_img(
                    img=image,
                    audio_base64=audio_base64,
                    prompt=video_prompt,
                    neg_prompt=VIDEO_NEG_PROMPT,
                    width=width,
                    height=height,
                    steps=num_steps,
                    task_id="animate",
                    deadline=self.get_submission_time(),
                )
            else:
                # Standard animation (FramePack / HunyuanVideo)
                self.logger.info(
                    f"Generating animation {width}x{height} "
                    f"for {video_duration_seconds:.1f} seconds.")
                scene_video_binary = await self.gen.gen_video(
                    img=image,
                    prompt=video_prompt,
                    neg_prompt=VIDEO_NEG_PROMPT,
                    width=width,
                    height=height,
                    video_seconds=video_duration_seconds,
                    steps=num_steps,
                    task_id="animate",
                    wait_request=True,
                    deadline=self.get_submission_time(),
                )

            await self.save_status(JobStatus.RUNNING)

            # --- Step 4: Merge audio and video, then save ---
            out_path = f"{self.job_path}/{self.job_id}.mp4"
            if scene_video_binary and audio_path:
                # Merge narration audio with generated video
                video_frames = await get_video_frames(scene_video_binary)
                video_file_info = get_video_file_info(scene_video_binary)
                video_info = video_file_info.get("video", {})
                _video_fps = video_info.get("fps")
                video_fps: float = _video_fps if _video_fps is not None else FANTASYTALKING_FPS

                out_path = await save_video_audio(
                    video_content=video_frames,
                    audio_path=audio_path,
                    out_video_path=out_path,
                    fps=video_fps)
            elif scene_video_binary:
                # Video without narration
                async with aiofiles.open(out_path, "wb") as file:
                    await file.write(scene_video_binary)
            elif audio_path:
                # Audio-only: no video was generated, save narration as final output
                self.logger.info("Audio-only mode: final output is the narration audio.")
                out_path = audio_path
            else:
                raise ValueError("Neither video nor audio was generated.")

            self.logger.info(
                f"Generated animation saved to '{out_path}'.")
