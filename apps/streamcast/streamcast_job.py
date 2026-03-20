"""
StreamCast job to generate a video podcast.
It coordinates the execution of the different models.
"""
import os
import sys
import time
import asyncio
import json
import aiofiles

from PIL import Image

from typing import override
from typing import List
from typing import Dict
from typing import Optional
from typing import Any
from typing import Tuple

# Local relative imports
sys.path.append("..")  # noqa: E402
sys.path.append("../..")  # noqa: E402

from streamwise_job import StreamWiseJob
from streamwise_job import JobStatus
from streamwise_job import OutputMode
from streamwise_job import MAX_LOG_TEXT

from resolutions import RESOLUTIONS

from lmm_service_manager import LMMServiceManager

from podcast_prompts import IMG_PROMPT_BASE
from podcast_prompts import IMG_PROMPT
from podcast_prompts import IMG_NEG_PROMPT
from podcast_prompts import IMG_ZOOM_PROMPT
from podcast_prompts import VIDEO_PROMPT
from podcast_prompts import VIDEO_NEG_PROMPT

from character import Character

from video import FANTASYTALKING_FPS
from video import HUNYUANFRAMEPACK_FPS
from video import MAX_FT_DURATION_SECS
from video import VAE_T

from gen_video_chunked import GenVideoChunked

from file_utils import read_file_base64
from file_utils import save_base64_as_binary
from file_utils import base64_to_binary

from media_utils import add_text_to_frame
from media_utils import get_video_with_text
from media_utils import get_font_size
from media_utils import get_video_frames
from media_utils import get_video_duration
from media_utils import get_audio_duration
from media_utils import get_video_fps
from media_utils import get_video_file_info
from media_utils import get_aligned_duration
from media_utils import get_audio_file_info
from media_utils import fit_audio_to_duration
from media_utils import save_video_frames
from media_utils import save_video_audio
from media_utils import concatenate_videos
from media_utils import split_text_lines

from console_utils import bytes_to_human

from tts_utils import strip_audio_file_silence


MAX_IMG_LINE_CHARS = 50  # Max characters per line in the output image/video


class StreamCastJob(StreamWiseJob):
    """A job to generate a podcast with images, audio, and video."""

    def __init__(
        self,
        job_id: str,
        service_manager: LMMServiceManager,
        config: Dict[str, Any] = {},
    ) -> None:
        super().__init__(
            "streamcast",
            job_id,
            service_manager,
            config)

    @override
    async def generate(
        self,
        job_config: Dict[str, Any],
    ) -> None:
        pdf_base64 = job_config.get("pdf_base64", None)
        await self.gen_podcast(pdf_base64)

    async def align_audio(
        self,
        audio_path: str,
    ) -> Tuple[str, float]:
        """
        Strip end silence and extend audio duration to align with the video constraints:
          1+4n frames for VAE (e.g., 4 for Fantasy Talking)
          Frames per second (e.g., 23 FPS for Fantasy Talking)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not audio_path.lower().endswith(".wav"):
            raise ValueError(f"Audio file must be a WAV file: {audio_path}")

        audio_info = get_audio_file_info(audio_path)
        audio_duration = audio_info["duration_seconds"]

        audio_path_trimmed = audio_path.rstrip(".wav")
        stripped_audio_path = f"{audio_path_trimmed}_stripped.wav"
        strip_audio_file_silence(
            audio_path,
            output_path=stripped_audio_path,
            strip_start=True,  # False to give more natural start
            strip_end=True)
        stripped_audio_info = get_audio_file_info(stripped_audio_path)
        stripped_audio_duration = stripped_audio_info["duration_seconds"]

        # Choose the right FPS alignment
        output_mode = self.get_config_output_mode()
        fps = FANTASYTALKING_FPS  # For video+audio synced use Fantasy Talking FPS
        if output_mode is not OutputMode.VIDEO_AUDIO_SYNCED:
            fps = HUNYUANFRAMEPACK_FPS  # Audio and unsynced use FramePack FPS

        new_audio_duration = get_aligned_duration(
            stripped_audio_duration,
            fps=fps,
            vae=VAE_T)

        if audio_duration == new_audio_duration:
            return audio_path, audio_duration  # It was already aligned

        aligned_audio_path = f"{audio_path_trimmed}_aligned.wav"
        aligned_audio_path = fit_audio_to_duration(
            stripped_audio_path,
            new_audio_duration,
            aligned_audio_path)
        num_frames = new_audio_duration * fps
        self.logger.info(
            f"Aligned audio with {audio_duration:.3f} to {new_audio_duration:.3f} "
            f"using {fps} FPS and {VAE_T} VAE for {num_frames:.1f} frames.")
        return aligned_audio_path, new_audio_duration

    async def gen_scene(
        self,
        scene_id: int,
        character: Character,
        text: str,
        video_prompt: str = VIDEO_PROMPT,
        video_neg_prompt: str = VIDEO_NEG_PROMPT,
    ) -> str:
        """
        Generate a scene with audio and video starting with image at high resolution.
        1. Generate video+audio at medium resolution with Fantasy Talking (single or by chunks).
        3. Upscale video+audio to high resolution.
        """
        t0 = time.time()
        width, height = self.width, self.height
        image = character.image
        if image:
            width, height = image.size
        log_text = text[:MAX_LOG_TEXT] + "..." if len(text) > MAX_LOG_TEXT else text
        self.logger.info(
            f"[{scene_id}] Generating scene for character {character.name} "
            f"with image {width}x{height} and text '{log_text}'...")

        # Generate audio (kokoro)
        audio_base64 = await self.gen.gen_audio(
            text,
            voice=character.voice,
            speed=character.speech_speed,
            task_id=f"{scene_id:03d}",
            deadline=self.get_scene_deadline(scene_id))
        if audio_base64 is None:
            raise ValueError(f"Cannot generate audio for scene {scene_id} with text '{log_text}'")
        audio_duration = get_audio_duration(audio_base64)
        audio_path = f"{self.job_path}/{scene_id:03d}.wav"
        await save_base64_as_binary(audio_path, audio_base64)
        self.logger.info(
            f"[{scene_id}] Generated audio with {bytes_to_human(len(audio_base64))} and {audio_duration:.3f} seconds.")

        aligned_audio_path, aligned_audio_duration = await self.align_audio(audio_path)
        if audio_duration != aligned_audio_duration:
            audio_path = aligned_audio_path
            audio_duration = aligned_audio_duration

        # Generate base video
        output_mode = self.get_config_output_mode()
        if output_mode is OutputMode.AUDIO_ONLY:
            # Generate video with text slides
            frame_text = f"{character.name}:\n" + "\n".join(split_text_lines(text, MAX_IMG_LINE_CHARS))
            font_size = get_font_size(width, height)
            font_color = self.characters.get_color(character.name)
            scene_video_binary = await get_video_with_text(
                width, height,
                frame_text,
                duration_seconds=audio_duration,
                fps=HUNYUANFRAMEPACK_FPS,
                font_size=font_size,
                font_color=font_color)
        elif audio_duration > MAX_FT_DURATION_SECS and output_mode == OutputMode.VIDEO_AUDIO_SYNCED:
            # Generate video synced with the audio (multiple sub-shots)
            self.logger.info(
                f"[{scene_id}] Scene is too long ({audio_duration:.3f}>{MAX_FT_DURATION_SECS:.3f} seconds), "
                "split into sub-shots.")
            gen_video_chunked = GenVideoChunked(
                video_id=scene_id,
                gen=self.gen,
                job_path=self.job_path,
                logger=self.logger)
            scene_video_binary = await gen_video_chunked.gen_video_chunked(
                audio_path=audio_path,
                image=self.image,
                prompt=video_prompt,
                neg_prompt=video_neg_prompt,
                width=width,
                height=height,
                num_steps=self.get_num_steps(),
                upscaling=self.get_config_bool("upscaling"),
                debug=self.get_config_bool("debug_image"),
                deadline=self.get_scene_deadline(scene_id),
            )
            """
            scene_video_binary = await self.gen_scene_chunks(
                scene_id,
                audio_path,
                image,
                video_prompt=video_prompt,
                video_neg_prompt=video_neg_prompt)
            """
        else:
            # Generate video synced with the audio
            scene_video_binary = await self.gen_scene_single(
                scene_id,
                audio_path,
                image,
                video_prompt=video_prompt,
                video_neg_prompt=video_neg_prompt)

        video_path = f"{self.job_path}/{scene_id:03d}_pre.mp4"
        async with aiofiles.open(video_path, "wb") as file:
            await file.write(scene_video_binary)

        self._log_video_info(f"[{scene_id}] Generated video", scene_video_binary)

        # Upscale video (without the audio) to high resolution
        video_file_info = get_video_file_info(scene_video_binary)
        video_info = video_file_info["video"]
        src_width, src_height = video_info["width"], video_info["height"]
        width, height = self.width, self.height
        if src_width < width or src_height < height:
            self.logger.info(
                f"[{scene_id}] Upscaling video from {src_width}x{src_height} to {width}x{height}...")
            scene_video_upscaled_binary = await self.gen.gen_video_upscale(
                video_binary=scene_video_binary,
                width=width,
                height=height,
                task_id=f"{scene_id:03d}_upscale",
                deadline=self.get_scene_deadline(scene_id))
            if scene_video_upscaled_binary is None:
                raise ValueError(f"Cannot upscale video for scene {scene_id}")

            self._log_video_info(
                f"[{scene_id}] Upscaled video with {src_width}x{src_height} to video ",
                scene_video_upscaled_binary)
            scene_video_binary = scene_video_upscaled_binary

        video_frames = await get_video_frames(scene_video_binary)

        # Check video vs audio duration
        video_file_info = get_video_file_info(scene_video_binary)
        video_info = video_file_info["video"]
        video_fps = video_info["fps"]
        video_duration = video_info["duration_seconds"]
        video_num_frames = video_info["num_frames"]
        if len(video_frames) != video_num_frames:
            self.logger.warning(
                f"[{scene_id}] Video frames mismatch: {len(video_frames)} != {video_num_frames}.")

        if audio_duration != video_duration:
            self.logger.warning(
                f"[{scene_id}] Audio duration ({audio_duration:.3f} seconds) is different than video "
                f"({video_duration:.3f} seconds "
                f"= {video_num_frames} / {video_fps} FPS).")

        # Add subtitles
        if self.get_config_bool("add_subtitles") and output_mode is not OutputMode.AUDIO_ONLY:
            # TODO it does not support multiple lines yet
            # frame_text = f"{character.name}:\n" + "\n".join(split_text_lines(text, MAX_IMG_LINE_CHARS))
            frame_text = f"{character.name}: {text}"
            video_frames = await self._overlay_subtitles_on_frames(scene_video_binary, frame_text)

        # Add audio back to the video
        video_audio_path = f"{self.job_path}/{scene_id:03d}.mp4"
        video_audio_path = await save_video_audio(
            video_content=video_frames,
            audio_path=audio_path,
            out_video_path=video_audio_path,
            fps=video_fps)

        self._log_video_info(f"[{scene_id}] Generated scene", video_audio_path)
        self.logger.info(f"[{scene_id}] Generated scene in {time.time() - t0:.3f} seconds.")

        return video_audio_path

    async def gen_scene_single(
        self,
        scene_id: int,
        audio_path: str,
        image: Image.Image,
        video_prompt: str = VIDEO_PROMPT,
        video_neg_prompt: str = VIDEO_NEG_PROMPT,
    ) -> bytes:
        """
        Generate video+audio in a single shot at medium resolution with Fantasy Talking.
        Returns video synced with the audio in base64.
        """
        audio_base64 = await read_file_base64(audio_path)
        audio_duration = get_audio_duration(audio_base64)

        width, height = self.width, self.height
        if self.get_config_bool("upscaling"):
            # width, height = RESOLUTIONS[self.aspect_ratio]["medium"]
            width = self.width // 2
            height = self.height // 2

        output_mode = self.get_config_output_mode()
        num_steps = self.get_num_steps()
        if output_mode == OutputMode.VIDEO_AUDIO_SYNCED:
            self.logger.info(f"[{scene_id}] Generating video+audio with {audio_duration:.3f} seconds...")
            video_binary = await self.gen.gen_video_audio_from_img(
                img=image,
                audio_base64=audio_base64,
                prompt=video_prompt,
                neg_prompt=video_neg_prompt,
                width=width,
                height=height,
                steps=num_steps,
                task_id=f"{scene_id:03d}",
                deadline=self.get_scene_deadline(scene_id),
            )
        else:
            self.logger.info(f"[{scene_id}] Generating video with {audio_duration:.3f} seconds...")
            video_binary = await self.gen.gen_video(
                img=image,
                prompt=video_prompt,
                neg_prompt=video_neg_prompt,
                width=width,
                height=height,
                video_seconds=audio_duration,  # Because of rounding, this may produce more frames than asked
                steps=num_steps,
                task_id=f"{scene_id:03d}",
                wait_request=True,
                deadline=self.get_scene_deadline(scene_id),
            )

        # Mismatch because we use ffmpeg for get_video_frames()
        video_frames = await get_video_frames(video_binary)
        video_num_frames = len(video_frames)
        video_file_info = get_video_file_info(video_binary)
        video_info = video_file_info["video"]
        video_fps: float = video_info["fps"]
        video_duration: float = video_info["duration_seconds"]
        if video_num_frames != video_info["num_frames"]:
            self.logger.warning(
                f"[{scene_id}] Video frames mismatch: {video_num_frames} != {video_info['num_frames']}.")

        video_path = f"{self.job_path}/{scene_id:03d}_{width}x{height}_single.mp4"
        async with aiofiles.open(video_path, "wb") as file:
            await file.write(video_binary)

        # Sanity check for one frame or more
        if abs(video_duration - audio_duration) >= 1.0 / video_fps:
            self.logger.warning(
                f"[{scene_id}] Generated video duration mismatch: "
                f"{video_duration:.3f} != {audio_duration:.3f} seconds.")

        self._log_video_info(f"[{scene_id}] Generated video", video_binary)

        if self.get_config_bool("debug_image"):
            frame_text = f"{scene_id:03d}"
            video_frames = [
                add_text_to_frame(frame, text=frame_text, position="top-left")
                for frame in video_frames
            ]
            video_path = f"{self.job_path}/{scene_id:03d}_{width}x{height}_single_debug.mp4"
            await save_video_frames(
                video_frames=video_frames,
                fps=video_fps,
                out_video_path=video_path)
            async with aiofiles.open(video_path, "rb") as file:
                video_binary = await file.read()

            self._log_video_info(
                f"[{scene_id}] Added debug text to video",
                video_binary)

        return video_binary

    async def gen_images(
        self,
        img_prompt: str,
        use_image_edit: bool = True,
    ) -> Tuple[Image.Image, List[Image.Image]]:
        """
        Generate main image and character images.
        """
        width, height = RESOLUTIONS[self.aspect_ratio]["high"]
        self.logger.info(f"Generating image with size {width}x{height} and prompt: {img_prompt}.")

        # Generate the main image
        img_neg_prompt = IMG_NEG_PROMPT
        image = await self.gen.gen_image(
            img_prompt,
            neg_prompt=img_neg_prompt,
            width=width,
            height=height,
            # TODO steps=25,
            task_id="main_image",
            deadline=self.get_submission_time())
        if image is not None:
            width, height = image.size
            image_path = f"{self.job_path}/main_image.png"
            image.save(image_path)
            self.logger.info(f"Image with {width}x{height} pixels saved to '{image_path}'.")

        # Get the sub-images from the main image
        num_characters = len(self.characters)
        character_images = await self.gen.gen_extract_characters(
            image,
            num_characters,
            task_id="extract_characters",
            deadline=self.get_submission_time())

        if not character_images:
            raise ValueError("No characters extracted from the image.")
        if len(character_images) != num_characters:
            raise ValueError(f"Expected {num_characters} characters, but got {len(character_images)}.")

        self.logger.info(f"Editing {len(character_images)} characters from the image.")
        for character_ix, character_image in enumerate(character_images):
            if use_image_edit:
                img_zoom_prompt = IMG_ZOOM_PROMPT
                character = self.characters.get_by_index(character_ix)
                if character and character.description:
                    img_zoom_prompt += f"\nThe character is a {character.gender} "
                    img_zoom_prompt += f"and their description is {character.description}."
                position = self.characters.get_position(character.name)
                if position == "left":
                    img_zoom_prompt += "\nThe character is looking to the right."
                elif position == "right":
                    img_zoom_prompt += "\nThe character is looking to the left."
                elif position == "center":
                    img_zoom_prompt += "\nThe character is looking forward."
                character_image = await self.gen.gen_edit_image(
                    character_image,
                    prompt=img_zoom_prompt,
                    neg_prompt=img_neg_prompt,
                    width=width,
                    height=height,
                    task_id=f"character_{character_ix:03d}",
                    deadline=self.get_submission_time())  # TODO deadlines for each image
            character_images[character_ix] = character_image

        self.logger.info(f"Extracted {len(character_images)} characters from the image.")
        for character_ix, character_image in enumerate(character_images):
            if character_ix >= len(self.characters):
                self.logger.warning(f"Character {character_ix} not found in {self.characters}.")
            character = self.characters.get_by_index(character_ix)
            character.image = character_image
            character_image_path = f"{self.job_path}/character_{character_ix + 1:03d}.png"
            character_image.save(character_image_path)
            width, height = character_image.size
            self.logger.info(f"Character {character_ix + 1} {width}x{height} saved to '{character_image_path}'.")

        return image, character_images

    async def gen_podcast(
        self,
        pdf_base64: str,
    ) -> None:
        """Generate a podcast."""
        async with self.job_status_handler():
            if not pdf_base64:
                self.logger.error("Document is required.")
                await self.save_status(JobStatus.FAILED)
                raise ValueError("Missing 'pdf_base64' in request")
            self.logger.info(f"Generating podcast transcript for document with {bytes_to_human(len(pdf_base64))}.")

            self.logger.info(f"Document base64 with {bytes_to_human(len(pdf_base64))}.")
            async with aiofiles.open(f"{self.job_path}/document.pdf", "wb") as file:
                pdf_binary = base64_to_binary(pdf_base64)
                await file.write(pdf_binary)

            # Default prompts
            img_prompt = IMG_PROMPT
            video_prompt = VIDEO_PROMPT
            video_neg_prompt = VIDEO_NEG_PROMPT

            await self.save_status(JobStatus.RUNNING)

            # Configurations
            num_characters = self.get_config_int("num_characters")
            output_mode = self.get_config_output_mode()

            # Podcast transcript
            future_images: Optional[asyncio.Task] = None
            scene_id = 0
            async with aiofiles.open(f"{self.job_path}/podcast_transcript.jsonl", "wb") as file:
                async for line_json in self.gen.gen_podcast_transcript(
                    task_id="transcript",
                    pdf_base64=pdf_base64,
                    num_characters=num_characters,
                    style_prompt=self.get_config_str("style_prompt"),
                    scene_prompt=self.get_config_str("scene_prompt"),
                    custom_prompt=self.get_config_str("custom_prompt"),
                    max_tokens=self.get_config_int("max_tokens"),
                    max_dialogues=self.get_config_int("max_dialogues"),
                    max_words_per_dialogue=self.get_config_int("max_words_per_dialogue"),
                ):
                    line = (json.dumps(line_json) + "\n").encode("utf-8")
                    await file.write(line)
                    await file.flush()

                    line_type = line_json.get("type", "")
                    if line_type == "image":
                        img_prompt = line_json.get("content", None)
                        img_prompt += "\n" + IMG_PROMPT_BASE
                        if num_characters == 1:
                            img_prompt += f"\nThe image shows {num_characters} character. "
                        elif num_characters > 1:
                            img_prompt += f"\nThe image shows {num_characters} characters. "
                        self.logger.info(f"Image prompt: {img_prompt}")
                    elif line_type == "character":
                        character_name = line_json.get("name", "Unknown")
                        character_gender = line_json.get("gender", "Unknown")
                        character_description = line_json.get("description", "")
                        character = Character(
                            name=character_name,
                            gender=character_gender,
                            description=character_description)
                        if "speech_speed" in self.config:
                            character.speech_speed = float(self.config["speech_speed"])
                        self.characters[character_name] = character
                        self.logger.info(f"Character: {character_name}, {character_gender}, {character_description}")
                        img_prompt += f"{character_name} is a {character_gender} "
                        img_prompt += f"and the description is {character_description}\n"

                        if len(self.characters) == num_characters and output_mode is not OutputMode.AUDIO_ONLY:
                            # Trigger async image generation
                            future_images = asyncio.create_task(self.gen_images(
                                img_prompt=img_prompt,
                                use_image_edit=self.get_config_bool("edit_image"),
                            ))
                    elif line_type == "dialogue":
                        character_name = line_json.get("character", "Unknown")
                        dialogue_content = line_json.get("content", "")
                        self.logger.info(f"Scene {scene_id}: [{character_name}] {dialogue_content}")
                        # TODO we could trigger the task for gen_scene()
                        self.transcript_scenes.append(line_json)
                        scene_id += 1
            self.logger.info("Transcript generated.")

            await self.save_status(JobStatus.RUNNING)

            # Generate images
            img_main: Optional[Image.Image] = None
            if future_images is None and output_mode is not OutputMode.AUDIO_ONLY:
                future_images = asyncio.create_task(self.gen_images(
                    img_prompt=img_prompt,
                    use_image_edit=self.get_config_bool("edit_image"),
                ))
            if output_mode is OutputMode.AUDIO_ONLY:
                self.logger.debug("Audio only mode, skipping image generation.")
            else:
                img_main, img_characters = await future_images
                self.logger.info(f"Main and {len(img_characters)} character images generated.")

            # Generate each scene
            scene_video_tasks: Dict[int, asyncio.Task] = {}
            for scene_id, scene_json in enumerate(self.transcript_scenes):
                character_name = scene_json.get("character", "Unknown")
                if character_name in self.characters:
                    character = self.characters[character_name]
                else:
                    self.logger.warning(f"Character '{character_name}' not found in characters. Using default.")
                    character = Character(name=character, gender="Unknown")
                if character.image is None and output_mode is not OutputMode.AUDIO_ONLY:
                    self.logger.warning(f"Character '{character_name}' has no image. Using main image as fallback.")
                    character.image = img_main  # Use the main image as a fallback
                character_position = self.characters.get_position(character_name)
                character_video_prompt = video_prompt % (character.gender, character_position)
                if "content" not in scene_json:
                    self.logger.warning(f"[{scene_id}] No content for scene. Skipping...")
                else:
                    scene_text = scene_json["content"]
                    scene_video_task = asyncio.create_task(
                        self.gen_scene(
                            scene_id=scene_id,
                            character=character,
                            text=scene_text,
                            video_prompt=character_video_prompt,
                            video_neg_prompt=video_neg_prompt
                        ))
                    scene_video_tasks[scene_id] = scene_video_task

            await self.save_status(JobStatus.RUNNING)

            scene_video_paths: List[str] = []
            results = await asyncio.gather(*scene_video_tasks.values(), return_exceptions=True)
            for scene_id, result in enumerate(results):
                if isinstance(result, Exception):
                    self._handle_scene_exception(scene_id, result)
                elif not result:
                    self.logger.warning(f"[{scene_id}] No video generated. Skipping...")
                else:
                    scene_video_paths.append(result)
                    self.logger.info(f"[{scene_id}] Video+audio saved to '{result}'.")

            if not scene_video_paths:
                raise ValueError("No scene videos generated. Cannot create final podcast video.")

            await self.save_status(JobStatus.RUNNING)

            # Concatenate all scene videos into a final podcast video
            self.logger.info(f"Concatenating {len(scene_video_paths)} scenes into final podcast video...")
            scene_binaries: List[bytes] = []
            for scene_video_path in scene_video_paths:
                async with aiofiles.open(scene_video_path, "rb") as file:
                    scene_binary = await file.read()
                    scene_binaries.append(scene_binary)

            video_binary = await concatenate_videos(
                scene_binaries,
                fast_copy=False)  # TODO move to True once we fix the durations
            if not video_binary:
                raise ValueError("Cannot concatenate scenes into final podcast video.")
            video_duration = await get_video_duration(video_binary)
            video_fps = get_video_fps(video_binary)

            video_path = f"{self.job_path}/{self.job_id}.mp4"
            async with aiofiles.open(video_path, "wb") as file:
                await file.write(video_binary)

            self.logger.info(
                f"Generated podcast video with {video_duration:.3f} seconds, "
                f"{video_fps} FPS, "
                f"{bytes_to_human(len(video_binary))}, and "
                f"'{video_path}'")
