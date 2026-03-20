"""
StreamDub job to generate a dubbed video.
It coordinates the execution of the different models.
"""

import sys
import json
import aiofiles

from dataclasses import asdict

from typing import override
from typing import Dict
from typing import Any
from typing import List
from typing import Optional

from dub_prompts import DUB_PROMPT
from dub_prompts import VIDEO_DUB_PROMPT
from dub_prompts import VIDEO_DUB_NEG_PROMPT

from video import MAX_FT_DURATION_SECS
from video import FANTASYTALKING_FPS

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

from file_utils import read_file_base64
from file_utils import save_base64_as_binary
from file_utils import read_file_bytes

from media_utils import concatenate_videos
from media_utils import fit_audio_to_duration
from media_utils import chunk_video_binary
from media_utils import get_video_file_info
from media_utils import get_video_frames_at_fps
from media_utils import get_audio_duration
from media_utils import extract_audio_from_video
from media_utils import get_video_frames
from media_utils import chunk_audio_base64
from media_utils import save_video_audio

from language_utils import to_language


class StreamDubJob(StreamWiseJob):
    """A job to generate a dubbed video."""

    def __init__(
        self,
        job_id: str,
        service_manager: LMMServiceManager,
        config: Dict[str, Any] = {},
    ) -> None:
        super().__init__(
            "streamdub",
            job_id,
            service_manager,
            config)

        self.service_manager = service_manager
        self.scenes: List[SceneSegment] = []

    @override
    async def generate(
        self,
        job_config: Dict[str, Any],
    ) -> None:
        video_base64 = job_config.get("video_base64", None)
        output_language = job_config.get("output_language", None)
        await self.gen_dub(
            video_base64,
            output_language)

    async def save_scenes(self) -> None:
        """Persist current scene metadata (including transcript/translation) to scenes.json."""
        scenes_path = f"{self.job_path}/scenes.json"
        async with aiofiles.open(scenes_path, "w") as scene_file:
            scenes_dict_list = [asdict(scene) for scene in self.scenes]
            scenes_json = json.dumps(scenes_dict_list, indent=2)
            await scene_file.write(scenes_json)

    async def gen_dub(
        self,
        video_base64: Optional[str] = None,
        output_language: Optional[str] = "e",  # Spanish
    ) -> None:
        """Generate a dubbed video."""
        async with self.job_status_handler():
            if not video_base64:
                self.logger.error("Video is required.")
                await self.save_status(JobStatus.FAILED)
                raise ValueError("Missing 'video_base64' in request")
            self.logger.info(f"Generating dubbed version for video with {bytes_to_human(len(video_base64))}.")

            await self.save_status(JobStatus.RUNNING)

            # Save as video for processing
            self.logger.info(f"Saving input video with {bytes_to_human(len(video_base64))}.")
            video_path = f"{self.job_path}/video.mp4"
            await save_base64_as_binary(video_path, video_base64)

            await self.save_status(JobStatus.RUNNING)

            # Detect scenes
            self.scenes = await self.detect_scenes()
            if not self.scenes:
                raise ValueError("No scenes detected in video.")
            self.logger.info(f"Detected {len(self.scenes)} scenes.")

            await self.save_status(JobStatus.RUNNING)

            # Extract audio from each scene
            await self.chunk_audio_into_scenes()

            # Persist scene metadata so the UI can display scenes with original audio
            self.logger.info("Scenes:")
            for idx, scene in enumerate(self.scenes):
                self.logger.info(f"  Scene {idx}: {scene}")
            await self.save_scenes()

            scene_binaries = []
            for scene in self.scenes:
                try:
                    video_binary = await self.gen_dub_scene(
                        scene,
                        output_language)
                    scene_binaries.append(video_binary)
                except Exception as ex:
                    self.logger.error(f"[{scene.scene_id}] Cannot generate dubbed scene: {ex}")

            await self.save_status(JobStatus.RUNNING)

            if not scene_binaries:
                raise ValueError("No scenes generated.")

            # Update scenes.json so the UI reflects the dubbed audio paths
            await self.save_scenes()

            # Concatenate scenes
            video_binary = await concatenate_videos(
                scene_binaries,
                fast_copy=False)
            if not video_binary:
                raise ValueError("Cannot concatenate scenes into final dubbed video.")
            video_info = get_video_file_info(video_binary)
            video_duration = video_info["video"]["duration_seconds"]
            video_fps = video_info["video"]["fps"]

            video_path = f"{self.job_path}/{self.job_id}.mp4"
            async with aiofiles.open(video_path, "wb") as file:
                await file.write(video_binary)

            self.logger.info(
                f"Generated dubbed video with {video_duration:.3f} seconds, "
                f"{video_fps} FPS, "
                f"{bytes_to_human(len(video_binary))}, and "
                f"'{video_path}'")

    async def detect_scenes(
        self,
        threshold: float = 27.0,
        min_scene_len: int = 15,
    ) -> List[SceneSegment]:
        """
        Return list of (start_frame, end_frame, start_sec, end_sec).
        TODO refactor with streamshort_job.py
        """
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
                scene.audio_path = f"scene_{scene.scene_id:03d}.wav"
                scene_audio_path = f"{self.job_path}/{scene.audio_path}"
                await save_base64_as_binary(scene_audio_path, scene_audio_base64)
                chunks.append(scene_audio_base64)
        except Exception as ex:
            self.logger.error(f"Error during audio chunking: {ex} [{type(ex)}]")
        return chunks

    async def gen_dub_scene(
        self,
        scene: SceneSegment,
        lang_code: Optional[str] = "e",  # Spanish
    ) -> bytes:
        """
        Generate scene with dubbed audio.
        """
        scene_id = scene.scene_id
        self.logger.info(
            f"[{scene_id}] Generating dubbed scene into {to_language(lang_code)} ('{lang_code}').")

        # Transcribe audio
        scene_transcript = await self.transcribe_audio(scene)
        scene.transcript = scene_transcript

        # TODO make this async into tasks

        if "♪" in scene.transcript:
            self.logger.info(f"[{scene_id}] Scene contains music ({scene.transcript}), skip dubbing.")
            return await self.get_video_scene(scene)
        if not scene.transcript.strip():
            self.logger.info(f"[{scene_id}] Scene has no transcription, skip dubbing.")
            return await self.get_video_scene(scene)

        self.logger.info(f"[{scene_id}] Transcript: {scene.transcript[0:80]}...")
        transcript_path = f"{self.job_path}/scene_{scene_id:03d}.txt"
        async with aiofiles.open(transcript_path, "w") as file:
            await file.write(scene.transcript)
        await self.save_scenes()

        # Translate transcription
        scene.translation = await self.translate_scene(
            scene,
            output_lang_code=lang_code)

        self.logger.info(f"[{scene_id}] Translation: {scene.translation[0:80]}...")
        transcript_path = f"{self.job_path}/scene_{scene_id:03d}_translation.txt"
        async with aiofiles.open(transcript_path, "w") as file:
            await file.write(scene.translation)
        await self.save_scenes()

        if not scene.translation.strip():
            self.logger.info(
                f"[{scene_id}] No translation available, using original audio and skipping lip sync.")
            return await self.get_video_scene(scene)

        await self.save_status(JobStatus.RUNNING)

        # Generate dubbed audio using the original scene audio as the voice reference
        # so that the speaker's voice identity is preserved in the dubbed output.
        deadline = self.get_submission_time() + scene.start_sec
        original_audio_path = f"{self.job_path}/scene_{scene_id:03d}.wav"
        voice_sample: Optional[str] = None
        if self.config.get("voice_cloning", True):
            try:
                voice_sample = await read_file_base64(original_audio_path)
                self.logger.info(
                    f"[{scene_id}] Using original scene audio for voice cloning "
                    f"({bytes_to_human(len(voice_sample))}).")
            except Exception as ex:
                self.logger.warning(
                    f"[{scene_id}] Could not read original scene audio for voice cloning: {ex}. "
                    "Falling back to default voice.")
        else:
            self.logger.info(f"[{scene_id}] Voice cloning disabled; using default voice.")
        if voice_sample is not None:
            try:
                audio_base64 = await self.gen.gen_clone_audio(
                    text=scene.translation,
                    voice_sample=voice_sample,
                    lang_code=lang_code,
                    task_id=f"{scene_id:03d}",
                    deadline=deadline,
                )
            except Exception as ex:
                self.logger.warning(
                    f"[{scene_id}] Voice cloning failed: {ex}. Falling back to default voice.")
                audio_base64 = await self.gen.gen_audio(
                    text=scene.translation,
                    lang_code=lang_code,
                    voice_sample=voice_sample,
                    task_id=f"{scene_id:03d}",
                    deadline=deadline,
                )
        else:
            audio_base64 = await self.gen.gen_audio(
                text=scene.translation,
                lang_code=lang_code,
                task_id=f"{scene_id:03d}",
                deadline=deadline,
            )
        scene.audio_path = f"scene_{scene_id:03d}_dubbed.wav"
        scene_audio_path = f"{self.job_path}/{scene.audio_path}"
        await save_base64_as_binary(scene_audio_path, audio_base64)

        # Lip sync video scenes
        # TODO if video too long, chunk it
        if scene.duration_sec > MAX_FT_DURATION_SECS:
            self.logger.warning(
                f"[{scene_id}] Scene too long: "
                f"{scene.duration_sec:.3f} > {MAX_FT_DURATION_SECS:.3f} seconds.")
        scene_dubbed_video_binary = await self.gen_video_lip_synced(scene)

        # Add subtitles (on by default)
        if self.config.get("add_subtitles", True):
            scene_dubbed_video_binary = await self._add_subtitles_to_video(
                scene, scene_dubbed_video_binary)

        # Save scene video
        scene_dubbed_video_path = f"{self.job_path}/scene_{scene_id:03d}_dubbed.mp4"
        async with aiofiles.open(scene_dubbed_video_path, "wb") as file:
            await file.write(scene_dubbed_video_binary)

        return scene_dubbed_video_binary

    async def get_video_scene(
        self,
        scene: SceneSegment
    ) -> bytes:
        """
        Get the video bytes for a scene.
        """
        video_path = f"{self.job_path}/video.mp4"
        video_binary = await read_file_bytes(video_path)
        scene_video_binary = chunk_video_binary(
            video_binary,
            start_seconds=scene.start_sec,
            end_seconds=scene.end_sec
        )

        scene_video_path = f"{self.job_path}/scene_{scene.scene_id:03d}.mp4"
        async with aiofiles.open(scene_video_path, "wb") as file:
            await file.write(scene_video_binary)

        return scene_video_binary

    async def _add_subtitles_to_video(
        self,
        scene: SceneSegment,
        video_binary: bytes,
    ) -> bytes:
        """
        Overlay the translated subtitle text onto every frame of the dubbed video.
        """
        if not scene.translation:
            return video_binary

        scene_id = scene.scene_id
        video_frames = await self._overlay_subtitles_on_frames(video_binary, scene.translation)
        video_fps = get_video_file_info(video_binary)["video"]["fps"]

        scene_audio_path = f"{self.job_path}/{scene.audio_path}"
        subtitled_path = f"{self.job_path}/scene_{scene_id:03d}_dubbed_subtitled.mp4"
        subtitled_path = await save_video_audio(
            video_content=video_frames,
            audio_path=scene_audio_path,
            fps=video_fps,
            out_video_path=subtitled_path,
        )
        async with aiofiles.open(subtitled_path, "rb") as subtitle_file:
            return await subtitle_file.read()

    async def transcribe_audio(
        self,
        scene: SceneSegment
    ) -> str:
        """
        Transcribe the audio of the video.
        TODO refactor with streamshort_job.py
        """
        if not scene.audio_path:
            return ""

        audio_path = f"{self.job_path}/{scene.audio_path}"
        audio_transcript, lang_code = await self.gen.gen_audio_transcript(
            audio_path,
            task_id=f"{scene.scene_id:03d}",
        )
        if not audio_transcript:
            return ""
        scene.language = lang_code
        audio_transcript = audio_transcript.strip()
        return audio_transcript

    async def translate_scene(
        self,
        scene: SceneSegment,
        input_lang_code: str = "a",  # American English
        output_lang_code: str = "e",  # Spanish
    ) -> str:
        """
        Translate text using LLM.
        """
        text = scene.transcript
        if not text:
            return ""

        dub_prompt = DUB_PROMPT.format(
            input_language=to_language(input_lang_code),
            output_language=to_language(output_lang_code)
        )
        messages = [
            {"role": "system", "content": dub_prompt},
            {"role": "user", "content": text}
        ]
        prompt_path = f"{self.job_path}/translate_prompt_{scene.scene_id:03d}.txt"
        async with aiofiles.open(prompt_path, "w") as prompt_file:
            messages_json = json.dumps(messages, indent=2)
            await prompt_file.write(messages_json)

        translated_text = await self.gen.gen_text(
            messages,
            task_id=f"translate{scene.scene_id:03d}",
        )

        return translated_text

    async def gen_video_lip_synced(
        self,
        scene: SceneSegment
    ) -> bytes:
        """
        Generate lip synced video for a scene.
        """
        scene_id = scene.scene_id

        # Chunk video for the scene
        scene_video_binary = await self.get_video_scene(scene)

        # Convert frames into the right sampling rate
        scene_video_frames = await get_video_frames(scene_video_binary)
        video_info = get_video_file_info(scene_video_binary)
        scene_video_fps = video_info["video"]["fps"]
        if scene_video_fps != FANTASYTALKING_FPS:
            self.logger.info(
                f"[{scene_id}] Resampling scene video from "
                f"{scene_video_fps:.2f} FPS to {FANTASYTALKING_FPS:.2f} FPS.")
            scene_video_frames = get_video_frames_at_fps(
                scene_video_frames,
                scene_video_fps,
                FANTASYTALKING_FPS,
            )

        # Merge video with dubbed audio
        scene_audio_path = f"{self.job_path}/{scene.audio_path}"
        scene_audio_base64 = await read_file_base64(scene_audio_path)
        scene_audio_seconds = get_audio_duration(scene_audio_base64)

        scene_video_frame = scene_video_frames[0]
        width = scene_video_frame.width
        height = scene_video_frame.height
        scene_video_seconds = len(scene_video_frames) / FANTASYTALKING_FPS
        num_steps = self.get_num_steps()

        self.logger.info(
            f"[{scene_id}] Generating lip synced video for scene "
            f"with video with {len(scene_video_frames)} frames, "
            f"{scene_video_seconds:.2f} seconds, "
            f"resolution {width}x{height}, "
            f"and audio with {scene_audio_seconds:.2f} seconds, "
            f"in steps {num_steps}.")

        # Pad video or audio if needed
        if scene_audio_seconds > scene_video_seconds + 0.1:
            extra_seconds = scene_audio_seconds - scene_video_seconds
            extra_frames = int(extra_seconds * FANTASYTALKING_FPS)
            last_frame = scene_video_frames[-1]
            for _ in range(extra_frames):
                scene_video_frames.append(last_frame)
            self.logger.warning(
                f"[{scene_id}] Audio is longer than video: "
                f"{scene_audio_seconds:.3f} > {scene_video_seconds:.3f} seconds. "
                f"Added {extra_frames} extra frames ({extra_seconds:.3f} seconds).")
            # TODO we may want to resample instead of extending the end
        elif scene_audio_seconds < scene_video_seconds - 0.1:
            self.logger.warning(
                f"[{scene_id}] Audio is shorter than video: "
                f"{scene_audio_seconds:.3f} < {scene_video_seconds:.3f} seconds.")
            fit_audio_to_duration(
                scene_audio_path,
                target_duration=scene_video_seconds,
                output_path=scene_audio_path)  # TODO new output path?

        deadline = self.get_submission_time() + scene.start_sec
        scene_dubbed_video_binary = await self.gen.gen_video_audio_from_video(
            video=scene_video_frames,
            audio_base64=scene_audio_base64,
            prompt=VIDEO_DUB_PROMPT,
            neg_prompt=VIDEO_DUB_NEG_PROMPT,
            width=width,
            height=height,
            steps=num_steps,
            task_id=f"{scene.scene_id:03d}",
            deadline=deadline,
        )

        return scene_dubbed_video_binary
