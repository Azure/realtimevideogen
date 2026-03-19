"""
StreamChat job to generate a video chat response.
"""
import asyncio
import json
import sys
import time
import aiofiles
import unicodedata

from PIL import Image

from typing import List, override
from typing import Any
from typing import Dict
from typing import Optional

from chat_prompts import IMG_PROMPT
from chat_prompts import IMG_NEG_PROMPT
from chat_prompts import CHAT_PROMPT
from chat_prompts import VIDEO_PROMPT
from chat_prompts import VIDEO_NEG_PROMPT

# Local relative imports
sys.path.append("..")  # noqa: E402
sys.path.append("../..")  # noqa: E402

from streamwise_job import StreamWiseJob
from streamwise_job import JobStatus

from lmm_service_manager import LMMServiceManager

from character import Character

from file_utils import save_base64_as_binary

from console_utils import bytes_to_human


class StreamChatJob(StreamWiseJob):
    """A job to generate a video chat response."""

    def __init__(
        self,
        job_id: str,
        service_manager: LMMServiceManager,
        config: Dict[str, Any] = {},
    ) -> None:
        super().__init__(
            "streamchat",
            job_id,
            service_manager,
            config)
        self.character: Optional[Character] = None
        self.image: Optional[Image.Image] = None
        self.messages = [{
            "role": "system",
            "content": CHAT_PROMPT
        }]

    def get_config_gender(self) -> str:
        return self.get_config_str(
            "gender_prompt",
            "female")

    @override
    async def generate(
        self,
        job_config: Dict[str, Any],
    ) -> None:
        await self.gen_chat_base()

    async def gen_chat_base(
        self,
    ) -> None:
        """Generate a chat video."""
        async with self.job_status_handler():
            self.character = Character(
                name="Assistant",
                gender=self.get_config_gender(),
                speech_speed=self.get_config_float("speech_speed", 1.1),
            )

            # Generate the base main image
            img_prompt = IMG_PROMPT
            if self.character.gender:
                img_prompt += "The gender of the character is a " + self.character.gender + "."
            style_prompt = self.get_config_str("style_prompt", "")
            if style_prompt:
                img_prompt += "The style of the image is: " + style_prompt + "."
            scene_prompt = self.get_config_str("scene_prompt", "")
            if scene_prompt:
                img_prompt += "The scene is: " + scene_prompt + "."
            custom_prompt = self.get_config_str("custom_prompt", "")
            if custom_prompt:
                img_prompt += "Additional details: " + custom_prompt + "."

            width, height = self.width, self.height
            image = await self.gen.gen_image(
                img_prompt,
                neg_prompt=IMG_NEG_PROMPT,
                width=width,
                height=height,
                task_id="main_image",
                deadline=self.get_submission_time(),
            )
            if image is None:
                raise Exception("Image generation failed.")
            self.image = image
            width, height = image.size
            image_path = f"{self.job_path}/main_image.png"
            image.save(image_path)
            self.logger.info(f"Image with {width}x{height} pixels saved to '{image_path}'.")

    def _get_msg_id(self) -> int:
        """Get the current message ID."""
        # Exclude system message and count pairs (Q/A)
        return (len(self.messages) - 1) // 2

    async def transcribe_audio(
        self,
        audio_path: str,
    ) -> str:
        msg_id = self._get_msg_id()
        audio_transcript, _ = await self.gen.gen_audio_transcript(
            audio_path,
            task_id=f"chat{msg_id:03d}",
        )
        return audio_transcript

    async def gen_chat(
        self,
        user_message: str
    ) -> Dict[str, str]:
        """
        Generate a chat response given a user message.
        """
        async with self.job_status_handler():
            msg_id = self._get_msg_id()

            # Generate text response
            response_text = await self.gen_chat_text(
                user_message=user_message,
                msg_id=msg_id,
            )

            await self.save_status(JobStatus.RUNNING)

            # Generate audio response
            audio_task = asyncio.create_task(
                self.gen_chat_audio(
                    response_text=response_text,
                    msg_id=msg_id,
                )
            )
            await self.save_status(JobStatus.RUNNING)

            # Generate video response
            # TODO await task somewhere?
            # video_task =
            asyncio.create_task(
                self.gen_chat_video(
                    audio_task=audio_task,
                    msg_id=msg_id,
                )
            )

            """
            video_binary = await video_task
            if video_binary is None:
                raise Exception("Video generation failed.")
            self.logger.info(f"Generated video with {bytes_to_human(len(video_binary))}.")
            """

            return {
                "id": msg_id,
                "reply": response_text,
            }

    async def gen_chat_text(
        self,
        user_message: str,
        msg_id: int,
    ) -> str:
        message_path = f"{self.job_path}/chat{msg_id:03d}_message.txt"
        async with aiofiles.open(message_path, "w") as file:
            await file.write(user_message)

        self.messages.append({
            "role": "user",
            "content": user_message
        })

        prompt_path = f"{self.job_path}/chat{msg_id:03d}_prompt.jsonl"
        async with aiofiles.open(prompt_path, "w") as file:
            for msg in self.messages:
                json_line = json.dumps(
                    msg,
                    ensure_ascii=False,
                    separators=(",", ":")
                )
                await file.write(json_line + "\n")

        # Generate the text response using the LLM
        response_text = await self.gen.gen_text(
            self.messages,
            task_id=f"chat{msg_id:03d}",
        )

        self.logger.info(f"[{msg_id}] Generated response: {response_text}")
        response_path = f"{self.job_path}/chat{msg_id:03d}_response.txt"
        async with aiofiles.open(response_path, "w") as file:
            await file.write(response_text)

        self.messages.append({
            "role": "assistant",
            "content": response_text
        })

        return response_text

    async def gen_chat_audio(
        self,
        response_text: str,
        msg_id: int,
    ) -> str:
        lang_code = "a"  # American English TODO
        response_text_clean = response_text.replace("\n", " ").strip()
        response_text_clean = remove_emojis(response_text_clean)
        audio_base64 = await self.gen.gen_audio(
            text=response_text_clean,
            voice=self.character.voice,
            speed=self.character.speech_speed,
            lang_code=lang_code,
            task_id=f"{msg_id:03d}",
            deadline=time.time(),  # Now
        )
        if audio_base64 is None:
            raise Exception("Audio generation failed.")

        self.logger.info(f"[{msg_id}] Generated audio with {bytes_to_human(len(audio_base64))}.")

        audio_path = f"{self.job_path}/chat{msg_id:03d}.wav"
        await save_base64_as_binary(
            audio_path,
            audio_base64)
        return audio_base64

    async def gen_chat_video(
        self,
        audio_task: asyncio.Task,
        msg_id: int,
    ) -> None:
        """
        Generate chat video from base image and audio.
        """
        if self.image is None:
            raise Exception("Base image not found for video generation.")

        audio_base64 = await audio_task
        if audio_base64 is None:
            raise Exception("Audio generation failed.")

        video_binary = await self.gen.gen_video_audio_from_img(
            img=self.image,
            audio_base64=audio_base64,
            prompt=VIDEO_PROMPT,
            neg_prompt=VIDEO_NEG_PROMPT,
            width=self.width,
            height=self.height,
            steps=self.get_num_steps(),
            task_id=f"{msg_id:03d}",
            deadline=time.time(),  # Now
        )
        if video_binary is None:
            raise Exception("Video generation failed.")

        self.logger.info(f"[{msg_id}] Generated video with {bytes_to_human(len(video_binary))}.")

        video_path = f"{self.job_path}/chat{msg_id:03d}.mp4"
        async with aiofiles.open(video_path, "wb") as file:
            await file.write(video_binary)

    async def get_chat_history(self) -> List[Dict[str, str]]:
        """Get the chat history."""
        history = []
        for msg in self.messages:
            history.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        return history


def remove_emojis(
    text: str
) -> str:
    """
    Remove emojis from the given text.
    Avoid TTS speaking them.
    """
    return "".join(
        ch for ch in text
        if not unicodedata.category(ch).startswith("So")
    )
