"""
Podcast transcript generation wrapper.
It uses an LLM to generate podcast transcripts from PDF documents.
"""

import asyncio
import logging
import requests
import re
import traceback
import json
import tempfile
import aiofiles

from azure.identity import DefaultAzureCredential
from azure.identity import get_bearer_token_provider
from openai import AsyncAzureOpenAI
from openai import AsyncOpenAI

from typing import override
from typing import List
from typing import Dict
from typing import Optional
from typing import Any
from typing import AsyncGenerator

from pydantic import BaseModel
from pydantic import Field
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_random_exponential

from wrapper_model import ModelGeneration

from file_utils import base64_to_binary
from media_utils import fix_json_like_string

from pdf_utils import parse_pdf

from transcript_prompts import QUESTION_MODIFIER
from transcript_prompts import SYSTEM_PROMPT
from transcript_prompts import IMG_STYLE_MODIFIERS
from transcript_prompts import IMG_SCENE_MODIFIERS

from http import HTTPStatus


DEFAULT_MAX_TOKENS = 5000
DEFAULT_MAX_DIALOGUES = 50
DEFAULT_NUM_CHARACTERS = 2
DEFAULT_TEMPERATURE = 0.7


class Dialogue(BaseModel):
    character: str = Field(
        ..., description="Name of the character delivering the dialogue."
    )
    transcript: str = Field(
        ..., description="Transcript of the dialogue. Must not be more than 50 words."
    )
    end_script: bool = Field(
        default=False,
        description="Indicates if the script is finished. If True, no more dialogues will be generated."
    )

    def __str__(self) -> str:
        str_ = f"{self.character}: {self.transcript}"
        return str_


class Script(BaseModel):
    dialogues: List[Dialogue] = Field(
        ...,
        description="Ordered list of dialogues in the script."
    )

    def __str__(self) -> str:
        str_ = "Script:\n\n"
        for dialogue in self.dialogues:
            str_ += f"{dialogue.__str__()}\n\n"
        return str_


class Scene(BaseModel):
    characters: List[str] = Field(
        ..., description="List of characters shown in this scene."
    )
    dialogues: List[Dialogue] = Field(
        ..., description="Ordered list of dialogues in the scene."
    )

    def __str__(self) -> str:
        str_ = f"Characters: {self.characters}\n\n"
        str_ += "Transcript:\n\n"
        for dialogue in self.dialogues:
            str_ += f"{dialogue.__str__()}\n\n"
        return str_


class Podcast(BaseModel):
    scenes: List[Scene] = Field(
        ..., description="Ordered list of scenes in the podcast."
    )

    def __str__(self) -> str:
        str_ = ""
        for scene_idx, scene in enumerate(self.scenes):
            str_ += f"Scene: {scene_idx}\n{scene.__str__()}"
        return str_


class PodcastTranscriptGenerator(ModelGeneration):
    """
    A class to generate podcast transcripts from PDF documents using an LLM.
    This class handles downloading PDFs, parsing them into text and images,
    and generating a structured podcast script with scenes and dialogues.
    """

    def __init__(
        self,
        llm_url: str = "http://localhost:8000/v1",
        llm_model: str = "meta-llama/Meta-Llama-3.1-8B",
        multi_modal: bool = False,
    ) -> None:
        super().__init__("podcasttranscript")
        self.set_llm(llm_model, llm_url, multi_modal)

    def load_model(self) -> None:
        logging.debug("No model for podcast transcript generation.")

    def init_model_parallelism(self) -> None:
        logging.debug("No parallelism for podcast transcript generation.")

    def model_compile(self) -> None:
        logging.debug("No compilation for podcast transcript generation.")

    def _get_azure_openai_client(self, base_url: str) -> AsyncAzureOpenAI:
        def _get_azure_api_version(base_url: str) -> Optional[str]:
            api_version_pattern = r"api-version=([^&]+)"
            match = re.search(api_version_pattern, base_url)
            if match:
                return match.group(1)
            logging.error("Failed to extract API version from URL")
            return None

        api_version = _get_azure_api_version(base_url)
        azure_token_url = "https://cognitiveservices.azure.com/.default"
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), azure_token_url
        )
        return AsyncAzureOpenAI(
            azure_ad_token_provider=token_provider,
            azure_endpoint=base_url,
            api_version=api_version)

    def set_llm(
        self,
        llm_model: str,
        llm_url: str,
        multi_modal: bool = False,
        api_key: str = "n/a",
    ) -> None:
        self.llm_model = llm_model
        self.llm_url = llm_url
        self.multi_modal = multi_modal
        self.llm_client: AsyncOpenAI
        if "azure" in llm_url:
            self.llm_client = self._get_azure_openai_client(llm_url)
            self.extra_body = None
        else:
            self.llm_client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.llm_url)
            # timeout=10.0)  # Set a timeout for LLM requests
            self.extra_body = dict(guided_decoding_backend="xgrammar")

    def download_pdf(
        self,
        url: str,
        job_id: Optional[str] = None,
    ) -> str:
        """
        Download the PDF from the given URL and save it to a temporary file.
        """
        response = requests.get(url, timeout=30)
        if response.status_code == HTTPStatus.OK:
            if not job_id:
                output_path = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name
            else:
                output_path = f"/tmp/{job_id}.pdf"
            with open(output_path, "wb") as file:
                file.write(response.content)
            return output_path
        raise Exception(f"Failed to download PDF, status code: {response.status_code}")

    async def gen_script(
        self,
        pdf_images: List[str],
        pdf_text: List[str],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_dialogues: int = DEFAULT_MAX_DIALOGUES,
        max_words_per_dialogue: int = -1,
        job_id: Optional[str] = None,
    ) -> Script:
        QUESTION = "What is the main theme of this paper?"

        # Prepare the user prompt text string
        user_prompt_text = "\n\n".join(pdf_text)
        user_prompt_text += f"\n\n{QUESTION_MODIFIER} {QUESTION}"

        # Add images to the user prompt
        user_prompt: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt_text}]
        if self.multi_modal:
            for image_url in pdf_images:
                user_prompt.append({"type": "image_url", "image_url": {"url": image_url}})

        # Prepare the system prompt text string
        system_prompt = SYSTEM_PROMPT
        system_prompt += f"\n\n{QUESTION_MODIFIER} {QUESTION}"

        # Combine the prompts into a single message
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        async with aiofiles.open(f"/tmp/{job_id}_prompt.json", "w") as prompt_file:
            await prompt_file.write(json.dumps(messages, indent=2))

        try:
            response = await self.llm_client.beta.chat.completions.parse(
                model=self.llm_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=self.extra_body,
                response_format=Script,
            )
            logging.info(f"LLM response: {response}")
        except Exception as ex:
            msg = f"Cannot query LLM for script at {self.llm_url}: {ex}."
            logging.error(msg)
            raise Exception(msg)

        response_message = response.choices[0].message
        if response_message.parsed:
            return response_message.parsed
        raise ValueError("LLM response did not contain a parsed Script object.")

    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=3))
    async def gen_script_stream(
        self,
        pdf_images: List[str],
        pdf_text: List[str],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_dialogues: int = DEFAULT_MAX_DIALOGUES,
        max_words_per_dialogue: int = -1,
        num_characters: int = DEFAULT_NUM_CHARACTERS,
        style_prompt: Optional[str] = None,
        scene_prompt: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        job_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict, None]:
        QUESTION = "What is the main theme of this paper?"
        system_prompt_text = SYSTEM_PROMPT + f"\n\n{QUESTION_MODIFIER} {QUESTION}"
        user_prompt_text = "\n\n".join(pdf_text) + f"\n\n{QUESTION_MODIFIER} {QUESTION}"

        contraint_instruction = ""
        if max_words_per_dialogue > 0:
            contraint_instruction += f"Each dialogue should not exceed {max_words_per_dialogue} words.\n"

        if max_dialogues < 0:
            logging.warning(f"'max_dialogues' not set, using default value of {DEFAULT_MAX_DIALOGUES}.")
            max_dialogues = DEFAULT_MAX_DIALOGUES
        if max_dialogues > 0:
            contraint_instruction += f"Generate as close as possible to {max_dialogues} dialogues.\n"
            contraint_instruction += f"Do not generate more than {max_dialogues} dialogues.\n"
        if max_dialogues > 2:
            contraint_instruction += "Make the last two dialogues conclude the discussion.\n"

        if num_characters == 1:
            contraint_instruction += "Generate an image description with 1 character. "
            contraint_instruction += "The image description must feature exactly 1 person. "
            contraint_instruction += "GENERATE ONLY 1 CHARACTER. "
            contraint_instruction += "The scene must feature a single person. "
            contraint_instruction += "All dialogues belong to this person. "
        elif num_characters > 1:
            contraint_instruction += f"Generate an image description with {num_characters} characters."
            contraint_instruction += f"The image description must feature exactly {num_characters} people. "
            contraint_instruction += f"GENERATE ONLY {num_characters} CHARACTERS. "
            contraint_instruction += f"The scene must feature {num_characters} people. "
            contraint_instruction += f"Make the {num_characters} characters have a dialogue. "
            contraint_instruction += f"There must be {num_characters} characters in the dialogues. "
        contraint_instruction += "Do NOT add any other characters.\n"

        if style_prompt and style_prompt in IMG_STYLE_MODIFIERS:
            style_modifier = IMG_STYLE_MODIFIERS[style_prompt]
            contraint_instruction += f"The style of the image should be {style_modifier}.\n"
        if scene_prompt and scene_prompt in IMG_SCENE_MODIFIERS:
            scene_modifier = IMG_SCENE_MODIFIERS[scene_prompt]
            contraint_instruction += f"The scene of the image should be in {scene_modifier}.\n"
        if custom_prompt:
            contraint_instruction += f"The image should have {custom_prompt}.\n"

        # Compose prompt
        system_prompt_text = (
            f"{system_prompt_text}\n\n"
            f"{contraint_instruction}"
        )
        user_prompt_text = (
            f"{user_prompt_text}\n\n"
            f"{contraint_instruction}"
        )

        # Build the messages
        user_msg: List[Dict[str, Any]] = [{
            "type": "text",
            "text": user_prompt_text
        }]
        if self.multi_modal:
            for image_url in pdf_images:
                user_msg.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}})

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt_text},
            {"role": "user", "content": user_msg}
        ]

        async with aiofiles.open(f"/tmp/{job_id}_prompt.json", "w") as prompt_file:
            await prompt_file.write(json.dumps(messages, indent=2))

        response_stream = await self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=self.extra_body,
            stream=True,
        )

        it = 0
        buffer_text = ""
        async for chunk in response_stream:  # type: ignore[union-attr]
            if self.interrupted:  # type: ignore[has-type]
                self.interrupted = False
                logging.info("Generation interrupted.")
                return

            delta = chunk.choices[0].delta.content or ""
            buffer_text += delta
            if delta.endswith("\n"):
                buffer_text = buffer_text.strip()
                if buffer_text.startswith("{") and buffer_text.endswith("}"):
                    try:
                        buffer_text = fix_json_like_string(buffer_text)
                        buffer_json = json.loads(buffer_text)
                        yield buffer_json
                    except json.JSONDecodeError as json_error:
                        logging.error(f"JSON error: {json_error} for buffer: {buffer_text}")
                else:
                    logging.info(f"Ignoring: {buffer_text}")
                buffer_text = ""
            it += 1

    async def gen_podcast(
        self,
        script: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        job_id: Optional[str] = None,
    ) -> Podcast:
        # Prepare the user prompt text string
        user_prompt = "Given the podcast description, split the dialogues into scenes such that some scenes focus on "
        user_prompt += "one character, others on multiple characters."
        user_prompt += f"\n\nThe transcript is as follows: {script}"

        # Prepare the system prompt text string
        system_prompt = "You are a world-class podcast director who can organize a script into a collection of scenes "
        system_prompt += "that will be later converted into a video."

        # Combine the prompts into a single message
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        async with aiofiles.open(f"/tmp/{job_id}_prompt.json", "w") as prompt_file:
            await prompt_file.write(json.dumps(messages, indent=2))

        try:
            response = await self.llm_client.beta.chat.completions.parse(
                model=self.llm_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=self.extra_body,
                response_format=Podcast,
            )
        except Exception as ex:
            msg = f"Cannot query LLM for podcast at {self.llm_url}: {ex}."
            logging.error(msg)
            raise Exception(msg)

        response_message = response.choices[0].message
        if response_message.parsed:
            return response_message.parsed
        raise ValueError("LLM response did not contain a parsed Podcast object.")

    @override
    async def generate(
        self,
        pdf_url: Optional[str] = None,
        pdf_text: Optional[List[str]] = None,
        pdf_images: Optional[List[str]] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        llm_model: Optional[str] = None,
        llm_url: Optional[str] = None,
        multi_modal: bool = False,
        max_dialogues: int = -1,
        max_words_per_dialogue: int = -1,
        job_id: Optional[str] = None,
    ) -> Podcast:
        gen_timer = self._new_gen_timer(job_id)

        self.running = True  # We can run in parallel but good to know if we are running

        if llm_model is not None and llm_url is not None:
            logging.info(f"Setting LLM model to {llm_model} at {llm_url}.")
            self.set_llm(
                llm_model,
                llm_url,
                multi_modal)

        try:
            if pdf_url:
                if not pdf_url.startswith("http"):
                    raise ValueError(f"PDF URL must start with 'http': {pdf_url}")
                gen_timer.start("download_pdf")
                pdf_path = self.download_pdf(pdf_url, job_id)
                pdf_text, pdf_images = parse_pdf(pdf_path)
                gen_timer.end("download_pdf")

            if pdf_text is None or pdf_images is None:
                raise ValueError("PDF text or images are not provided or could not be parsed.")

            gen_timer.start("gen_script")
            script = await self.gen_script(
                pdf_images,
                pdf_text,
                max_tokens=max_tokens,
                temperature=temperature,
                max_dialogues=max_dialogues,
                max_words_per_dialogue=max_words_per_dialogue,
                job_id=job_id,
            )
            gen_timer.end("gen_script")

            gen_timer.start("gen_podcast")
            podcast = await self.gen_podcast(
                str(script),
                max_tokens=max_tokens,
                temperature=temperature)
            gen_timer.end("gen_podcast")
        finally:
            self.running = False
            gen_timer.end("total")

        return podcast

    async def generate_stream(
        self,
        pdf_url: Optional[str] = None,
        pdf_text: Optional[List[str]] = None,
        pdf_images: Optional[List[str]] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        llm_model: Optional[str] = None,
        llm_url: Optional[str] = None,
        multi_modal: bool = False,
        max_dialogues: int = -1,
        max_words_per_dialogue: int = -1,
        num_characters: int = DEFAULT_NUM_CHARACTERS,
        style_prompt: Optional[str] = None,
        scene_prompt: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict, None]:
        gen_timer = self._new_gen_timer(job_id)

        self.running = True  # We can run in parallel but good to know if we are running

        if llm_model is not None and llm_url is not None:
            logging.info(f"Setting LLM model to {llm_model} at {llm_url}.")
            self.set_llm(
                llm_model,
                llm_url,
                multi_modal)

        try:
            if pdf_url:
                if not pdf_url.startswith("http"):
                    raise ValueError(f"PDF URL must start with 'http': {pdf_url}")
                gen_timer.start("download_pdf")
                pdf_path = self.download_pdf(pdf_url)
                pdf_text, pdf_images = parse_pdf(pdf_path)
                gen_timer.end("download_pdf")

            if pdf_text is None or pdf_images is None:
                raise ValueError("PDF text or images are not provided or could not be parsed.")

            it = 0
            gen_timer.start("gen_script_stream")
            gen_timer.start(f"gen_script_stream_{it}")
            async for dialogue in self.gen_script_stream(
                pdf_images,
                pdf_text,
                max_tokens=max_tokens,
                temperature=temperature,
                max_dialogues=max_dialogues,
                max_words_per_dialogue=max_words_per_dialogue,
                num_characters=num_characters,
                style_prompt=style_prompt,
                scene_prompt=scene_prompt,
                custom_prompt=custom_prompt,
                job_id=job_id,
            ):
                yield dialogue
                gen_timer.end(f"gen_script_stream_{it}")
                it += 1
                gen_timer.start(f"gen_script_stream_{it}")
            gen_timer.end(f"gen_script_stream_{it}")
            gen_timer.end("gen_script_stream")
        finally:
            self.running = False
            gen_timer.end("total")

    async def warmup(self) -> None:
        pass  # No specific warmup needed for this model

    async def get_rest_args(
        self,
        data_json: Dict[str, str]
    ) -> Dict[str, Any]:
        if data_json is None:
            raise ValueError("Missing JSON body")

        job_id = data_json.get("job_id", None)

        pdf_url = data_json.get("pdf_url", None)

        pdf_base64 = data_json.get("doc", None)
        pdf_text = None
        pdf_images = None
        if pdf_base64 is not None:
            if not job_id:
                output_path = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name
            else:
                output_path = f"/tmp/{job_id}.pdf"
            doc_bytes = base64_to_binary(pdf_base64)
            async with aiofiles.open(output_path, "wb") as doc_file:
                await doc_file.write(doc_bytes)
            pdf_text, pdf_images = parse_pdf(output_path)

        if pdf_url is None and pdf_text is None and pdf_images is None:
            raise ValueError("Either 'pdf_url' or 'doc' must be provided")

        rest_args: Dict[str, Any] = {
            "job_id": job_id,
            "temperature": float(data_json.get("temperature", 0.6)),
            "max_tokens": int(data_json.get("max_tokens", DEFAULT_MAX_TOKENS)),
            "llm_url": data_json.get("llm_url", "http://localhost:8000/v1"),
            "llm_model": data_json.get("llm_model", "google/gemma-3-27b-it"),
            "multi_modal": bool(data_json.get("multi_modal", False)),
            "max_dialogues": int(data_json.get("max_dialogues", DEFAULT_MAX_DIALOGUES)),
            "max_words_per_dialogue": int(data_json.get("max_words_per_dialogue", -1)),
            "num_characters": int(data_json.get("num_characters", DEFAULT_NUM_CHARACTERS)),
        }

        if pdf_url is not None:
            rest_args["pdf_url"] = pdf_url
        if pdf_text is not None:
            rest_args["pdf_text"] = pdf_text
        if pdf_images is not None:
            rest_args["pdf_images"] = pdf_images

        if "style_prompt" in data_json:
            rest_args["style_prompt"] = data_json["style_prompt"]
        if "scene_prompt" in data_json:
            rest_args["scene_prompt"] = data_json["scene_prompt"]
        if "custom_prompt" in data_json:
            rest_args["custom_prompt"] = data_json["custom_prompt"]

        return {
            "task": self.model_name,
            "args": rest_args
        }

    def get_health(self) -> Dict[str, Any]:
        ret = super().get_health()
        ret.update({
            # "gpu": torch.cuda.get_device_name(0)
            "gpu": None,
            "llm_url": self.llm_url,
            "llm_model": self.llm_model,
            "multi_modal": self.multi_modal,
        })
        return ret


def log_podcast(scene: Dict[str, Any]) -> None:
    if isinstance(scene, dict):
        assert "type" in scene
        line_type = scene["type"]
        if line_type == "image" and "content" in scene:
            scene_img_description = scene['content']
            logging.info(f"IMAGE: {scene_img_description}")
        elif line_type == "character":
            gender = scene.get('gender', 'Unknown')
            description = scene.get('description', 'No description provided')
            logging.info(
                f"Character: {scene['name']} ({gender}) - {description}")
        elif line_type == "dialogue" and "character" in scene and "content" in scene:
            logging.info(f"[{scene['character']}] {scene['content']}")
        else:
            logging.info(f"Scene: {scene}")
    else:
        logging.info(f"Scene: {scene}.")


async def main() -> None:
    LLM_MODEL = "meta-llama/Meta-Llama-3.1-8B"
    LMM_MULTI_MODAL = False

    LLM_MODEL = "google/gemma-3-27b-it"
    LMM_MULTI_MODAL = True

    LLM_URL = "http://localhost:8000/v1"
    LLM_URL = "http://localhost:18086/v1"

    # Azure OpenAI
    LLM_MODEL = "gpt-4o-2024-08-06"
    LLM_URL = "https://girfan-az-openai-001.openai.azure.com/openai/" \
              "deployments/gpt-4o-2/chat/completions?api-version=2025-01-01-preview"

    # Gemma on Kubernetes
    LLM_MODEL = "google/gemma-3-27b-it"
    LMM_MULTI_MODAL = True
    LLM_URL = "http://10.244.22.5:8000/v1"

    logging.basicConfig(level=logging.INFO)

    logging.info("Starting Podcast Transcript Generation...")
    logging.info(f"Using LLM model: {LLM_MODEL}.")
    logging.info(f"Using LLM URL: {LLM_URL}.")
    logging.info(f"Multi-modal support: {LMM_MULTI_MODAL}.")

    try:
        podcast_generator = PodcastTranscriptGenerator(
            llm_url=LLM_URL,
            llm_model=LLM_MODEL,
            multi_modal=LMM_MULTI_MODAL,
        )

        PDF_URL = "https://arxiv.org/pdf/2501.16634"

        # Output podcast transcript using the async generator
        # Podcast 1
        logging.info("Generating podcast 1 with a max of 7 dialogues:")
        async for scene in podcast_generator.generate_stream(
            pdf_url=PDF_URL,
            max_dialogues=7,
            max_words_per_dialogue=20,
        ):
            log_podcast(scene)

        # Podcast 2
        logging.info("Generating podcast 2 with a max of 20 dialogues:")
        async for scene in podcast_generator.generate_stream(
            pdf_url=PDF_URL,
            max_dialogues=20,
            max_words_per_dialogue=50,
        ):
            log_podcast(scene)

        # Podcast 3
        logging.info("Generating podcast 3 with 3 characters and a max of 15 dialogues:")
        async for scene in podcast_generator.generate_stream(
            pdf_url=PDF_URL,
            max_dialogues=15,
            num_characters=3,
            max_words_per_dialogue=10,
        ):
            log_podcast(scene)

        # Output podcast transcript in a single pass
        podcast = await podcast_generator.generate(PDF_URL)
        for podcast_scene in podcast.scenes:
            logging.info(f"Scene with characters: {podcast_scene.characters}.")
            for dialogue in podcast_scene.dialogues:
                logging.info(f"{dialogue.character}: {dialogue.transcript}.")
        # We can also output the podcast in JSON
        podcast_json = podcast.model_dump_json(indent=2)
        logging.info(f"Podcast JSON:\n{podcast_json}")

    except Exception as ex:
        exc_str = ''.join(traceback.format_tb(ex.__traceback__))
        logging.error(f"An error occurred: {ex}:\n{exc_str}.")


if __name__ == "__main__":
    asyncio.run(main())
