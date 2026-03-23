"""
Wrapper for Kokoro text-to-speech model.
"""
import logging
import tempfile
import asyncio

import torch

from torch import inference_mode
from torch import Tensor

from typing import override
from typing import List
from typing import Dict
from typing import Any
from typing import Union
from typing import Optional

from enum import Enum

from kokoro import KPipeline

from model_timing import GenTimer
from wrapper_model import ModelGeneration
from media_utils import save_audio

VOICES = {
    "female": [
        "af_heart",
        "af_bella",
        "af_kore",
        "af_nicole",
    ],
    "male": [
        "am_adam",
        "am_puck",
        "am_michael",
        "am_fenrir",
    ]
}


class Language(str, Enum):
    """
    Supported languages.
    https://github.com/hexgrad/kokoro/blob/main/kokoro/pipeline.py
    """
    AMERICAN_ENGLISH = "a"
    BRITISH_ENGLISH = "b"
    SPANISH = "e"
    FRENCH = "f"
    HINDI = "h"
    ITALIAN = "i"
    BRAZILIAN_PORTUGUESE = "p"
    JAPANESE = "j"
    MANDARIN_CHINESE = "z"


class KokoroGeneration(ModelGeneration):
    """Handle audio generation using the Kokoro model."""

    def __init__(self) -> None:
        super().__init__("kokoro")

        # Model components: language -> model pipeline
        self.kokoro: Dict[str, KPipeline] = {}

    def __del__(self) -> None:
        if self.kokoro:
            self.kokoro = {}  # Release model pipelines

    def init_parallelism(self) -> None:
        self.load_timer.start("torch_dist")
        # No real parallelism as it runs with a single GPU or CPU
        if torch.cuda.is_available():
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.device_id: Union[int, str] = self.local_rank
            self.device = torch.device(f"cuda:{self.device_id}")
            torch.cuda.set_device(self.local_rank)
        else:
            self.device_id = "cpu"
            self.device = torch.device(self.device_id)
        self.load_timer.end("torch_dist")

    def load_model(self) -> None:
        self.load_timer.start("kokoro")
        model_name = "hexgrad/Kokoro-82M"
        for lang in Language:
            lang_code = lang.value
            try:
                logging.info(f"Loading Kokoro model for language {lang} ({lang_code})")
                self.load_timer.start(f"kokoro_{lang_code}")
                kokoro = KPipeline(
                    repo_id=model_name,
                    device=self.device,
                    lang_code=lang_code)
                self.kokoro[lang_code] = kokoro
                self.load_timer.end(f"kokoro_{lang_code}")
            except Exception as ex:
                logging.error(f"Cannot load language {lang} ({lang_code}): {str(ex)}")
        self.load_timer.end("kokoro")

    def init_model_parallelism(self) -> None:
        if self.world_size > 1:
            logging.warning("Kokoro does not support distributed parallelism.")

    def model_compile(self) -> None:
        if not self.torch_compile:
            return
        self.load_timer.start("compile")
        for lang in Language:
            lang_code = lang.value
            if lang_code in self.kokoro:
                self.kokoro[lang_code] = torch.compile(
                    self.kokoro[lang_code],
                    mode="reduce-overhead")
        self.load_timer.end("compile")

    async def get_rest_args(self, data_json: dict) -> Dict[str, Any]:
        if data_json is None:
            raise ValueError("Missing JSON body")

        job_id = data_json.get("job_id", None)

        text = data_json.get("text", None)
        if text is None:
            raise ValueError("Missing 'text' parameter")
        # https://github.com/hexgrad/kokoro/blob/main/kokoro.js/src/voices.js
        voice = data_json.get("voice", "af_heart")
        speed = float(data_json.get("speed", 1.0))
        lang_code = data_json.get("lang_code", Language.AMERICAN_ENGLISH.value)
        return {
            "task": self.model_name,
            "args": {
                "job_id": job_id,
                "text": text,
                "voice": voice,
                "speed": speed,
                "lang_code": lang_code,
            }
        }

    @torch.inference_mode()
    async def warmup(self) -> None:
        logging.info("Warmup for Kokoro generation.")
        await self.generate(text="Warmup")

    @override
    @inference_mode()
    async def generate(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang_code: str = Language.AMERICAN_ENGLISH.value,
        job_id: Optional[str] = None,
        output_type: str = "audio_path",
    ) -> Optional[Union[str, Tensor]]:
        gen_timer = self._new_gen_timer(job_id)

        self.running = True  # We can run in parallel but good to know if we are running

        audios = []
        try:
            # Clean up text
            text = text.replace("*", "")

            if lang_code not in self.kokoro:
                raise ValueError(f"Unsupported language code: {lang_code}")

            gen_timer.start("kokoro")
            audio_generator = await asyncio.to_thread(
                self.kokoro[lang_code],
                text=text,
                voice=voice,
                speed=speed
            )
            gen_timer.end("kokoro")

            # text, phonemes, audio = gs, ps, audio
            for gs, ps, audio in audio_generator:
                audios.append(audio)

            return await self._output_audio(
                job_id=job_id,
                gen_timer=gen_timer,
                audios=audios,
                output_type=output_type)
        finally:
            self.running = False
            gen_timer.end("total")

    async def _output_audio(
        self,
        job_id: Optional[str],
        gen_timer: GenTimer,
        audios: List[Tensor],
        output_type: str = "audio_path",  # "audio_path"
    ) -> Optional[Union[str, Tensor]]:
        gen_timer.start("output_audio")
        if len(audios) > 1:
            logging.warning("Multiple audio chunks generated, returning the first one.")
        try:
            for audio in audios:
                if output_type == "tensor":
                    return audio

                if not job_id:
                    audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                else:
                    audio_path = f"/tmp/{job_id}.wav"

                audio_path = save_audio(
                    audio=audio,
                    audio_path=audio_path)
                return audio_path
            return None
        finally:
            gen_timer.end("output_audio")

    def get_health(self) -> Dict[str, Any]:
        ret = super().get_health()
        if torch.cuda.is_available():
            ret["gpu"] = torch.cuda.get_device_name(0)
        return ret
