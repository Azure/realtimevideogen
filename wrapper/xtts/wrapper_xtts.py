import logging

import torch
from torch import inference_mode

from typing_extensions import override  # Python 3.11
from typing import Optional
from typing import Dict
from typing import Any
from typing import Union

from wrapper_model import ModelGeneration

import numpy as np

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


class XTTSGeneration(ModelGeneration):
    def __init__(self) -> None:
        super().__init__("xtts")

        # Model components
        self.xtts_config: Optional[XttsConfig] = None
        self.xtts: Optional[Xtts] = None

    def __del__(self) -> None:
        if self.xtts is not None:
            self.xtts = None

    def init_parallelism(self) -> None:
        self.load_timer.start("torch_dist")
        # No real parallelism as it runs with a single GPU or CPU
        if torch.cuda.is_available():
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.device_id = self.local_rank
            self.device = torch.device(f"cuda:{self.device_id}")
            torch.cuda.set_device(self.local_rank)
        else:
            self.device_id = "cpu"
            self.device = torch.device(self.device_id)
        self.load_timer.end("torch_dist")

    def load_model(self) -> None:
        self.load_timer.start("xtts")
        self.xtts_config = XttsConfig()
        # TODO fix this with the DockerFile
        self.xtts_config.load_json("coqui/XTTS-v2/config.json")
        self.xtts = Xtts.init_from_config(self.xtts_config)
        self.xtts.load_checkpoint(
            self.xtts_config,
            checkpoint_dir="coqui/XTTS-v2",
            eval=True)
        self.xtts.cuda()
        self.load_timer.end("xtts")

    def init_model_parallelism(self) -> None:
        if self.world_size > 1:
            logging.warning("XTTS does not support distributed parallelism.")

    def model_compile(self) -> None:
        if not self.torch_compile:
            return
        self.load_timer.start("compile")
        self.xtts = torch.compile(
            self.xtts,
            mode="reduce-overhead")
        self.load_timer.end("compile")

    async def get_rest_args(
        self,
        data_json: Dict[str, Union[str, int, float]]
    ) -> Dict[str, Any]:
        if data_json is None:
            raise ValueError("Missing JSON body")
        text = data_json.get("text", None)
        if text is None:
            raise ValueError("Missing 'text' parameter")
        return {
            "task": self.model_name,
            "args": {
                "text": text,
            }
        }

    @inference_mode()
    async def warmup(self) -> None:
        logging.info("Warmup for XTTS generation")
        await self.generate(text="Warmup")

    def _assert_model_init(self) -> None:
        super()._assert_model_init()
        if not self.xtts:
            raise ValueError("XTTS not loaded.")

    @override
    @inference_mode()
    async def generate(  # type: ignore[override]
        self,
        text: str,
        job_id: Optional[str] = None,
    ) -> np.ndarray:
        gen_timer = self._new_gen_timer(job_id)

        self._assert_model_init()

        self.running = True  # We can run in parallel but good to know if we are running

        try:
            # Clean up text
            text = text.replace("*", "")

            outputs = self.xtts.synthesize(
                text,
                self.xtts_config,
                speaker_wav="tests/data/ljspeech/wavs/LJ001-0001.wav",
                gpt_cond_len=3,
                language="en",
            )
            wav_np = outputs["wav"]
            return wav_np
        finally:
            self.running = False
            gen_timer.end("total")

    def get_health(self) -> Dict[str, Any]:
        ret = super().get_health()
        if torch.cuda.is_available():
            ret["gpu"] = torch.cuda.get_device_name(0)
        return ret
