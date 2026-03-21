import torch
import logging
import tempfile
import traceback

from torch import inference_mode

from typing import Optional
from typing import Union
from typing import Dict
from typing import Any

from model_timing import GenTimer
from wrapper_model import ModelGeneration

from transformers import AutoProcessor
from transformers import DiaForConditionalGeneration

from dia.model import Dia


class DiaGeneration(ModelGeneration):
    """Handle image generation using the Dia model."""

    def __init__(self) -> None:
        super().__init__("dia")

        # Model components
        self.processor: Optional[AutoProcessor] = None
        self.dia: Optional[DiaForConditionalGeneration] = None

    def __del__(self) -> None:
        super().__del__()
        if self.processor is not None:
            self.processor = None
        if self.dia is not None:
            self.dia = None

    def init_parallelism(self) -> None:
        self.load_timer.start("torch_dist")
        # No real parallelism as it runs with a single GPU or CPU
        device_id: Union[int, str]
        if torch.cuda.is_available():
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            device_id = self.local_rank
            self.device = torch.device(f"cuda:{device_id}")
            torch.cuda.set_device(self.local_rank)
        else:
            device_id = "cpu"
            self.device = torch.device(device_id)
        self.device_id = device_id
        self.load_timer.end("torch_dist")

    def load_model(self) -> None:
        self.load_timer.start("processor")
        self.processor = AutoProcessor.from_pretrained(
            "nari-labs/Dia-1.6B-0626")
        self.load_timer.end("processor")

        self.load_timer.start("dia")
        self.dia = DiaForConditionalGeneration.from_pretrained(
            "nari-labs/Dia-1.6B-0626")
        self.dia = self.dia.to(self.device)
        self.load_timer.end("dia")

        try:
            self.load_timer.start("dia_test")
            logging.warning("TESTING DIA FANCY")
            _ = Dia.from_pretrained(
                # "nari-labs/Dia-1.6B-0626",
                "nari-labs/Dia-1.6B",
                device=self.device)
            self.load_timer.end("dia_test")
        except Exception as ex:
            logging.error(f"Failed to load test Dia model: {ex}")
            logging.error(traceback.format_exc())

    def init_model_parallelism(self) -> None:
        if self.world_size > 1:
            logging.warning("Dia does not support distributed parallelism.")

    def _assert_model_init(self) -> None:
        super()._assert_model_init()
        assert self.dia is not None, "Dia model is not initialized."

    def model_compile(self) -> None:
        if not self.torch_compile:
            return
        self.load_timer.start("compile")
        self.dia = torch.compile(
            self.dia,
            fullgraph=True,
            mode="max-autotune")
        self.load_timer.end("compile")

    def get_rest_args(self, data_json: Dict[str, str]) -> Dict[str, Any]:
        if data_json is None:
            raise ValueError("Missing JSON body")

        job_id = data_json.get("job_id", None)

        text = data_json.get("text", None)
        if text is None:
            raise ValueError("Missing 'text' parameter")

        return {
            "task": self.model_name,
            "args": {
                "job_id": job_id,
                "text": text,
                "max_new_tokens": data_json.get("max_new_tokens", None),
                "guidance_scale": data_json.get("guidance_scale", 3.0),
                "temperature": data_json.get("temperature", 1.8),
                "top_p": data_json.get("top_p", 0.90),
                "top_k": data_json.get("top_k", 45),
            }
        }

    @inference_mode()
    async def warmup(self) -> None:
        logging.info("Warmup for Dia generation.")
        await self.generate(text="Warmup")

    @inference_mode()
    async def generate(
        self,
        text: str,
        job_id: Optional[str] = None,
        max_new_tokens: int = 256,
        guidance_scale: float = 3.0,
        temperature: float = 1.8,
        top_p: float = 0.90,
        top_k: int = 45,
    ) -> str:
        gen_timer = self._new_gen_timer(job_id)

        self._assert_model_init()
        assert self.processor is not None
        assert self.dia is not None

        self.running = True  # We can run in parallel but good to know if we are running

        logging.info(f"Generating audio for: '{text[0:100]}'.")

        try:
            # Clean up text
            text = text.replace("*", "")
            text = text.strip()

            gen_timer.start("encode")
            inputs = self.processor(
                text=text,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            gen_timer.end("encode")

            gen_timer.start("dia")
            outputs = self.dia.generate(
                **inputs,
                # use_torch_compile=self.torch_compile,
                # max_new_tokens=max_new_tokens,
                max_new_tokens=None,
                guidance_scale=guidance_scale,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                early_stopping=True)
            gen_timer.end("dia")

            gen_timer.start("decode")
            outputs = self.processor.batch_decode(outputs)
            gen_timer.end("decode")

            logging.info(f"Generated {len(outputs)} output(s) for '{job_id}'.")

            return self._output_audio(
                job_id=job_id or "",
                gen_timer=gen_timer,
                outputs=outputs)
        finally:
            self.running = False
            gen_timer.end("total")

    def _output_audio(
        self,
        job_id: str,
        gen_timer: GenTimer,
        outputs: Any,
        output_type: str = "audio_path",
    ) -> str:
        gen_timer.start("output_audio")
        assert self.processor is not None
        try:
            if not job_id:
                audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            else:
                audio_path = f"/tmp/{job_id}.wav"
            self.processor.save_audio(
                outputs,
                audio_path)
            return audio_path
        finally:
            gen_timer.end("output_audio")

    def get_health(self) -> Dict[str, Any]:
        ret = super().get_health()
        if torch.cuda.is_available():
            ret["gpu"] = torch.cuda.get_device_name(0)
        return ret
