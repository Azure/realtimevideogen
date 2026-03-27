"""
Wrapper for HunyuanImage model.
"""
import logging
import os
import asyncio
import datetime

from typing import override
from typing import Any
from typing import Dict
from typing import Optional

from PIL import Image

import torch
import torch.distributed as dist
from torch import inference_mode

from wrapper_model import ModelGeneration

from transformers import AutoModelForCausalLM
# from hunyuan_image_3_pipeline import HunyuanImage3Text2ImagePipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from xfuser.config import EngineConfig


class HunyuanImageGeneration(ModelGeneration):
    """Handle image generation using the HunyuanImage model."""

    # HF_MODEL_NAME = "tencent/HunyuanImage-3.0"
    HF_MODEL_NAME = "./HunyuanImage-3"
    MAX_LOG_TEXT_LEN = 64

    def __init__(
        self,
        model_name: str = "hunyuanimage",
        engine_config: EngineConfig = None,
        param_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(model_name)

        self.engine_config = engine_config
        if self.engine_config is not None:
            self.torch_compile = self.engine_config.runtime_config.use_torch_compile
        self.param_dtype = param_dtype

        # Model components
        self.model: Optional[AutoModelForCausalLM] = None
        self.pipeline: Optional[DiffusionPipeline] = None

    def __del__(self) -> None:
        if self.model is not None:
            self.model = None

    def init_parallelism(self) -> None:
        self.load_timer.start("torch_dist")

        self.rank = int(os.getenv("RANK", 0))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))

        self.device_id = self.local_rank

        if not torch.cuda.is_available():
            self.device_id = 0
            self.device = torch.device("cpu")
            logging.warning("CUDA is not available. Running on CPU.")
            self.load_timer.end("torch_dist")
            return  # Single GPU mode, no parallelism needed

        self.device = torch.device(f"cuda:{self.device_id}")

        torch.cuda.set_device(self.local_rank)

        if self.world_size <= 1:
            self.load_timer.end("torch_dist")
            return  # Single GPU mode, no parallelism needed

        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=self.rank,
                world_size=self.world_size,
                timeout=datetime.timedelta(hours=24),  # Prevent NCCL timeout
            )

        self.load_timer.end("torch_dist")

        if not dist.is_initialized():
            raise RuntimeError("Distributed process group not initialized")

    def init_model_parallelism(self) -> None:
        logging.info(f"[{self.rank}] Hunyuan Image parallelism.")

    def load_model(self) -> None:
        assert torch.cuda.is_available()

        if self.rank > 0:
            logging.info(f"[{self.rank}] Model loaded only on rank 0.")
            return

        self.load_timer.start("model")
        self.model = AutoModelForCausalLM.from_pretrained(  # type: ignore[assignment]
            self.HF_MODEL_NAME,
            attn_implementation="sdpa",  # Use "flash_attention_2" if FlashAttention is installed
            trust_remote_code=True,
            torch_dtype=self.param_dtype,
            device_map="auto",
            moe_impl="eager",  # Use "flashinfer" if FlashInfer is installed
            # low_cpu_mem_usage=True, # TODO ?
            moe_drop_tokens=True,
        )  # nosec B615 - local path
        assert self.model is not None
        self.model.load_tokenizer(self.HF_MODEL_NAME)  # type: ignore[attr-defined]
        self.pipeline = self.model.pipeline  # type: ignore[attr-defined]
        self.load_timer.end("model")

    def model_compile(self) -> None:
        """Compile the model using torch.compile if enabled."""
        if not self.torch_compile:
            return

        if self.model:
            self.load_timer.start("dit_compile")
            self.model = torch.compile(  # type: ignore[call-overload]
                self.model,
                mode="max-autotune-no-cudagraphs"
            )
            self.load_timer.end("dit_compile")

    def _assert_args(
        self,
        height: int,
        width: int,
    ) -> None:
        # height_latent = height // self.pipeline.vae_scale_factor
        # width_latent = width // self.pipeline.vae_scale_factor
        # self.model.vae is AutoencoderKLConv3D
        assert self.model is not None
        vae_config = self.model.vae.config  # type: ignore[attr-defined]
        if width % vae_config.ffactor_spatial != 0:
            raise ValueError(f"Width {width} not supported. Must be multiple of {vae_config.ffactor_spatial}.")
        if height % vae_config.ffactor_spatial != 0:
            raise ValueError(f"Height {height} not supported. Must be multiple of {vae_config.ffactor_spatial}.")
        """
        height x width:
        ("1:1", "1024x1024"),
        ("4:3", "896x1152"),
        ("3:4", "1152x896"),
        ("16:9", "768x1280"),
        ("9:16", "1280x768"),
        ("21:9", "640x1408"),
        """
        if width * height > 1024 * 1024:
            raise ValueError(f"{width}x{height} too large. Max is 1024 x 1024.")

    def _assert_model_init(self) -> None:
        super()._assert_model_init()
        if self.model is None:
            raise ValueError("HunyuanImage model not loaded.")

    @inference_mode()
    async def warmup(self) -> None:
        logging.info(f"[{self.rank}] Warmup for Hunyuan Image generation.")
        await self.generate(
            height=1024,  # 1:1
            width=1024,
            prompt="A warmup image to initialize the model.",
            sampling_steps=5,  # It needs at least 5 steps to warm up properly
        )

    @override
    @inference_mode()
    async def generate(
        self,
        height: int,
        width: int,
        prompt: str,
        sampling_steps: int = 25,
        cfg: float = 0.5,
        seed: Optional[int] = None,
        job_id: Optional[str] = None,
    ) -> Optional[Image.Image]:
        """Generate an image using HunyuanImage 3."""
        if self.rank > 0:
            logging.info(f"[{self.rank}] Image generation only rank 0.")
            return None

        gen_timer = self._new_gen_timer(job_id)

        self._assert_model_init()
        self._assert_args(height, width)

        self.running = True  # Mark running to avoid concurrent calls

        try:
            logging.info(
                f"[{self.rank}] Generating image with {width}x{height}, "
                f"{cfg:.1f} CFG, and "
                f"{sampling_steps} steps, and "
                f"'{prompt[:self.MAX_LOG_TEXT_LEN]}'.")

            def callback_gen_timer(
                pipeline: DiffusionPipeline,
                step: int,
                timestep: int,
                callback_kwargs: Dict[str, Any],
            ) -> Dict[str, Any]:
                gen_timer.end(f"step_{step:03d}")
                logging.info(f"[{self.rank}] Step {step + 1}/{sampling_steps}.")

                if step < sampling_steps - 1:
                    gen_timer.start(f"step_{step + 1:03d}")
                self.check_interrupted()
                return callback_kwargs

            image = await asyncio.to_thread(
                self.model.generate_image,  # type: ignore[union-attr]
                prompt=prompt,
                # image_size="auto",
                # image_size=f"{height}x{width}",
                image_size=(height, width),
                diff_infer_steps=sampling_steps,
                # diff_guidance_scale=cfg,  # TODO figure if this breaks
                seed=seed,  # TODO figure if this breaks
                stream=True,
            )
            """
            # Pipeline mode would allow callback_on_step_end
            cot_text = None  # change for "think", "recaption"
            system_prompt = None  # get_system_prompt(use_system_prompt, bot_task, system_prompt)
            model_inputs = self.model.prepare_model_inputs(
                prompt=prompt,
                cot_text=cot_text,
                system_prompt=system_prompt,
                mode="gen_image",
                seed=seed,
                image_size=image_size,
            )
            images = await asyncio.to_thread(
                self.pipeline,
                batch_size=1,
                image_size=[width, height],
                prompt=prompt,
                num_inference_steps=sampling_steps,
                guidance_scale=cfg,
                output_type="pil",  # TODO different output modes
                callback_on_step_end=callback_gen_timer,
                seed=seed,
                **model_inputs
            )
            if not images or len(images) != 1:
                raise RuntimeError(f"Wrong image generated: {images}")
            image = images[0]
            """
            return image
        finally:
            self.running = False
            gen_timer.end("total")

    async def get_rest_args(self, data_json: Dict[str, str]) -> Dict[str, Any]:
        if data_json is None:
            raise ValueError("Missing JSON body")
        prompt = data_json.get("prompt", None)
        if prompt is None:
            raise ValueError("Missing 'prompt' parameter")
        height = int(data_json.get("height", 480))
        width = int(data_json.get("width", 640))
        steps = int(data_json.get("sampling_steps", 25))
        seed = data_json.get("seed", None)
        return {
            "task": self.model_name,
            "args": {
                "prompt": prompt,
                "width": width,
                "height": height,
                "sampling_steps": steps,
                "seed": seed,
            }
        }
