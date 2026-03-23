import logging
import os
import sys
import random

from typing import override
from typing import Optional
from typing import Dict
from typing import Any
from typing import Union

from PIL import Image

import torch
from torch import inference_mode

from wrapper_model import ModelGeneration
from wrapper_model import GenerationInterruptedError

from diffusers import QwenImagePipeline

from xfuser.config import EngineConfig


class QwenImageGeneration(ModelGeneration):
    """Handle image generation using the QwenImage model."""

    HF_MODEL_NAME = "Qwen/Qwen-Image"

    def __init__(
        self,
        model_name: str = "qwenimage",
        engine_config: EngineConfig = None,
        param_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(model_name)

        self.engine_config = engine_config
        if self.engine_config is not None:
            self.torch_compile = self.engine_config.runtime_config.use_torch_compile
        self.param_dtype = param_dtype

        # Model components
        self.pipeline: QwenImagePipeline = None

    def __del__(self) -> None:
        """Cleanup resources on deletion."""
        if self.pipeline is not None:
            self.pipeline = None

    def init_parallelism(self) -> None:
        """Initialize distributed settings for multi-GPU setups."""
        self.load_timer.start("torch_dist")

        self.rank = int(os.getenv("RANK", 0))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))

        self.device_id = self.local_rank
        self.device = torch.device(f"cuda:{self.device_id}")

        torch.cuda.set_device(self.local_rank)

        if self.world_size > 1:
            logging.warning("Qwen Image is not optimized for multi-GPU setups (yet).")
            self.world_size = 1

        self.load_timer.end("torch_dist")

    def load_model(self) -> None:
        """Load the Qwen Image model from Hugging Face."""
        assert torch.cuda.is_available()

        self.load_timer.start("pipeline")
        self.pipeline = QwenImagePipeline.from_pretrained(
            pretrained_model_name_or_path=self.HF_MODEL_NAME,
            torch_dtype=self.param_dtype,
        )
        self.pipeline = self.pipeline.to(self.device)
        self.load_timer.end("pipeline")

    def init_model_parallelism(self) -> None:
        """Qwen Image does not support parallelism yet."""
        if self.world_size > 1:
            logging.warning("Parallelism not supported.")

    def model_compile(self) -> None:
        """Compile the model using torch.compile if enabled."""
        if not self.torch_compile:
            return

        self.load_timer.start("dit_compile")
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.pipeline.transformer = torch.compile(
            self.pipeline.transformer,
            mode="max-autotune-no-cudagraphs"
        )
        self.load_timer.end("dit_compile")

    def _assert_model_init(self) -> None:
        """Check if the model has been initialized."""
        super()._assert_model_init()
        assert self.pipeline is not None

    def _assert_args(
        self,
        height: int,
        width: int,
    ) -> None:
        """Check if the image size is supported for the current parallelism setting."""
        height_latent = height // self.pipeline.vae_scale_factor
        width_latent = width // self.pipeline.vae_scale_factor
        img_latent_shape = (height_latent // 2) * (width_latent // 2)
        if img_latent_shape % self.world_size != 0:
            raise ValueError(f"{height}x{width} not supported for {self.world_size} GPUs.")

    @inference_mode()
    async def warmup(self) -> None:
        """Warmup the Qwen Image model with a sample generation."""
        logging.info(f"[{self.rank}] Warmup for Qwen Image generation.")
        await self.generate(
            # Ideally, we would use smaller sizes, but it has issues with 8 GPUs
            width=1280,
            height=800,
            prompt="A warmup image to initialize the model.",
            neg_prompt="",
            sampling_steps=5)  # It needs at least 5 steps to warm up properly

    @override
    @inference_mode()
    async def generate(
        self,
        height: int,
        width: int,
        prompt: str,
        neg_prompt: str = "",
        sampling_steps: int = 25,  # 10
        seed: Optional[int] = None,
        job_id: Optional[str] = None,
    ) -> Image.Image:
        """Generate an image using QwenImage."""
        gen_timer = self._new_gen_timer(job_id)

        self._assert_model_init()
        self._assert_args(height, width)

        self.running = True  # Mark running to avoid concurrent calls

        try:
            if seed is None or seed < 0:
                seed = random.randint(0, sys.maxsize)
            seed_g = torch.Generator(device=self.device)
            seed_g.manual_seed(seed)

            def callback_gen_timer(
                pipeline: QwenImagePipeline,
                step: int,
                timestep: int,
                callback_kwargs: dict
            ) -> dict:
                gen_timer.end(f"step_{step:03d}")
                if step < sampling_steps - 1:
                    gen_timer.start(f"step_{step + 1:03d}")
                if self.interrupted:  # type: ignore[has-type]
                    self.interrupted = False
                    raise GenerationInterruptedError(
                        f"Generation interrupted at step {step + 1} ({timestep}).")
                return callback_kwargs

            gen_timer.start(f"step_{0:03d}")
            output = self.pipeline(
                height=height,
                width=width,
                prompt=prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=sampling_steps,
                output_type="pil",
                generator=seed_g,
                callback_on_step_end=callback_gen_timer,
            )

            if not output or len(output.images) != 1:
                raise ValueError(f"Expected 1 image, but got {len(output.images)} images")
            image = output.images[0]
            return image
        finally:
            self.running = False
            gen_timer.end("total")

    async def get_rest_args(
        self,
        data_json: Dict[str, Union[str, int, float]]
    ) -> Dict[str, Any]:
        """Extract and validate REST API arguments for Qwen Image generation."""
        if data_json is None:
            raise ValueError("Missing JSON body")
        prompt = data_json.get("prompt", None)
        if prompt is None:
            raise ValueError("Missing 'prompt' parameter")
        neg_prompt = data_json.get("neg_prompt", "")
        height = int(data_json.get("height", 480))
        width = int(data_json.get("width", 640))
        steps = int(data_json.get("sampling_steps", 20))
        return {
            "task": self.model_name,
            "args": {
                "prompt": prompt,
                "neg_prompt": neg_prompt,
                "width": width,
                "height": height,
                "sampling_steps": steps,
            }
        }
