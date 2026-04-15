"""
from diffusers import CogView4Pipeline
import torch

pipe = CogView4Pipeline.from_pretrained("THUDM/CogView4-6B", torch_dtype=torch.bfloat16).to("cuda")

# Open it for reduce GPU memory usage
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

prompt = "A vibrant cherry red sports car sits proudly under the gleaming sun, its polished exterior smooth "
prompt += "and flawless, casting a mirror-like reflection. The car features a low, aerodynamic body, angular "
prompt += "headlights that gaze forward like predatory eyes, and a set of black, high-gloss racing rims that "
prompt += "contrast starkly with the red. "
prompt += "A subtle hint of chrome embellishes the grille and exhaust, while the tinted windows suggest a "
prompt += "luxurious and private interior. The scene conveys a sense of speed and elegance, the car appearing as "
prompt += "if it's about to burst into a sprint along a coastal road, with the ocean's azure waves crashing in "
prompt += "the background."
image = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    num_images_per_prompt=1,
    num_inference_steps=50,
    width=1024,
    height=1024,
).images[0]

image.save("cogview4.png")
"""

import logging
import os
import sys
import random

from typing import Optional
from typing import Dict
from typing import Union
from typing import Any

from PIL import Image

import torch
from torch import inference_mode

from wrapper_model import ModelGeneration

from diffusers import CogView4Pipeline

from xfuser.config import EngineConfig


class CogViewGeneration(ModelGeneration):
    """Handle image generation using the CogView model."""
    MODEL_NAME = "THUDM/CogView4-6B"

    def __init__(
        self,
        model_name: str = "cogview",
        engine_config: Optional[EngineConfig] = None,
        param_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(model_name)

        self.engine_config = engine_config
        if self.engine_config is not None:
            self.torch_compile = self.engine_config.runtime_config.use_torch_compile
        self.param_dtype = param_dtype

        # Parallelism
        self.GPU = None
        if torch.cuda.is_available():
            self.GPU = torch.cuda.get_device_name(0)

        self.base_seed = random.randint(0, sys.maxsize)

        # Model components
        self.pipeline: Optional[CogView4Pipeline] = None

    def __del__(self) -> None:
        # Clean models
        if self.pipeline is not None:
            self.pipeline = None

    def init_parallelism(self) -> None:
        self.load_timer.start("torch_dist")

        self.rank = int(os.getenv("RANK", 0))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))

        self.device_id = self.local_rank
        self.device = torch.device(f"cuda:{self.device_id}")

        torch.cuda.set_device(self.local_rank)

        if self.world_size > 1:
            logging.warning("CogView is not optimized for multi-GPU setups (yet).")
            self.world_size = 1

        self.load_timer.end("torch_dist")

    def load_model(self) -> None:
        assert torch.cuda.is_available()

        self.load_timer.start("pipeline")
        self.pipeline = CogView4Pipeline.from_pretrained(
            pretrained_model_name_or_path=self.MODEL_NAME,
            torch_dtype=self.param_dtype,
        )
        assert self.pipeline is not None
        self.pipeline = self.pipeline.to(self.device)  # type: ignore[union-attr]

        # Enable memory optimizations
        """
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.vae.enable_slicing()
        self.pipeline.vae.enable_tiling()
        """

        self.load_timer.end("pipeline")

        logging.info(f"Loaded CogView4Pipeline: {self.MODEL_NAME} device:{self.device} dtype:{self.param_dtype}.")

    def init_model_parallelism(self) -> None:
        # CogView4 doesn't support model parallelism yet
        pass

    def model_compile(self) -> None:
        if not self.torch_compile:
            return

        # TODO this is not likely supported
        self.load_timer.start("compile")
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        assert self.pipeline is not None
        if hasattr(self.pipeline, 'transformer'):
            self.pipeline.transformer = torch.compile(
                self.pipeline.transformer,
                mode="max-autotune-no-cudagraphs"
            )
        self.load_timer.end("compile")

    def _assert_model_init(self) -> None:
        super()._assert_model_init()
        assert self.pipeline is not None

    def _assert_args(
        self,
        height: int,
        width: int,
    ) -> None:
        """
        Check if the image size is supported for the current parallelism setting.
        """
        # CogView4 has specific size requirements
        if height % 64 != 0 or width % 64 != 0:
            raise ValueError(f"Height and width must be divisible by 64, got {height}x{width}")

    @inference_mode()
    async def warmup(self) -> None:
        logging.info(f"[{self.rank}] Warmup for CogView4 generation.")
        await self.generate(
            height=512,
            width=512,
            prompt="A warmup image to initialize the model.",
            sampling_steps=2
        )

    @inference_mode()
    async def generate(
        self,
        height: int,
        width: int,
        prompt: str,
        guidance_scale: float = 3.5,
        sampling_steps: int = 50,
        job_id: Optional[str] = None,
    ) -> Image.Image:
        """
        Generate an image from a prompt using the CogView4 model.
        Args:
            height (int): Height of the generated image.
            width (int): Width of the generated image.
            prompt (str): Text prompt to guide the image generation.
            guidance_scale (float, optional): Guidance scale for generation. Default is 3.5.
            sampling_steps (int, optional): Number of inference steps for sampling. Default is 50.
        """
        gen_timer = self._new_gen_timer(job_id)

        self._assert_model_init()
        self._assert_args(height, width)
        assert self.pipeline is not None

        self.running = True  # Mark running to avoid concurrent calls

        try:
            seed = self.base_seed if self.base_seed >= 0 else random.randint(0, sys.maxsize)
            seed_g = torch.Generator(device=self.device)
            seed_g.manual_seed(seed)

            def callback_gen_timer(
                pipeline: CogView4Pipeline,
                step: int,
                timestep: int,
                callback_kwargs: dict
            ) -> dict:
                gen_timer.end(f"step_{step:03d}")
                if step < sampling_steps - 1:
                    gen_timer.start(f"step_{step + 1:03d}")
                return callback_kwargs

            gen_timer.start(f"step_{0:03d}")
            output = self.pipeline(  # type: ignore[operator]
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_images_per_prompt=1,
                num_inference_steps=sampling_steps,
                output_type="pil",
                generator=seed_g,
                callback_on_step_end=callback_gen_timer,
            )
            images = output.images

            assert len(images) == 1, f"Expected 1 image, but got {len(images)} images."

            return images[0]
        finally:
            self.running = False
            gen_timer.end("total")

    def get_health(self) -> Dict[str, Any]:
        ret = super().get_health()
        ret.update({
            "gpu": self.GPU,
            "rank": self.rank,
            "world_size": self.world_size,
            "torch_compile": self.torch_compile,
            "dtype": str(self.param_dtype),
            "device_map": getattr(self.pipeline, 'hf_device_map', None) if self.pipeline else None,
        })
        return ret

    async def get_rest_args(
        self,
        data_json: Dict[str, Union[str, int, float]]
    ) -> Dict[str, Any]:
        if data_json is None or not isinstance(data_json, dict):
            raise ValueError("Missing JSON body")

        prompt = data_json.get("prompt", None)
        if prompt is None:
            raise ValueError("Missing 'prompt' parameter")

        height = int(data_json.get("height", 1024))
        width = int(data_json.get("width", 1024))
        guidance_scale = float(data_json.get("guidance_scale", 3.5))
        steps = int(data_json.get("sampling_steps", 50))
        return {
            "task": self.model_name,
            "args": {
                "prompt": prompt,
                "height": height,
                "width": width,
                "guidance_scale": guidance_scale,
                "sampling_steps": steps,
            }
        }
