"""
Wrapper class for Flux Kontext model generation.
"""
import logging
import sys
import random
import asyncio

from typing import override
from typing import Optional
from typing import Dict
from typing import Any

from PIL import Image

import torch
import torch.distributed as dist
from torch import inference_mode

from image_utils import base64_to_img
from wrapper_model import GenerationInterruptedError
from wrapper_flux import FluxGeneration

from flux_xfuser import parallelize_transformer

from diffusers import FluxKontextPipeline

from xfuser.config import EngineConfig
from xfuser.core.distributed import get_runtime_state
from xfuser.core.distributed import initialize_runtime_state
from xfuser.core.distributed import get_pipeline_parallel_world_size


class FluxKontextGeneration(FluxGeneration):
    """Class for generating images using the Flux Kontext model."""

    def __init__(
        self,
        model_name: str = "fluxkontext",
        engine_config: EngineConfig = None,
        param_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(
            model_name=model_name,
            engine_config=engine_config,
            param_dtype=param_dtype)

    def load_model(self) -> None:
        """Load the Flux Kontext model from Hugging Face."""
        assert torch.cuda.is_available()

        self.load_timer.start("pipeline")
        cache_args = None
        self.MODEL_NAME = "black-forest-labs/FLUX.1-Kontext-dev"
        self.pipeline = FluxKontextPipeline.from_pretrained(
            pretrained_model_name_or_path=self.MODEL_NAME,
            engine_config=self.engine_config,
            cache_args=cache_args,
            torch_dtype=self.param_dtype,
            # device_map="auto", # TODO check if needed
        )
        self.pipeline = self.pipeline.to(self.device)
        self.load_timer.end("pipeline")

        logging.info(
            f"Loaded FluxKontextPipeline: {self.MODEL_NAME} device:{self.device} dtype:{self.param_dtype}.")

    def init_model_parallelism(self) -> None:
        """Initialize model parallelism using xfuser."""
        if not dist.is_initialized() or self.world_size <= 1:
            return

        self.load_timer.start("dit_parallel")
        initialize_runtime_state(self.pipeline, self.engine_config)
        get_runtime_state().set_input_parameters(
            batch_size=1,
            # height=self.input_config.height,
            # width=self.input_config.width,
            # num_inference_steps=self.input_config.num_inference_steps,
            max_condition_sequence_length=512,
            split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
        )

        parallelize_transformer(self.pipeline)
        self.load_timer.end("dit_parallel")

    def model_compile(self) -> None:
        """Compile the model using torch compile if enabled."""
        if not self.torch_compile:
            return

        self.load_timer.start("dit_compile")
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.pipeline.transformer = torch.compile(
            self.pipeline.transformer,
            mode="max-autotune-no-cudagraphs"
        )
        self.load_timer.end("dit_compile")

    @inference_mode()
    async def warmup(self) -> None:
        """Warmup the model with a dummy generation to initialize everything."""
        logging.info(f"[{self.rank}] Warmup for Flux Kontext generation.")
        empty_img = Image.new("RGB", (512, 512), (255, 255, 255))
        await self.generate(
            empty_img,
            width=1280,
            height=800,
            prompt="A warmup image to initialize the model.",
            neg_prompt="",
            sampling_steps=5)

    @override
    @inference_mode()
    async def generate(
        self,
        img: Image.Image,
        height: int,
        width: int,
        prompt: str,
        neg_prompt: str = "",
        sampling_steps: int = 25,  # 10
        seed: Optional[int] = None,
        job_id: Optional[str] = None,
    ) -> Image.Image:
        """
        Generate an image from another image using the Flux Kontext model.
        Args:
            img (Image.Image): Input image to guide the generation.
            height (int): Height of the generated image.
            width (int): Width of the generated image.
            prompt (str): Text prompt to guide the image generation.
            negative_prompt (str, optional): Negative prompt to avoid certain features in the image.
            sampling_steps (int, optional): Number of inference steps for sampling. Default is 25.
        """
        gen_timer = self._new_gen_timer(job_id)

        # Check if the image size is supported for the current parallelism setting
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py
        height_latent = height // self.pipeline.vae_scale_factor
        width_latent = width // self.pipeline.vae_scale_factor
        img_latent_shape = (height_latent // 2) * (width_latent // 2)
        if img_latent_shape % self.world_size != 0:
            raise ValueError(f"{height}x{width} not supported for {self.world_size} GPUs.")

        gen_timer.start("image_preprocess")
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        gen_timer.end("image_preprocess")

        self.running = True  # Mark running to avoid concurrent calls

        try:
            if seed is not None and seed >= 0:
                self.set_seed(seed)
            seed = self.base_seed if self.base_seed >= 0 else random.randint(0, sys.maxsize)
            seed_g = torch.Generator(device=self.device)
            seed_g.manual_seed(seed)

            def callback_gen_timer(
                pipeline: FluxKontextPipeline,
                step: int,
                timestep: int,
                callback_kwargs: dict
            ) -> dict:
                gen_timer.end(f"step_{step:03d}")
                if step < sampling_steps - 1:
                    gen_timer.start(f"step_{step + 1:03d}")
                if self.interrupted:  # type: ignore[has-type]
                    self.interrupted = False
                    raise GenerationInterruptedError(f"Generation interrupted at step {step + 1}.")
                return callback_kwargs

            gen_timer.start(f"step_{0:03d}")
            output = await asyncio.to_thread(
                self.pipeline,
                image=img,
                height=height,
                width=width,
                prompt=prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=sampling_steps,
                output_type="pil",
                generator=seed_g,
                callback_on_step_end=callback_gen_timer,
            )

            assert len(output.images) == 1, f"Expected 1 image, but got {len(output.images)} images."

            return output.images[0]
        finally:
            self.running = False
            gen_timer.end("total")

    async def get_rest_args(self, data_json: Dict[str, str]) -> Dict[str, Any]:
        if data_json is None:
            raise ValueError("Missing JSON body")
        img_base64 = data_json.get("img", None)
        if img_base64 is None:
            raise ValueError("Missing 'img' parameter")
        img = base64_to_img(img_base64)
        prompt = data_json.get("prompt", None)
        if prompt is None:
            raise ValueError("Missing 'prompt' parameter")
        neg_prompt = data_json.get("neg_prompt", "")
        height = int(data_json.get("height", 480))
        width = int(data_json.get("width", 640))
        steps = int(data_json.get("sampling_steps", 20))
        seed = data_json.get("seed", None)
        return {
            "task": self.model_name,
            "args": {
                "img": img,
                "prompt": prompt,
                "neg_prompt": neg_prompt,
                "width": width,
                "height": height,
                "sampling_steps": steps,
                "seed": seed,
            }
        }
