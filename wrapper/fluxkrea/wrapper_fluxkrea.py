import logging
import sys
import random

from typing import Optional
from typing import Dict
from typing import Any

from PIL import Image

import torch
import torch.distributed as dist
from torch import inference_mode

from wrapper_model import GenerationInterruptedError
from wrapper_flux import FluxGeneration

from flux_xfuser import parallelize_transformer

from diffusers import FluxPipeline

from xfuser.config import EngineConfig
from xfuser.core.distributed import get_runtime_state
from xfuser.core.distributed import initialize_runtime_state
from xfuser.core.distributed import get_pipeline_parallel_world_size


class FluxKreaGeneration(FluxGeneration):
    """Handle image generation using the Flux Krea model."""
    HF_MODEL_NAME = "black-forest-labs/FLUX.1-Krea-dev"

    def __init__(
        self,
        model_name: str = "fluxkrea",
        engine_config: EngineConfig = None,
        param_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(
            model_name=model_name,
            engine_config=engine_config,
            param_dtype=param_dtype)

        self.pipeline: Optional[FluxPipeline] = None

    def load_model(self) -> None:
        """Load the Flux Krea model."""
        assert torch.cuda.is_available()

        self.load_timer.start("pipeline")
        cache_args = None
        self.pipeline = FluxPipeline.from_pretrained(
            pretrained_model_name_or_path=self.HF_MODEL_NAME,
            engine_config=self.engine_config,
            cache_args=cache_args,
            torch_dtype=self.param_dtype,
            # device_map="auto", # TODO check if needed
        )
        if not self.pipeline:
            raise ValueError("Failed to load FluxKrea pipeline.")
        assert isinstance(self.pipeline, FluxPipeline)
        self.pipeline = self.pipeline.to(self.device)  # type: ignore[attr-defined]
        self.load_timer.end("pipeline")

        logging.info(
            "Loaded FluxKreaPipeline: %s device:%s dtype:%s.",
            self.HF_MODEL_NAME, self.device, self.param_dtype)

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
        """Compile the model using torch.compile if enabled."""
        if not self.torch_compile:
            return
        if self.pipeline is None:
            return

        self.load_timer.start("dit_compile")
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.pipeline.transformer = torch.compile(  # type: ignore[attr-defined]
            self.pipeline.transformer,  # type: ignore[attr-defined]
            mode="max-autotune-no-cudagraphs"
        )
        self.load_timer.end("dit_compile")

    @inference_mode()
    async def generate(
        self,
        width: int,
        height: int,
        prompt: str,
        neg_prompt: str = "",
        sampling_steps: int = 25,  # 10
        seed: Optional[int] = None,
        job_id: Optional[str] = None,
    ) -> Image.Image:
        """
        Generate an image from another image using the Flux Krea model.
        Args:
            height (int): Height of the generated image.
            width (int): Width of the generated image.
            prompt (str): Text prompt to guide the image generation.
            negative_prompt (str, optional): Negative prompt to avoid certain features in the image.
            sampling_steps (int, optional): Number of inference steps for sampling. Default is 25.
        """
        gen_timer = self._new_gen_timer(job_id)

        self._assert_model_init()
        # Check if the image size is supported for the current parallelism setting
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py
        assert self.pipeline is not None, "FluxKrea pipeline not initialized."
        height_latent = height // self.pipeline.vae_scale_factor  # type: ignore[attr-defined]
        width_latent = width // self.pipeline.vae_scale_factor  # type: ignore[attr-defined]
        img_latent_shape = (height_latent // 2) * (width_latent // 2)
        if img_latent_shape % self.world_size != 0:
            raise ValueError(f"{height}x{width} not supported for {self.world_size} GPUs.")

        self.running = True  # Mark running to avoid concurrent calls

        try:
            if seed is not None and seed >= 0:
                self.set_seed(seed)
            else:
                self.reset_seed()
            seed = self.base_seed if self.base_seed >= 0 else random.randint(0, sys.maxsize)
            seed_g = torch.Generator(device=self.device)
            seed_g.manual_seed(seed)

            def callback_gen_timer(
                pipeline: FluxPipeline,
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
            output = self.pipeline(  # type: ignore[operator]
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
        """Extract and validate arguments from the REST API request."""
        if data_json is None:
            raise ValueError("Missing JSON body")
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
                "prompt": prompt,
                "neg_prompt": neg_prompt,
                "height": height,
                "width": width,
                "sampling_steps": steps,
                "seed": seed,
            }
        }
