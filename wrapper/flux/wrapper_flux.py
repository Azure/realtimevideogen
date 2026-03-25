"""
Wrapper class for FLUX model generation using Hugging Face Diffusers and Xfuser.
"""
import logging
import sys
import random
import asyncio

from typing import override
from typing import Optional
from typing import Dict
from typing import Any
from typing import Union

from PIL import Image

import torch
import torch.distributed as dist
from torch import inference_mode

from wrapper_model import GenerationInterruptedError
from wrapper_usp import USPGeneration

from flux_xfuser import parallelize_transformer

from diffusers import FluxPipeline

from xfuser.config import EngineConfig
from xfuser.core.distributed import get_runtime_state
from xfuser.core.distributed import initialize_runtime_state
from xfuser.core.distributed import get_pipeline_parallel_world_size


class FluxGeneration(USPGeneration):
    """Wrapper class for FLUX model generation using Hugging Face Diffusers and Xfuser."""

    MAX_LOG_TEXT_LEN = 64

    def __init__(
        self,
        model_name: str = "flux",
        engine_config: EngineConfig = None,
        param_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(
            model_name=model_name,
            engine_config=engine_config,
            param_dtype=param_dtype,
        )

        # Model components
        self.pipeline: Optional[FluxPipeline] = None

    def __del__(self) -> None:
        # Clean models
        if self.pipeline is not None:
            self.pipeline = None
        super().__del__()

    def load_model(self) -> None:
        self.load_timer.start("pipeline")
        cache_args = None
        """
        cache_args = {
            "use_teacache": engine_args.use_teacache,
            "use_fbcache": engine_args.use_fbcache,
            "rel_l1_thresh": 0.12,
            "return_hidden_states_first": False,
            "num_steps": input_config.num_inference_steps,
        }
        """
        self.MODEL_NAME = "black-forest-labs/FLUX.1-dev"
        self.pipeline = FluxPipeline.from_pretrained(
            pretrained_model_name_or_path=self.MODEL_NAME,
            engine_config=self.engine_config,
            cache_args=cache_args,
            torch_dtype=self.param_dtype,
            # device_map="auto", # TODO check if needed
        )
        if not self.pipeline:
            raise ValueError("Failed to load FLUX pipeline.")
        assert isinstance(self.pipeline, FluxPipeline)
        # TODO save some memory for V100 32GB
        # https://huggingface.co/docs/diffusers/main/en/optimization/memory
        # https://huggingface.co/docs/diffusers/main/en/optimization/memory#reduce-memory-usage
        # self.pipeline.enable_sequential_cpu_offload()
        # self.pipeline.enable_model_cpu_offload()
        # https://huggingface.co/docs/diffusers/en/training/distributed_inference#model-sharding
        self.pipeline = self.pipeline.to(self.device)  # type: ignore[attr-defined]
        self.load_timer.end("pipeline")

        logging.info(
            f"Loaded FluxPipeline: {self.MODEL_NAME} device:{self.device} dtype:{self.param_dtype}.")

    def init_model_parallelism(self) -> None:
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
        if not self.torch_compile:
            return
        if not self.pipeline:
            raise ValueError("FLUX pipeline not initialized.")

        self.load_timer.start("dit_compile")
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.pipeline.transformer = torch.compile(  # type: ignore[attr-defined]
            self.pipeline.transformer,  # type: ignore[attr-defined]
            mode="max-autotune-no-cudagraphs"
        )
        self.load_timer.end("dit_compile")

    def _assert_model_init(self) -> None:
        super()._assert_model_init()
        if self.pipeline is None:
            raise ValueError("FLUX pipeline not initialized.")

    def _assert_args(
        self,
        height: int,
        width: int,
    ) -> None:
        # Check if the image size is supported for the current parallelism setting
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py
        if not self.pipeline:
            raise ValueError("FLUX pipeline not initialized.")
        height_latent = height // self.pipeline.vae_scale_factor  # type: ignore[attr-defined]
        width_latent = width // self.pipeline.vae_scale_factor  # type: ignore[attr-defined]
        img_latent_shape = (height_latent // 2) * (width_latent // 2)
        if img_latent_shape % self.world_size != 0:
            raise ValueError(f"{width}x{height} not supported for {self.world_size} GPUs.")

    @inference_mode()
    async def warmup(self) -> None:
        """Warmup the model with a sample generation."""
        logging.info(f"[{self.rank}] Warmup for FLUX generation.")
        await self.generate(
            # Ideally, we would use smaller sizes, but it has issues with 8 GPUs
            job_id="warmup",
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
        sampling_steps: int = 25,
        seed: Optional[int] = None,
        job_id: Optional[str] = None,
    ) -> Image.Image:
        """Generate an image from a prompt using the FLUX model."""
        gen_timer = self._new_gen_timer(job_id)

        self._assert_model_init()
        self._assert_args(height, width)
        assert self.pipeline is not None

        self.running = True  # Mark running to avoid concurrent calls

        try:
            if seed is not None and seed >= 0:
                self.set_seed(seed)
            seed = random.randint(0, sys.maxsize)
            if self.base_seed is not None and self.base_seed >= 0:
                seed = self.base_seed
            seed_g = torch.Generator(device=self.device)
            seed_g.manual_seed(seed)

            def callback_gen_timer(
                pipeline: FluxPipeline,
                step: int,
                timestep: int,
                callback_kwargs: Dict[str, Any],
            ) -> Dict[str, Any]:
                gen_timer.end(f"step_{step:03d}")
                logging.info(f"[{self.rank}] Step {step + 1}/{sampling_steps}.")

                if step < sampling_steps - 1:
                    gen_timer.start(f"step_{step + 1:03d}")
                if self.interrupted:  # type: ignore[has-type]
                    self.interrupted = False
                    raise GenerationInterruptedError(f"Generation interrupted at step {step + 1}")
                return callback_kwargs

            logging.info(
                f"[{self.rank}] Generating image with {width}x{height} and '{prompt[:self.MAX_LOG_TEXT_LEN]}'...")
            gen_timer.start(f"step_{0:03d}")
            output: Any = await asyncio.to_thread(
                lambda: self.pipeline(  # type: ignore[operator, misc]
                    width=width,
                    height=height,
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    num_inference_steps=sampling_steps,
                    output_type="pil",
                    generator=seed_g,
                    callback_on_step_end=callback_gen_timer,
                )
            )

            if not output or len(output.images) != 1:
                raise ValueError(f"Expected 1 image, but got {len(output.images)} images")
            image = output.images[0]
            return image
        finally:
            self.running = False
            torch.cuda.empty_cache()
            gen_timer.end("total")

    def get_health(self) -> Dict[str, Any]:
        ret = super().get_health()
        ret.update({
            "device_map": getattr(self.pipeline, "hf_device_map", None) if self.pipeline else None,
        })
        return ret

    async def get_rest_args(
        self,
        data_json: Dict[str, Union[str, int, float]]
    ) -> Dict[str, Any]:
        if data_json is None or not isinstance(data_json, dict):
            raise ValueError("Missing JSON body")

        job_id = data_json.get("job_id", None)

        prompt = data_json.get("prompt", None)
        if prompt is None:
            raise ValueError("Missing 'prompt' parameter")
        neg_prompt = data_json.get("neg_prompt", "")

        height = int(data_json.get("height", 480))
        width = int(data_json.get("width", 640))
        steps = int(data_json.get("sampling_steps", 20))

        rest_args: Dict[str, Any] = {
            "task": self.model_name,
            "args": {
                "job_id": job_id,
                "prompt": prompt,
                "neg_prompt": neg_prompt,
                "height": height,
                "width": width,
                "sampling_steps": steps,
            }
        }
        if "seed" in data_json:
            seed = data_json.get("seed", -1)
            if seed is not None:
                rest_args["args"]["seed"] = int(seed)
        return rest_args
