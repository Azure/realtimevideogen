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
import torch.distributed as dist
from torch import inference_mode

from wrapper_model import ModelGeneration
from wrapper_model import GenerationInterruptedError

from diffusers import HiDreamImagePipeline

from transformers import AutoTokenizer
from transformers import LlamaForCausalLM

from xfuser.config import EngineConfig


class HiDreamGeneration(ModelGeneration):
    """Handle image generation using the HiDream model."""
    HF_MODEL_NAME = "HiDream-ai/HiDream-I1-Full"

    def __init__(
        self,
        model_name: str = "hidream",
        engine_config: Optional[EngineConfig] = None,
        param_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(model_name)

        self.engine_config = engine_config
        if self.engine_config is not None:
            self.torch_compile = self.engine_config.runtime_config.use_torch_compile
        self.param_dtype = param_dtype

        self.gpu = torch.cuda.get_device_name(0)

        # Model components
        self.pipeline: Optional[HiDreamImagePipeline] = None
        self.text_encoder: Optional[LlamaForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None

    def __del__(self) -> None:
        # Clean models
        if hasattr(self, "pipeline") and self.pipeline is not None:
            self.pipeline = None
        if hasattr(self, "text_encoder") and self.text_encoder is not None:
            self.text_encoder = None
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            self.tokenizer = None
        if dist.is_initialized():
            dist.destroy_process_group()

    def init_parallelism(self) -> None:
        self.load_timer.start("torch_dist")

        self.rank = int(os.getenv("RANK", 0))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))

        self.device_id = self.local_rank
        self.device = torch.device(f"cuda:{self.device_id}")

        torch.cuda.set_device(self.local_rank)

        if self.world_size > 1:
            # https://github.com/dw763j/HiDream-I1-multigpu
            # https://github.com/HiDream-ai/HiDream-I1/pull/30/files
            logging.warning("HiDream is not optimized for multi-GPU setups (yet).")
            self.world_size = 1

        self.load_timer.end("torch_dist")

    def load_model(self) -> None:
        assert torch.cuda.is_available()

        self.load_timer.start("pipeline")

        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")  # nosec B615
        self.text_encoder = LlamaForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            output_hidden_states=True,
            output_attentions=True,
            torch_dtype=self.param_dtype,
        )  # nosec B615

        self.pipeline = HiDreamImagePipeline.from_pretrained(
            pretrained_model_name_or_path=self.HF_MODEL_NAME,
            tokenizer_4=self.tokenizer,
            text_encoder_4=self.text_encoder,
            torch_dtype=self.param_dtype,
        )
        self.pipeline = self.pipeline.to(self.device)  # type: ignore[attr-defined]
        self.load_timer.end("pipeline")

        logging.info(
            f"Loaded HiDreamImagePipeline: {self.HF_MODEL_NAME} device:{self.device} dtype:{self.param_dtype} "
            f"device_map:{self.pipeline.hf_device_map}.")  # type: ignore[attr-defined]

    def init_model_parallelism(self) -> None:
        """HiDream does not support parallelism yet."""
        if self.world_size > 1:
            logging.warning("Parallelism not supported.")

    def model_compile(self) -> None:
        """Compile the model using torch.compile if enabled."""
        if not self.torch_compile:
            return

        self.load_timer.start("dit_compile")
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.pipeline.transformer = torch.compile(  # type: ignore[attr-defined]
            self.pipeline.transformer,  # type: ignore[attr-defined]
            mode="max-autotune-no-cudagraphs"
        )
        assert self.pipeline.transformer is not None  # type: ignore[attr-defined]
        self.load_timer.end("dit_compile")

    def _assert_model_init(self) -> None:
        super()._assert_model_init()
        assert self.pipeline is not None

    def _assert_args(
        self,
        height: int,
        width: int,
    ) -> None:
        """Check if the image size is supported for the current parallelism setting."""
        assert self.pipeline is not None
        height_latent = height // self.pipeline.vae_scale_factor  # type: ignore[attr-defined]
        width_latent = width // self.pipeline.vae_scale_factor  # type: ignore[attr-defined]
        img_latent_shape = (height_latent // 2) * (width_latent // 2)
        if img_latent_shape % self.world_size != 0:
            raise ValueError(f"{height}x{width} not supported for {self.world_size} GPUs.")

    @inference_mode()
    async def warmup(self) -> None:
        logging.info(f"[{self.rank}] Warmup for HiDream generation.")
        await self.generate(
            height=720,
            width=1280,
            prompt="A warmup image to initialize the model.",
            neg_prompt="",
            sampling_steps=2)

    @override
    @inference_mode()
    async def generate(
        self,
        height: int,
        width: int,
        prompt: str,
        neg_prompt: str = "",
        sampling_steps: int = 25,  # 10\
        seed: Optional[int] = None,
        job_id: Optional[str] = None,
    ) -> Image.Image:
        """
        Generate an image from a prompt using the HiDream model.
        Args:
            height (int): Height of the generated image.
            width (int): Width of the generated image.
            prompt (str): Text prompt to guide the image generation.
            negative_prompt (str, optional): Negative prompt to avoid certain features in the image.
            sampling_steps (int, optional): Number of inference steps for sampling. Default is 25.
            seed (int, optional): Random seed for reproducibility. If None, a random seed will be generated.
            job_id (str, optional): Job ID for tracking the generation process.
        """
        gen_timer = self._new_gen_timer(job_id)

        self._assert_args(height, width)
        self._assert_model_init()
        assert self.pipeline is not None

        self.running = True  # Mark running to avoid concurrent calls

        try:
            if seed is None or seed < 0:
                seed = random.randint(0, sys.maxsize)
            seed_g = torch.Generator(device=self.device)
            seed_g.manual_seed(seed)

            def callback_gen_timer(
                pipeline: HiDreamImagePipeline,
                step: int,
                timestep: int,
                callback_kwargs: dict
            ) -> dict:
                gen_timer.end(f"step_{step:03d}")
                if step < sampling_steps - 1:
                    gen_timer.start(f"step_{step + 1:03d}")
                if self.interrupted:  # type: ignore[has-type]
                    self.interrupted = False
                    raise GenerationInterruptedError(f"Generation interrupted at step {step + 1}")
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

            if not output or len(output.images) != 1:
                raise ValueError(f"Expected 1 image, but got {len(output.images)} images")
            image = output.images[0]
            return image
        finally:
            self.running = False
            gen_timer.end("total")

    def get_health(self) -> Dict[str, Any]:
        ret = super().get_health()
        ret.update({
            "gpu": self.gpu,
            "rank": self.rank,
            "world_size": self.world_size,
            "torch_compile": self.torch_compile,
            "dtype": str(self.param_dtype),
            "device_map": self.pipeline.hf_device_map if self.pipeline else None,  # type: ignore[attr-defined]
        })
        return ret

    async def get_rest_args(
        self,
        data_json: Dict[str, Union[str, int, float]]
    ) -> Dict[str, Any]:
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
