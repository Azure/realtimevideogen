"""
https://github.com/deepseek-ai/Janus/blob/1daa72fa409002d40931bd7b36a9280362469ead/demo/app_januspro.py#L15
"""
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

import numpy as np

import torch
import torch.distributed as dist
from torch import inference_mode

from wrapper_model import ModelGeneration

from transformers import AutoModelForCausalLM
from transformers import AutoConfig
from janus.models import VLChatProcessor

from xfuser.config import EngineConfig


class JanusProGeneration(ModelGeneration):
    """Wrapper class for Janus Pro model generation."""

    def __init__(
        self,
        model_name: str = "januspro",
        engine_config: EngineConfig = None,
        param_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(model_name)

        self.engine_config = engine_config
        if self.engine_config is not None:
            self.torch_compile = self.engine_config.runtime_config.use_torch_compile
        self.param_dtype = param_dtype

        # Parallelism
        self.gpu: Optional[str] = None
        if torch.cuda.is_available():
            self.gpu = torch.cuda.get_device_name(0)

        self.base_seed = random.randint(0, sys.maxsize)

        # Model components
        self.vl_gpt: Optional[torch.nn.Module] = None
        self.vl_chat_processor: Optional[Any] = None
        self.tokenizer: Optional[Any] = None

    def __del__(self) -> None:
        # Clean models
        if self.vl_gpt is not None:
            self.vl_gpt = None
        if self.vl_chat_processor is not None:
            self.vl_chat_processor = None
        if self.tokenizer is not None:
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
            logging.warning("Janus is not optimized for multi-GPU setups (yet).")
            self.world_size = 1

        self.load_timer.end("torch_dist")

    def load_model(self) -> None:
        assert torch.cuda.is_available()

        self.load_timer.start("processor")
        self.MODEL_NAME = "deepseek-ai/Janus-Pro-7B"
        self.vl_chat_processor = VLChatProcessor.from_pretrained(
            self.MODEL_NAME
        )
        self.tokenizer = self.vl_chat_processor.tokenizer
        self.load_timer.end("processor")

        self.load_timer.start("config")
        config = AutoConfig.from_pretrained(self.MODEL_NAME)  # nosec B615
        language_config = config.language_config
        language_config._attn_implementation = 'eager'
        self.load_timer.end("config")

        self.load_timer.start("model")
        self.vl_gpt = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            language_config=language_config,
            trust_remote_code=True
        )  # nosec B615
        self.vl_gpt = self.vl_gpt.to(self.param_dtype)
        self.vl_gpt = self.vl_gpt.to(self.device)
        self.vl_gpt = self.vl_gpt.eval()
        self.load_timer.end("model")

        logging.info(f"Loaded Janus Pro: {self.MODEL_NAME} device:{self.device} dtype:{self.param_dtype}.")

    def init_model_parallelism(self) -> None:
        if self.world_size > 1:
            logging.warning("Janus Pro does not support model parallelism yet.")

    def model_compile(self) -> None:
        if not self.torch_compile:
            return

        self.load_timer.start("model_compile")
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        # Note: Janus has complex architecture, be careful with compilation
        # self.vl_gpt = torch.compile(self.vl_gpt, mode="max-autotune-no-cudagraphs")
        self.load_timer.end("model_compile")

    def _assert_model_init(self) -> None:
        super()._assert_model_init()
        assert self.vl_gpt is not None
        assert self.vl_chat_processor is not None
        assert self.tokenizer is not None

    def _assert_args(
        self,
        img_size: int,
        patch_size: int,
    ) -> None:
        if img_size % patch_size != 0:
            raise ValueError(f"Image size {img_size} must be divisible by patch size {patch_size}")
        if img_size < 384:
            raise ValueError(f"Image size {img_size} must be at least 384")

    def _prepare_prompt(self, prompt: str) -> str:
        assert self.vl_chat_processor is not None
        messages = [
            {'role': '<|User|>', 'content': prompt},
            {'role': '<|Assistant|>', 'content': ''}
        ]
        text = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=messages,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt=''
        )
        return text + self.vl_chat_processor.image_start_tag

    @inference_mode()
    async def warmup(self) -> None:
        logging.info(f"[{self.rank}] Warmup for Janus Pro generation.")
        await self.generate(
            prompt="A warmup image to initialize the model.",
            img_size=384,
            image_token_num_per_image=576
        )

    @override
    @inference_mode()
    async def generate(
        self,
        prompt: str,
        temperature: float = 1.0,
        cfg_weight: float = 5.0,
        image_token_num_per_image: int = 576,
        img_size: int = 384,
        patch_size: int = 16,
        job_id: Optional[str] = None,
    ) -> Image.Image:
        """
        Generate images from a prompt using the Janus Pro model.
        Args:
            prompt (str): Text prompt to guide the image generation.
            temperature (float): Sampling temperature for generation.
            parallel_size (int): Number of images to generate in parallel.
            cfg_weight (float): Classifier-free guidance weight.
            image_token_num_per_image (int): Number of tokens per image.
            img_size (int): Size of the generated images.
            patch_size (int): Patch size for the vision model.
        Returns:
            list[Image.Image]: List of generated PIL Images.
        """
        gen_timer = self._new_gen_timer(job_id)

        self._assert_model_init()
        assert self.vl_gpt is not None
        assert self.vl_chat_processor is not None
        assert self.tokenizer is not None
        self._assert_args(img_size, patch_size)

        width = img_size // patch_size * patch_size
        height = img_size // patch_size * patch_size

        # Single image generation for now
        parallel_size = 1

        self.running = True  # Mark running to avoid concurrent calls

        try:
            torch.cuda.empty_cache()

            gen_timer.start("prepare_prompt")
            formatted_prompt = self._prepare_prompt(prompt)
            gen_timer.end("prepare_prompt")

            gen_timer.start("tokenize")
            input_ids = torch.LongTensor(self.tokenizer.encode(formatted_prompt))
            tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).to(self.device)
            for i in range(parallel_size * 2):
                tokens[i, :] = input_ids
                if i % 2 != 0:
                    tokens[i, 1:-1] = self.vl_chat_processor.pad_id
            inputs_embeds = self.vl_gpt.language_model.get_input_embeddings()(tokens)
            gen_timer.end("tokenize")

            gen_timer.start("generate_tokens")
            generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(self.device)
            pkv = None
            for ix in range(image_token_num_per_image):
                gen_timer.start(f"generate_token_{ix:03d}")
                outputs = self.vl_gpt.language_model.model(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    past_key_values=pkv
                )
                pkv = outputs.past_key_values
                hidden_states = outputs.last_hidden_state
                logits = self.vl_gpt.gen_head(hidden_states[:, -1, :])
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]
                logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens[:, ix] = next_token.squeeze(dim=-1)
                next_token = torch.cat([
                    next_token.unsqueeze(dim=1),
                    next_token.unsqueeze(dim=1)
                ], dim=1).view(-1)

                img_embeds = self.vl_gpt.prepare_gen_img_embeds(next_token)
                inputs_embeds = img_embeds.unsqueeze(dim=1)
                gen_timer.end(f"generate_token_{ix:03d}")
            gen_timer.end("generate_tokens")

            gen_timer.start("decode_images")
            # TODO fix failure
            # shape '[2, 24, 24, 8]' is invalid for input of size 1600.
            patches = self.vl_gpt.gen_vision_model.decode_code(
                generated_tokens.to(dtype=torch.int),
                shape=[parallel_size, 8, width // patch_size, height // patch_size]
            )
            dec = patches.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
            dec = np.clip((dec + 1) / 2 * 255, 0, 255)
            visual_img = np.zeros((parallel_size, width, height, 3), dtype=np.uint8)
            visual_img[:, :, :] = dec
            gen_timer.end("decode_images")

            gen_timer.start("convert_pil")
            images = []
            for i in range(parallel_size):
                pil_image = Image.fromarray(visual_img[i]).resize((768, 768), Image.Resampling.LANCZOS)
                images.append(pil_image)
            gen_timer.end("convert_pil")

            logging.info(f"[{self.rank}] Generated {len(images)} images. Return just 1.")

            return images[0]
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

        temperature = float(data_json.get("temperature", 1.0))
        cfg_weight = float(data_json.get("cfg_weight", 5.0))
        image_token_num_per_image = int(data_json.get("image_token_num_per_image", 576))
        img_size = int(data_json.get("img_size", 384))
        patch_size = int(data_json.get("patch_size", 16))

        return {
            "task": self.model_name,
            "args": {
                "prompt": prompt,
                "temperature": temperature,
                "cfg_weight": cfg_weight,
                "image_token_num_per_image": image_token_num_per_image,
                "img_size": img_size,
                "patch_size": patch_size,
            }
        }
