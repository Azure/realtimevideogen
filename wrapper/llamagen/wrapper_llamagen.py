"""
Based on:
https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/sample/sample_t2i.py
"""

import logging
import os
import sys
import random
import numpy as np

from typing import override
from typing import Optional
from typing import Dict
from typing import List
from typing import Tuple
from typing import Any
from typing import Union

from PIL import Image

import torch
import torch.distributed as dist
from torch import inference_mode

from wrapper_model import ModelGeneration

from tokenizer.tokenizer_image.vq_model import VQ_models
from language.t5 import T5Embedder
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate

from xfuser.config import EngineConfig


class LlamaGenGeneration(ModelGeneration):
    """LlamaGen model for text-to-image generation."""

    def __init__(
        self,
        model_name: str = "llamagen",
        engine_config: EngineConfig = None,
        param_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(model_name)

        self.engine_config = engine_config
        self.torch_compile = True
        if self.engine_config is not None:
            self.torch_compile = self.engine_config.runtime_config.use_torch_compile
        self.param_dtype = param_dtype

        # LlamaGen specific configuration
        self.REPO_ID = "peizesun/llamagen_t2i"
        self.GPT_TYPE = "t2i"
        self.VQ_MODEL_NAME = "VQ-16"
        # GPT-B  111M
        # GPT-L  343M
        # GPT-XL 775M
        # GPT-3B 3.1B
        self.GPT_MODEL_NAME = "GPT-XL"

        # LlamaGen specific parameters
        self.downsample_size = 16
        self.codebook_size = 16384
        self.codebook_embed_dim = 8
        self.cls_token_num = 120

        # T5 model configuration - match this with your trained GPT model
        # Common T5 models and their dimensions:
        # - flan-t5-xl: 2048 dimensions
        # - t5-v1_1-xl: 2048 dimensions
        # - t5-v1_1-xxl: 4096 dimensions
        # - flan-t5-xxl: 4096 dimensions
        self.T5_MODEL_TYPE = "flan-t5-xl"
        self.t5_feature_max_len = 120
        self.t5_feature_dim = 2048  # This should match the T5 model's hidden size
        # It supports 256 and 512
        self.image_size = 512  # pixels
        self.latent_size = self.image_size // self.downsample_size

        # Parallelism
        self.gpu: Optional[str] = None
        if torch.cuda.is_available():
            self.gpu = torch.cuda.get_device_name(0)

        # Model components
        self.vq_model: Optional[torch.nn.Module] = None
        self.gpt_model: Optional[torch.nn.Module] = None
        self.t5_model: Optional[T5Embedder] = None

    def __del__(self) -> None:
        if self.vq_model is not None:
            self.vq_model = None
        if self.gpt_model is not None:
            self.gpt_model = None
        if self.t5_model is not None:
            self.t5_model = None
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
            logging.warning("LlamaGen is not optimized for multi-GPU setups (yet).")
            self.world_size = 1

        self.load_timer.end("torch_dist")

    def load_model(self) -> None:
        assert torch.cuda.is_available()

        # Setup PyTorch optimizations like in the original code
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
        setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.load_timer.start("vq_model")
        # Load VQ model
        self.vq_model = VQ_models[self.VQ_MODEL_NAME](
            codebook_size=self.codebook_size,
            codebook_embed_dim=self.codebook_embed_dim
        )
        self.vq_model.to(self.device)
        self.vq_model.eval()
        vq_file = f"{self.REPO_ID}/vq_ds16_t2i.pt"
        checkpoint = torch.load(
            vq_file,
            map_location="cpu",
            weights_only=False)  # nosec B614 - trusted HuggingFace model checkpoint
        self.vq_model.load_state_dict(checkpoint["model"])
        self.load_timer.end("vq_model")

        self.load_timer.start("gpt_model")
        self.gpt_model = GPT_models[self.GPT_MODEL_NAME](
            block_size=self.latent_size ** 2,
            cls_token_num=self.cls_token_num,
            model_type=self.gpt_type,
        )
        self.gpt_model = self.gpt_model.to(dtype=self.param_dtype)
        self.gpt_model = self.gpt_model.to(device=self.device)
        if self.image_size not in [256, 512]:
            raise ValueError(f"Image size {self.image_size} not supported. Must be 256 or 512.")
        t2i_file = f"{self.REPO_ID}/t2i_XL_stage2_{self.image_size}.pt"
        checkpoint = torch.load(
            t2i_file,
            map_location="cpu",
            weights_only=False)  # nosec B614 - trusted HuggingFace model checkpoint
        model_weight = checkpoint.get("model", checkpoint.get("module", checkpoint.get("state_dict", checkpoint)))
        self.gpt_model.load_state_dict(model_weight, strict=False)
        self.gpt_model.eval()
        self.load_timer.end("gpt_model")

        self.load_timer.start("t5_model")
        # Load T5 model for text embeddings which should be already downloaded:
        # For flan-t5-xl: huggingface-cli download google/flan-t5-xl --local-dir google/flan-t5-xl
        # https://github.com/FoundationVision/LlamaGen/blob/main/language/t5.py
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # self.model = T5EncoderModel.from_pretrained(path, **t5_model_kwargs).eval()
        # t5_path = "DeepFloyd"  # "t5-v1_1-xxl"
        t5_path = "google"  # "flan-t5-xl"
        logging.info(f"Loading T5 model from '{t5_path}'.")
        if not os.path.exists(t5_path):
            raise FileNotFoundError(
                f"T5 model directory '{t5_path}' does not exist. Please download the model using huggingface-cli.")
        if self.T5_MODEL_TYPE != ["flan-t5-xl"]:
            raise ValueError("Only 'flan-t5-xl' is available.")
        self.t5_model = T5Embedder(
            device=self.device,
            local_cache=True,
            cache_dir=t5_path if os.path.exists(t5_path) else None,
            dir_or_name=self.T5_MODEL_TYPE,
            torch_dtype=self.param_dtype,
            model_max_length=self.t5_feature_max_len,
        )
        self.load_timer.end("t5_model")

        logging.info(f"Loaded LlamaGen. VQ:{self.vq_model_name} GPT:{self.gpt_model_name} T5:{self.t5_model_type}.")

    def init_model_parallelism(self) -> None:
        if self.world_size > 1:
            logging.warning("LlamaGen does not support model parallelism yet.")

    def model_compile(self) -> None:
        if not self.torch_compile:
            return

        self.load_timer.start("model_compile")
        self.gpt_model = torch.compile(  # type: ignore[assignment]
            self.gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        )
        self.load_timer.end("model_compile")

    def _assert_model_init(self) -> None:
        assert self.vq_model is not None
        assert self.gpt_model is not None
        assert self.t5_model is not None

    def _assert_args(self, image_size: int) -> None:
        if image_size not in [256, 384, 512]:
            raise ValueError(f"Image size {image_size} not supported. Must be 256, 384, or 512.")
        if image_size != self.image_size:
            raise ValueError(f"Image size {image_size} does not match model's size {self.image_size}.")

    def _prepare_text_embeddings(
            self,
            prompts: List[str],
            no_left_padding: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare text embeddings using T5 model.
        """
        caption_embs, emb_masks = self.t5_model.get_text_embeddings(prompts)  # type: ignore[union-attr]

        if not no_left_padding:
            # Implement left-padding as in the original code
            new_emb_masks = torch.flip(emb_masks, dims=[-1])
            new_caption_embs = []
            for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
                valid_num = int(emb_mask.sum().item())
                logging.debug(f'Prompt {idx} token len: {valid_num}')
                new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
                new_caption_embs.append(new_caption_emb)
            new_caption_embs = torch.stack(new_caption_embs)  # type: ignore[assignment]
        else:
            new_caption_embs, new_emb_masks = caption_embs, emb_masks

        c_indices = new_caption_embs * new_emb_masks[:, :, None]
        c_emb_masks = new_emb_masks

        return c_indices, c_emb_masks  # type: ignore[return-value]

    def _sample_to_image(self, sample: torch.Tensor) -> Image.Image:
        # "sample" is a tensor with shape [C, H, W] [3, H, W] and values in [-1, 1]
        sample = (sample + 1) / 2  # Convert from [-1, 1] to [0, 1]
        sample = torch.clamp(sample, 0, 1)
        # Convert to numpy and transpose to HWC format
        sample_np = sample.cpu().numpy().transpose(1, 2, 0)
        sample_np = (sample_np * 255).astype(np.uint8)  # Convert to [0, 255] uint8
        pil_image = Image.fromarray(sample_np)
        return pil_image

    @inference_mode()
    async def warmup(self) -> None:
        logging.info(f"[{self.rank}] Warmup for LlamaGen generation.")
        await self.generate(
            prompt="A warmup image to initialize the model.",
            image_size=512,
            cfg_scale=7.5,
            temperature=1.0,
            top_k=1000,
        )

    @override
    @inference_mode()
    async def generate(
        self,
        prompt: str,
        image_size: int = 512,  # pixels (512 x 512)
        cfg_scale: float = 7.5,
        temperature: float = 1.0,
        top_k: int = 1000,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        no_left_padding: bool = False,
        job_id: Optional[str] = None,
    ) -> Image.Image:
        """
        Generate an image from a prompt using the LlamaGen model.
        Args:
            prompt (str): Text prompt to guide the image generation.
            image_size (int): Size of the generated image (256, 384, or 512).
            cfg_scale (float): Classifier-free guidance scale.
            temperature (float): Sampling temperature.
            top_k (int): Top-k sampling parameter.
            top_p (float): Top-p (nucleus) sampling parameter.
            seed (int): Random seed for reproducibility.
            no_left_padding (bool): Whether to skip left padding for text embeddings.
        Returns:
            Image.Image: Generated PIL Image.
        """
        gen_timer = self._new_gen_timer(job_id)

        self._assert_model_init()
        self._assert_args(image_size)

        self.running = True  # Mark running to avoid concurrent calls

        try:
            # Set seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
            else:
                seed = random.randint(0, sys.maxsize)
                torch.manual_seed(seed)

            torch.set_grad_enabled(False)
            latent_size = image_size // self.downsample_size

            logging.info(f"Generating image with prompt: '{prompt}'.")
            logging.info(f"Image size: {image_size}, Latent size: {latent_size}.")

            # Prepare text embeddings
            gen_timer.start("text_embeddings")
            prompts = [prompt]  # LlamaGen expects a list
            c_indices, c_emb_masks = self._prepare_text_embeddings(prompts, no_left_padding)
            gen_timer.end("text_embeddings")

            # Generate token indices
            gen_timer.start("sampling")
            qzshape = [len(c_indices), self.codebook_embed_dim, latent_size, latent_size]

            index_sample = generate(
                self.gpt_model,
                c_indices,
                latent_size ** 2,
                c_emb_masks,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                sample_logits=True,
            )
            gen_timer.end("sampling")

            # Decode to image
            gen_timer.start("decoding")
            samples = self.vq_model.decode_code(  # type: ignore[union-attr, operator]
                index_sample, qzshape)  # output in [-1, 1]
            sample = samples[0]
            gen_timer.end("decoding")

            # Convert to PIL Image
            gen_timer.start("convert_pil")
            pil_image = self._sample_to_image(sample)
            gen_timer.end("convert_pil")

            return pil_image
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
            "vq_model": self.vq_model_name,
            "gpt_model": self.gpt_model_name,
            "gpt_type": self.gpt_type,
            "t5_model": self.t5_model_type,
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

        image_size = int(data_json.get("image_size", 512))
        cfg_scale = float(data_json.get("cfg_scale", 7.5))
        temperature = float(data_json.get("temperature", 1.0))
        top_k = int(data_json.get("top_k", 1000))
        top_p = float(data_json.get("top_p", 1.0))
        seed = data_json.get("seed", None)
        if seed is not None:
            seed = int(seed)
        no_left_padding = bool(data_json.get("no_left_padding", False))

        return {
            "task": self.model_name,
            "args": {
                "prompt": prompt,
                "image_size": image_size,
                "cfg_scale": cfg_scale,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "seed": seed,
                "no_left_padding": no_left_padding,
            }
        }
