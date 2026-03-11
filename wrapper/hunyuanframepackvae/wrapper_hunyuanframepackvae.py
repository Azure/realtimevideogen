"""
Handle video VAE encoding and decoding using the Hunyuan Framepack VAE model.
"""
import logging
import os
import tempfile
import aiofiles
import asyncio

import torch

from typing import override
from typing import Union
from typing import Optional
from typing import Dict
from typing import Any

from torch import inference_mode

from wrapper_model import ModelGeneration

from model_timing import GenTimer

from diffusers import AutoencoderKLHunyuanVideo

from media_utils import base64_to_tensor
from media_utils import save_bcthw_as_mp4


class HunyuanFramepackVAEGeneration(ModelGeneration):
    """Handle video VAE encoding and decoding using the Hunyuan Framepack VAE model."""

    def __init__(
        self,
        param_dtype: torch.dtype = torch.float16,
        enable_tiling: bool = False,
        enable_slicing: bool = False,
    ) -> None:
        super().__init__("hunyuanframepackvae")

        self.param_dtype = param_dtype
        self.enable_tiling = enable_tiling
        self.enable_slicing = enable_slicing

        # Parallelism
        self.GPU = None
        if torch.cuda.is_available():
            self.GPU = torch.cuda.get_device_name(0)

        # Model features
        self.latent_channels = 16  # Latent channels for Hunyuan VAE
        self.vae_stride = (4, 8, 8)  # time, height, width
        self.FPS = 30  # This is technically a constant for the model

        # Model components
        self.vae = None

    def __del__(self) -> None:
        # Clean models
        if self.vae is not None:
            del self.vae
        super().__del__()

    def init_parallelism(self) -> None:
        self.load_timer.start("torch_dist")
        self.rank = int(os.getenv("RANK", 0))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        if self.world_size > 1:
            logging.warning(f"[{self.rank}] No distributed mode available.")

        if torch.cuda.is_available():
            self.device_id = self.local_rank
            self.device = torch.device(f"cuda:{self.device_id}")
            torch.cuda.set_device(self.local_rank)
        else:
            # Running on CPU is very slow, but it is supported
            self.device_id = 0
            self.device = torch.device("cpu")
            # MAX_NUM_CPUS = 16
            # num_threads = min(MAX_NUM_CPUS, os.cpu_count() or 1)
            # torch.set_num_threads(num_threads)
            num_threads = torch.get_num_threads()
            logging.warning(f"CUDA is not available. Running VAE with {num_threads} CPU threads.")

        self.load_timer.end("torch_dist")

    def load_model(self) -> None:
        prev_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        self.load_timer.start("vae")
        self.vae = AutoencoderKLHunyuanVideo.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            subfolder="vae",
            torch_dtype=self.param_dtype,
        ).to(self.device)
        self.vae.eval().requires_grad_(False)

        if not self.enable_tiling:
            logging.info(f"[{self.rank}] Disabling tiling for VAE.")
            self.vae.disable_tiling()
        else:
            logging.info(f"[{self.rank}] Enabling tiling for VAE.")
            self.vae.enable_tiling()

        if not self.enable_slicing:
            logging.info(f"[{self.rank}] Disabling slicing for VAE.")
            self.vae.disable_slicing()
        else:
            logging.info(f"[{self.rank}] Enabling slicing for VAE.")
            self.vae.enable_slicing()
        self.load_timer.end("vae")

        if torch.cuda.is_available():
            diff_memory = torch.cuda.memory_allocated() - prev_memory
            logging.info(f"[{self.rank}] VAE memory allocated: {diff_memory / 1024 / 1024 ** 2:.2f} GB.")

    def init_model_parallelism(self) -> None:
        if self.world_size > 1:
            logging.warning(f"[{self.rank}] No distributed mode available for Hunyuan VAE.")

    def model_compile(self) -> None:
        if not self.torch_compile:
            return

        logging.info(f"[{self.rank}] Compiling VAE with torch.compile().")
        self.load_timer.start("vae_compile")
        self.vae = torch.compile(
            self.vae,
            mode="max-autotune-no-cudagraphs",
        )
        self.load_timer.end("vae_compile")

    def _assert_model_init(self) -> None:
        super()._assert_model_init()
        assert self.vae is not None

    def _assert_args(
        self,
        latents: torch.Tensor,
    ) -> None:
        if latents is None:
            raise ValueError("Latents cannot be None.")
        if not isinstance(latents, torch.Tensor):
            raise TypeError(f"Expected latents to be a torch.Tensor, got {type(latents)}.")
        if latents.ndim != 5:
            raise ValueError(f"Expected latents with 5D [B, C, T, H, W], got {latents.ndim} dimensions.")
        if latents.shape[1] != 16:
            raise ValueError(f"Expected latents with 16 channels, got {latents.shape[1]} channels.")
        if latents.shape[2] <= 0:
            raise ValueError(f"Latents must have a positive number of frames, got {latents.shape[2]} frames.")

    @inference_mode()
    async def warmup(self) -> None:
        logging.info(f"[{self.rank}] Warmup for Hunyuan Framepack VAE generation.")
        latents = torch.randn(
            (1, 16, 4, 64, 64),  # B, C, T, H, W
            device=self.device,
            dtype=self.param_dtype
        )
        await self.generate(latents)

    @override
    @inference_mode()
    async def generate(  # type: ignore[override]
        self,
        latents: torch.Tensor,
        job_id: Optional[str] = None,
        output_type: str = "tensor",  # "tensor", "video_binary", "video_path"
    ) -> torch.Tensor:
        return await self.vae_decode(
            latents,
            job_id=job_id,
            output_type=output_type)

    @inference_mode()
    async def vae_decode(
        self,
        latents: torch.Tensor,
        job_id: Optional[str] = None,
        output_type: str = "tensor",  # "tensor", "video_binary", "video_path"
    ) -> torch.Tensor:
        """
        Latent -> Pixels.
        """
        gen_timer = self._new_gen_timer(job_id)

        self._assert_model_init()
        self._assert_args(latents)

        self.running = True

        try:
            gen_timer.start("vae_decoder")
            latents = latents / self.vae.config.scaling_factor
            latents = latents.to(self.vae.device, dtype=self.param_dtype)
            pixels = await asyncio.to_thread(
                self.vae.decode,
                latents)
            pixels = pixels.sample
            gen_timer.end("vae_decoder")

            return await self._output_video(
                job_id,
                gen_timer,
                pixels,
                output_type)
        finally:
            self.running = False
            gen_timer.end("total")

    async def _output_video(
        self,
        job_id: Optional[str],
        gen_timer: GenTimer,
        pixels: torch.Tensor,
        output_type: str = "tensor",  # "tensor", "video_binary", "video_path"
    ) -> Union[torch.Tensor, str, bytes]:
        # TODO use the one in HunyuanFramePackBase
        gen_timer.start("output")
        try:
            if output_type == "tensor":
                return pixels

            if output_type in ("video_binary", "video_path"):
                if not job_id:
                    video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                else:
                    video_path = f"/tmp/{job_id}.mp4"
                video_path = save_bcthw_as_mp4(
                    pixels,
                    video_path,
                    fps=self.FPS)
                if output_type == "video_path":
                    return video_path

                # video_binary
                async with aiofiles.open(video_path, "rb") as f:
                    video_binary = await f.read()
                return video_binary

            logging.error(f"Unknown output type: {output_type}")
            return None
        finally:
            gen_timer.end("output")

    @inference_mode()
    def vae_encode(
        self,
        pixels: torch.Tensor,
        job_id: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Pixels -> Latent.
        """
        gen_timer = self._new_gen_timer(job_id)

        self._assert_model_init()
        assert pixels is not None
        assert isinstance(pixels, torch.Tensor)
        assert pixels.ndim == 5  # B, C, T, H, W
        assert pixels.shape[1] == 3  # RGB channels

        self.running = True

        try:
            gen_timer.start("vae_encoder")
            pixels = pixels.to(self.device, dtype=self.param_dtype)
            latents = self.vae.encode(pixels).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            gen_timer.end("vae_encoder")
            return latents
        finally:
            self.running = False
            gen_timer.end("total")

    async def get_rest_args(self, data_json: Dict[str, str]) -> Dict[str, Any]:
        if data_json is None or not isinstance(data_json, dict):
            raise ValueError("Missing JSON body")

        latents_base64 = data_json.get("latents", None)
        if latents_base64 is None:
            raise ValueError("Missing 'latents' parameter")
        latents = base64_to_tensor(latents_base64)

        return {
            "task": self.model_name,
            "args": {
                "latents": latents
            }
        }

    def get_health(self) -> Dict[str, Any]:
        ret = super().get_health()
        ret.update({
            "gpu": self.GPU,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "torch_compile": self.torch_compile,
            "dtype": str(self.param_dtype),
            "vae_stride": self.vae_stride,
            "latent_channels": self.latent_channels,
            "fps": self.FPS,
        })
        return ret
