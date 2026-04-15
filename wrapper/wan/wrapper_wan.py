"""
Wrapper for Wan video generation model.
"""
import os
import logging
import tempfile
import aiofiles

from typing import Optional
from typing import Dict
from typing import Any
from typing import Union
from typing import Tuple
from typing import List

from abc import abstractmethod

from PIL import Image

import torch

from torch import inference_mode

import torchvision

from functools import partial

from model_timing import GenTimer
from wrapper_usp import USPGeneration
from image_utils import base64_to_img

from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE
from wan.distributed.fsdp import shard_model

from xfuser.config import EngineConfig


class WanVideoGeneration(USPGeneration):
    """Generic class handle video generation using the Wan model."""

    FPS = 16
    NUM_HEADS = 40
    interrupted: bool

    def __init__(
        self,
        model_name: str = "wan",
        ckpt_dir: str = "./Wan2.1-I2V-14B-480P",
        engine_config: EngineConfig = None,
        param_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(
            model_name=model_name,
            engine_config=engine_config,
            param_dtype=param_dtype,
        )

        self.ckpt_dir = ckpt_dir

        # Model components
        self.text_encoder: Optional[T5EncoderModel] = None
        self.vae: Optional[WanVAE] = None

        # Model features
        self.sp_size = 1
        # https://replicate.com/blog/wan-21-parameter-sweep
        self.shift = 3.0  # Low values -> Less movement and smoother
        self.guide_scale = 5.0  # Higher values -> Follow the prompt closer (maybe also the image???)

        # https://github.com/Wan-Video/Wan2.1/blob/main/wan/configs/wan_i2v_14B.py
        self.patch_size: Tuple[int, int, int] = (1, 2, 2)
        self.vae_stride = (4, 8, 8)  # time, height, width

    def __del__(self) -> None:
        """Clean models."""
        if self.text_encoder is not None:
            del self.text_encoder
        if self.vae is not None:
            del self.vae
        super().__del__()

    def init_vae(self) -> None:
        """Initialize the VAE model."""
        assert torch.cuda.is_available()

        prev_memory = torch.cuda.memory_allocated()
        self.load_timer.start("vae")
        self.vae = WanVAE(
            vae_pth=os.path.join(self.ckpt_dir, "Wan2.1_VAE.pth"),
            # dtype=torch.float,
            device=self.device,  # device_img_encoder,
        )
        self.load_timer.end("vae")
        diff_memory = torch.cuda.memory_allocated() - prev_memory
        logging.info(f"[{self.rank}] VAE memory allocated: {diff_memory / 1024 / 1024 ** 2:.2f} GB.")

    def load_model(self) -> None:
        """Load the model into memory."""
        assert torch.cuda.is_available()

        # Load across GPUs
        shard_fn = None
        if self.world_size > 1:
            shard_fn = partial(shard_model, device_id=self.device_id)

        prev_memory = torch.cuda.memory_allocated()
        self.load_timer.start("text_encoder")
        self.text_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=torch.device('cpu'),  # device_txt_encoder,
            checkpoint_path=os.path.join(self.ckpt_dir, 'models_t5_umt5-xxl-enc-bf16.pth'),
            tokenizer_path=os.path.join(self.ckpt_dir, 'google/umt5-xxl'),
            shard_fn=shard_fn,  # Load across GPUs
        )
        self.load_timer.end("text_encoder")
        diff_memory = torch.cuda.memory_allocated() - prev_memory
        logging.info(f"[{self.rank}] Text encoder memory allocated: {diff_memory / 1024 / 1024 ** 2:.2f} GB.")

        self.init_vae()

    def _assert_model_init(self) -> None:
        super()._assert_model_init()
        if self.text_encoder is None:
            raise ValueError("Text encoder not initialized.")
        if self.vae is None:
            raise ValueError("VAE not initialized.")

    def _assert_args(
        self,
        height: int,
        width: int,
        num_frames: int,
        start_frames: int = 1,
    ) -> None:
        if not self.vae_stride:
            raise ValueError("VAE stride not set.")
        if height % self.vae_stride[1] != 0:
            raise ValueError(f"Height {height} should be divisible by VAE factor {self.vae_stride[1]}")
        if width % self.vae_stride[2] != 0:
            raise ValueError(f"Width {width} should be divisible by VAE factor {self.vae_stride[2]}")
        # TODO check world size divisible

        # Latent space is the first frame (input image) + empty frames//4
        if (num_frames - start_frames) % self.vae_stride[0] != 0:
            raise ValueError(f"num_frames {num_frames} should be {self.vae_stride[0]}*n")

        # The number of frames should be less than 1+80
        # Over 1+80 frames, it triggers weird video effects
        if num_frames < 1 or num_frames > start_frames + 80:
            raise ValueError(f"num_frames {num_frames} should be between 1 and 1+80")

    def _get_mask(
        self,
        lat_h: int,
        lat_w: int,
        total_frames: int,
        start_frames: int,
        end_frames: int,
    ) -> torch.Tensor:
        """
        Create a mask for the latent frames.
        Converts from raw frame mask [1, total_frames, H, W] to latent mask [4, latent_frames, H, W]

        latent_frames = start_frames + (middle_frames // 4) + end_frames
        where middle_frames = total_frames - start_frames - end_frames

        Start and end frames are repeated 4x; middle frames are grouped into latent frames (1 per 4 frames).
        """
        assert self.vae_stride is not None
        assert (total_frames - start_frames
                - end_frames) % self.vae_stride[0] == 0, "Middle frames must be divisible by {f}"

        # Step 1: Create base mask
        mask = torch.ones(1, total_frames, lat_h, lat_w, device=self.device)
        mask[:, start_frames:total_frames - end_frames] = 0  # zero out middle frames

        # Step 2: Expand start and end frames (repeated 4x each)
        start_frames_repeated = torch.repeat_interleave(
            mask[:, 0:start_frames],
            repeats=4,
            dim=1
        )

        end_frames_repeated = torch.repeat_interleave(
            mask[:, total_frames - end_frames:],
            repeats=4,
            dim=1
        ) if end_frames > 0 else torch.zeros(1, 0, lat_h, lat_w, device=self.device)

        # Step 3: Combine into final mask
        middle_mask = mask[:, start_frames:total_frames - end_frames]
        mask = torch.cat([
            start_frames_repeated,
            middle_mask,
            end_frames_repeated
        ], dim=1)

        # Step 4: Reshape to latent format: [1, latent_frames, 4, H, W] -> [4, latent_frames, H, W]
        mask = mask.view(1, mask.shape[1] // self.vae_stride[0], 4, lat_h, lat_w)
        mask = mask.transpose(1, 2)[0]  # remove batch dim

        # Step 5: Sanity check
        num_lat_frames = mask.shape[1]
        expected_lat_frames = start_frames + (total_frames - start_frames
                                              - end_frames) // self.vae_stride[0] + end_frames
        assert num_lat_frames == expected_lat_frames, f"Latent frames {num_lat_frames} != {expected_lat_frames}"

        return mask

    @inference_mode()
    async def warmup(self) -> None:
        logging.info(f"[{self.rank}] Warmup for Wan generation.")
        await self.generate(
            img=Image.new("RGB", (1280, 800), (255, 255, 255)),
            prompt="Warmup prompt",
            neg_prompt="",
            width=1280,
            height=720,
            num_frames=1 + 4,
            sampling_steps=2)

    @inference_mode()
    async def generate(
        self,
        img: Image.Image,
        prompt: str,
        neg_prompt: str = "",
        width: int = 640,
        height: int = 480,
        num_frames: int = 1 + 80,
        sampling_steps: int = 50,
        job_id: Optional[str] = None,
        output_type: str = "tensor"
    ) -> Union[List[Image.Image], str, bytes, torch.Tensor]:
        raise NotImplementedError("Method should be implemented in subclasses.")

    @inference_mode()
    def vae_decode(
        self,
        latents: torch.Tensor,
        job_id: Optional[str] = None,
    ) -> torch.Tensor:
        """Latent -> Pixels."""
        gen_timer = self._new_gen_timer(job_id)

        assert self.vae is not None
        # Assert arguments
        assert latents is not None
        assert isinstance(latents, torch.Tensor)
        assert latents.dim() == 4  # C, T, H, W
        assert latents.shape[0] == 20

        try:
            gen_timer.start("vae_decoder")
            latents = latents.to(self.device, dtype=self.param_dtype)
            pixels = self.vae.decode([latents])[0]
            gen_timer.end("vae_decoder")
            return pixels
        finally:
            gen_timer.end("total")

    @inference_mode()
    def vae_encode(
        self,
        pixels: torch.Tensor,
        job_id: Optional[str] = None,
    ) -> torch.Tensor:
        """Pixels -> Latent."""
        gen_timer = self._new_gen_timer(job_id)

        assert self.vae is not None
        # Assert arguments
        assert pixels is not None
        assert isinstance(pixels, torch.Tensor)
        assert pixels.dim() == 4  # C, T, H, W
        assert pixels.shape[1] == 3  # RGB

        try:
            gen_timer.start("vae_encoder")
            pixels = pixels.to(self.device, dtype=self.param_dtype)
            latents = self.vae.encode([pixels])[0]
            gen_timer.end("vae_encoder")
            return latents
        finally:
            gen_timer.end("total")

    async def _output_video(
        self,
        job_id: Optional[str],
        gen_timer: GenTimer,
        video_tensor: torch.Tensor,  # C, T, H, W
        output_type: str = "tensor",  # "tensor", "pil", "video_binary", "video_path"
    ) -> Union[List[Image.Image], str, bytes, torch.Tensor, None]:
        gen_timer.start("output")
        try:
            if output_type == "tensor":
                return video_tensor

            if output_type == "pil":
                return self._tensor_to_pil(video_tensor)

            if output_type in ("video_binary", "video_path"):
                if not job_id:
                    video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                else:
                    video_path = f"/tmp/{job_id}.mp4"
                self._save_video(
                    video_tensor=video_tensor,
                    video_path=video_path)
                if output_type == "video_path":
                    return video_path

                # video_binary
                async with aiofiles.open(video_path, "rb") as file:
                    video_binary = await file.read()
                return video_binary

            logging.error(f"Unknown output type: {output_type}")
            return None
        finally:
            gen_timer.end("output")

    @abstractmethod
    def _save_video(
        self,
        video_tensor: torch.Tensor,  # C, T, H, W (not B, C, T, H, W)
        video_path: str
    ) -> Optional[str]:
        raise NotImplementedError("Method should be implemented in subclasses.")

    def _tensor_to_pil(
        self,
        tensor: torch.Tensor,  # C, T, H, W
        nrow: int = 8,
        normalize: bool = True,
        value_range: Union[tuple, List] = (-1, 1),
    ) -> List[Image.Image]:
        assert tensor is not None
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dim() == 4  # C, T, H, W

        tensor = tensor.clamp(min(value_range), max(value_range))
        tensor = torch.stack([
            torchvision.utils.make_grid(u, nrow=nrow, normalize=normalize, value_range=value_range)
            for u in tensor.unbind(2)
        ], dim=1).permute(1, 2, 3, 0)
        tensor = (tensor * 255).type(torch.uint8).cpu()

        return [Image.fromarray(frame) for frame in tensor.numpy()]

    async def get_rest_args(
        self,
        data_json: Dict[str, Union[str, int, float]]
    ) -> Dict[str, Any]:
        if data_json is None:
            raise ValueError("Missing JSON body")

        job_id = data_json.get("job_id", None)

        img_base64 = data_json.get("img", None)
        if not img_base64:
            raise ValueError("Missing 'img' parameter")
        if not isinstance(img_base64, str):
            raise ValueError("'img' parameter must be a base64-encoded string")
        img = base64_to_img(img_base64)

        prompt = data_json.get("prompt", None)
        if prompt is None:
            raise ValueError("Missing 'prompt' parameter")
        neg_prompt = data_json.get("neg_prompt", "")

        width = int(data_json.get("width", 640))
        height = int(data_json.get("height", 480))
        sampling_steps = int(data_json.get("sampling_steps", 5)) or int(data_json.get("steps", 5))
        output_type = data_json.get("output_type", "tensor")
        num_frames = int(data_json.get("num_frames", 1 + 16))

        if height <= 0:
            raise ValueError(f"height {height} must be positive.")
        if width <= 0:
            raise ValueError(f"width {width} must be positive.")
        if sampling_steps <= 0:
            raise ValueError(f"sampling_steps {sampling_steps} must be positive.")

        video_seconds = data_json.get("video_seconds", 0.0)
        if video_seconds:
            if float(video_seconds) <= 0:
                raise ValueError(f"video_seconds {video_seconds} must be positive.")
            if not self.vae_stride:
                raise ValueError("VAE stride not set.")
            vae_frames = self.vae_stride[0]
            num_frames = int(video_seconds * self.FPS)
            num_frames = 1 + ((num_frames - 1) // vae_frames) * vae_frames  # 4n + 1

        if num_frames <= 0:
            raise ValueError(f"num_frames {num_frames} must be positive.")

        return {
            "task": self.model_name,
            "args": {
                "job_id": job_id,
                "img": img,
                "prompt": prompt,
                "neg_prompt": neg_prompt,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "sampling_steps": sampling_steps,
                "output_type": output_type
            }
        }
