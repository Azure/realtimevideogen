"""
Handle video generation using the Wan 2.2 model.
"""

import sys
import logging
import math
import random
import types

from typing import Union
from typing import List
from typing import Optional
from typing import Any

from PIL import Image

from contextlib import contextmanager

import torch
import torch.amp as amp
import torch.distributed as dist
from torch import inference_mode

import torchvision.transforms.functional as TF

from functools import partial

from wrapper_model import GenerationInterruptedError
from wrapper_wan import WanVideoGeneration

from wan.modules.model import WanModel
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.distributed.fsdp import shard_model
from wan.distributed.sequence_parallel import sp_attn_forward
from wan.distributed.sequence_parallel import sp_dit_forward
from wan.utils.utils import save_video

from xfuser.config import EngineConfig
from xfuser.core.distributed import get_sequence_parallel_world_size


class Wan22VideoGeneration(WanVideoGeneration):
    """Handle video generation using the Wan 2.2 model."""

    interrupted: bool

    def __init__(
        self,
        model_name: str = "wan22",
        ckpt_dir: str = "./Wan2.2-I2V-14B-480P",
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
        self.model_high_noise: Optional[WanModel] = None
        self.model_low_noise: Optional[WanModel] = None

        # Model features
        self.boundary_high_low = 0.9

    def __del__(self) -> None:
        # Clean models
        if self.model_high_noise is not None:
            del self.model_high_noise
        if self.model_low_noise is not None:
            del self.model_low_noise
        super().__del__()

    def load_model(self) -> None:
        """Load the Wan 2.2 model."""
        super().load_model()

        prev_memory = torch.cuda.memory_allocated()

        # Wan 2.2 uses two models for low and high noise
        self.load_timer.start("dit_high")
        self.model_high_noise = WanModel.from_pretrained(
            self.ckpt_dir,
            subfolder="high_noise_model",
            torch_dtype=self.param_dtype,
        )
        if not self.model_high_noise:
            raise ValueError("Failed to load high noise model")
        self.model_high_noise.eval()
        self.model_high_noise.requires_grad_(False)
        self.load_timer.end("dit_high")

        self.load_timer.start("dit_low")
        self.model_low_noise = WanModel.from_pretrained(
            self.ckpt_dir,
            subfolder="low_noise_model",
            torch_dtype=self.param_dtype,
        )
        if not self.model_low_noise:
            raise ValueError("Failed to load low noise model")
        self.model_low_noise.eval()
        self.model_low_noise.requires_grad_(False)
        self.load_timer.end("dit_low")

        diff_memory = torch.cuda.memory_allocated() - prev_memory
        logging.info(f"[{self.rank}] DiT Memory allocated: {diff_memory / 1024 / 1024 ** 2:.2f} GB.")

    def init_model_parallelism(self) -> None:
        """Initialize model parallelism for Wan 2.2."""
        if self.model_high_noise is None:
            raise ValueError("Model not loaded")
        if self.model_low_noise is None:
            raise ValueError("Model not loaded")

        if not dist.is_initialized() or self.world_size <= 1:
            self.model_high_noise = self.model_high_noise.to(self.device)
            self.model_low_noise = self.model_low_noise.to(self.device)
            return

        self.sp_size = 1
        self.load_timer.start("dit_parallel")
        for model in [self.model_high_noise, self.model_low_noise]:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)
            self.sp_size = get_sequence_parallel_world_size()
        self.load_timer.end("dit_parallel")

        if dist.is_initialized():
            dist.barrier()

        # Load the DiT model across GPUs
        if self.world_size > 1:
            shard_fn = partial(shard_model, device_id=self.device_id)
            self.model_high_noise = shard_fn(self.model_high_noise)
            self.model_low_noise = shard_fn(self.model_low_noise)
        if not self.model_high_noise:
            raise ValueError("Failed to shard high noise model")
        if not self.model_low_noise:
            raise ValueError("Failed to shard low noise model")
        self.model_high_noise = self.model_high_noise.to(self.device)
        self.model_low_noise = self.model_low_noise.to(self.device)

    def model_compile(self) -> None:
        """Compile the Wan 2.2 model with torch.compile()."""
        if not self.torch_compile:
            return
        logging.info(f"[{self.rank}] Compiling DiT with torch.compile().")
        self.load_timer.start("dit_high_compile")
        self.model_high_noise = torch.compile(
            self.model_high_noise,
            mode="max-autotune-no-cudagraphs",
        )
        self.load_timer.end("dit_high_compile")

        self.load_timer.start("dit_low_compile")
        self.model_low_noise = torch.compile(
            self.model_low_noise,
            mode="max-autotune-no-cudagraphs",
        )
        self.load_timer.end("dit_low_compile")

    def _assert_model_init(self) -> None:
        super()._assert_model_init()
        if self.model_high_noise is None:
            raise ValueError("High noise model not initialized")
        if self.model_low_noise is None:
            raise ValueError("Low noise model not initialized")

    def _get_model(self, t: torch.Tensor) -> WanModel:
        if t.item() >= self.boundary_high_low:
            return self.model_high_noise
        return self.model_low_noise

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
    ) -> Union[List[Image.Image], str, bytes, torch.Tensor, None]:
        """
        Generate a video from an image and a prompt.
        """
        gen_timer = self._new_gen_timer(job_id)

        start_frames = 1

        self._assert_model_init()
        self._assert_args(height, width, num_frames, start_frames)

        self.running = True  # Mark running to avoid concurrent calls

        try:
            # Convert image to normalized tensor
            img_resized = img.resize((width, height), Image.Resampling.LANCZOS)
            img_tensor_norm = TF.to_tensor(img_resized).sub_(0.5).div_(0.5).to(self.device)
            # img is PIL.Image.Image image mode=RGB size=640x480 at 0x7FA09029B1D0
            # img_tensor_norm: [3, 480, 640] [RGB, height, width]

            h, w = img_tensor_norm.shape[1:]
            lat_h = h // self.vae_stride[1]
            lat_w = w // self.vae_stride[2]

            vae_t = self.vae_stride[0]
            max_seq_len = ((num_frames - start_frames)
                           // vae_t + start_frames) * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])
            max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
            if self.rank == 0:
                logging.info(f"[{self.rank}] size:{w}x{h}, lat_size:{lat_w}x{lat_h}, "
                             f"#frames:{num_frames}, #start_frames:{start_frames}, max_seq_len:{max_seq_len}, "
                             f"sp_size:{self.sp_size}, patch:{self.patch_size}, stride:{self.vae_stride}.")

            seed = self.base_seed if self.base_seed >= 0 else random.randint(0, sys.maxsize)
            seed_g = torch.Generator(device=self.device)
            seed_g.manual_seed(seed)

            # Adjusted from https://github.com/Wan-Video/Wan2.1/pull/100 -> 80+1 -> 20+1
            latent_num_frames = (num_frames - start_frames) // self.vae_stride[0] + start_frames
            noise = torch.randn(
                16,  # latent channel
                latent_num_frames,
                lat_h,
                lat_w,
                dtype=torch.float32,
                generator=seed_g,
                device=self.device,
            )  # [lat_channel, #lat_frames, lat_h, lat_w] [16, 21, 68, 90]

            # Preprocess
            # Text encoder
            gen_timer.start("text_encoder")
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([prompt], self.device)
            context_null = self.text_encoder([neg_prompt], self.device)
            gen_timer.end("text_encoder")

            # VAE encoder: Pixels -> Latent
            gen_timer.start("vae_encoder")

            # Mask+image (in latent space) with 1s for the first frame (input image) and empty (0s) for the rest
            y_1_frame = None
            if img is not None:
                msk_1_frame = self._get_mask(lat_h, lat_w, num_frames, 1, 0)
                img_frame = torch.nn.functional.interpolate(
                    img_tensor_norm[None].cpu(),
                    size=(h, w),
                    mode='bicubic'
                ).transpose(0, 1)  # Shape: [RGB, 1, h, w]

                num_empty_frames = num_frames - 1
                empty_frames = torch.zeros(3, num_empty_frames, h, w)  # Shape: [RGB, #EmptyFrames, h, w]
                vid_frames = [
                    # img_frame * prev_frame_weight, # TODO check how to use this
                    img_frame,
                    empty_frames
                ]

                # Create mask + y: pixels (img/video) to latent
                y_1_frame = self.vae.encode([
                    torch.concat(vid_frames, dim=1).to(self.device)
                ])[0]
                y_1_frame = torch.concat([
                    msk_1_frame,
                    y_1_frame
                ])  # [latent channels, #latent frames, lat_h, lat_w] [20, 21, 68, 90]

            gen_timer.end("vae_encoder")

            @contextmanager
            def noop_no_sync() -> Any:
                yield

            no_sync_high = getattr(self.model_high_noise, "no_sync", noop_no_sync)
            no_sync_low = getattr(self.model_low_noise, "no_sync", noop_no_sync)

            # DiT sampling
            x0 = []
            with amp.autocast('cuda', dtype=self.param_dtype), torch.no_grad(), no_sync_high(), no_sync_low():
                # Setup scheduler
                gen_timer.start("scheduler_setup")

                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=1000,
                    shift=1,
                    use_dynamic_shifting=False
                )

                sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=self.shift)
                timesteps = sample_scheduler.timesteps
                gen_timer.end("scheduler_setup")

                # Sample videos
                latent = noise

                for it, t in enumerate(timesteps):
                    logging.debug(f"[{self.rank}] Running step {it + 1}/{len(timesteps)}.")

                    if self.interrupted:
                        self.interrupted = False
                        raise GenerationInterruptedError(f"Generation interrupted at step {it + 1}.")

                    gen_timer.start(f"dit_{it:03d}")

                    # TODO test batching
                    latent_model_input = [latent.to(self.device)]
                    timestep_list = [t]

                    timestep = torch.stack(timestep_list).to(self.device)

                    # Choose the mask depending on the previous video and stage
                    y = y_1_frame  # We use the mask starting from the image

                    arg_c = {
                        "t": timestep,
                        "context": [context[0]],
                        "seq_len": max_seq_len,
                        "y": [y],
                    }
                    arg_null = {
                        "t": timestep,
                        "context": context_null,
                        "seq_len": max_seq_len,
                        "y": [y],
                    }

                    # Select the model
                    model = self._get_model(t)

                    # TODO we can make this parallel
                    # TODO we could batch the two of them
                    # This takes a lot of memory because of the KV cache
                    gen_timer.start(f"dit_cond_{it:03d}")
                    noise_pred_cond = model(
                        latent_model_input,
                        **arg_c
                    )[0].to(self.device)
                    gen_timer.end(f"dit_cond_{it:03d}")

                    gen_timer.start(f"dit_uncond_{it:03d}")
                    noise_pred_uncond = model(
                        latent_model_input,
                        **arg_null
                    )[0].to(self.device)
                    gen_timer.end(f"dit_uncond_{it:03d}")

                    noise_pred = noise_pred_uncond + self.guide_scale * (noise_pred_cond - noise_pred_uncond)
                    gen_timer.end(f"dit_{it:03d}")

                    gen_timer.start(f"scheduler_{it:03d}")
                    latent = latent.to(self.device)
                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latent.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g
                    )[0]
                    latent = temp_x0.squeeze(0)
                    gen_timer.end(f"scheduler_{it:03d}")

                    x0 = [latent.to(self.device)]
                    del latent_model_input, timestep

                videos = None
                if self.rank == 0 and output_type != "latent":
                    # Latent -> Pixels (video)
                    gen_timer.start("vae_decoder")
                    videos = self.vae.decode(x0, start_frames=1, end_frames=0)
                    gen_timer.end("vae_decoder")

            del noise
            del sample_scheduler
            if dist.is_initialized():
                dist.barrier()

            if self.rank != 0:
                return None

            if output_type == "latent":
                x0 = x0[0]
                return x0

            if not videos or len(videos) == 0:
                raise ValueError("No videos generated")
            video_tensor = videos[0]
            return await self._output_video(
                job_id,
                gen_timer,
                video_tensor,  # C, T, H, W
                output_type)
        finally:
            self.running = False
            gen_timer.end("total")

    def _save_video(
        self,
        video_tensor: torch.Tensor,  # C, T, H, W
        video_path: str,
    ) -> str:
        assert video_tensor is not None
        assert isinstance(video_tensor, torch.Tensor)
        assert video_tensor.dim() == 4
        assert video_tensor.shape[0] == 3  # Channels
        return save_video(
            tensor=video_tensor[None],  # C, T, H, W -> B, C, T, H, W
            save_file=video_path,
            fps=self.FPS,
            nrow=1)
