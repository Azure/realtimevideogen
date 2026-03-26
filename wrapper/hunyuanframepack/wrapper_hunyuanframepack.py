"""
Wrapper class for HunyuanFramePack model generation using Hugging Face Diffusers and Xfuser.
"""
import sys
import logging
import time
import random
import math
import torch
import asyncio

from torch import inference_mode

from typing import Optional

from wrapper_hunyuanframepack_base import HunyuanFramePackBase

from PIL import Image

from diffusers_helper.hunyuan import vae_decode
from diffusers_helper.hunyuan import vae_encode
from diffusers_helper.utils import soft_append_bcthw

from xfuser.config import EngineConfig


class HunyuanFramepackGeneration(HunyuanFramePackBase):
    """Handle video generation using the HunyuanFramePack model."""

    def __init__(
        self,
        engine_config: EngineConfig = None,
        param_dtype: torch.dtype = torch.bfloat16,
        enable_tiling: bool = False,
        enable_slicing: bool = False,
    ) -> None:
        super().__init__(
            model_name="hunyuanframepack",
            framepack_model_name="lllyasviel/FramePackI2V_HY",
            engine_config=engine_config,
            param_dtype=param_dtype,
            enable_tiling=enable_tiling,
            enable_slicing=enable_slicing,
        )

    @inference_mode()
    async def generate(
        self,
        img: Image.Image,
        prompt: str,
        neg_prompt: str = "",
        height: int = 512,
        width: int = 768,
        num_frames: int = 1 + 80,
        sampling_steps: int = 25,  # 10
        # latent frames for every Hunyuan Video inference window
        # 9->36 pixel frames -> 1.2 seconds
        latent_window_size: int = 9,
        cfg: float = 1.0,
        distilled_guidance_scale: float = 10.0,
        guidance_rescale: int = 0,
        save_intermediate: Optional[str] = None,
        job_id: Optional[str] = None,
        output_type: str = "tensor",  # "tensor", "video_binary", "video_path"
    ) -> Optional[torch.Tensor]:
        """
        Generate a video from an input image and a prompt.
        Based on:
        https://raw.githubusercontent.com/lllyasviel/FramePack/refs/heads/main/demo_gradio.py
        TODO move more to the base class so we can unify regular and F1.
        """
        gen_timer = self._new_gen_timer(job_id)

        self._assert_model_init()
        self._assert_args(height, width)

        video_seconds = num_frames / self.FPS
        total_latent_sections = (video_seconds * self.FPS) / (latent_window_size * self.vae_stride[0])
        total_latent_sections = int(math.ceil(total_latent_sections))
        total_latent_sections = max(1, total_latent_sections)

        num_frames_it = latent_window_size * self.vae_stride[0] - 3

        if self.rank == 0:
            logging.info(
                f"[{self.rank}] Length:{video_seconds:.2f}secs Frames:{num_frames} "
                f"Frames/iteration:{num_frames_it} Iterations:{total_latent_sections} "
                f"FPS:{self.FPS} CFG:{cfg}.")

        self.running = True  # Mark running to avoid concurrent calls

        try:
            # Text encoder
            llama_vec, llama_attention_mask, clip_l_pooler, \
                llama_vec_n, llama_attention_mask_n, clip_l_pooler_n = self._encode_text(
                    prompt,
                    neg_prompt,
                    cfg)

            # Process input image
            input_image_np, input_image_pt = self._process_image(
                gen_timer,
                img,
                height,
                width)

            # CLIP Vision
            image_encoder_last_hidden_state = self._clip_vision(
                gen_timer,
                input_image_np)

            # VAE encoding
            # 1,RGB,1,h,w -> 1,lat_channels,1,lat_h,lat_w ([1,3,1,544,704] -> [1,16,1,68,88])
            gen_timer.start("vae_encoder")
            t0_vae = time.time()
            start_latent = vae_encode(input_image_pt, self.vae)
            gen_timer.end("vae_encoder")
            if self.rank == 0:
                logging.info(f"[{self.rank}] VAE encoding time: {time.time() - t0_vae:.3f} seconds.")

            # Prepare latent space
            seed = self.base_seed if self.base_seed >= 0 else random.randint(0, sys.maxsize)
            seed_g = torch.Generator(device=self.device)
            seed_g.manual_seed(seed)

            lat_h = height // self.vae_stride[1]
            lat_w = width // self.vae_stride[2]
            history_latents = torch.zeros(
                # B, C, T, H, W
                size=(1, self.LAT_CHANNELS, 1 + 2 + 16, lat_h, lat_w),
                dtype=torch.float32
            ).cpu()
            history_pixels = None
            total_generated_latent_frames = 0

            latent_paddings: list[int] = list(reversed(range(total_latent_sections)))

            if total_latent_sections > 4:
                # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
                # items looks better than expanding it when total_latent_sections > 4
                # One can try to remove below trick and just
                # use `latent_paddings = list(reversed(range(total_latent_sections)))` to compare
                latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

            # DiT sampling
            # We do 2 loops:
            # 1. Outer loop sticks together chunks into a final of num_frames
            # 2. Inner loop generate chunks of N frames (9 latent frames = 36 video frames = 1.2 seconds)
            for it, latent_padding in enumerate(latent_paddings):
                logging.debug(f"Running step {it + 1}.")

                self.check_interrupted()

                gen_timer.start(f"dit_{it:03d}")

                is_last_section = latent_padding == 0
                latent_padding_size = latent_padding * latent_window_size

                indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
                (clean_latent_indices_pre, _,  # blank_indices
                 latent_indices,
                 clean_latent_indices_post, clean_latent_2x_indices,
                 clean_latent_4x_indices) = indices.split(
                     [1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

                clean_latents_pre = start_latent.to(history_latents)
                clean_latents_post, clean_latents_2x, clean_latents_4x = \
                    history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
                clean_latents = torch.cat([
                    clean_latents_pre,
                    clean_latents_post
                ], dim=2)

                if self.engine_config is not None and self.engine_config.runtime_config.use_teacache:
                    self.transformer.initialize_teacache(enable_teacache=True, num_steps=sampling_steps)
                else:
                    self.transformer.initialize_teacache(enable_teacache=False)

                # [B, C, T, H, W]
                generated_latents = await asyncio.to_thread(
                    self._sample_hunyuan,
                    it0=it,
                    gen_timer=gen_timer,
                    width=width,
                    height=height,
                    frames=num_frames_it,
                    real_guidance_scale=cfg,
                    distilled_guidance_scale=distilled_guidance_scale,
                    guidance_rescale=guidance_rescale,
                    num_inference_steps=sampling_steps,
                    generator=seed_g,
                    prompt_embeds=llama_vec,
                    prompt_embeds_mask=llama_attention_mask,
                    prompt_poolers=clip_l_pooler,
                    negative_prompt_embeds=llama_vec_n,
                    negative_prompt_embeds_mask=llama_attention_mask_n,
                    negative_prompt_poolers=clip_l_pooler_n,
                    image_embeddings=image_encoder_last_hidden_state,
                    latent_indices=latent_indices,
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                )

                if is_last_section:
                    generated_latents = torch.cat([
                        start_latent.to(generated_latents),
                        generated_latents
                    ], dim=2)

                total_generated_latent_frames += int(generated_latents.shape[2])
                history_latents = torch.cat([
                    generated_latents.to(history_latents),
                    history_latents
                ], dim=2)

                # [B, C, T, H, W]
                real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

                gen_timer.end(f"dit_{it:03d}")

                if self.rank == 0:
                    lat_frames = real_history_latents.shape[2]
                    # Approximation, we should add and remove appropriately
                    cur_frames = lat_frames * self.vae_stride[0]
                    logging.info(
                        f"[{self.rank}] it:{it} lat_frames:{lat_frames} frames:{cur_frames} "
                        f"{cur_frames / self.FPS:.1f}/{video_seconds:.3f} seconds.")
                    if save_intermediate is not None:
                        torch.save(
                            real_history_latents,  # We are saving the whole history latents
                            f"/tmp/{save_intermediate}_latents_{it:03d}.pt")

            if self.rank != 0:
                return None  # other workers do not need to return anything or VAE decode

            if output_type == "latent":
                if history_pixels is None:
                    return real_history_latents
                section_latent_frames = latent_window_size * 2
                return real_history_latents[:, :, :section_latent_frames]

            # VAE decode
            # Can be done in the background yielding frames
            # [1, 16, lat_frames, lat_h, lat_w] -> [1, 3, 1+2+16, h, w]
            # ([1, 16, 37, 68, 88] -> [1, 3, 145, 544, 704])
            gen_timer.start("vae_decoder")
            if history_pixels is None:
                # This is the common path
                history_pixels = vae_decode(real_history_latents, self.vae).cpu()
            else:
                section_latent_frames = latent_window_size * 2
                if is_last_section:
                    section_latent_frames = latent_window_size * 2 + 1
                # vae.config.temporal_compression_ratio=4
                overlapped_frames = latent_window_size * self.vae_stride[0] - 3
                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], self.vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
            gen_timer.end("vae_decoder")

            if self.rank == 0:
                logging.info(
                    f"[{self.rank}] VAE decode. "
                    f"Latent:{real_history_latents.shape}->Pixel:{history_pixels.shape}.")

                out_num_frames = history_pixels.shape[2]
                if out_num_frames < num_frames:
                    logging.warning(f"[{self.rank}] Output frames {out_num_frames} < requested {num_frames}.")
                elif out_num_frames > num_frames:
                    logging.warning(f"[{self.rank}] Output frames {out_num_frames} > requested {num_frames}. Trimming.")
                    history_pixels = history_pixels[:, :, :num_frames, :, :]

            return await self._output_video(
                job_id,
                gen_timer,
                history_pixels,
                output_type)
        finally:
            self.running = False
            gen_timer.end("total")
