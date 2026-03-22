"""
Handle video generation using the Wan 2.2 S2V (Sound-to-Video) model.

Reference: https://github.com/Wan-Video/Wan2.2
"""

import asyncio
import logging
import os
import tempfile

from typing import Dict
from typing import Union
from typing import List
from typing import Optional
from typing import Any

from PIL import Image

import torch
import torch.distributed as dist
from torch import inference_mode

from wrapper_wan import WanVideoGeneration
from media_utils import base64_to_audio_file as _base64_to_audio_file
from media_utils import empty_audio_file as _empty_audio_file

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video

from xfuser.config import EngineConfig


class Wan22VideoGeneration(WanVideoGeneration):
    """Handle video generation using the Wan 2.2 S2V (Sound-to-Video) model.

    Uses the official wan.WanS2V pipeline to generate video clips driven by
    speech/audio and conditioned on a reference image and text prompt.

    Supports:
    - Direct audio input (WAV/MP3 file path)
    - TTS audio synthesis via CosyVoice (enable_tts=True)
    - Automatic video length based on audio duration (num_clip=None)
    """

    def __init__(
        self,
        model_name: str = "wan22",
        ckpt_dir: str = "./Wan2.2-S2V-14B",
        engine_config: EngineConfig = None,
        param_dtype: torch.dtype = torch.bfloat16,
        offload_model: bool = True,
    ) -> None:
        super().__init__(
            model_name=model_name,
            ckpt_dir=ckpt_dir,
            engine_config=engine_config,
            param_dtype=param_dtype,
        )
        self.offload_model = offload_model

        # WanS2V pipeline instance (set in load_model)
        self.wan_s2v: Optional[wan.WanS2V] = None

        # S2V defaults matching wan_s2v_14B config
        self.shift = 3.0
        self.guide_scale = 4.5

    def __del__(self) -> None:
        if self.wan_s2v is not None:
            del self.wan_s2v
        super().__del__()

    def load_model(self) -> None:
        """Load the Wan 2.2 S2V model using the official WanS2V pipeline."""
        assert torch.cuda.is_available()

        cfg = WAN_CONFIGS['s2v-14B']

        use_fsdp = self.world_size > 1

        prev_memory = torch.cuda.memory_allocated()
        self.load_timer.start("wan_s2v")
        self.wan_s2v = wan.WanS2V(
            config=cfg,
            checkpoint_dir=self.ckpt_dir,
            device_id=self.device_id,
            rank=self.rank,
            t5_fsdp=use_fsdp,
            dit_fsdp=use_fsdp,
            use_sp=False,
            t5_cpu=True,
            convert_model_dtype=True,
        )
        self.load_timer.end("wan_s2v")

        diff_memory = torch.cuda.memory_allocated() - prev_memory
        logging.info(f"[{self.rank}] WanS2V memory allocated: {diff_memory / 1024 / 1024 ** 2:.2f} GB.")

        # Expose pipeline sub-components for _assert_model_init compatibility
        self.text_encoder = self.wan_s2v.text_encoder
        self.vae = self.wan_s2v.vae

    def init_model_parallelism(self) -> None:
        """Model parallelism is configured inside WanS2V.__init__; no additional setup needed."""

    def model_compile(self) -> None:
        """Compile the DiT model with torch.compile()."""
        if not self.torch_compile:
            return
        if self.wan_s2v is None:
            return
        logging.info(f"[{self.rank}] Compiling WanS2V DiT with torch.compile().")
        self.load_timer.start("dit_compile")
        self.wan_s2v.noise_model = torch.compile(
            self.wan_s2v.noise_model,
            mode="max-autotune-no-cudagraphs",
        )
        self.load_timer.end("dit_compile")

    def _assert_model_init(self) -> None:
        super()._assert_model_init()
        if self.wan_s2v is None:
            raise ValueError("WanS2V model not initialized")

    @inference_mode()
    async def warmup(self) -> None:
        logging.info(f"[{self.rank}] Warmup for Wan 2.2 S2V generation.")
        audio_path: Optional[str] = None
        try:
            audio_path = _empty_audio_file(duration_seconds=1.0)
            await self.generate(
                img=Image.new("RGB", (704, 480), (128, 128, 128)),
                prompt="Warmup prompt",
                neg_prompt="",
                max_area=480 * 704,
                sampling_steps=2,
                audio_path=audio_path,
                infer_frames=4,
                num_clip=1,
            )
        finally:
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)

    @inference_mode()
    async def generate(
        self,
        img: Image.Image,
        prompt: str,
        neg_prompt: str = "",
        max_area: int = 1024 * 704,
        sampling_steps: int = 40,
        audio_path: Optional[str] = None,
        enable_tts: bool = False,
        tts_prompt_audio: Optional[str] = None,
        tts_prompt_text: Optional[str] = None,
        tts_text: Optional[str] = None,
        num_clip: Optional[int] = None,
        infer_frames: int = 80,
        job_id: Optional[str] = None,
        output_type: str = "tensor",
    ) -> Union[List[Image.Image], str, bytes, torch.Tensor, None]:
        """Generate a video clip driven by audio and conditioned on an image and prompt.

        Args:
            img: Reference image used as the visual anchor for generation.
            prompt: Text description guiding video content.
            neg_prompt: Negative prompt to suppress unwanted content.
            max_area: Maximum pixel area for the output video (width * height).
                      The actual resolution is derived from the input image aspect ratio.
            sampling_steps: Number of diffusion denoising steps.
            audio_path: Path to the driving audio file (WAV/MP3).
                        Required when enable_tts is False.
            enable_tts: If True, synthesise audio from text via CosyVoice instead
                        of using a pre-recorded audio file.
            tts_prompt_audio: Path to reference speaker audio for zero-shot TTS.
                              Used only when enable_tts is True.
            tts_prompt_text: Transcript matching tts_prompt_audio.
                             Used only when enable_tts is True.
            tts_text: Text to synthesise into speech.
                      Used only when enable_tts is True.
            num_clip: Number of video clips to generate. When None the pipeline
                      infers the count automatically from the audio length.
            infer_frames: Frames generated per clip (must be a multiple of 4).
            job_id: Identifier for this generation job (used for temp file naming).
            output_type: One of "tensor", "pil", "video_binary", or "video_path".

        Returns:
            Generated video in the requested format, or None on non-primary ranks.
        """
        gen_timer = self._new_gen_timer(job_id)
        self._assert_model_init()
        assert self.wan_s2v is not None  # guaranteed by _assert_model_init
        self.running = True

        img_path: Optional[str] = None
        try:
            # Save PIL image to a temporary file; WanS2V expects a file path
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                img_path = tmp.name
            img.save(img_path)

            gen_timer.start("s2v_generate")
            video_tensor = await asyncio.to_thread(
                self.wan_s2v.generate,
                input_prompt=prompt,
                ref_image_path=img_path,
                audio_path=audio_path,
                enable_tts=enable_tts,
                tts_prompt_audio=tts_prompt_audio,
                tts_prompt_text=tts_prompt_text,
                tts_text=tts_text,
                num_repeat=num_clip,
                max_area=max_area,
                infer_frames=infer_frames,
                shift=self.shift,
                sampling_steps=sampling_steps,
                guide_scale=self.guide_scale,
                n_prompt=neg_prompt,
                seed=self.base_seed,
                offload_model=self.offload_model,
            )
            gen_timer.end("s2v_generate")

            if dist.is_initialized():
                dist.barrier()

            if self.rank != 0:
                return None

            if video_tensor is None:
                raise ValueError("No video generated")

            return await self._output_video(job_id, gen_timer, video_tensor, output_type)
        finally:
            self.running = False
            gen_timer.end("total")
            if img_path and os.path.exists(img_path):
                os.unlink(img_path)

    def _save_video(
        self,
        video_tensor: torch.Tensor,  # C, T, H, W
        video_path: str,
    ) -> str:
        assert video_tensor is not None
        assert isinstance(video_tensor, torch.Tensor)
        assert video_tensor.dim() == 4
        assert video_tensor.shape[0] == 3  # RGB channels
        return save_video(
            tensor=video_tensor[None],  # C, T, H, W -> B, C, T, H, W
            save_file=video_path,
            fps=self.FPS,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )

    async def get_rest_args(
        self,
        data_json: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Parse REST API request for Wan 2.2 S2V generation.

        Expected JSON fields:
            img (str): Base64-encoded reference image.
            prompt (str): Text prompt describing the video.
            neg_prompt (str, optional): Negative prompt.
            max_area (int, optional): Maximum output pixel area (default 1024*704).
            sampling_steps (int, optional): Diffusion steps (default 40).
            audio (str, optional): Base64-encoded audio file. Required when
                enable_tts is false/absent.
            enable_tts (bool, optional): Use CosyVoice TTS to generate audio.
            tts_prompt_audio (str, optional): Base64-encoded TTS reference audio.
            tts_prompt_text (str, optional): Transcript for TTS reference audio.
            tts_text (str, optional): Text to synthesise when enable_tts is True.
            num_clip (int, optional): Number of video clips (auto from audio if absent).
            infer_frames (int, optional): Frames per clip (default 80).
            output_type (str, optional): "tensor", "pil", "video_binary", or "video_path".
        """
        if data_json is None:
            raise ValueError("Missing JSON body")

        from image_utils import base64_to_img

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

        enable_tts = bool(data_json.get("enable_tts", False))

        audio_path: Optional[str] = None
        tts_prompt_audio_path: Optional[str] = None

        if enable_tts:
            tts_prompt_audio_b64 = data_json.get("tts_prompt_audio", None)
            if tts_prompt_audio_b64:
                if not isinstance(tts_prompt_audio_b64, str):
                    raise ValueError("'tts_prompt_audio' must be a base64-encoded string")
                tts_prompt_audio_path = await _base64_to_audio_file(tts_prompt_audio_b64)
        else:
            audio_b64 = data_json.get("audio", None)
            if not audio_b64:
                raise ValueError("Missing 'audio' parameter (or set enable_tts=true)")
            if not isinstance(audio_b64, str):
                raise ValueError("'audio' parameter must be a base64-encoded string")
            audio_path_dest: Optional[str] = None
            if job_id:
                audio_path_dest = f"/tmp/{job_id}.wav"
            audio_path = await _base64_to_audio_file(audio_b64, audio_path=audio_path_dest)

        return {
            "task": self.model_name,
            "args": {
                "job_id": job_id,
                "img": img,
                "prompt": prompt,
                "neg_prompt": neg_prompt,
                "max_area": int(data_json.get("max_area", 1024 * 704)),
                "sampling_steps": int(data_json.get("sampling_steps", 40)),
                "audio_path": audio_path,
                "enable_tts": enable_tts,
                "tts_prompt_audio": tts_prompt_audio_path,
                "tts_prompt_text": data_json.get("tts_prompt_text", None),
                "tts_text": data_json.get("tts_text", None),
                "num_clip": int(data_json["num_clip"]) if data_json.get("num_clip") is not None else None,
                "infer_frames": int(data_json.get("infer_frames", 80)),
                "output_type": data_json.get("output_type", "tensor"),
            }
        }
