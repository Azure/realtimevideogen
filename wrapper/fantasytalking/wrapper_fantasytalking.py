"""
Wrapper for Fantasy Talking video generation.
"""
import os
import sys
import types
import logging
import tempfile
import math
import asyncio
import aiofiles
import aiofiles.os

from typing import override
from typing import List
from typing import Union
from typing import Optional
from typing import Dict
from typing import Any
from typing import Tuple

from PIL import Image

import torchvision.transforms.functional as TF

import torch
import torch.distributed as dist
from torch import inference_mode

import numpy as np

from functools import partial

from model_timing import GenTimer
from wrapper_model import ModelGeneration
from wrapper_usp import USPGeneration

from image_utils import base64_to_img
from media_utils import base64_to_video_frames
from media_utils import base64_to_audio_file
from media_utils import empty_audio_file
from media_utils import save_video_audio

import librosa

from transformers import Wav2Vec2Model
from transformers import Wav2Vec2Processor

from diffsynth import ModelManager
from diffsynth import WanVideoPipeline
from model import FantasyTalkingAudioConditionModel

from utils import get_audio_features

from xfuser.config import EngineConfig

sys.path.append("/wan")  # noqa: E402
from fantasytalking_xfuser import usp_fantasytalking_forward
from wan.distributed.xdit_context_parallel import usp_attn_forward
from wan.distributed.fsdp import shard_model


def resample_frames(
    video_frames: List[Image.Image],
    src_fps: float,
    dst_fps: float,
    audio_duration: Optional[float] = None
) -> List[Image.Image]:
    """
    Instead of simply trimming or padding, we resample the frames to match the desired number of frames.
    # if video_num_frames < num_frames:
    #     logging.warning(
    #         f"[{self.rank}] Video {video_num_frames} < Output {num_frames}, "
    #         "padding video with last frame.")
    #     video += [video[-1]] * (num_frames - video_num_frames)
    """

    if src_fps > dst_fps and audio_duration:
        # Truncate the video to the first audio_duration seconds
        num_frames = int(audio_duration * src_fps)
        if num_frames < len(video_frames):
            logging.info(f"Truncating video from {len(video_frames)} to {num_frames} frames to match audio duration.")
            video_frames = video_frames[:num_frames]

    duration = len(video_frames) / src_fps
    num_dst_frames = int(round(duration * dst_fps))
    idxs = np.linspace(0, len(video_frames) - 1, num_dst_frames)
    idxs = idxs.astype(int)
    return [
        video_frames[i]
        for i in idxs
    ]


class FantasyTalking(USPGeneration):
    """
    Fantasy Talking video generation wrapper.
    """

    # Weird FPS, but it is what Fantasy Talking uses
    FPS = 23.0
    SRC_FPS = 30.0  # from HunyuanVideo
    # Original limit for Wan at 16 FPS -> 5.1 seconds
    # self.MAX_FRAMES = 1 + 80
    # Increased the limit from 1+80 to 1+116 which is ~5.1 seconds
    MAX_FRAMES = 1 + 116
    # Wan values for number of attention heads
    NUM_HEADS = 40

    def __init__(
        self,
        model_name: str = "fantasytalking",
        engine_config: EngineConfig = None,
        param_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(
            model_name,
            engine_config,
            param_dtype)

        # Model components
        self.pipeline_wan: Optional[WanVideoPipeline] = None
        self.fantasytalking: Optional[FantasyTalkingAudioConditionModel] = None
        self.wav2vec_processor: Optional[Wav2Vec2Processor] = None
        self.wav2vec: Optional[Wav2Vec2Model] = None

        # Model features
        self.vae_stride = (4, 8, 8)  # time, height, width

    def __del__(self) -> None:
        # Clean models
        if self.pipeline_wan is not None:
            self.pipeline_wan = None
        if self.fantasytalking is not None:
            self.fantasytalking = None
        if self.wav2vec_processor is not None:
            self.wav2vec_processor = None
        if self.wav2vec is not None:
            self.wav2vec = None
        super().__del__()

    def load_model(self) -> None:
        assert torch.cuda.is_available()

        self.load_timer.start("wan")
        model_manager = ModelManager(device="cpu")
        BASE_MODELS_FOLDER = "/fantasytalking"
        NUM_MODEL_CHUNKS = 7
        wan_model_path = f"{BASE_MODELS_FOLDER}/Wan2.1-I2V-14B-720P"
        model_manager.load_models(
            [
                [
                    f"{wan_model_path}/diffusion_pytorch_model-{idx:05d}-of-00007.safetensors"
                    for idx in range(1, NUM_MODEL_CHUNKS + 1)
                ],
                f"{wan_model_path}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                f"{wan_model_path}/models_t5_umt5-xxl-enc-bf16.pth",
                f"{wan_model_path}/Wan2.1_VAE.pth",
            ],
            torch_dtype=self.param_dtype
        )
        # self.pipeline_wan = WanVideoPipeline.from_model_manager(
        self.pipeline_wan = CustomWanVideoPipeline(
            device=self.device,
            torch_dtype=self.param_dtype)
        if not self.pipeline_wan:
            raise ValueError("Wan model not initialized")
        self.pipeline_wan.fetch_models(model_manager)
        # self.pipeline_wan.enable_vram_management(num_persistent_param_in_dit=None)
        self.load_timer.end("wan")

        # Load FantasyTalking weights
        self.load_timer.start("fantasytalking")
        self.fantasytalking = FantasyTalkingAudioConditionModel(
            self.pipeline_wan.dit,
            audio_in_dim=768,
            audio_proj_dim=2048
        ).to(self.device)
        if self.fantasytalking is None:
            raise ValueError("Fantasy Talking model not initialized")
        self.fantasytalking.load_audio_processor(
            f"{BASE_MODELS_FOLDER}/fantasytalking_model.ckpt",
            self.pipeline_wan.dit
        )
        self.load_timer.end("fantasytalking")

        # Load wav2vec models
        self.load_timer.start("wav2vec")
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(
            f"{BASE_MODELS_FOLDER}/wav2vec2-base-960h"  # nosec B615 - local path
        )
        self.wav2vec = Wav2Vec2Model.from_pretrained(
            f"{BASE_MODELS_FOLDER}/wav2vec2-base-960h"  # nosec B615 - local path
        ).to(self.device)
        self.load_timer.end("wav2vec")

    def init_model_parallelism(self) -> None:
        if self.pipeline_wan is None:
            raise RuntimeError("Pipeline WAN not initialized")

        if not dist.is_initialized() or self.world_size <= 1:
            self.pipeline_wan.to(self.device)
            return

        self.load_timer.start("dit_parallel")
        for block in self.pipeline_wan.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.pipeline_wan.dit.forward = types.MethodType(usp_fantasytalking_forward, self.pipeline_wan.dit)

        # Load across GPUs
        shard_fn = None
        if self.world_size > 1:
            shard_fn = partial(shard_model, device_id=self.device_id)
            self.pipeline_wan = self.pipeline_wan.to(self.param_dtype)
            if not self.pipeline_wan:
                raise ValueError("Wan not initialized for sharding")
            self.pipeline_wan.dit = shard_fn(self.pipeline_wan.dit)
        self.load_timer.end("dit_parallel")

        self.pipeline_wan.to(self.device)

    def model_compile(self) -> None:
        if not self.torch_compile:
            return
        if not self.pipeline_wan:
            raise RuntimeError("Pipeline Wan not initialized")
        logging.info(f"[{self.rank}] Compiling transformer with torch.compile().")
        self.load_timer.start("compile")
        self.pipeline_wan.dit = torch.compile(
            self.pipeline_wan.dit,
            mode="max-autotune-no-cudagraphs",
        )
        # This may cause issues in the video processing, and doesn't really give much speedup
        logging.info(f"[{self.rank}] Compiling VAE with torch.compile().")
        self.pipeline_wan.vae = torch.compile(
            self.pipeline_wan.vae,
            mode="max-autotune-no-cudagraphs",
        )
        logging.info(f"[{self.rank}] Compiling Fantasy Talking with torch.compile().")
        self.fantasytalking = torch.compile(
            self.fantasytalking,
            mode="max-autotune-no-cudagraphs",
        )
        self.load_timer.end("compile")

    def _assert_model_init(self) -> None:
        super()._assert_model_init()
        assert self.pipeline_wan is not None, "Pipeline WAN not initialized"
        assert self.fantasytalking is not None, "Fantasy Talking model not initialized"
        assert self.wav2vec_processor is not None, "Wav2Vec processor not initialized"
        assert self.wav2vec is not None, "Wav2Vec model not initialized"

    def _assert_args(self, height: int, width: int) -> None:
        if not self.vae_stride:
            raise ValueError("VAE stride not set.")
        if height % self.vae_stride[1] != 0:
            raise ValueError(f"Height {height} is not divisible by VAE scale factor {self.vae_stride}")
        if width % self.vae_stride[2] != 0:
            raise ValueError(f"Width {width} is not divisible by VAE scale factor {self.vae_stride}")
        # TODO add check for sizes based on world_size similar to what we do in Flux
        """
        # Check if the image size is supported for the current parallelism setting
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py
        height_latent = height // self.vae_stride[1]
        width_latent = width // self.vae_stride[2]
        img_latent_shape = (height_latent // 2) * (width_latent // 2)
        if img_latent_shape % self.world_size != 0:
            raise ValueError(f"{height}x{width} not supported for {self.world_size} GPUs.")
        """

    @inference_mode()
    async def warmup(self) -> None:
        logging.info(f"[{self.rank}] Warmup for Fantasy Talking generation.")
        empty_img = Image.new("RGB", (640, 480), color=(255, 255, 255))
        empty_audio_path = empty_audio_file(0.5)  # 0.5 seconds of silence
        await self.generate(
            job_id="warmup",
            img=empty_img,
            video=None,
            audio_path=empty_audio_path,
            prompt="A warmup generation",
            neg_prompt="",
            width=640,
            height=480,
            sampling_steps=2)
        if os.path.exists(empty_audio_path):
            os.remove(empty_audio_path)

    def _to_num_latent_frames(self, num_frames: int) -> int:
        if not self.vae_stride:
            raise ValueError("VAE stride not set.")
        return (num_frames - 1) // self.vae_stride[0] + 1

    def _to_num_frames(self, num_latent_frames: int) -> int:
        if not self.vae_stride:
            raise ValueError("VAE stride not set.")
        return num_latent_frames * self.vae_stride[0] + 1

    def _get_audio_num_frames(
        self,
        audio_path: str
    ) -> Tuple[float, int, int]:
        # We need to align the audio frame number with (1 + 4n) frames
        if not self.vae_stride:
            raise ValueError("VAE stride not set.")
        audio_duration = librosa.get_duration(path=audio_path)
        audio_num_frames = int(math.ceil(self.FPS * audio_duration))
        video_num_frames = int(1 + math.ceil((audio_num_frames - 1) / self.vae_stride[0]) * self.vae_stride[0])
        return audio_duration, audio_num_frames, video_num_frames

    @override
    @inference_mode()
    async def generate(
        self,
        img: Image.Image,
        video: Optional[List[Image.Image]],
        audio_path: str,
        prompt: str,
        neg_prompt: str = "",
        width: int = 1280,
        height: int = 720,
        sampling_steps: int = 30,  # 10
        audio_scale: float = 1.0,  # 1.0 for audio, 0.0 for no audio influence
        cfg_scale: float = 5.0,  # how much does video follow prompt
        audio_cfg_scale: float = 5.0,  # how much does video follow audio
        # We use this to avoid first frame: https://github.com/Fantasy-AMAP/fantasy-talking/issues/52
        end_percent: float = 0.9,
        adjust_durations: bool = True,
        job_id: Optional[str] = None,
        output_type: str = "pil",  # "pil", "video_binary", "video_path"
    ) -> Union[List[Image.Image], str, bytes, None]:
        """
        Generate a video from an image, and audio, and a prompt.
        Args:
            img (Image.Image): Input image.
            audio_path (str): Path to the audio file.
            prompt (str): Text prompt for video generation.
            neg_prompt (str, optional): Negative text prompt. Defaults to "".
            width (int, optional): Width of the output video. Defaults to 1280.
            height (int, optional): Height of the output video. Defaults to 720.
            sampling_steps (int, optional): Number of sampling steps. Defaults to 30.
            audio_scale (float, optional): Scale for audio influence. Defaults to 1.0.
            cfg_scale (float, optional): Scale for prompt influence. Defaults to 5.0.
            audio_cfg_scale (float, optional): Scale for audio prompt influence. Defaults to 5.0.
            end_percent (float, optional): Percentage of de-noising steps considering audio inputs. Defaults to 0.9.
            adjust_durations (bool, optional): Whether to adjust video durations to match audio. Defaults to True.
            job_id (str, optional): Job ID for tracking. Defaults to None.
            output_type (str, optional): Output type. Can be "pil", "video_binary", or "video_path". Defaults to "pil".
        """
        gen_timer = self._new_gen_timer(job_id)

        logging.info(f"[{self.rank}] Generating video for '{job_id if job_id else prompt[0:80]}'.")

        self._assert_model_init()
        self._assert_args(height, width)

        if not await aiofiles.os.path.exists(audio_path):
            raise ValueError(f"Audio file '{audio_path}' does not exist")

        self.running = True  # Mark running to avoid concurrent calls

        try:
            audio_duration, audio_num_frames, num_frames = self._get_audio_num_frames(audio_path)

            if num_frames > self.MAX_FRAMES:
                raise ValueError(
                    f"Audio {audio_duration:.3f}s exceeds maximum frames {self.MAX_FRAMES} at {self.FPS} FPS "
                    f"-> {num_frames} frames")

            lat_num_frames = self._to_num_latent_frames(num_frames)
            if self.rank == 0:
                logging.info(f"[{self.rank}] Audio:{audio_duration:.3f}s #audio_frames:{audio_num_frames} "
                             f"#video_frames:{num_frames} #lat_frames:{lat_num_frames} FPS:{self.FPS}.")

            if not img and not video:
                raise ValueError("No image or video provided")

            if video is not None and len(video) > 0:
                gen_timer.start("video_preprocess")
                video = [frame.resize((width, height), Image.Resampling.LANCZOS) for frame in video]

                # We assume the input video has the same FPS
                video_num_frames = len(video)
                video_duration = num_frames / self.FPS
                if video_num_frames != num_frames:
                    if not adjust_durations:
                        raise ValueError(f"Video frames {video_num_frames} != Output frames {num_frames}")
                    logging.warning(
                        f"[{self.rank}] Resampling video from {video_num_frames} (source) to {num_frames} "
                        "(destination) frames to match output.")
                    video_resampled = resample_frames(
                        video,
                        src_fps=self.SRC_FPS, dst_fps=self.FPS,
                        audio_duration=audio_duration)
                    # Normalize to exactly num_frames to prevent latent/noise tensor shape mismatch.
                    # resample_frames may return a different count than num_frames due to
                    # floating-point rounding in the FPS conversion (e.g. 30→23 FPS).
                    if len(video_resampled) < num_frames and len(video_resampled) > 0:
                        logging.warning(
                            f"[{self.rank}] Resampled video has {len(video_resampled)} frames, "
                            f"padding to {num_frames} with last frame.")
                        video_resampled = video_resampled + [video_resampled[-1]] * (num_frames - len(video_resampled))
                    elif len(video_resampled) > num_frames:
                        logging.warning(
                            f"[{self.rank}] Resampled video has {len(video_resampled)} frames, "
                            f"trimming to {num_frames}.")
                        video_resampled = video_resampled[:num_frames]
                    video = video_resampled

                if img is None:
                    if self.rank == 0:
                        logging.info(f"[{self.rank}] Using first video frame as starting image.")
                    img = video[0]
                if self.rank == 0:
                    video_num_frames = len(video)
                    logging.info(
                        f"[{self.rank}] Input video:{video_duration:.3f}s #frames:{video_num_frames} FPS:{self.FPS}.")
                gen_timer.end("video_preprocess")

            if img.size != (width, height):
                gen_timer.start("img_preprocess")
                if self.rank == 0:
                    logging.info(f"[{self.rank}] Image:{img.size}->{(width, height)}.")
                img = img.resize((width, height), Image.Resampling.LANCZOS)
                gen_timer.end("img_preprocess")

            gen_timer.start("audio_encoder")
            audio_wav2vec_fea = get_audio_features(
                self.wav2vec,
                self.wav2vec_processor,
                audio_path,
                self.FPS,
                num_frames
            )
            if self.fantasytalking is None:
                raise ValueError("Fantasy Talking model not initialized")
            audio_proj_fea = self.fantasytalking.get_proj_fea(audio_wav2vec_fea)
            pos_idx_ranges = self.fantasytalking.split_audio_sequence(
                audio_proj_fea.size(1),
                num_frames=num_frames
            )
            audio_proj_split, audio_context_lens = self.fantasytalking.split_tensor_with_padding(
                audio_proj_fea,
                pos_idx_ranges,
                expand_length=4,
            )
            gen_timer.end("audio_encoder")

            gen_timer.start("wan")
            if not self.pipeline_wan:
                raise ValueError("Pipeline Wan not initialized")
            self.pipeline_wan.to(self.device)
            video_frames = await asyncio.to_thread(
                self.pipeline_wan,
                parent=self,
                prompt=prompt,
                negative_prompt=neg_prompt,
                input_image=img,
                input_video=video,
                width=width,
                height=height,
                num_frames=num_frames,
                latents_num_frames=lat_num_frames,
                num_inference_steps=sampling_steps,
                seed=self.base_seed,
                tiled=True,
                audio_scale=audio_scale,
                cfg_scale=cfg_scale,
                audio_cfg_scale=audio_cfg_scale,
                audio_proj=audio_proj_split,
                audio_context_lens=audio_context_lens,
                gen_timer=gen_timer,
                end_percent=end_percent,  # Use 0.9 to control how much de-noising steps consider audio inputs
            )
            gen_timer.end("wan")

            if self.rank != 0:
                return None  # Only rank 0 returns something

            # Because of the 1+4n alignment, we might have generated more frames than audio, trim the video to match
            if audio_num_frames < len(video_frames):
                logging.info(
                    f"[{self.rank}] Trimming number of video frames from {len(video_frames)} to "
                    f"{audio_num_frames} to match audio frames.")
                video_frames = video_frames[0:audio_num_frames]

            return await self._output_video(
                job_id,
                gen_timer,
                audio_path,
                video_frames,
                output_type)
        finally:
            self.running = False
            torch.cuda.empty_cache()
            gen_timer.end("total")

    async def _output_video(
        self,
        job_id: Optional[str],
        gen_timer: GenTimer,
        audio_path: str,
        video_frames: List[Image.Image],
        output_type: str = "pil",  # "pil", "video_binary", "video_path"
    ) -> Union[List[Image.Image], str, bytes, None]:
        gen_timer.start("output_video")
        try:
            if output_type == "pil":
                return video_frames

            if output_type in ("video_binary", "video_path"):
                if not job_id:
                    video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                else:
                    video_path = f"/tmp/{job_id}.mp4"
                video_path = await save_video_audio(
                    video_content=video_frames,
                    audio_path=audio_path,
                    out_video_path=video_path,
                    fps=self.FPS)
                if output_type == "video_path":
                    return video_path

                # video_binary
                async with aiofiles.open(video_path, "rb") as file:
                    video_binary = await file.read()
                return video_binary

            logging.error(f"Unknown output type: {output_type}")
            return None
        finally:
            gen_timer.end("output_video")

    async def get_rest_args(
        self,
        data_json: Dict[str, Union[str, int, float]]
    ) -> Dict[str, Any]:
        if data_json is None:
            raise ValueError("Missing JSON body")

        job_id = data_json.get("job_id", None)

        img_base64 = data_json.get("img", None)
        img = None
        if img_base64:
            if not isinstance(img_base64, str):
                raise ValueError("Invalid 'img' parameter")
            img = base64_to_img(img_base64)

        video_base64 = data_json.get("video", None)
        video = None
        if video_base64:
            if not isinstance(video_base64, str):
                raise ValueError("Invalid 'video' parameter")
            video = base64_to_video_frames(video_base64)

        audio_base64 = data_json.get("audio", None)
        if not audio_base64:
            raise ValueError("Missing 'audio' parameter")
        if not isinstance(audio_base64, str):
            raise ValueError("Invalid 'audio' parameter")
        audio_path = None
        if not job_id:
            audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        else:
            audio_path = f"/tmp/{job_id}.wav"
        audio_path = await base64_to_audio_file(
            audio_base64,
            audio_path=audio_path)

        prompt = data_json.get("prompt", None)
        if prompt is None:
            raise ValueError("Missing 'prompt' parameter")
        neg_prompt = data_json.get("neg_prompt", "")

        gen_args = {
            "task": self.model_name,
            "args": {
                "job_id": job_id,
                "img": img,
                "video": video,
                "prompt": prompt,
                "neg_prompt": neg_prompt,
                "audio_path": audio_path,
                "width": int(data_json.get("width", 640)),
                "height": int(data_json.get("height", 480)),
                "sampling_steps": int(data_json.get("sampling_steps", 10)),
                "audio_scale": float(data_json.get("audio_scale", 1.0)),
                "cfg_scale": float(data_json.get("cfg_scale", 5.0)),
                "audio_cfg_scale": float(data_json.get("audio_cfg_scale", 5.0)),
                "end_percent": float(data_json.get("end_percent", 0.9)),
                "output_type": data_json.get("output_type", "pil"),
            }
        }
        return gen_args


class CustomWanVideoPipeline(WanVideoPipeline):
    """
    Customize WanVideoPipeline to support end_percent (0.0-1.0).
    This is used to control how much de-noising steps are considering audio inputs.
    E.g., a 0.9 end_percent means that the last 10% of de-noising steps will not consider audio inputs.
    https://github.com/Fantasy-AMAP/fantasy-talking/issues/52
    We also add a timer.
    """
    @staticmethod
    def from_model_manager(
        model_manager: ModelManager,
        torch_dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
    ) -> "CustomWanVideoPipeline":
        if device is None:
            device = model_manager.device
        if torch_dtype is None:
            torch_dtype = model_manager.torch_dtype
        pipe = CustomWanVideoPipeline(
            device=device,
            torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        parent: ModelGeneration,
        prompt: str,
        negative_prompt: str = "",
        input_image: Optional[Image.Image] = None,
        input_video: Optional[List[Image.Image]] = None,
        denoising_strength: float = 1.0,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        height: int = 480,
        width: int = 832,
        num_frames: int = 1 + 80,
        cfg_scale: float = 5.0,
        audio_cfg_scale: Optional[float] = None,
        num_inference_steps: int = 50,
        sigma_shift: float = 5.0,
        tiled: bool = True,
        tile_size: Tuple[int, int] = (30, 52),
        tile_stride: Tuple[int, int] = (15, 26),
        gen_timer: Optional[GenTimer] = None,
        end_percent: float = 0.9,
        **kwargs: Any,
    ) -> List[Image.Image]:
        # Parameter check
        vae_stride = (4, 8, 8)  # time, height, width
        if gen_timer is None:
            raise ValueError("gen_timer is required for timing")
        height, width = self.check_resize_height_width(height, width)
        if num_frames % vae_stride[0] != 1:
            num_frames = (num_frames + 2) // vae_stride[0] * vae_stride[0] + 1
            logging.info(f"Only 'num_frames % 4 != 1' is acceptable. We round it up to {num_frames}.")
        if end_percent < 0.0 or end_percent > 1.0:
            raise ValueError(f"'end_percent' must be in [0.0, 1.0], got {end_percent}")

        # Tiler parameters
        tiler_kwargs = {
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }

        # Scheduler
        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength,
            shift=sigma_shift
        )

        # Initialize noise
        gen_timer.start("vae_encode")
        BATCH_SIZE = 1
        LAT_CHANNELS = 16  # VAE latent channels
        num_lat_frames = (num_frames - 1) // vae_stride[0] + 1
        lat_h = height // vae_stride[1]
        lat_w = width // vae_stride[2]
        noise = self.generate_noise(
            (BATCH_SIZE, LAT_CHANNELS, num_lat_frames, lat_h, lat_w),
            seed=seed,
            device=rand_device,
            dtype=torch.float32,
        ).to(self.device)

        if input_video is not None:
            # Resize each frame to match adjusted height/width from self.check_resize_height_width
            input_video = [
                TF.resize(frame, size=(height, width), antialias=True)
                for frame in input_video
            ]
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2)  # type: ignore[arg-type, assignment]
            latents = self.encode_video(input_video, **tiler_kwargs).to(
                dtype=noise.dtype,
                device=noise.device)
            latents = self.scheduler.add_noise(
                latents,
                noise,
                timestep=self.scheduler.timesteps[0])
        else:
            latents = noise
        gen_timer.end("vae_encode")

        # Encode prompts
        gen_timer.start("encode_prompt")
        prompt_emb_nega = {}
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
        gen_timer.end("encode_prompt")

        # Encode image
        gen_timer.start("encode_image")
        image_emb = {}
        if input_image is not None and self.image_encoder is not None:
            image_emb = self.encode_image(input_image, num_frames, height, width)
        gen_timer.end("encode_image")

        # Extra input
        extra_input = self.prepare_extra_input(latents)

        # De-noise steps
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self.device).type):
            total_steps = len(self.scheduler.timesteps)
            total_steps_considering_audio = min(int(total_steps * end_percent), total_steps)
            for progress_id, timestep in enumerate(self.scheduler.timesteps):
                logging.debug(f"Running step {progress_id + 1}/{total_steps}.")

                parent.check_interrupted()

                gen_timer.start(f"dit_{progress_id:03d}")
                if progress_id >= total_steps_considering_audio:
                    logging.debug(f"Skipping audio at step {progress_id}.")
                    audio_cfg_scale = 0.0

                timestep = timestep.unsqueeze(0).to(
                    dtype=torch.float32,
                    device=self.device)

                # Inference
                noise_pred_posi = self.dit(
                    latents,
                    timestep=timestep,
                    **prompt_emb_posi,
                    **image_emb,
                    **extra_input,
                    **kwargs,
                )  # (zt,audio,prompt)
                if audio_cfg_scale is not None:
                    audio_scale = kwargs["audio_scale"]
                    kwargs["audio_scale"] = 0.0
                    noise_pred_noaudio = self.dit(
                        latents,
                        timestep=timestep,
                        **prompt_emb_posi,
                        **image_emb,
                        **extra_input,
                        **kwargs,
                    )  # (zt,0,prompt)
                    # kwargs['ip_scale'] = ip_scale
                    if cfg_scale != 1.0:  # prompt cfg
                        noise_pred_no_cond = self.dit(
                            latents,
                            timestep=timestep,
                            **prompt_emb_nega,
                            **image_emb,
                            **extra_input,
                            **kwargs,
                        )  # (zt,0,0)
                        noise_pred = (
                            noise_pred_no_cond
                            + cfg_scale * (noise_pred_noaudio - noise_pred_no_cond)
                            + audio_cfg_scale * (noise_pred_posi - noise_pred_noaudio)
                        )
                    else:
                        noise_pred = noise_pred_noaudio + audio_cfg_scale * (
                            noise_pred_posi - noise_pred_noaudio
                        )
                    kwargs["audio_scale"] = audio_scale
                elif cfg_scale != 1.0:
                    noise_pred_nega = self.dit(
                        latents,
                        timestep=timestep,
                        **prompt_emb_nega,
                        **image_emb,
                        **extra_input,
                        **kwargs,
                    )  # (zt,audio,0)
                    noise_pred = noise_pred_nega + cfg_scale * (
                        noise_pred_posi - noise_pred_nega
                    )
                else:
                    noise_pred = noise_pred_posi

                # Scheduler
                latents = self.scheduler.step(
                    noise_pred,
                    self.scheduler.timesteps[progress_id],
                    latents
                )

                gen_timer.end(f"dit_{progress_id:03d}")

        # Decode
        gen_timer.start("vae_decode")
        frames = self.decode_video(latents, **tiler_kwargs)
        frames = self.tensor2video(frames[0])
        gen_timer.end("vae_decode")

        return frames
