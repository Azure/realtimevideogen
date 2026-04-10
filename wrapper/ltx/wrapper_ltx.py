import os
import logging
import torch
import tempfile

from torch import inference_mode

from typing import override
from typing import List
from typing import Optional
from typing import Dict
from typing import Union
from typing import Any
from typing import Tuple

from diffusers import LTXConditionPipeline
from diffusers import LTXLatentUpsamplePipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video
from diffusers.utils import load_video

from PIL import Image

from wrapper_model import ModelGeneration

from model_timing import GenTimer

from image_utils import base64_to_img
from media_utils import save_diffusers_video


class LTXVideoGeneration(ModelGeneration):
    """
    Base class for generation using LTX video model.
    """

    def __init__(
        self,
        param_dtype: torch.dtype = torch.bfloat16,
        torch_compile: bool = True,
    ) -> None:
        super().__init__(model_name="ltx")

        # Model components
        self.pipeline: Optional[LTXConditionPipeline] = None
        self.pipe_upsample: Optional[LTXLatentUpsamplePipeline] = None

        self.param_dtype = param_dtype

        # Parallelism
        self.gpu: Optional[str] = None
        if torch.cuda.is_available():
            self.gpu = torch.cuda.get_device_name(0)

        # Model features
        self.FPS = 24
        self.vae_stride = (4, 32, 32)  # time, height, width

    def init_parallelism(self) -> None:
        self.load_timer.start("torch_dist")

        self.rank = int(os.getenv("RANK", 0))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))

        self.device_id = self.local_rank
        self.device = torch.device(f"cuda:{self.device_id}")

        if self.world_size > 1:
            # TODO implement xfuser parallelism
            logging.warning("Parallelism is not supported in LTX generation, running on single device.")

        self.load_timer.end("torch_dist")

    def load_model(self) -> None:
        assert torch.cuda.is_available()

        self.load_timer.start("pipeline")
        self.pipeline = LTXConditionPipeline.from_pretrained(
            # "Lightricks/LTX-Video",
            "Lightricks/LTX-Video-0.9.7-dev",
            torch_dtype=self.param_dtype)
        assert self.pipeline is not None
        self.pipeline.to(self.device)  # type: ignore[union-attr]
        self.load_timer.end("pipeline")

        self.load_timer.start("upsample")
        self.pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
            "Lightricks/ltxv-spatial-upscaler-0.9.7",
            vae=self.pipeline.vae,  # type: ignore[union-attr]
            torch_dtype=self.param_dtype)
        assert self.pipe_upsample is not None
        self.pipe_upsample.to(self.device)  # type: ignore[union-attr]
        self.load_timer.end("upsample")

        self.pipeline.vae.enable_tiling()  # type: ignore[union-attr]

    def init_model_parallelism(self) -> None:
        if self.world_size > 1:
            logging.warning("LTX video generation does not support distributed parallelism.")

    def model_compile(self) -> None:
        if not self.torch_compile:
            return
        logging.info("Compiling transformer with torch.compile().")
        self.load_timer.start("dit_compile")
        assert self.pipeline is not None
        self.pipeline.transformer = torch.compile(  # type: ignore[attr-defined]
            self.pipeline.transformer,  # type: ignore[attr-defined]
            mode="max-autotune-no-cudagraphs",
        )
        self.load_timer.end("dit_compile")

    def round_to_nearest_resolution_acceptable_by_vae(
        self,
        height: int,
        width: int
    ) -> Tuple[int, int]:
        assert self.pipeline is not None
        vae_spatial_compression_ratio = self.pipeline.vae_spatial_compression_ratio  # type: ignore[attr-defined]
        height = height - (height % vae_spatial_compression_ratio)
        width = width - (width % vae_spatial_compression_ratio)
        return height, width

    def __del__(self) -> None:
        if self.pipeline is not None:
            del self.pipeline
        if self.pipe_upsample is not None:
            del self.pipe_upsample
        super().__del__()

    def _assert_model_init(self) -> None:
        assert self.pipeline is not None
        assert self.pipe_upsample is not None

    @torch.inference_mode()
    async def warmup(self) -> None:
        logging.info(f"[{self.rank}] Warmup for LTX video generation.")
        await self.generate(
            img=Image.new("RGB", (480, 480), color=(255, 255, 255)),
            prompt="Warmup prompt",
            neg_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
            height=480,
            width=480,
            num_frames=1 + 8
        )

    def _assert_args(
        self,
        height: int,
        width: int,
        num_frames: int
    ) -> None:
        if height % 32 != 0:
            raise ValueError(f"Height {height} is not divisible by 32.")
        if width % 32 != 0:
            raise ValueError(f"Width {width} is not divisible by 32.")
        if num_frames <= 1:
            raise ValueError(f"Number of frames {num_frames} must be greater than 1.")
        if num_frames % 8 != 1:
            raise ValueError(f"Number of frames {num_frames} must be 1 + 8n, where n >= 0.")

    @override
    @inference_mode()
    async def generate(
        self,
        img: Image.Image,
        prompt: str,
        neg_prompt: str = "",
        height: int = 512,
        width: int = 832,
        num_frames: int = 1 + 96,
        sampling_steps: int = 30,
        sampling_upscale_steps: int = 10,
        job_id: Optional[str] = None,
        output_type: str = "pil",  # "pil", "video_binary", "video_path"
    ) -> Union[List[Image.Image], bytes, str, None]:
        gen_timer = self._new_gen_timer(job_id)

        self._assert_model_init()
        self._assert_args(height, width, num_frames)
        assert self.pipeline is not None
        assert self.pipe_upsample is not None

        self.running = True  # Mark running to avoid concurrent calls

        try:
            # image = load_image(
            # "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/penguin.png")
            # compress the image using video compression as the model was trained on videos
            video = load_video(export_to_video([img]))
            condition1 = LTXVideoCondition(video=video, frame_index=0)

            prompt = "A cute little penguin takes out a book and starts reading it"
            negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
            # expected_height = 480
            # expected_width = 832
            downscale_factor = 2 / 3
            # num_frames = 96

            # Part 1. Generate video at smaller resolution
            downscaled_height = int(height * downscale_factor)
            downscaled_width = int(width * downscale_factor)
            downscaled_height, downscaled_width = self.round_to_nearest_resolution_acceptable_by_vae(
                downscaled_height,
                downscaled_width)
            seed = torch.Generator()
            seed.manual_seed(0)
            latents = self.pipeline(  # type: ignore[operator]
                conditions=[condition1],
                prompt=prompt,
                negative_prompt=neg_prompt,
                width=downscaled_width,
                height=downscaled_height,
                num_frames=num_frames,
                num_inference_steps=sampling_steps,
                generator=seed,
                output_type="latent",
            ).frames

            # Part 2. Upscale generated video using latent up-sampler with fewer inference steps
            # The available latent up-sampler up-scales the height/width by 2x
            upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
            upscaled_latents = self.pipe_upsample(  # type: ignore[operator]
                latents=latents,
                output_type="latent"
            ).frames

            # Part 3. De-noise the upscaled video with few steps to improve texture (optional, but recommended)
            seed = torch.Generator()
            seed.manual_seed(0)
            video_frames = self.pipeline(  # type: ignore[operator]
                conditions=[condition1],
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=upscaled_width,
                height=upscaled_height,
                num_frames=num_frames,
                denoise_strength=0.4,  # Effectively, 4 inference steps out of 10
                num_inference_steps=sampling_upscale_steps,
                latents=upscaled_latents,
                decode_timestep=0.05,
                image_cond_noise_scale=0.025,
                generator=seed,
                output_type="pil",
            ).frames[0]

            # Part 4. Downscale the video to the expected resolution
            video_frames = [frame.resize((width, height)) for frame in video_frames]

            return self._output_video(
                job_id,
                gen_timer,
                video_frames,
                output_type=output_type)
        finally:
            self.running = False
            gen_timer.end("total")

    def _output_video(
        self,
        job_id: Optional[str],
        gen_timer: GenTimer,
        video_frames: List[Image.Image],
        output_type: str = "pil",  # "pil", "video_binary", "video_path"
    ) -> Union[List[Image.Image], bytes, str, None]:
        gen_timer.start("output")
        try:
            if output_type == "pil":
                return video_frames

            if output_type in ("video_binary", "video_path"):
                if not job_id:
                    video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                else:
                    video_path = f"/tmp/{job_id}.mp4"
                save_diffusers_video(
                    video_frames,
                    out_video_path=video_path,
                    fps=self.FPS)
                if output_type == "video_path":
                    return video_path

                # video_binary
                with open(video_path, "rb") as f:
                    video_binary = f.read()
                return video_binary

            logging.error(f"Unknown output type: {output_type}")
            return None
        finally:
            gen_timer.end("output")

    async def get_rest_args(
        self,
        data_json: Dict[str, Union[str, int, float]]
    ) -> Dict[str, Any]:
        if data_json is None or not isinstance(data_json, dict):
            raise ValueError("Missing JSON body.")

        job_id = data_json.get("job_id", None)

        img_base64 = data_json.get("img", None)
        if img_base64 is None:
            raise ValueError("Missing 'img' parameter.")
        assert isinstance(img_base64, str)
        img = base64_to_img(img_base64)

        prompt = data_json.get("prompt", None)
        if prompt is None:
            raise ValueError("Missing 'prompt' parameter.")

        neg_prompt = data_json.get("neg_prompt", "")

        height = int(data_json.get("height", 512))
        width = int(data_json.get("width", 832))
        num_frames = int(data_json.get("num_frames", 1 + 16))
        steps = int(data_json.get("sampling_steps", 30))

        if height <= 0:
            raise ValueError(f"height {height} must be positive.")
        if width <= 0:
            raise ValueError(f"width {width} must be positive.")
        if steps <= 0:
            raise ValueError(f"sampling_steps {steps} must be positive.")

        video_seconds = data_json.get("video_seconds", None)
        if video_seconds is not None:
            if float(video_seconds) <= 0:
                raise ValueError(f"video_seconds {video_seconds} must be positive.")
            VAE_FRAMES = self.vae_stride[0]
            num_frames = int(video_seconds * self.FPS)
            num_frames = 1 + ((num_frames - 1) // VAE_FRAMES) * VAE_FRAMES  # 4n + 1

        if num_frames <= 0:
            raise ValueError(f"num_frames {num_frames} must be positive.")

        gen_args = {
            "task": self.model_name,
            "args": {
                "job_id": job_id,
                "img": img,
                "prompt": prompt,
                "neg_prompt": neg_prompt,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "sampling_steps": steps,
            }
        }
        return gen_args

    def get_health(self) -> Dict[str, Any]:
        ret = super().get_health()
        ret.update({
            "gpu": self.gpu,
            "rank": self.rank,
            "world_size": self.world_size,
            "torch_compile": self.torch_compile,
            "dtype": str(self.param_dtype),
            "vae_stride": self.vae_stride,
            "fps": str(self.FPS),
        })
        return ret
