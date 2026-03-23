"""
Wrapper class for Flux Upscaler model generation.
"""
import logging
import sys
import random
import aiofiles
import asyncio

from PIL import Image

from typing import override
from typing import List
from typing import Union
from typing import Optional
from typing import Dict
from typing import Any

import torch
import torch.distributed as dist
from torch import inference_mode

from model_timing import GenTimer
from wrapper_model import GenerationInterruptedError
from wrapper_usp import USPGeneration

from flux_xfuser import parallelize_transformer

from diffusers import FluxPipeline
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline

from image_utils import base64_to_img
from media_utils import save_video_frames
from media_utils import base64_to_video_frames
from file_utils import base64_to_binary
from media_utils import get_video_fps

from xfuser.config import EngineConfig
from xfuser.core.distributed import get_runtime_state
from xfuser.core.distributed import initialize_runtime_state
from xfuser.core.distributed import get_pipeline_parallel_world_size


class FluxUpscalerGeneration(USPGeneration):
    """Class for image and video upscaling using the Flux model with ControlNet."""

    def __init__(
        self,
        model_name: str = "fluxupscaler",
        engine_config: EngineConfig = None,
        param_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(
            model_name=model_name,
            engine_config=engine_config,
            param_dtype=param_dtype,
        )

        # Model components
        self.controlnet: Optional[FluxControlNetModel] = None
        self.pipeline: Optional[FluxControlNetPipeline] = None

    def __del__(self) -> None:
        if self.pipeline is not None:
            del self.pipeline.transformer
            self.pipeline = None
        if self.controlnet is not None:
            del self.controlnet
            self.controlnet = None
        super().__del__()

    def load_model(self) -> None:
        assert torch.cuda.is_available()

        self.load_timer.start("controlnet")
        self.CONTROL_NET_NAME = "jasperai/Flux.1-dev-Controlnet-Upscaler"
        self.controlnet = FluxControlNetModel.from_pretrained(
            self.CONTROL_NET_NAME,
            torch_dtype=self.param_dtype,
        )
        self.load_timer.end("controlnet")

        self.load_timer.start("pipeline")
        cache_args = None
        self.MODEL_NAME = "black-forest-labs/FLUX.1-dev"
        self.pipeline = FluxControlNetPipeline.from_pretrained(
            pretrained_model_name_or_path=self.MODEL_NAME,
            controlnet=self.controlnet,
            engine_config=self.engine_config,
            cache_args=cache_args,
            torch_dtype=self.param_dtype,
        )
        self.pipeline = self.pipeline.to(self.device)  # type: ignore[union-attr]
        self.load_timer.end("pipeline")

        logging.info(
            f"[{self.rank}] Loaded FluxUpscalerGeneration: {self.MODEL_NAME} and {self.CONTROL_NET_NAME} "
            f"device:{self.device} dtype:{self.param_dtype}.")

    def init_model_parallelism(self) -> None:
        if not dist.is_initialized() or self.world_size <= 1:
            return

        self.load_timer.start("dit_parallel")
        initialize_runtime_state(self.pipeline, self.engine_config)
        get_runtime_state().set_input_parameters(
            batch_size=1,
            # height=self.input_config.height,
            # width=self.input_config.width,
            # num_inference_steps=self.input_config.num_inference_steps,
            max_condition_sequence_length=512,
            split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
        )

        parallelize_transformer(self.pipeline)
        self.load_timer.end("dit_parallel")

    def model_compile(self) -> None:
        if not self.torch_compile:
            return
        if self.pipeline is None:
            return

        self.load_timer.start("dit_compile")
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.pipeline.transformer = torch.compile(  # type: ignore[attr-defined]
            self.pipeline.transformer,  # type: ignore[attr-defined]
            mode="max-autotune-no-cudagraphs"
        )
        self.load_timer.end("dit_compile")

    def _assert_model_init(self) -> None:
        super()._assert_model_init()
        assert self.pipeline is not None
        assert self.controlnet is not None

    def _assert_args(
        self,
        height: int,
        width: int,
    ) -> None:
        # Check if the image size is supported for the current parallelism setting
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py
        assert self.pipeline is not None
        height_latent = height // self.pipeline.vae_scale_factor  # type: ignore[attr-defined]
        width_latent = width // self.pipeline.vae_scale_factor  # type: ignore[attr-defined]
        img_latent_shape = (height_latent // 2) * (width_latent // 2)
        if img_latent_shape % self.world_size != 0:
            raise ValueError(f"{height}x{width} not supported for {self.world_size} GPUs.")

    @inference_mode()
    async def warmup(self) -> None:
        logging.info(f"[{self.rank}] Warmup for Flux Upscaler generation.")
        await self.generate(
            img=Image.new("RGB", (640, 480), color=(255, 255, 255)),
            width=1280,
            height=720,
            prompt="A warmup image to initialize the model.",
            neg_prompt="",
            sampling_steps=2)

    @override
    @inference_mode()
    async def generate(
        self,
        img: Optional[Image.Image] = None,
        video: Optional[List[Image.Image]] = None,
        height: int = 960,
        width: int = 1280,
        prompt: str = "",
        neg_prompt: str = "",
        sampling_steps: int = 28,
        controlnet_conditioning_scale: float = 0.6,
        guidance_scale: float = 3.5,
        video_fps: int = 30,
        job_id: Optional[str] = None,
        output_type: str = "pil",  # "pil", "video_binary", "video_path"
    ) -> Any:  # returns Image for images, or List[Image]/str/bytes for video
        gen_timer = self._new_gen_timer(job_id)

        self._assert_model_init()
        self._assert_args(height, width)

        self.running = True  # Mark running to avoid concurrent calls

        try:
            # Video upscaling
            if video is not None:
                ret: List[Optional[Image.Image]] = []
                video_frames = video
                for it, video_frame in enumerate(video_frames):
                    if video_frame is None:
                        ret.append(None)
                    else:
                        gen_timer.start(f"frame_{it:03d}")
                        resized_video_frame = await asyncio.to_thread(
                            self.generate_image,
                            gen_timer=gen_timer,
                            img_id=it,
                            img=video_frame,
                            height=height,
                            width=width,
                            prompt=prompt,
                            neg_prompt=neg_prompt,
                            sampling_steps=sampling_steps,
                            controlnet_conditioning_scale=controlnet_conditioning_scale,
                            guidance_scale=guidance_scale)
                        ret.append(resized_video_frame)
                        gen_timer.end(f"frame_{it:03d}")
                if self.rank == 0:
                    logging.info(f"[{self.rank}] Generated {len(video)}->{len(ret)} video frames.")
                return await self._output_video(
                    job_id,
                    gen_timer,
                    ret,
                    video_fps,
                    output_type)

            # Image upscaling
            if img is not None:
                out_image = await asyncio.to_thread(
                    self.generate_image,
                    gen_timer=gen_timer,
                    img=img,
                    height=height,
                    width=width,
                    prompt=prompt,
                    neg_prompt=neg_prompt,
                    sampling_steps=sampling_steps,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    guidance_scale=guidance_scale)
                return out_image

            # Missing inputs
            raise ValueError("Image or video required for Flux Upscaling generation.")
        finally:
            self.running = False
            gen_timer.end("total")

    async def _output_video(
        self,
        job_id: Optional[str],
        gen_timer: GenTimer,
        video_frames: List[Optional[Image.Image]],
        video_fps: int = 30,
        output_type: str = "pil",  # "pil", "video_binary", "video_path"
    ) -> Union[List[Optional[Image.Image]], str, bytes, None]:
        gen_timer.start("output")
        try:
            if output_type == "pil":
                return video_frames

            if output_type in ("video_binary", "video_path"):
                video_path = None
                if job_id:
                    video_path = f"/tmp/{job_id}.mp4"
                video_path = await save_video_frames(
                    [f for f in video_frames if f is not None],
                    out_video_path=video_path,
                    fps=video_fps)
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
    def generate_image(
        self,
        gen_timer: GenTimer,
        img_id: int = 0,
        img: Optional[Image.Image] = None,
        height: int = 960,
        width: int = 1280,
        prompt: str = "",
        neg_prompt: str = "",
        sampling_steps: int = 28,
        controlnet_conditioning_scale: float = 0.6,
        guidance_scale: float = 3.5,
    ) -> Image.Image:
        """
        Upscale one image from a prompt using the Flux model.
        """
        assert img is not None, "Image is required for Flux Upscaling."
        img = img.resize((width, height), Image.Resampling.LANCZOS)

        seed = self.base_seed if self.base_seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        def callback_gen_timer(
            pipeline: FluxPipeline,
            step: int,
            timestep: int,
            callback_kwargs: dict
        ) -> dict:
            gen_timer.end(f"step_{img_id:03d}_{step:03d}")
            if step < sampling_steps - 1:
                gen_timer.start(f"step_{img_id:03d}_{step + 1:03d}")
            if self.interrupted:  # type: ignore[has-type]
                self.interrupted = False
                raise GenerationInterruptedError(f"Generation interrupted at step {step + 1}.")
            return callback_kwargs

        assert self.pipeline is not None, "Flux pipeline not initialized."
        gen_timer.start(f"step_{img_id:03d}_{0:03d}")
        output: Any = self.pipeline(  # type: ignore[operator]
            control_image=img,
            height=height,
            width=width,
            prompt=prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=sampling_steps,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guidance_scale=guidance_scale,
            output_type="pil",
            generator=seed_g,
            callback_on_step_end=callback_gen_timer,
        )

        if len(output.images) != 1:
            raise ValueError(f"Expected 1 image, but got {len(output.images)} images.")

        return output.images[0]

    def get_health(self) -> Dict[str, Any]:
        ret = super().get_health()
        ret.update({
            "device_map": getattr(self.pipeline, "hf_device_map", None) if self.pipeline else None,
        })
        return ret

    async def get_rest_args(
        self,
        data_json: Dict[str, Union[str, int, float]]
    ) -> Dict[str, Any]:
        if data_json is None or not isinstance(data_json, dict):
            raise ValueError("Missing JSON body")

        job_id = data_json.get("job_id", None)

        img_base64 = data_json.get("img", None)
        img = None
        if img_base64 is not None:
            img = base64_to_img(str(img_base64))

        video_base64 = data_json.get("video", None)
        video_frames = None
        video_fps: float = -1
        if video_base64 is not None:
            video_frames = base64_to_video_frames(str(video_base64))
            video_binary = base64_to_binary(str(video_base64))
            video_fps = get_video_fps(video_binary)

        prompt = data_json.get("prompt", "")
        neg_prompt = data_json.get("neg_prompt", "")

        rest_args: Dict[str, Any] = {
            "task": self.model_name,
            "args": {
                "job_id": job_id,
                "img": img,
                "video": video_frames,
                "prompt": prompt,
                "neg_prompt": neg_prompt,
                "width": int(data_json.get("width", 640)),
                "height": int(data_json.get("height", 480)),
                "sampling_steps": int(data_json.get("sampling_steps", 28)),
                "controlnet_conditioning_scale": data_json.get("controlnet_conditioning_scale", 0.6),
                "guidance_scale": data_json.get("guidance_scale", 3.5),
            }
        }
        if video_fps > 0:
            rest_args["args"]["video_fps"] = video_fps
        return rest_args
