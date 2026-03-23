"""
Wrapper for Real-ESRGAN image and video upscaling.
"""
import os
import logging
import tempfile
import aiofiles
import asyncio

from PIL import Image

from datetime import timedelta

import torch
from torch import distributed as dist
from torch import inference_mode

from typing import override
from typing import List
from typing import Optional
from typing import Union
from typing import Dict
from typing import Any

from model_timing import GenTimer
from wrapper_model import ModelGeneration

from console_utils import bytes_to_human

from image_utils import base64_to_img
from media_utils import base64_to_video_frames
from file_utils import base64_to_binary
from media_utils import get_video_fps
from media_utils import save_video_frames
from media_utils import get_video_size

from RealESRGAN import RealESRGAN


class RealESRGANGeneration(ModelGeneration):
    """Handle image and video upscaling using Real-ESRGAN."""

    def __init__(self) -> None:
        super().__init__("realesrgan")

        # Model components
        self.SCALING_FACTORS = [2, 4, 8]
        self.models: Dict[int, RealESRGAN] = {}

        self.GPU: Optional[str] = None
        if torch.cuda.is_available():
            self.GPU = torch.cuda.get_device_name(0)

    def __del__(self) -> None:
        if self.models is not None:
            del self.models
        super().__del__()

    def init_parallelism(self) -> None:
        self.load_timer.start("torch_dist")

        self.rank = int(os.getenv("RANK", 0))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))

        self.device_id = self.local_rank

        if not torch.cuda.is_available():
            self.device_id = 0
            self.device = torch.device("cpu")
            logging.warning("CUDA is not available. Running on CPU.")
            self.load_timer.end("torch_dist")
            return  # CPU mode, no parallelism needed

        self.device = torch.device(f"cuda:{self.device_id}")

        torch.cuda.set_device(self.local_rank)

        if self.world_size <= 1:
            self.load_timer.end("torch_dist")
            return  # Single GPU mode, no parallelism needed

        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=self.rank,
                world_size=self.world_size,
                timeout=timedelta(hours=24),  # Prevent NCCL timeout
            )

        self.load_timer.end("torch_dist")

    def load_model(self) -> None:
        self.load_timer.start("realesrgan")
        for scaling_factor in self.SCALING_FACTORS:
            self.load_timer.start(f"realesrgan{scaling_factor}x")
            self.models[scaling_factor] = RealESRGAN(self.device, scale=scaling_factor)
            self.models[scaling_factor].load_weights(
                f"ai-forever/Real-ESRGAN/RealESRGAN_x{scaling_factor}.pth",
                download=False,
            )
            self.load_timer.end(f"realesrgan{scaling_factor}x")
        logging.info("Loaded Real-ESRGAN.")
        self.load_timer.end("realesrgan")

    def init_model_parallelism(self) -> None:
        if not dist.is_initialized() or self.world_size <= 1:
            return
        logging.info("Real-ESRGAN supports video parallelism.")

    def model_compile(self) -> None:
        if not self.torch_compile:
            return
        self.load_timer.start("compile")
        logging.warning("torch.compile() not supported.")
        # Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run.
        # RealESRGAN/rrdbnet_arch.py", line 120, in forward
        # out = self.conv_last(self.lrelu(self.conv_hr(feat))).
        # To prevent overwriting, clone the tensor outside of torch.compile() or
        # call torch.compiler.cudagraph_mark_step_begin() before each model invocation.
        """
        for scaling_factor in self.SCALING_FACTORS:
            self.load_timer.start(f"compile{scaling_factor}x")
            self.models[scaling_factor].model = torch.compile(
                self.models[scaling_factor].model,
                mode="reduce-overhead")
            self.load_timer.end(f"compile{scaling_factor}x")
        """
        self.load_timer.end("compile")

    def _assert_model_init(self) -> None:
        super()._assert_model_init()
        if not self.models:
            raise ValueError("Real-ESRGAN not initialized.")

    @inference_mode()
    async def warmup(self) -> None:
        logging.info(f"[{self.rank}] Warmup for Real-ESRGAN generation.")
        video = [Image.new("RGB", (640, 480), color=(255, 255, 255))] * self.world_size * 2
        await self.generate(
            video=video,
            width=1280,
            height=960)

    def _chunk_list_image(
        self,
        images: List[Image.Image]
    ) -> List[Optional[Image.Image]]:
        """
        Chunk the list of images based on the world size.
        This is used to distribute the workload across multiple ranks.
        Each rank will process only its assigned images (not None).
        """
        if self.world_size == 1 or len(images) < 1:
            return images
        # Chunk one image per rank
        ret = []
        for it, image in enumerate(images):
            if it % self.world_size == self.rank:
                ret.append(image)
            else:
                ret.append(None)
        return ret

    def _gather_chunks(
        self,
        chunked_images: List[Optional[Image.Image]]
    ) -> List[Image.Image]:
        """
        Gather the images from all ranks and put them into a sinle one.
        Each rank will return its assigned images (not None).
        This is used to collect the results after processing.
        """
        if self.world_size == 1:
            return chunked_images

        gathered_lists = None
        if self.rank == 0:
            gathered_lists = [None] * self.world_size
        dist.gather_object(chunked_images, gathered_lists, dst=0)

        if self.rank != 0:
            return []

        ret = []
        for position_images in zip(*gathered_lists):  # type: ignore[misc]
            for img in position_images:
                if img is not None:
                    ret.append(img)
                    break
        if len(ret) != len(chunked_images):
            raise ValueError("Gathered images do not match the original chunked images length.")
        return ret

    @override
    @inference_mode()
    async def generate(
        self,
        job_id: Optional[str] = None,
        image: Optional[Image.Image] = None,
        video: Optional[List[Image.Image]] = None,
        height: int = 960,
        width: int = 1280,
        batch_size: int = 4,
        patches_size: int = 192,
        padding: int = 24,
        pad_size: int = 15,
        video_fps: int = 30,
        output_type: str = "pil",  # "pil", "video_binary", "video_path"
    ) -> List[Image.Image]:
        gen_timer = self._new_gen_timer(job_id)

        self._assert_model_init()

        self.running = True  # Mark running to avoid concurrent calls

        try:
            # Video upscaling
            if video is not None:
                if self.rank == 0:
                    video_len = get_video_size(video)
                    logging.info(
                        f"[{self.rank}] Upscaling video with {len(video)} frames and {bytes_to_human(video_len)}.")
                ret: list[Any] = []
                video_frames = video
                chunked_video_frames = self._chunk_list_image(video_frames)
                for it, frame in enumerate(chunked_video_frames):
                    if frame is None:
                        ret.append(None)
                    else:
                        gen_timer.start(f"frame_{it:03d}")
                        resized_frame = await asyncio.to_thread(
                            self.generate_image,
                            image=frame,
                            height=height,
                            width=width,
                            batch_size=batch_size,
                            patches_size=patches_size,
                            padding=padding,
                            pad_size=pad_size)
                        ret.append(resized_frame)
                        gen_timer.end(f"frame_{it:03d}")
                ret = self._gather_chunks(ret)
                if self.rank == 0:
                    logging.info(f"[{self.rank}] Generated {len(video)}->{len(ret)} upscaled video frames.")
                else:
                    return None  # type: ignore[return-value]  # Skip non-rank 0 processes
                return await self._output_video(  # type: ignore[return-value]
                    job_id,
                    gen_timer,
                    ret,
                    video_fps,
                    output_type)

            # Image upscaling
            if image is not None:
                if self.rank != 0:
                    logging.debug(f"[{self.rank}] Skipping image upscaling, not rank 0.")
                    return [None]
                out_image = await asyncio.to_thread(
                    self.generate_image,
                    image=image,
                    height=height,
                    width=width,
                    batch_size=batch_size,
                    patches_size=patches_size,
                    padding=padding,
                    pad_size=pad_size)
                if self.rank == 0:
                    logging.info(f"[{self.rank}] Generated one upscaled image.")
                return [out_image]

            # Missing inputs
            raise ValueError("Image or video required for Real-ESRGAN generation.")
        finally:
            self.running = False
            gen_timer.end("total")

    async def _output_video(
        self,
        job_id: Optional[str],
        gen_timer: GenTimer,
        video_frames: List[Image.Image],
        video_fps: int = 30,
        output_type: str = "pil",  # "pil", "video_binary", "video_path"
    ) -> Union[List[Image.Image], str, bytes]:
        gen_timer.start("output")
        try:
            if output_type == "pil":
                return video_frames

            if output_type in ("video_binary", "video_path"):
                if not job_id:
                    video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                else:
                    video_path = f"/tmp/{job_id}.mp4"
                video_path = await save_video_frames(
                    video_frames,
                    out_video_path=video_path,
                    fps=video_fps)
                if output_type == "video_path":
                    return video_path

                # video_binary
                async with aiofiles.open(video_path, "rb") as f:
                    video_binary = await f.read()
                return video_binary

            logging.error(f"Unknown output type: {output_type}")
            return None  # type: ignore[return-value]
        finally:
            gen_timer.end("output")

    def get_model_scaling_factor(
        self,
        input_width: int,
        input_height: int,
        output_width: int,
        output_height: int,
    ) -> int:
        if output_width < input_width or output_height < input_height:
            raise ValueError(
                f"Output size {output_width}x{output_height} must be larger than input {input_width}x{input_height}.")
        max_scaling_factor = max(
            output_width / input_width,
            output_height / input_height)
        if max_scaling_factor > self.SCALING_FACTORS[-1]:
            raise ValueError(f"Scaling factor {max_scaling_factor}x > max {self.SCALING_FACTORS[-1]}x.")
        # Get the next scaling factor in self.SCALING_FACTORS
        model_scaling_factor = min(
            [sf for sf in self.SCALING_FACTORS if sf >= max_scaling_factor],
            default=self.SCALING_FACTORS[-1]
        )
        if model_scaling_factor not in self.models:
            raise ValueError(f"Model for {model_scaling_factor}x not loaded.")
        return model_scaling_factor

    @torch.inference_mode()
    def generate_image(
        self,
        image: Image.Image,
        height: int = 768,
        width: int = 1024,
        batch_size: int = 4,
        patches_size: int = 192,
        padding: int = 24,
        pad_size: int = 15,
    ) -> Image.Image:
        input_width, input_height = image.size
        model_scaling_factor = self.get_model_scaling_factor(
            input_width, input_height,
            width, height)
        logging.debug(
            f"[{self.rank}] {input_width}x{input_height}->{width}x{height} Using scaling {model_scaling_factor}x.")

        model = self.models.get(model_scaling_factor, None)
        if model is None:
            raise ValueError(f"Model for {model_scaling_factor}x not loaded.")
        output_image = model.predict(
            image,
            batch_size=batch_size,
            patches_size=patches_size,
            padding=padding,
            pad_size=pad_size)

        logging.debug(f"[{self.rank}] Generated image with size {output_image.size}")
        if output_image.size[0] != width or output_image.size[1] != height:
            logging.debug(f"[{self.rank}] Downscaling now to {width}x{height}.")
            output_image = output_image.resize((width, height), Image.LANCZOS)

        return output_image

    def get_health(self) -> Dict[str, Any]:
        ret = super().get_health()
        ret.update({
            "gpu": self.GPU,
            "rank": self.rank,
            "world_size": self.world_size,
            "torch_compile": self.torch_compile,
        })
        return ret

    async def get_rest_args(self, data_json: Dict[str, str]) -> Dict[str, Any]:
        if data_json is None or not isinstance(data_json, dict):
            raise ValueError("Missing JSON body")

        job_id = data_json.get("job_id", None)

        img_base64 = data_json.get("img", None)
        img = None
        if img_base64 is not None:
            img = base64_to_img(img_base64)

        video_base64 = data_json.get("video", None)
        video_frames = None
        video_fps: float = -1.0
        if video_base64 is not None:
            video_frames = base64_to_video_frames(video_base64)
            video_binary = base64_to_binary(video_base64)
            video_fps = get_video_fps(video_binary)

        rest_args = {
            "task": self.model_name,
            "args": {
                "job_id": job_id,
                "image": img,
                "video": video_frames,
                "width": int(data_json.get("width", 640)),
                "height": int(data_json.get("height", 480)),
                "batch_size": int(data_json.get("batch_size", 4)),
                "patches_size": int(data_json.get("patches_size", 192)),
                "padding": int(data_json.get("padding", 24)),
                "pad_size": int(data_json.get("pad_size", 15)),
                "output_type": data_json.get("output_type", "pil"),
            }
        }
        if video_fps > 0:
            rest_args["args"]["video_fps"] = video_fps
        return rest_args
