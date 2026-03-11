"""
Wrapper class for LongCat-Video model generation.
TODO
https://github.com/meituan-longcat/LongCat-Video/blob/main/run_demo_image_to_video.py
"""

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

from PIL import Image

from wrapper_model import ModelGeneration

from model_timing import GenTimer

from image_utils import base64_to_img
from media_utils import save_diffusers_video


class LongCatVideoGeneration(ModelGeneration):
    """
    Base class for generation using LongCat-Video model.
    """

    def __init__(
        self,
        param_dtype: torch.dtype = torch.bfloat16,
        torch_compile: bool = True,
    ) -> None:
        super().__init__(model_name="longcatvideo")

        # TODO

    def init_parallelism(self) -> None:
        self.load_timer.start("torch_dist")

        self.rank = int(os.getenv("RANK", 0))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))

        self.device_id = self.local_rank
        self.device = torch.device(f"cuda:{self.device_id}")

        if self.world_size > 1:
            # TODO implement xfuser parallelism
            logging.warning("Parallelism is not supported in LongCat-Video generation, running on single device.")

        self.load_timer.end("torch_dist")

    def load_model(self) -> None:
        assert torch.cuda.is_available()

        self.load_timer.start("pipeline")
        # TODO
        self.load_timer.end("pipeline")

    def init_model_parallelism(self) -> None:
        if self.world_size > 1:
            logging.warning("LongCat-Video video generation does not support distributed parallelism.")

    def model_compile(self) -> None:
        if not self.torch_compile:
            return
        logging.info("Compiling transformer with torch.compile().")
        self.load_timer.start("dit_compile")
        # TODO
        self.load_timer.end("dit_compile")

    def __del__(self) -> None:
        # TODO
        super().__del__()

    def _assert_model_init(self) -> None:
        pass  # TODO

    def _assert_args(
        self,
        height: int,
        width: int,
        num_frames: int
    ) -> None:
        pass  # TODO

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

    @override
    @inference_mode()
    async def generate(  # type: ignore[override]
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

        self.running = True  # Mark running to avoid concurrent calls

        try:
            # TODO

            return None
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
        img = base64_to_img(img_base64)

        prompt = data_json.get("prompt", None)
        if prompt is None:
            raise ValueError("Missing 'prompt' parameter.")

        neg_prompt = data_json.get("neg_prompt", "")

        height = int(data_json.get("height", 512))
        width = int(data_json.get("width", 832))
        num_frames = int(data_json.get("num_frames", 1 + 16))
        steps = int(data_json.get("sampling_steps", 30))

        video_seconds = data_json.get("video_seconds", None)
        if video_seconds is not None:
            VAE_FRAMES = self.vae_stride[0]
            num_frames = int(video_seconds * self.FPS)
            num_frames = 1 + ((num_frames - 1) // VAE_FRAMES) * VAE_FRAMES  # 4n + 1

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
