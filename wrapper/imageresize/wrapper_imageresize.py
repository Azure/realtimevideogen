import logging

from PIL import Image

from typing import override
from typing import List
from typing import Optional
from typing import Dict
from typing import Any
from typing import Union

from wrapper_model import ModelGeneration

from image_utils import base64_to_img
from media_utils import base64_to_video_frames


class ImageResize(ModelGeneration):
    """Image resizing model generation."""

    def __init__(self) -> None:
        super().__init__("imageresize")

    async def warmup(self) -> None:
        logging.info("Warmup for ImageResize generation")
        dummy_image = Image.new('RGB', (128, 128), color="red")
        await self.generate(image=dummy_image)

    @override
    async def generate(
        self,
        image: Image.Image,
        video: Optional[List[Image.Image]] = None,
        height: int = 1024,
        width: int = 1024,
        job_id: Optional[str] = None,
    ) -> Image.Image:
        gen_timer = self._new_gen_timer(job_id)

        self.running = True

        try:
            # Video
            if video is not None:
                return [
                    frame.resize((width, height), Image.Resampling.LANCZOS)
                    for frame in video
                ]
            # Image
            if image is not None:
                return image.resize((width, height), Image.Resampling.LANCZOS)
            # Missing inputs
            raise ValueError("Image is required for resizing generation.")
        finally:
            gen_timer.end("total")
            self.running = False

    async def get_rest_args(
        self,
        data_json: Dict[str, Union[str, int, float]],
    ) -> Dict[str, Any]:
        if data_json is None:
            raise ValueError("Missing JSON body")

        img_base64 = data_json.get("img", None)
        img = None
        if img_base64 is not None:
            img = base64_to_img(img_base64)

        video_base64 = data_json.get("video", None)
        video = None
        if video_base64 is not None:
            video = base64_to_video_frames(video_base64)

        return {
            "task": self.model_name,
            "args": {
                "image": img,
                "video": video,
                "height": data_json.get("height", 1024),
                "width": data_json.get("width", 1024),
            }
        }
