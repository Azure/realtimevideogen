import math
import cv2
import logging
import os
import numpy as np

from typing import override
from typing import Optional
from typing import List
from typing import Tuple
from typing import Dict
from typing import Any
from typing import Union

import torch
from torch import inference_mode

from PIL import Image

from wrapper_model import ModelGeneration
from image_utils import base64_to_img

from ultralytics import YOLO
"""
WARNING  Ultralytics settings reset to default values. This may be due to a possible problem with your settings or a
recent ultralytics package update.
View Ultralytics Settings with 'yolo settings' or at '/home/azureuser/.config/Ultralytics/settings.json'
Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'.
For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings.
"""


def zoom_image_old(
    image: Image.Image,
    x: int,
    y: int,
    w: int,
    h: int,
    zoom_factor: float = 2
) -> Image.Image:
    image_np = np.array(image)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    w_zoom = int(image.width // zoom_factor)
    h_zoom = int(image.height // zoom_factor)
    x_center = x + w // 2
    y_center = y + h // 2
    x_zoom = int(x_center - w_zoom // 2)
    y_zoom = int(y_center - h_zoom // 2)

    x_zoom = max(0, x_zoom)
    y_zoom = max(0, y_zoom)
    x_zoom = min(image.width - w_zoom, x_zoom)
    y_zoom = min(image.height - h_zoom, y_zoom)
    cropped_person_zoom = image_cv2[
        y_zoom:y_zoom + h_zoom,
        x_zoom:x_zoom + w_zoom
    ]
    cropped_person_zoom_rgb = cv2.cvtColor(cropped_person_zoom, cv2.COLOR_BGR2RGB)
    zoomed_image = Image.fromarray(cropped_person_zoom_rgb)
    return zoomed_image


def zoom_image(
    image: Image.Image,
    x: int,
    y: int,
    w: int,
    h: int,
    zoom_factor: float = 2.0
) -> Image.Image:
    orig_w, orig_h = image.size
    aspect = orig_w / orig_h

    # Initial crop size based on zoom factor
    target_crop_w = max(w, int(orig_w / zoom_factor))
    target_crop_h = max(h, int(orig_h / zoom_factor))

    # Adjust crop size to match original aspect ratio
    if target_crop_w / target_crop_h > aspect:
        target_crop_h = int(target_crop_w / aspect)
    else:
        target_crop_w = int(target_crop_h * aspect)

    # Center the crop on the box
    center_x = x + w // 2
    center_y = y + h // 2

    # Compute crop box
    left = max(0, center_x - target_crop_w // 2)
    upper = max(0, center_y - target_crop_h // 2)
    right = left + target_crop_w
    lower = upper + target_crop_h

    # Adjust if crop goes out of bounds
    if right > orig_w:
        overflow = right - orig_w
        left = max(0, left - overflow)
        right = orig_w
    if lower > orig_h:
        overflow = lower - orig_h
        upper = max(0, upper - overflow)
        lower = orig_h

    # Final sanity check on box
    left = int(left)
    upper = int(upper)
    right = int(right)
    lower = int(lower)

    # Crop and resize
    image = image.crop((left, upper, right, lower))
    # image = image.resize((orig_w, orig_h), Image.LANCZOS)
    return image


def take_top_characters(
    images: List[Tuple[int, float, Image.Image]],
    num_characters: int
) -> List[Image.Image]:
    # Based on confidence
    sorted_images = sorted(
        images,
        key=lambda x: x[1],
        reverse=True
    )[:num_characters]

    # From left to right
    sorted_images = sorted(
        sorted_images,
        key=lambda x: x[0],
        reverse=False
    )
    result: List[Image.Image] = [image[2] for image in sorted_images]
    return result


class ImageCharacterExtractor(ModelGeneration):
    """
    A class to zoom into a specific area of an image.
    """

    def __init__(self) -> None:
        super().__init__("yolo")

        # Model components
        self.obj_recognition: Optional[YOLO] = None

    def __del__(self) -> None:
        if self.obj_recognition is not None:
            self.obj_recognition = None

    def init_parallelism(self) -> None:
        self.load_timer.start("torch_dist")
        # No real parallelism as it runs with a single GPU or CPU
        self.gpu: Optional[str] = None
        device_id: Union[int, str]
        if torch.cuda.is_available():
            self.rank = int(os.getenv("RANK", 0))
            self.local_rank = int(os.getenv("LOCAL_RANK", 0))
            self.world_size = int(os.getenv("WORLD_SIZE", 1))
            device_id = self.local_rank
            self.device = torch.device(f"cuda:{device_id}")
            self.gpu = torch.cuda.get_device_name(device_id)
            torch.cuda.set_device(self.local_rank)
        else:
            device_id = "cpu"
            self.device = torch.device(device_id)
        self.device_id = device_id
        self.load_timer.end("torch_dist")

    def load_model(self) -> None:
        self.load_timer.start("yolo")
        # pretrained YOLO11n model
        self.obj_recognition = YOLO("yolo11n.pt")
        # TODO expand this list if we go fancier
        self.CHARACTER_CLASSES = [
            "person",
            "teddy bear",
        ]
        self.load_timer.end("yolo")

    def init_model_parallelism(self) -> None:
        if self.world_size > 1:
            logging.warning("YOLO does not support distributed parallelism.")

    def model_compile(self) -> None:
        if not self.torch_compile:
            return
        self.load_timer.start("compile")
        assert self.obj_recognition is not None
        self.obj_recognition.model = torch.compile(
            self.obj_recognition.model,
            mode="reduce-overhead")
        self.load_timer.end("compile")

    @inference_mode()
    async def warmup(self) -> None:
        logging.info("Warmup for YOLO generation")
        empty_img = Image.new("RGB", (640, 480), (255, 255, 255))
        self.extract_characters(empty_img)

    def _assert_model_init(self) -> None:
        super()._assert_model_init()
        if self.obj_recognition is None:
            raise ValueError("YOLO not loaded.")

    @inference_mode()
    def extract_characters(
        self,
        img: Image.Image,
        num_characters: int = 2,
        zoom_factor: float = 1.6,
        job_id: Optional[str] = None,
    ) -> List[Optional[Image.Image]]:
        gen_timer = self._new_gen_timer(job_id)

        self._assert_model_init()
        assert self.obj_recognition is not None

        self.running = True  # We can run in parallel but good to know if we are running

        try:
            results = self.obj_recognition.predict(
                img,
                verbose=False,
                device=self.device)

            if len(results) == 0:
                logging.warning("No results for the image.")
                return []

            # Original image with the boxes on top for debugging
            debug_img_np = results[0].plot()
            debug_img_np = cv2.cvtColor(debug_img_np, cv2.COLOR_BGR2RGB)
            debug_img = Image.fromarray(debug_img_np)

            if len(results[0].boxes) == 0:
                logging.warning("No characters detected in the image.")
                return [debug_img] + [None] * num_characters

            logging.info(f"Detected {len(results[0].boxes)} objects in the image.")
            person_zoom_images = []
            for box in results[0].boxes:
                obj_class = box.cls  # class index
                obj_class = self.obj_recognition.model.names[int(obj_class)]  # It takes < 3 milliseconds
                obj_conf = box.conf.cpu().numpy()[0]  # confidence score
                x1, y1, x2, y2 = box.xyxy.cpu().numpy().tolist()[0]  # xyxy format (x1, y1, x2, y2)
                if obj_class in self.CHARACTER_CLASSES:
                    x_center = math.floor((x2 + x1) / 2.0)
                    person_zoom_image = zoom_image(
                        img,
                        x1, y1,
                        x2 - x1, y2 - y1,
                        zoom_factor=zoom_factor)
                    person_zoom_images.append((x_center, obj_conf, person_zoom_image))

            # Sort by confidence and take the top NUM_CHARACTERS and sort by x
            top_images: List[Optional[Image.Image]] = take_top_characters(person_zoom_images, num_characters)
            return [debug_img] + top_images
        finally:
            self.running = False
            gen_timer.end("total")

    @override
    @inference_mode()
    async def generate(
        self,
        img: Image.Image,
        num_characters: int = 2,
        zoom_factor: float = 1.6,
        job_id: Optional[str] = None,
    ) -> List[Optional[Image.Image]]:
        return self.extract_characters(
            img=img,
            num_characters=num_characters,
            zoom_factor=zoom_factor,
            job_id=job_id)

    def get_health(self) -> Dict[str, Any]:
        ret = super().get_health()
        ret["gpu"] = self.gpu
        return ret

    async def get_rest_args(
        self,
        data_json: Dict[str, Union[str, int, float]]
    ) -> Dict[str, Any]:
        if data_json is None or not isinstance(data_json, dict):
            raise ValueError("Missing JSON body")

        job_id = data_json.get("job_id", None)

        img_base64 = data_json.get("img", None)
        if img_base64 is None:
            raise ValueError("Missing 'img' parameter")
        if not isinstance(img_base64, str):
            raise ValueError("'img' parameter must be a base64 string")
        img = base64_to_img(img_base64)
        num_characters = int(data_json.get("num_characters", 2))
        zoom_factor = float(data_json.get("zoom_factor", 1.6))

        return {
            "task": self.model_name,
            "args": {
                "job_id": job_id,
                "img": img,
                "num_characters": num_characters,
                "zoom_factor": zoom_factor
            }
        }


if __name__ == "__main__":
    character_extractor = ImageCharacterExtractor()

    img_filename = "generated_image_hidream_20250416T171929.png"
    image = Image.open(img_filename).convert("RGB")
    character_extractor.extract_characters(image, 2)
