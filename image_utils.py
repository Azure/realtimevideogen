from typing import Optional

from io import BytesIO

from PIL import Image

from file_utils import binary_to_base64
from file_utils import base64_to_binary


def img_to_base64(image: Optional[Image.Image]) -> Optional[str]:
    if image is None:
        return None
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected Image.Image for image, got {type(image)}")
    image_buffer = BytesIO()
    image.save(image_buffer, format="PNG")
    image_bytes = image_buffer.getvalue()
    image_base64 = binary_to_base64(image_bytes)
    return image_base64


def img_to_bytesio(image: Optional[Image.Image]) -> Optional[BytesIO]:
    if image is None:
        return None
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected Image.Image for image, got {type(image)}")
    image_buffer = BytesIO()
    image.save(image_buffer, format="PNG")
    image_buffer.seek(0)
    return image_buffer


def base64_to_img(image_base64: str) -> Image.Image:
    """Converts a base64-encoded string to a PIL Image object."""
    if not isinstance(image_base64, str):
        raise TypeError(f"Expected str for image_base64, got {type(image_base64)}")
    image_bytes = base64_to_binary(image_base64)
    image_buffer = BytesIO(image_bytes)
    image = Image.open(image_buffer).convert("RGB")
    return image
