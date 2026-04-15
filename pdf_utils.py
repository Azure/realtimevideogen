import io
import base64
import fitz

from PIL import Image

from typing import List
from typing import Tuple


def parse_pdf(
    pdf_path: str
) -> Tuple[List[str], List[str]]:
    """
    Parse the PDF document to extract text and images.
    """
    pdf_text = []
    pdf_images = []

    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text()
            images = page.get_images(full=True)
            pdf_text.append(text)
            for img in images:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                encoded_image = encode_image(image)
                pdf_images.append(encoded_image)
        for page in doc:
            image = page_to_image(page)
            encoded_image = encode_image(image)
            pdf_images.append(encoded_image)

    return pdf_text, pdf_images


def page_to_image(
    page: fitz.Page
) -> Image.Image:
    """
    Render the full page to a pixel map (as RGB image).
    """
    ZOOM = 2  # Increase resolution (1 = 72 DPI, 2 = 144 DPI)
    mat = fitz.Matrix(ZOOM, ZOOM)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return image


def encode_image(
    image: Image.Image,
    format: str = "JPEG"
) -> str:
    """
    Encode a PIL Image to a base64 data URL.
    """
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    val = buffered.getvalue()
    encoded = base64.b64encode(val).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"
