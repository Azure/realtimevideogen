"""
Utilities to process PPTX files.
"""

import logging
import fitz
import subprocess
import os

from pptx import Presentation

from typing import Optional
from typing import List


IMAGE_DPI = 100


def get_num_slides(
    pptx_path: str,
    count_hidden: bool = False
) -> int:
    """
    Get the number of slides in a PPTX file.
    """
    presentation = Presentation(pptx_path)
    num_slides = 0
    for slide in presentation.slides:
        is_hidden = slide._element.get("show") == "0"
        if count_hidden or not is_hidden:
            num_slides += 1
    return num_slides


def pptx_to_images(
    pptx_path: str,
    output_path: str,
    dpi: int = IMAGE_DPI,
    width: Optional[int] = 1280,
    height: Optional[int] = 800,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """
    Render PPTX slides to images using libreoffice.
    """
    # PPTX to PDF
    cmd = [
        "libreoffice",
        "--headless",
        "--convert-to", "pdf",
        "--outdir", output_path,
        pptx_path
    ]
    if logger:
        logger.debug(f"Rendering PPTX slides to PDF with command: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True)
    if logger and result.stdout:
        logger.debug("LibreOffice:\n%s", result.stdout)
    if logger and result.stderr:
        logger.warning("LibreOffice:\n%s", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to generate images from PPTX. Return code: {result.returncode}")

    # PDF to PNG
    image_paths = []
    pdf_path = pptx_path.replace(".pptx", ".pdf")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Expected PDF file not found: {pdf_path}")

    doc = fitz.open(pdf_path)

    doc_rect = doc[0].rect
    if width and height:
        matrix_width = width / doc_rect.width
        matrix_height = height / doc_rect.height
        matrix = fitz.Matrix(matrix_width, matrix_height)
    else:
        matrix = fitz.Matrix(dpi / 72, dpi / 72)

    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        pix = page.get_pixmap(matrix=matrix)
        image_path = f"{output_path}/slide_{page_number + 1:03d}.png"
        pix.save(image_path)
        image_paths.append(image_path)
    doc.close()

    return image_paths
