"""
StreamLecture job to generate a lecture video.
"""
import asyncio
import sys
import aiofiles

from typing import override
from typing import Any
from typing import Dict

from lecture_prompts import IMG_PROMPT
from lecture_prompts import IMG_NEG_PROMPT

# Local relative imports
sys.path.append("..")  # noqa: E402
sys.path.append("../..")  # noqa: E402

from streamwise_job import StreamWiseJob
from streamwise_job import JobStatus

from lmm_service_manager import LMMServiceManager

from console_utils import bytes_to_human

from file_utils import base64_to_binary

from pdf_utils import parse_pdf


class StreamLectureJob(StreamWiseJob):
    """A job to generate a lecture video."""

    def __init__(
        self,
        job_id: str,
        service_manager: LMMServiceManager,
        config: Dict[str, Any] = {},
    ) -> None:
        super().__init__(
            "streamlecture",
            job_id,
            service_manager,
            config)

    @override
    async def generate(
        self,
        job_config: Dict[str, Any],
    ) -> None:
        pdf_base64 = job_config.get("pdf_base64", None)
        await self.gen_lecture(pdf_base64)

    async def gen_lecture(
        self,
        pdf_base64: str,
    ) -> None:
        """
        Generate a lecture video from a PDF.
        """
        async with self.job_status_handler():
            if not pdf_base64:
                self.logger.error("Document is required.")
                await self.save_status(JobStatus.FAILED)
                raise ValueError("Missing 'pdf_base64' in request")
            self.logger.info(f"Generating lecture video for document with {bytes_to_human(len(pdf_base64))}.")

            self.logger.info(f"Document base64 with {bytes_to_human(len(pdf_base64))}.")
            pdf_path = f"{self.job_path}/document.pdf"
            async with aiofiles.open(pdf_path, "wb") as file:
                pdf_binary = base64_to_binary(pdf_base64)
                await file.write(pdf_binary)

            # Extract the materials from the PDF
            # TODO
            parse_pdf(pdf_path)

            # Generate the main image
            image_task = asyncio.create_task(
                self.gen.gen_image(
                    prompt=IMG_PROMPT,
                    neg_prompt=IMG_NEG_PROMPT,
                    width=self.width,
                    height=self.height,
                    # TODO
                    task_id="main_image",
                    deadline=self.get_submission_time(),
                )
            )
            self.image = await image_task

            # Generate the transcript
            # TODO

            # Generate each scene
            # TODO
