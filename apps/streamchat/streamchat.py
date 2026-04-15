"""
StreamChat: Generate a video chat response.
Starts an HTTP server to accept job submissions and monitor job status.
"""

import json
import re
import sys
import os
import logging
import aiofiles
import aiofiles.os

from typing import override
from typing import Dict
from typing import Any
from typing import List

from http import HTTPStatus

from quart import request
from quart import jsonify

from streamchat_job import StreamChatJob

# Local relative imports
sys.path.append("..")  # noqa: E402
sys.path.append("../..")  # noqa: E402

from streamwise_job import StreamWiseJob
from streamwise_app import StreamWiseApp
from streamwise_app import run_app

from quart_utils import QuartReturn


# Find latest chatXXX_prompt.jsonl file
# matches chat001_prompt.jsonl, chat23_prompt.jsonl, etc.
HISTORY_JSONL_PATTERN = re.compile(r"^chat\d+_prompt\.jsonl$")


class StreamChatApp(StreamWiseApp):
    """Quart app for StreamChat video chat generation."""

    def __init__(self) -> None:
        super().__init__("streamchat")

        # Register chat route
        route = self.app.route

        @route("/chat/<job_id>", methods=["POST"])
        async def chat(job_id: str) -> QuartReturn:
            logging.info(f"Received chat request for job {job_id}.")
            job = self.jobs.get(job_id, None)
            if not job:
                logging.error(f"Job {job_id} not found.")
                return jsonify({
                    "status": "error",
                    "error": f"Job {job_id} not found."
                }), HTTPStatus.NOT_FOUND

            form = await request.form
            files = await request.files
            user_message = form.get("message", "").strip()
            audio_message = files.get("audio")
            if not user_message and not audio_message:
                job.logger.error("No text or audio message provided.")
                return jsonify({
                    "status": "error",
                    "error": "No user message or audio provided."
                }), HTTPStatus.BAD_REQUEST

            if audio_message:
                self.logger.info("Transcribing audio message.")
                audio_path = f"/tmp/{job_id}_audio_input.webm"
                audio_message.save(audio_path)  # TODO this may not work
                user_message = await job.transcribe_audio(audio_path)
                self.logger.info(f"Transcribed audio message: {user_message}")

            try:
                reply = await job.gen_chat(user_message)
                response = {
                    "status": "ok"
                }
                response.update(reply)
                return jsonify(response), HTTPStatus.OK
            except Exception as ex:
                job.logger.error(f"Error generating chat response: {ex}")
                return jsonify({
                    "status": "error",
                    "error": str(ex)
                }), HTTPStatus.INTERNAL_SERVER_ERROR

        @route("/chat/<job_id>/history", methods=["GET"])
        async def chat_history(
            job_id: str
        ) -> QuartReturn:
            job = self.jobs.get(job_id, None)
            if job:
                history = await job.get_chat_history()
            else:
                job_path = f"{self.tmp_dir}/{job_id}"
                history = await get_chat_history_from_file(job_path, job_id)
            return jsonify({
                "status": "ok",
                "history": history
            }), HTTPStatus.OK

    @override
    def create_job(
        self,
        job_id: str,
        job_config: Dict[str, Any]
    ) -> StreamWiseJob:
        return StreamChatJob(
            job_id=job_id,
            config=job_config,
            service_manager=self.service_manager
        )


async def get_chat_history_from_file(
    job_path: str,
    job_id: str
) -> List[Dict[str, str]]:
    """
    Load chat history from local JSONL file.
    """
    if not await aiofiles.os.path.exists(job_path):
        raise FileNotFoundError(f"Job path '{job_path}' does not exist.")

    jsonl_files = [
        file
        for file in await aiofiles.os.listdir(job_path)
        if HISTORY_JSONL_PATTERN.match(file)]
    if not jsonl_files:
        raise FileNotFoundError(f"History for '{job_id}' does not exist.")

    jsonl_files.sort(
        key=lambda f: os.path.getmtime(os.path.join(job_path, f)),
        reverse=True)
    latest_file = os.path.join(job_path, jsonl_files[0])
    history = await parse_chat_history(latest_file)
    return history


async def parse_chat_history(
    file_path: str
) -> List[Dict[str, str]]:
    """Parse chat history from a JSONL file."""
    history: List[Dict[str, str]] = []
    if not await aiofiles.os.path.exists(file_path):
        return history
    async with aiofiles.open(file_path, "r", encoding="utf-8") as jsonl_file:
        async for line in jsonl_file:
            msg = json.loads(line.strip())
            history.append(msg)
    return history


if __name__ == "__main__":
    run_app(
        StreamChatApp,
        tmp_dir="/tmp/streamchat",
        log_files=[
            "streamwise.log",
            "streamchat.log"
        ],
        app_name="StreamChat",
    )
