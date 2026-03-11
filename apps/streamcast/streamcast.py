"""
StreamCast: Generate podcasts from documents using AI services.
Starts an HTTP server to accept job submissions and monitor job status.
"""

import sys
import logging

from typing import override
from typing import Dict
from typing import Any

from streamcast_job import StreamCastJob

# Local relative imports
sys.path.append("..")  # noqa: E402
sys.path.append("../..")  # noqa: E402

from streamwise_job import StreamWiseJob
from streamwise_app import StreamWiseApp
from streamwise_app import run_app

from tts_utils import estimate_num_words_from_audio_duration


class StreamCastApp(StreamWiseApp):
    """Quart app for StreamCast podcast generation."""

    def __init__(self) -> None:
        super().__init__("streamcast")

    def get_job_config_from_request(
        self,
        job_id: str,
        request_json: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process and return the job configuration from the request JSON."""
        ret = super().get_job_config_from_request(job_id, request_json)

        # Estimate dialogue length for a user requested duration
        if "video_duration_seconds" in ret:
            video_duration_seconds = float(ret["video_duration_seconds"])
            del ret["video_duration_seconds"]
            DIALOGUE_DURATION_SECONDS = 5.0  # Each dialogue should be ~5 seconds
            num_words = estimate_num_words_from_audio_duration(DIALOGUE_DURATION_SECONDS)
            ret["max_words_per_dialogue"] = num_words
            num_dialogues = max(2, int(video_duration_seconds // DIALOGUE_DURATION_SECONDS))
            ret["max_dialogues"] = num_dialogues
            logging.info(
                f"Estimated {num_dialogues} dialogues with {num_words} words each for "
                f"{video_duration_seconds} seconds video.")

        return ret

    @override
    def create_job(
        self,
        job_id: str,
        job_config: Dict[str, Any]
    ) -> StreamWiseJob:
        return StreamCastJob(
            job_id=job_id,
            config=job_config,
            service_manager=self.service_manager
        )


if __name__ == "__main__":
    run_app(
        StreamCastApp,
        tmp_dir="/tmp/streamcast",
        log_files=[
            "streamwise.log",
            "streamcast.log"
        ],
        app_name="streamcast",
    )
