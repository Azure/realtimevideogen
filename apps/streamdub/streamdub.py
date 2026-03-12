"""
StreamDub: Generate a dubbed video from an original video.
Starts an HTTP server to accept job submissions and monitor job status.
"""

import sys

from typing import override
from typing import Dict
from typing import Any

from streamdub_job import StreamDubJob

# Local relative imports
sys.path.append("..")  # noqa: E402
sys.path.append("../..")  # noqa: E402

from streamwise_job import StreamWiseJob
from streamwise_app import StreamWiseApp
from streamwise_app import run_app


class StreamDubApp(StreamWiseApp):
    """Quart app for StreamDub video dubbing."""

    def __init__(self) -> None:
        super().__init__("streamdub")

    @override
    def create_job(
        self,
        job_id: str,
        job_config: Dict[str, Any]
    ) -> StreamWiseJob:
        return StreamDubJob(
            job_id=job_id,
            config=job_config,
            service_manager=self.service_manager
        )


if __name__ == "__main__":
    run_app(
        StreamDubApp,
        tmp_dir="/tmp/streamdub",
        log_files=[
            "streamwise.log",
            "streamdub.log"
        ],
        app_name="streamdub",
    )
