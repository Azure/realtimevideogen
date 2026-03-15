"""
StreamLecture: Generate a lecture video from slides and audio.
Starts an HTTP server to accept job submissions and monitor job status.
"""

import sys

from typing import override
from typing import Dict
from typing import Any

from streamlecture_job import StreamLectureJob

# Local relative imports
sys.path.append("..")  # noqa: E402
sys.path.append("../..")  # noqa: E402

from streamwise_job import StreamWiseJob
from streamwise_app import StreamWiseApp
from streamwise_app import run_app


class StreamLectureApp(StreamWiseApp):
    """Quart app for StreamLecture video lecture generation."""

    def __init__(self) -> None:
        super().__init__("streamlecture")

    @override
    def create_job(
        self,
        job_id: str,
        job_config: Dict[str, Any]
    ) -> StreamWiseJob:
        return StreamLectureJob(
            job_id=job_id,
            config=job_config,
            service_manager=self.service_manager
        )


if __name__ == "__main__":
    run_app(
        StreamLectureApp,
        tmp_dir="/tmp/streamlecture",
        log_files=[
            "streamwise.log",
            "streamlecture.log"
        ],
        app_name="StreamLecture",
    )
