"""
StreamAnimate: Generate an animated video.
Starts an HTTP server to accept job submissions and monitor job status.
"""

import sys

from typing import override
from typing import Dict
from typing import Any

from streamanimate_job import StreamAnimateJob

# Local relative imports
sys.path.append("..")  # noqa: E402
sys.path.append("../..")  # noqa: E402

from streamwise_job import StreamWiseJob
from streamwise_app import StreamWiseApp
from streamwise_app import run_app


class StreamAnimateApp(StreamWiseApp):
    """Quart app for StreamAnimate video generation."""

    def __init__(self) -> None:
        super().__init__("streamanimate")

    @override
    def create_job(
        self,
        job_id: str,
        job_config: Dict[str, Any]
    ) -> StreamWiseJob:
        return StreamAnimateJob(
            job_id=job_id,
            config=job_config,
            service_manager=self.service_manager
        )


if __name__ == "__main__":
    run_app(
        StreamAnimateApp,
        tmp_dir="/tmp/streamanimate",
        log_files=[
            "streamwise.log",
            "streamanimate.log"
        ],
        app_name="StreamAnimate",
    )
