"""
StreamShort: Generate a short video summary from a longer video.
Starts an HTTP server to accept job submissions and monitor job status.
"""

import sys

from typing import override
from typing import Dict
from typing import Any

from streamshort_job import StreamShortJob

# Local relative imports
sys.path.append("..")  # noqa: E402
sys.path.append("../..")  # noqa: E402

from streamwise_job import StreamWiseJob
from streamwise_app import StreamWiseApp
from streamwise_app import run_app


class StreamShortApp(StreamWiseApp):
    """Quart app for StreamShort video summary generation."""

    def __init__(self) -> None:
        super().__init__("streamshort")

    @override
    def create_job(
        self,
        job_id: str,
        job_config: Dict[str, Any]
    ) -> StreamWiseJob:
        return StreamShortJob(
            job_id=job_id,
            config=job_config,
            service_manager=self.service_manager
        )


if __name__ == "__main__":
    run_app(
        StreamShortApp,
        tmp_dir="/tmp/streamshort",
        log_files=[
            "streamwise.log",
            "streamshort.log"
        ],
        app_name="StreamShort",
    )
