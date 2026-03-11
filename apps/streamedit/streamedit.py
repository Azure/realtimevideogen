"""
StreamEdit: Generate an edited video.
Starts an HTTP server to accept job submissions and monitor job status.
"""

import sys

from typing import override
from typing import Dict
from typing import Any

from streamedit_job import StreamEditJob

# Local relative imports
sys.path.append("../..")  # noqa: E402

from streamwise_job import StreamWiseJob
from streamwise_app import StreamWiseApp
from streamwise_app import run_app


class StreamEditApp(StreamWiseApp):
    """Quart app for StreamEdit video editing."""

    def __init__(self) -> None:
        super().__init__("streamedit")

    @override
    def create_job(
        self,
        job_id: str,
        job_config: Dict[str, Any]
    ) -> StreamWiseJob:
        return StreamEditJob(
            job_id=job_id,
            config=job_config,
            service_manager=self.service_manager
        )


if __name__ == "__main__":
    run_app(
        StreamEditApp,
        tmp_dir="/tmp/streamedit",
        log_files=[
            "streamwise.log",
            "streamedit.log"
        ],
        app_name="StreamEdit",
    )
