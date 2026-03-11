"""
StreamPersona: Generate a persona presenting some content.
Starts an HTTP server to accept job submissions and monitor job status.
"""

import sys

from typing import override
from typing import Dict
from typing import Any

from streampersona_job import StreamPersonaJob

# Local relative imports
sys.path.append("..")  # noqa: E402
sys.path.append("../..")  # noqa: E402

from streamwise_job import StreamWiseJob
from streamwise_app import StreamWiseApp
from streamwise_app import run_app


class StreamPersonaApp(StreamWiseApp):
    """Quart app for StreamPersona persona generation."""

    def __init__(self) -> None:
        super().__init__("streampersona")

    @override
    def create_job(
        self,
        job_id: str,
        job_config: Dict[str, Any]
    ) -> StreamWiseJob:
        return StreamPersonaJob(
            job_id=job_id,
            config=job_config,
            service_manager=self.service_manager
        )


if __name__ == "__main__":
    run_app(
        StreamPersonaApp,
        tmp_dir="/tmp/streampersona",
        log_files=[
            "streamwise.log",
            "streampersona.log"
        ],
        app_name="StreamPersona",
    )
