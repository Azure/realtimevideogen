"""
StreamMovie: Generate a movie.
Starts an HTTP server to accept job submissions and monitor job status.
"""

import sys

from typing import override
from typing import Dict
from typing import Any

from streammovie_job import StreamMovieJob

# Local relative imports
sys.path.append("..")  # noqa: E402
sys.path.append("../..")  # noqa: E402

from streamwise_job import StreamWiseJob
from streamwise_app import StreamWiseApp
from streamwise_app import run_app


class StreamMovieApp(StreamWiseApp):
    """Quart app for StreamMovie movie generation."""

    def __init__(self) -> None:
        super().__init__("streammovie")

    @override
    def create_job(
        self,
        job_id: str,
        job_config: Dict[str, Any]
    ) -> StreamWiseJob:
        return StreamMovieJob(
            job_id=job_id,
            config=job_config,
            service_manager=self.service_manager
        )


if __name__ == "__main__":
    run_app(
        StreamMovieApp,
        tmp_dir="/tmp/streammovie",
        log_files=[
            "streamwise.log",
            "streammovie.log"
        ],
        app_name="StreamMovie",
    )
