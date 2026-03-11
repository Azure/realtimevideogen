"""
StreamAnimate job to generate an animated video.
"""
import sys

from typing import override
from typing import Any
from typing import Dict

# Local relative imports
sys.path.append("..")  # noqa: E402
sys.path.append("../..")  # noqa: E402

from streamwise_job import StreamWiseJob

from lmm_service_manager import LMMServiceManager


class StreamAnimateJob(StreamWiseJob):
    """A job to generate an animated video."""

    def __init__(
        self,
        job_id: str,
        service_manager: LMMServiceManager,
        config: Dict[str, Any] = {},
    ) -> None:
        super().__init__(
            "streamanimate",
            job_id,
            service_manager,
            config)

    @override
    async def generate(
        self,
        job_config: Dict[str, Any],
    ) -> None:
        # TODO
        raise NotImplementedError("Not implemented yet.")
