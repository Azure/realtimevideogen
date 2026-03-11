"""
A mock for Fantasy Talking modules used in testing.
"""

from unittest.mock import MagicMock

from typing import Dict
from typing import Any


class FantasyTalkingMock(MagicMock):

    def get_sub_modules(self) -> Dict[str, Any]:
        return {
            "torchvision": MagicMock(),
            "torchvision.transforms": MagicMock(),
            "torchvision.transforms.functional": MagicMock(),
            "xfuser": MagicMock(),
            "xfuser.config": MagicMock(),
            "xfuser.core": MagicMock(),
            "xfuser.core.distributed": MagicMock(),
            "diffsynth": MagicMock(),
            "diffsynth.models": MagicMock(),
            "diffsynth.models.wan_video_dit": MagicMock(),
            "wan.distributed": MagicMock(),
            "wan.distributed.xdit_context_parallel": MagicMock(),
            "wan.distributed.fsdp": MagicMock(),
            "transformers": MagicMock(),
            "model": MagicMock(),
            "utils": MagicMock(),
        }
