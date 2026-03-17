from typing import Any
from typing import Dict
from typing import Optional

from wrapper_model import ModelGeneration

from PIL import Image


class MockGeneration(ModelGeneration):
    """Mock model generation for testing purposes."""

    def __init__(self) -> None:
        super().__init__("mock")

    async def warmup(self) -> None:
        await self.generate()

    async def generate(
        self,
        job_id: Optional[str] = None,
        *args: Any,
        **kwargs: Dict[str, Any]
    ) -> Any:
        gen_timer = self._new_gen_timer(job_id)

        self.running = True

        try:
            if "output_type" not in kwargs:
                return None
            width = kwargs.get("width", 128)
            height = kwargs.get("height", 64)
            if kwargs["output_type"] == "pil":
                return Image.new("RGB", (int(width), int(height)), color='blue')
            if kwargs["output_type"] == "jsonl":
                return '{"mock": "data"}'
            if kwargs["output_type"] == "audio_path":
                return "TODO audio_path"
            if kwargs["output_type"] == "video_binary":
                return "TODO video_binary"
            if kwargs["output_type"] == "video_path":
                return "TODO video_path"
            if kwargs["output_type"] == "tensor":
                return "TODO tensor"
            return kwargs["output_type"]
        finally:
            gen_timer.end("total")
            self.running = False

    async def get_rest_args(
        self,
        data_json: Dict[str, str]
    ) -> Dict[str, Any]:
        return {}
