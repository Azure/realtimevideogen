import math
import sys
import os
import asyncio
import tempfile
import aiofiles

from PIL import Image

from typing import Any
from typing import Optional
from typing import Dict
from typing import List
from typing import AsyncGenerator
from typing import cast

from unittest.mock import patch
from unittest.mock import MagicMock

# Add current path
sys.path.append(os.getcwd())

from video import FANTASYTALKING_FPS
from video import HUNYUANFRAMEPACK_FPS

from file_utils import read_file_base64
from file_utils import save_base64_as_binary

from tests.torch_mock import TorchMock
from tests.k8s_mock import K8sMock
from tests.fantasytalking_mock import FantasyTalkingMock

mock_torch = TorchMock()
mock_k8s = K8sMock()
mock_ft = FantasyTalkingMock()

mock_modules = {
    "imageio": MagicMock(),
    "tabulate": MagicMock(),
    "soundfile": MagicMock(),
}
mock_modules.update(mock_torch.get_sub_modules())
mock_modules.update(mock_k8s.get_sub_modules())
mock_modules.update(mock_ft.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from media_utils import save_video_frames
    from media_utils import get_video_frames
    from media_utils import get_frame_with_text
    from media_utils import get_video_file_info
    from media_utils import save_video_audio
    from media_utils import get_audio_duration

    from apps.lmm_generator import LMMGenerator

    from apps.client import ServiceRequest
    from apps.client import RequestStatus


# Each mock function outputs a color for easy asserting
MOCK_COLORS = {
    "default": "pink",
    "gen_image": "white",
    "gen_edit_image": "lightgray",
    "gen_extract_characters": "gray",
    "gen_video": "blue",
    "gen_video_audio_from_img": "green",
    "gen_video_audio_from_video": "red",
    "gen_video_upscale": "yellow",
    "gen_video_from_latents": "purple",
    "gen_intermediate_video_frames": "orange",
}
MOCK_COLORS_RGB = {
    "default": (255, 192, 203),
    "gen_image": (255, 255, 255),
    "gen_edit_image": (211, 211, 211),
    "gen_extract_characters": (128, 128, 128),
    "gen_video": (0, 0, 255),
    "gen_video_audio_from_img": (0, 128, 0),
    "gen_video_audio_from_video": (253, 0, 0),
    "gen_video_upscale": (255, 255, 0),
    "gen_video_from_latents": (128, 0, 128),
    "gen_intermediate_video_frames": (255, 165, 0),
}


class LMMGeneratorMock(LMMGenerator):
    """Mock LMMGenerator for testing."""

    def __init__(
        self,
        *_: Any,
        **__: Any,
    ) -> None:
        self.app_name = "mock"
        self.job_id = "mock"

        self.request_executor = MagicMock()
        self.service_manager = MagicMock()
        # The mock uses a dict for O(1) lookup, while the base class uses a list for ordered
        # request tracking. Both serve the same role (request registry) but with different APIs.
        self.requests: Dict[str, ServiceRequest] = {}  # type: ignore[assignment]

    async def gen_podcast_transcript(
        self, *args: Any, **kwargs: Any
    ) -> AsyncGenerator[Dict[str, str], None]:
        yield {"type": "image", "content": "Image prompt."}
        yield {"type": "character", "name": "Jane", "gender": "Female"}
        yield {"type": "character", "name": "Joe", "gender": "Male"}
        yield {"type": "dialogue", "character": "Jane", "content": "Hello"}
        yield {"type": "dialogue", "character": "Joe", "content": "World"}
        yield {"type": "dialogue", "character": "Unknown", "content": "Unknown character"}
        yield {"type": "dialogue", "content": "No character"}
        yield {"type": "dialogue"}

    async def gen_image(  # type: ignore[override]
        self,
        *_: Any,
        width: int,
        height: int,
        **__: Any,
    ) -> Image.Image:
        return Image.new("RGB", (width, height), color=MOCK_COLORS["gen_image"])

    async def gen_edit_image(  # type: ignore[override]
        self,
        *_: Any,
        width: int,
        height: int,
        **__: Any,
    ) -> Image.Image:
        return Image.new("RGB", (width, height), color=MOCK_COLORS["gen_edit_image"])

    async def gen_extract_characters(
        self,
        image: Image.Image,
        *_: Any,
        num_characters: int = 2,
        **__: Any,
    ) -> List[Image.Image]:
        width, height = image.size
        return [
            Image.new("RGB", (width, height), color=MOCK_COLORS["gen_extract_characters"])
            for _ in range(num_characters)
        ]

    async def gen_audio(self, *args: Any, **kwargs: Any) -> str:
        mock_audio_path = "tests/data/audio_4675.wav"
        # TODO cut it to a size?
        return await read_file_base64(mock_audio_path)

    async def _gen_synthetic_video(
        self,
        num_frames: int = 81,
        width: int = 160,
        height: int = 100,
        fps: int = FANTASYTALKING_FPS,
        audio_path: Optional[str] = None,
        color: str = MOCK_COLORS["default"],
    ) -> bytes:
        """Synthetic test video."""
        frames: List[Image.Image] = [
            cast(Image.Image, get_frame_with_text(
                width,
                height,
                f"frame{idx:02d}",
                output_type="pil",
                background_color=color,
            ))
            for idx in range(num_frames)
        ]

        if audio_path and await aiofiles.os.path.exists(audio_path):
            video_path = await save_video_audio(
                frames,
                audio_path=audio_path,
                fps=fps
            )
        else:
            video_path = await save_video_frames(
                frames,
                fps=fps
            )

        async with aiofiles.open(video_path, "rb") as file:
            video_binary = await file.read()

        return video_binary

    async def gen_video(  # type: ignore[override]
        self,
        img: Image.Image,
        prompt: str,
        neg_prompt: str = "",
        width: int = 640,
        height: int = 400,
        num_frames: int = 81,
        wait_request: bool = True,
        task_id: Optional[str] = None,
        **_: Any,
    ) -> ServiceRequest | bytes:
        video_binary = await self._gen_synthetic_video(
            width=width,
            height=height,
            num_frames=num_frames,
            fps=HUNYUANFRAMEPACK_FPS,
            color=MOCK_COLORS["gen_video"],
        )

        if wait_request is True:
            return video_binary

        service_name = "hunyuanframepackf1"
        request = ServiceRequest(
            request_id=f"{self.job_id}_{task_id}_{service_name}",
            service_name=service_name,
            payload_json={
                "job_id": self.job_id,
                "img": b"AAAAA",  # This would come from img
                "width": width,
                "height": height,
                "num_frames": num_frames,
            },
        )
        request.status = RequestStatus.COMPLETED
        request.future = asyncio.Future()
        request.future.set_result(("video/mp4", video_binary))
        return request

    async def gen_video_audio_from_img(  # type: ignore[override]
        self,
        *,
        audio_base64: str,
        width: int = 640,
        height: int = 400,
        **_: Any,
    ) -> bytes:
        audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        audio_path = await save_base64_as_binary(audio_path, audio_base64)
        audio_duration = get_audio_duration(audio_base64)
        num_frames = (((math.ceil(audio_duration * FANTASYTALKING_FPS) - 1) // 4) * 4) + 1
        return await self._gen_synthetic_video(
            width=width,
            height=height,
            num_frames=num_frames,
            fps=FANTASYTALKING_FPS,
            audio_path=audio_path,
            color=MOCK_COLORS["gen_video_audio_from_img"],
        )

    async def gen_video_audio_from_video(  # type: ignore[override]
        self,
        video: List[Image.Image],
        audio_base64: str,
        prompt: str,
        neg_prompt: str = "",
        width: int = 640,
        height: int = 400,
        **_: Any,
    ) -> bytes:
        audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        audio_path = await save_base64_as_binary(audio_path, audio_base64)
        return await self._gen_synthetic_video(
            width=width,
            height=height,
            num_frames=len(video),
            fps=FANTASYTALKING_FPS,
            audio_path=audio_path,
            color=MOCK_COLORS["gen_video_audio_from_video"],
        )

    async def gen_video_upscale(  # type: ignore[override]
        self,
        *,
        video_binary: bytes,
        width: int = 640,
        height: int = 400,
        **_: Any,
    ) -> bytes:
        video_file_info = get_video_file_info(video_binary)
        video_info = video_file_info["video"]
        num_frames = video_info["num_frames"] or 81
        fps = int(video_info["fps"] or FANTASYTALKING_FPS)
        return await self._gen_synthetic_video(
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            color=MOCK_COLORS["gen_video_upscale"],
        )

    async def gen_video_from_latents(
        self,
        *_: Any,
        # latents: torch.Tensor,
        **__: Any,
    ) -> bytes:
        # TODO guess parameters from latents
        return await self._gen_synthetic_video(
            width=320,
            height=200,
            num_frames=81,
            # fps=FANTASYTALKING_FPS,
            color=MOCK_COLORS["gen_video_from_latents"],
        )

    async def gen_intermediate_video_frames(  # type: ignore[override]
        self,
        *_: Any,
        video_gen_request: ServiceRequest,
        **__: Any,
    ) -> AsyncGenerator[Image.Image, None]:
        assert video_gen_request.payload_json is not None
        video_binary = await self._gen_synthetic_video(
            width=video_gen_request.payload_json["width"],
            height=video_gen_request.payload_json["height"],
            num_frames=video_gen_request.payload_json["num_frames"],
            color=MOCK_COLORS["gen_intermediate_video_frames"],
        )
        frames = await get_video_frames(video_binary)
        for frame in frames:
            yield frame

    def get_requests(self) -> Dict[str, ServiceRequest]:
        return self.requests

    def get_queued_requests(self) -> List[str]:
        return []  # TODO

    async def gen_text(  # type: ignore[override]
        self,
        messages: List[Dict],
        *_: Any,
        **__: Any,
    ) -> str:
        """Return a canned text reply for testing."""
        return "This is a mock response."

    async def gen_audio_transcript(  # type: ignore[override]
        self,
        audio_path: str,
        *_: Any,
        **__: Any,
    ) -> tuple[str, str]:
        """Return a canned transcript for testing."""
        return ("Mock transcript text.", "en")


"""
async def _mock_generation() -> MagicMock:
    "" "
    Mock multi-modal generation components.
    "" "
    gen = MagicMock()

    gen.job_id = "mock_chunked_job_id"

    gen.job_path = f"/tmp/{gen.job_id}"
    os.makedirs(gen.job_path, exist_ok=True)

    loop = asyncio.get_running_loop()
    gen.image_task = loop.create_future()
    gen.image_task.set_result(Image.new("RGB", (640, 480), color="blue"))

    video_frames = [
        Image.new("RGB", (640, 480), color=color)
        for color in ["blue", "green", "red", "yellow"]
    ]
    video_path = await save_video_frames(video_frames)
    video_binary = await read_file_bytes(video_path)
    gen.gen_video_audio_from_img = AsyncMock(return_value=video_binary)

    gen.gen_video_audio_from_video = AsyncMock(return_value=video_binary)

    async def gen_video_mock(*args, **kwargs) -> Any:
        if kwargs.get("wait_request", None):
            return video_binary
        req = ServiceRequest(
            request_id="mock_request_id",
            service_name="hunyuanframepackf1",
            payload_json={"job_id": gen.job_id},
        )
        req.status = RequestStatus.COMPLETED
        req.future = asyncio.Future()
        req.future.set_result(("video/mp4", video_binary))
        return req

    gen.gen_video = AsyncMock(side_effect=gen_video_mock)
    return gen
"""
