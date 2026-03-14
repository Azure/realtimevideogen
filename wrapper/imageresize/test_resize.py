#!/usr/bin/env python3
"""CI Docker test: verify the imageresize service can resize images."""
import sys
import asyncio
from unittest.mock import MagicMock
from PIL import Image

# Mock packages not needed for image resizing
mock_torch = MagicMock()
mock_torch.distributed = MagicMock()
mock_torch.cuda.memory_allocated.return_value = 0.0
mock_torch.cuda.is_available.return_value = False
sys.modules['torch'] = mock_torch
sys.modules['torch.distributed'] = mock_torch.distributed
sys.modules['nvidia_smi'] = MagicMock()
sys.modules['imageio'] = MagicMock()
sys.modules['imageio_ffmpeg'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['numpy'] = MagicMock()


from wrapper_imageresize import ImageResize  # noqa: E402


async def main() -> None:
    model = ImageResize()
    model.init()

    img = Image.new('RGB', (100, 100), color='red')

    result = await model.generate(image=img, width=200, height=200)
    assert result is not None, "Result should not be None"
    assert result.size == (200, 200), f"Expected (200, 200), got {result.size}"
    print(f"PASS: image resized from (100, 100) to {result.size}")

    video = [Image.new('RGB', (50, 25), color='blue'), Image.new('RGB', (50, 25), color='blue')]
    frames = await model.generate(image=None, video=video, width=100, height=50)
    assert frames is not None, "Frames should not be None"
    assert len(frames) == 2, f"Expected 2 frames, got {len(frames)}"
    assert frames[0].size == (100, 50), f"Expected frame size (100, 50), got {frames[0].size}"
    print(f"PASS: video frames resized from (50, 25) to {frames[0].size}")


if __name__ == "__main__":
    asyncio.run(main())
