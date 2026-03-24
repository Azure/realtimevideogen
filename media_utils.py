import os
import torch
import tempfile
import math
import logging
import wave
import asyncio
import subprocess
import json
import re
import aiofiles
import aiofiles.os
import textwrap

import imageio
# from imageio import formats

import imageio_ffmpeg as ffmpeg

from contextlib import asynccontextmanager

import numpy as np

from typing import List
from typing import Sequence
from typing import TypedDict
from typing import Optional
from typing import Union
from typing import Dict
from typing import Tuple
from typing import Any
from typing import AsyncIterator

from aiofiles.threadpool.binary import AsyncBufferedIOBase

from io import BytesIO
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from file_utils import read_file_bytes
from file_utils import base64_to_binary
from file_utils import save_base64_as_binary
from file_utils import binary_to_base64

PIX_FMT_RGB24 = "rgb24"

# DEFAULT_VIDEO_CODEC = "libx264rgb"
# DEFAULT_PIX_FMT = "rgb24"
DEFAULT_VIDEO_CODEC = "libx264"
DEFAULT_AUDIO_CODEC = "aac"
DEFAULT_PIX_FMT = "yuv420p"


def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Converts a PyTorch tensor to a base64-encoded string."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor for tensor, got {type(tensor)}")
    buffer = BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    tensor_bytes = buffer.read()
    base64_str = binary_to_base64(tensor_bytes)
    return base64_str


def bytes_to_tensor(binary_data: bytes) -> torch.Tensor:
    """Converts binary data to a PyTorch tensor."""
    if not isinstance(binary_data, bytes):
        raise TypeError(f"Expected bytes for binary_data, got {type(binary_data)}")
    buffer = BytesIO(binary_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.load(buffer, map_location=device, weights_only=True)
    return tensor


def base64_to_tensor(base64_str: str) -> torch.Tensor:
    """Converts a base64-encoded string to a PyTorch tensor."""
    if not isinstance(base64_str, str):
        raise TypeError(f"Expected str for base64_str, got {type(base64_str)}")
    tensor_bytes = base64_to_binary(base64_str)
    tensor = bytes_to_tensor(tensor_bytes)
    return tensor


def base64_to_video_frames(
    video_base64: str,
    video_format: str = "mp4",
) -> List[Image.Image]:
    """Converts a base64-encoded string to a list of PIL Image frames."""
    if not isinstance(video_base64, str):
        raise TypeError(f"Expected str for video_base64, got {type(video_base64)}")
    video_bytes = base64_to_binary(video_base64)
    video_buffer = BytesIO(video_bytes)
    # fmt: imageio.Format = formats[video_format]
    video_frames = imageio.get_reader(
        video_buffer,
        format=video_format)  # type: ignore[arg-type]
    frames = [
        Image.fromarray(frame_np).convert("RGB")
        for frame_np in video_frames  # type: ignore[attr-defined]
    ]
    return frames


def video_frames_to_base64(
    video_frames: List[Image.Image],
    fps: float = 30.0,
    format: str = "mp4"
) -> str:
    """Converts a list of PIL Image frames to a base64-encoded video string."""
    if not isinstance(video_frames, list):
        raise TypeError(f"Expected list for video_frames, got {type(video_frames)}")
    if len(video_frames) <= 0:
        raise ValueError("video_frames cannot be empty")
    video_buffer = BytesIO()
    # fmt: imageio.Format = formats[format]
    with imageio.get_writer(
        video_buffer,
        format=format,  # type: ignore[arg-type]
        fps=fps
    ) as writer:
        for frame in video_frames:
            frame_np = np.array(frame)
            writer.append_data(frame_np)  # type: ignore[attr-defined]
    video_bytes = video_buffer.getvalue()
    video_base64 = binary_to_base64(video_bytes)
    return video_base64


def fix_json_like_string(s: str) -> str:
    """Add double quotes around keys (only if not already quoted)."""
    s = re.sub(r'([{,]\s*)(\w+)(\s*:\s*)', r'\1"\2"\3', s)
    return s


def chunk_audio_base64(
    audio_base64: str,
    start_seconds: float = 0.0,
    end_seconds: float = 2.0,
) -> str:
    """Chunk audio base64 between start_seconds and end_seconds."""
    if not isinstance(audio_base64, str):
        raise TypeError(f"Expected str for audio_base64, got {type(audio_base64)}")

    import soundfile

    audio_bytes = base64_to_binary(audio_base64)
    audio_buffer = BytesIO(audio_bytes)
    audio_data, sample_rate = soundfile.read(audio_buffer)
    start_sample = int(start_seconds * sample_rate)
    end_sample = int(end_seconds * sample_rate)
    if not (0 <= start_sample < end_sample <= len(audio_data)):
        raise ValueError(
            f"Invalid trim range {start_sample} to {end_sample} sample for audio with {len(audio_data)} samples.")
    trimmed_audio_data = audio_data[start_sample:end_sample]
    trimmed_buffer = BytesIO()
    soundfile.write(
        trimmed_buffer,
        trimmed_audio_data,
        sample_rate,
        format="WAV")
    trimmed_buffer.seek(0)
    return binary_to_base64(trimmed_buffer.getvalue())


def chunk_video_binary(
    video_binary: bytes,
    start_seconds: float = 0.0,
    end_seconds: float = 2.0,
    video_codec: str = DEFAULT_VIDEO_CODEC,
    audio_codec: str = DEFAULT_AUDIO_CODEC,
    pix_fmt: str = DEFAULT_PIX_FMT,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> bytes:
    """Chunk video binary between start_seconds and end_seconds."""
    if not video_binary:
        raise ValueError("Video input is empty.")
    if not isinstance(video_binary, bytes):
        raise TypeError(f"Expected bytes for video binary, got {type(video_binary)}")
    if start_seconds < 0:
        raise ValueError("start_seconds must be non-negative")
    if end_seconds <= start_seconds:
        raise ValueError("end_seconds must be greater than start_seconds")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as input_file, \
         tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as output_file:
        try:
            # Write input bytes
            input_file.write(video_binary)
            input_file.flush()

            # Run ffmpeg to trim and re-encode
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-y",  # Overwrite created output file
                "-ss", str(start_seconds),
                "-to", str(end_seconds),
                "-i", input_file.name,
            ]

            if width and height:
                cmd += ["-vf", f"scale={width}:{height}"]

            cmd += [
                "-c:v", video_codec,
                "-c:a", audio_codec,
                "-pix_fmt", pix_fmt,
                output_file.name
            ]
            subprocess.run(cmd, check=True)

            # Return trimmed video as bytes
            output_file.seek(0)
            return output_file.read()
        finally:
            # Clean up temporary files
            for f in (input_file.name, output_file.name):
                try:
                    os.remove(f)
                except OSError:
                    pass


def get_audio_duration(
    audio_content: Union[str, bytes]
) -> float:
    """Get audio duration in seconds from binary content or base64 string."""
    if isinstance(audio_content, bytes):
        audio_bytes = audio_content
    elif isinstance(audio_content, str):
        audio_base64 = audio_content
        audio_bytes = base64_to_binary(audio_base64)
    else:
        raise TypeError(f"Expected str|bytes for audio_content, got {type(audio_content)}")
    if not isinstance(audio_bytes, bytes):
        raise TypeError(f"Expected str|bytes for audio_content, got {type(audio_content)}")

    import soundfile

    audio_buffer = BytesIO(audio_bytes)
    audio_data, sample_rate = soundfile.read(audio_buffer)
    duration_seconds = len(audio_data) / sample_rate
    return duration_seconds


class AudioFileInfo(TypedDict):
    num_frames: int
    samplerate: int
    channels: int
    duration_seconds: float


def get_audio_file_info(
    audio_path: str
) -> AudioFileInfo:
    """Get audio file info from file path."""
    if not isinstance(audio_path, str):
        raise TypeError(f"Expected str for audio_path, got {type(audio_path)}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    import soundfile

    info = soundfile.info(audio_path)
    return {
        "num_frames": info.frames,
        "samplerate": info.samplerate,
        "channels": info.channels,
        "duration_seconds": info.frames / info.samplerate
    }


class OverallInfo(TypedDict):
    duration_seconds: float
    bitrate: Optional[int]
    num_bytes: int


class VideoInfo(TypedDict, total=False):
    codec: Optional[str]
    pix_fmt: Optional[str]
    bitrate: Optional[int]
    num_frames: Optional[int]
    fps: Optional[float]
    duration_seconds: Optional[float]
    width: Optional[int]
    height: Optional[int]
    aspect_ratio: Optional[float]


class AudioInfo(TypedDict, total=False):
    codec: Optional[str]
    bitrate: Optional[int]
    sample_rate: Optional[int]
    channels: Optional[int]
    duration_seconds: Optional[float]


class VideoFileInfo(TypedDict, total=False):
    overall: OverallInfo
    video: VideoInfo
    audio: AudioInfo


def get_video_file_info(
    video_content: Union[bytes, str]
) -> VideoFileInfo:
    """Get video file info from binary content or file path."""
    if not video_content:
        raise ValueError("Video input is empty.")
    if isinstance(video_content, bytes):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_video:
            video_binary = video_content
            temp_video.write(video_binary)
            temp_video.flush()
            video_path = temp_video.name
            return get_video_file_info_path(video_path)
    if isinstance(video_content, str):
        video_path = video_content
        return get_video_file_info_path(video_path)
    raise TypeError(f"Expected bytes|str for video content, got {type(video_content)}")


def get_video_file_info_path(
    video_path: str
) -> VideoFileInfo:
    """Get video file info from file path."""
    if not isinstance(video_path, str):
        raise TypeError(f"Expected str for video_path, got {type(video_path)}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
    cmd = [
        "ffprobe",
        "-v", "error",
        "-count_frames",
        "-show_entries",
        (
            "stream=codec_name,codec_type,avg_frame_rate,nb_frames,nb_read_frames,"
            "duration,width,height,bit_rate,sample_rate,channels,pix_fmt"
        ),
        "-show_entries",
        "format=duration,bit_rate,size",
        "-of", "json",
        video_path,
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True)
    info: Dict[str, Any] = json.loads(result.stdout)

    if "format" not in info:
        ffprobe_error = result.stderr.strip()
        logging.error(f"No format found in ffprobe error: {ffprobe_error}")
        raise ValueError("The video binary is corrupted or has an unsupported format")

    overall_info: OverallInfo = {
        "duration_seconds": float(info["format"].get("duration", 0)),
        "bitrate": int(info["format"].get("bit_rate", 0)) if "bit_rate" in info["format"] else None,
        "num_bytes": int(info["format"].get("size", 0))
    }
    file_info: VideoFileInfo = {
        "overall": overall_info
    }

    if "streams" not in info:
        logging.error(f"No streams found in ffprobe output: {result.stdout}")

    # Video info
    video_stream = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if video_stream:
        fps_val: Optional[float] = None
        fps = video_stream.get("avg_frame_rate", "0/0")
        if fps != "0/0":
            num, den = map(int, fps.split("/"))
            fps_val = num / den if den else None
        num_frames: Optional[int] = None
        if "nb_frames" in video_stream and video_stream["nb_frames"].isdigit():
            num_frames = int(video_stream["nb_frames"])
        elif "nb_read_frames" in video_stream and video_stream["nb_read_frames"].isdigit():
            num_frames = int(video_stream["nb_read_frames"])

        video_info: VideoInfo = {
            "codec": video_stream.get("codec_name"),
            "pix_fmt": video_stream.get("pix_fmt"),
            "bitrate": int(video_stream["bit_rate"]) if "bit_rate" in video_stream else None,
            "num_frames": num_frames,
            "fps": fps_val,
            # "duration_seconds": float(video_stream.get("duration", 0)),
        }
        # Formats like MKV may not have duration in stream
        if "duration" in video_stream and float(video_stream["duration"]) > 0:
            video_info["duration_seconds"] = float(video_stream["duration"])

        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        if width > 0 and height > 0:
            video_info["width"] = width
            video_info["height"] = height
            video_info["aspect_ratio"] = width / height

        file_info["video"] = video_info

    # Audio info
    audio_stream = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    if audio_stream:
        audio_info: AudioInfo = {
            "codec": audio_stream.get("codec_name"),
            "bitrate": int(audio_stream["bit_rate"]) if "bit_rate" in audio_stream else None,
            "sample_rate": int(audio_stream["sample_rate"]) if "sample_rate" in audio_stream else None,
            "channels": int(audio_stream["channels"]) if "channels" in audio_stream else None,
        }
        # Formats like MKV may not have duration in stream
        if "duration" in audio_stream and float(audio_stream["duration"]) > 0:
            audio_info["duration_seconds"] = float(audio_stream["duration"])
        file_info["audio"] = audio_info

    return file_info


class ImageFileInfo(TypedDict):
    width: int
    height: int
    aspect_ratio: float


def get_image_file_info(
    image_path: str
) -> ImageFileInfo:
    if not isinstance(image_path, str):
        raise TypeError(f"Expected str for image_path, got {type(image_path)}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    img = Image.open(image_path)
    return {
        "width": img.width,
        "height": img.height,
        "aspect_ratio": img.width / img.height,
    }


class TextFileInfo(TypedDict):
    num_chars: int
    num_words: int
    num_lines: int


def get_text_file_info(
    text_path: str
) -> TextFileInfo:
    if not isinstance(text_path, str):
        raise TypeError(f"Expected str for text_path, got {type(text_path)}")
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"Text file does not exist: {text_path}")
    with open(text_path, 'r', encoding="utf-8", errors="ignore") as f:
        content = f.read()
    lines = content.split('\n')
    words = content.split()
    return {
        "num_chars": len(content),
        "num_words": len(words),
        "num_lines": len(lines),
    }


class TensorFileInfo(TypedDict):
    dtype: str
    shape: str
    device: str
    mean: str
    min: str
    max: str
    numel: int


def get_tensor_file_info(
    tensor_path: str
) -> TensorFileInfo:
    if not isinstance(tensor_path, str):
        raise TypeError(f"Expected str for tensor_path, got {type(tensor_path)}")
    if not os.path.exists(tensor_path):
        raise FileNotFoundError(f"Tensor file does not exist: {tensor_path}")
    tensor = torch.load(tensor_path, weights_only=True)
    return {
        "dtype": str(tensor.dtype),
        "shape": str(tensor.shape),
        "device": str(tensor.device),
        "mean": str(tensor.mean().item()),
        "min": str(tensor.min().item()),
        "max": str(tensor.max().item()),
        "numel": tensor.numel(),
    }


async def base64_to_audio_file(
    audio_base64: str,
    audio_path: Optional[str] = None
) -> str:
    if not isinstance(audio_base64, str):
        raise TypeError(f"Expected str for audio_base64, got {type(audio_base64)}")
    if audio_path is None:
        audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    return await save_base64_as_binary(
        audio_path,
        audio_base64)


def empty_audio_file(
    duration_seconds: float = 0.2,
    sample_rate: int = 44100,  # KHz
    num_channels: int = 1,
    sample_width: int = 2
) -> str:
    n_samples = int(sample_rate * duration_seconds)
    silence = np.zeros(n_samples, dtype=np.int16)  # 16-bit PCM
    audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    with wave.open(audio_path, "w") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(silence.tobytes())
    return audio_path


def fit_audio_to_duration(
    input_path: str,
    target_duration: float,
    output_path: Optional[str] = None
) -> str:
    if not isinstance(input_path, str):
        raise TypeError(f"Expected str for input_path, got {type(input_path)}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input WAV file does not exist: {input_path}")
    if not input_path.lower().endswith(".wav"):
        raise ValueError(f"Input file must be a WAV file: {input_path}")
    if target_duration <= 0:
        raise ValueError("target_duration must be positive")

    with wave.open(input_path, "rb") as wf:
        params = wf.getparams()
        framerate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        nframes = wf.getnframes()
        frames = wf.readframes(nframes)

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sampwidth]
    audio = np.frombuffer(frames, dtype=dtype)

    if not output_path:
        output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

    target_samples = int(target_duration * framerate * n_channels)

    if len(audio) > target_samples:
        audio = audio[:target_samples]  # Truncate
    elif len(audio) < target_samples:
        silence = np.zeros(target_samples - len(audio), dtype=dtype)
        audio = np.concatenate([audio, silence])  # Pad with silence

    with wave.open(output_path, "wb") as wf:
        wf.setparams(params._replace(nframes=target_samples))
        wf.writeframes(audio.tobytes())
    return output_path


@asynccontextmanager
async def async_tempfile(
    suffix: str = "",
    delete: bool = True
) -> AsyncIterator[AsyncBufferedIOBase]:
    """Asynchronous context manager for a temporary file."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)  # close fd so aiofiles can open
    try:
        file = await aiofiles.open(path, mode="w+b")
        try:
            yield file
        finally:
            await file.close()
    finally:
        if delete and await aiofiles.os.path.exists(path):
            await aiofiles.os.remove(path)


async def get_video_frame(
    video_binary: bytes,
    frame_index: int = 0  # Get first frame by default
) -> Image.Image:
    """Extracts a specific frame from a video binary using imageio-ffmpeg."""
    if not isinstance(video_binary, bytes):
        raise TypeError(f"Expected bytes for video binary, got {type(video_binary)}")
    if len(video_binary) == 0:
        raise ValueError("Video binary is empty.")
    if frame_index < 0:
        raise ValueError("Frame index must be non-negative.")

    video_frames = await get_video_frames(video_binary)
    if frame_index >= len(video_frames):
        raise ValueError(f"Frame index {frame_index} exceeds number of frames {len(video_frames)}.")
    return video_frames[frame_index]


def get_video_frames_at_fps(
    video_frames: List[Image.Image],
    src_fps: float = 30.0,  # Hunyuan FramePack
    dst_fps: float = 23.0,  # Fantasy Talking
) -> List[Image.Image]:
    """
    Adjusts the frame rate of a list of video frames by either up-sampling or down-sampling.
    Args:
        video_frames (List[Image.Image]): List of input frames.
        src_fps (int): Original frames per second.
        dst_fps (int): Desired frames per second.
    Returns:
        List[Image.Image]: New list of frames adjusted to the target FPS.
    """
    if not video_frames:
        return []
    if src_fps <= 0 or dst_fps <= 0:
        raise ValueError("FPS must be a positive integer.")
    if src_fps == dst_fps:
        return video_frames  # No FPS change needed

    num_src_frames = len(video_frames)
    duration_secs = 1.0 * num_src_frames / src_fps
    # target_frame_count = int(math.ceil(duration_secs * dst_fps))
    target_frame_count = int(round(duration_secs * dst_fps))

    new_frames = []
    for ix in range(target_frame_count):
        t = 1.0 * ix / dst_fps  # target time in seconds
        src_index = min(int(t * src_fps), num_src_frames - 1)
        video_frame = video_frames[src_index]
        new_frames.append(video_frame)

    return new_frames


def get_ffmpeg_version() -> str:
    # Use dpkg -s ffmpeg to get version on Ubuntu
    # Example: Version: 7:4.2.7-0ubuntu0.1
    if os.path.exists("/usr/bin/dpkg"):
        result = subprocess.run(
            ["dpkg", "-s", "ffmpeg"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True)
        for line in result.stdout.split('\n'):
            if line.startswith("Version:"):
                return line.split("Version:")[1].strip()

    # Otherwise use ffmpeg -version
    result = subprocess.run(
        ["ffmpeg", "-version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True)
    first_line = result.stdout.split('\n')[0]
    # Extract from line: ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023 the FFmpeg developers
    if not first_line.startswith("ffmpeg version"):
        raise ValueError(f"Unexpected ffmpeg version output: {first_line}")
    version = first_line.split(" ")[2].strip()
    return version


def change_video_fps(
    video_binary: bytes,
    new_fps: float,
    video_codec: str = DEFAULT_VIDEO_CODEC,
    audio_codec: str = DEFAULT_AUDIO_CODEC,
    pix_fmt: str = DEFAULT_PIX_FMT,
) -> bytes:
    if not isinstance(video_binary, bytes):
        raise TypeError(f"Expected bytes for video binary, got {type(video_binary)}")
    if new_fps <= 0:
        raise ValueError("FPS must be a positive integer.")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as input_file, \
         tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as output_file:
        input_file.write(video_binary)
        input_file.flush()

        subprocess.run([
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",  # Overwrite created output file
            "-i", input_file.name,
            "-filter:v", f"fps={new_fps}",
            "-c:v", video_codec,
            "-c:a", audio_codec,
            "-pix_fmt", pix_fmt,
            output_file.name
        ], check=True)

        output_file.seek(0)
        return output_file.read()


def get_video_size(
    video_frames: List[Image.Image]
) -> int:
    """
    Get approximate size in bytes of a list of video frames.
    This is a rough estimate based on width, height, and RGB channels.
    """
    if not video_frames:
        return 0
    ret = 0
    NUM_RGB_CHANNELS = 3
    for frame in video_frames:
        if frame is not None:
            ret += frame.width * frame.height * NUM_RGB_CHANNELS
    return ret


async def get_video_frames(
    video_binary: Optional[bytes],
    extend_last_frame: bool = False
) -> List[Image.Image]:
    """
    Extracts frames from a video binary using imageio-ffmpeg.
    This is slow.
    """
    if not video_binary:
        return []
    if not isinstance(video_binary, bytes):
        raise TypeError(f"Expected bytes for video binary, got {type(video_binary)}")
    if len(video_binary) == 0:
        raise ValueError("Video binary is empty.")

    NUM_RGB_CHANNELS = 3

    async with async_tempfile(suffix=".mp4") as temp_video:
        await temp_video.write(video_binary)
        await temp_video.flush()
        video_path = temp_video.name

        reader = ffmpeg.read_frames(video_path, pix_fmt=PIX_FMT_RGB24)
        meta = reader.__next__()
        # 'ffmpeg_version', 'codec', 'pix_fmt', 'fps', 'source_size', 'size', 'rotate', 'duration'
        width, height = meta["size"]

        frames: List[Image.Image] = []
        for frame_bytes in reader:
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame_array = frame_array.reshape((height, width, NUM_RGB_CHANNELS))
            frame_img_pil = Image.fromarray(frame_array)
            frames.append(frame_img_pil)

        # Extend last frame if needed
        # expected_num_frames = int(math.ceil(meta["fps"] * meta["duration"]))
        expected_num_frames = int(round(meta["fps"] * meta["duration"]))
        if len(frames) < expected_num_frames:
            logging.warning(
                f"Video has {len(frames)} frames but expected {expected_num_frames} "
                f"({meta['fps']}x{meta['duration']}). "
                "Exending last frame.")
            if extend_last_frame:
                frames.extend([frames[-1]] * (expected_num_frames - len(frames)))

        return frames


async def get_video_duration(
    video_content: Union[bytes, str]
) -> float:
    """Get video duration in seconds from binary content or file path."""
    video_file_info = get_video_file_info(video_content)
    video_info = video_file_info.get("video", {})
    result = video_info.get("duration_seconds", -1.0)
    return result if result is not None else -1.0


def get_video_fps(
    video_content: Union[bytes, str]
) -> float:
    """Get video frames per second (FPS) from binary content or file path."""
    video_file_info = get_video_file_info(video_content)
    video_info = video_file_info.get("video", {})
    result = video_info.get("fps", -1.0)
    return result if result is not None else -1.0


async def get_video_num_frames(
    video_content: Union[bytes, str]
) -> int:
    """Get number of video frames from binary content or file path."""
    video_file_info = get_video_file_info(video_content)
    video_info = video_file_info.get("video", {})
    result = video_info.get("num_frames", -1)
    return result if result is not None else -1


def save_audio(
    audio: torch.Tensor,
    audio_path: str,
    sample_rate: int = 24000
) -> str:
    if not isinstance(audio, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor for audio, got {type(audio)}")
    if not isinstance(audio_path, str):
        raise TypeError(f"Expected str for audio_path, got {type(audio_path)}")
    import soundfile
    soundfile.write(
        audio_path,
        audio,
        sample_rate,
        format="WAV",
        subtype="PCM_16")
    return audio_path


def save_audio_np(
    wav: np.ndarray,
    path: str,
    sample_rate: Optional[int] = None
) -> None:
    import scipy

    # https://github.com/coqui-ai/TTS/blob/main/TTS/utils/audio/numpy_transforms.py#L430
    FREQ_32KHz = (32 * 1024) - 1
    wav_norm = wav * (FREQ_32KHz / max(0.01, np.max(np.abs(wav))))

    wav_norm = wav_norm.astype(np.int16)  # PCM16
    scipy.io.wavfile.write(path, sample_rate, wav_norm)


async def save_video_frames(
    video_frames: Union[Sequence[Union[Image.Image, np.ndarray, torch.Tensor]], np.ndarray],
    out_video_path: Optional[str] = None,
    fps: float = 30.0,
    video_codec: str = DEFAULT_VIDEO_CODEC,
    pix_fmt: str = DEFAULT_PIX_FMT,
    output_format: str = "mp4",  # "mkv"
) -> str:
    """Save video frames and optional audio to a file asynchronously using ffmpeg."""
    if not video_frames:
        raise ValueError("No video frames provided.")
    if not isinstance(video_frames, (list, np.ndarray)):
        raise TypeError(f"Expected list or np.ndarray for video_frames, got {type(video_frames)}")

    frame0 = video_frames[0]
    if isinstance(frame0, np.ndarray):
        width, height = frame0.shape[1], frame0.shape[0]
    else:
        width, height = frame0.size

    if not out_video_path:
        out_video_path = tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False).name

    num_frames = len(video_frames)
    duration = float(num_frames) / float(fps)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",  # Overwrite created output file
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",  # input frames from stdin
        "-frames:v", str(num_frames),
        "-t", str(duration),
        "-c:v", video_codec,
        "-pix_fmt", pix_fmt,
        "-preset", "fast",  # "slow",
        out_video_path
    ]
    cmd = [c for c in cmd if c]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL)
    if proc.stdin is None:
        raise RuntimeError("Failed to open ffmpeg stdin for writing frames.")

    for frame in video_frames[:num_frames]:
        if isinstance(frame, Image.Image):
            frame_array = np.array(frame.convert("RGB"))
        elif isinstance(frame, np.ndarray):
            frame_array = frame
        elif isinstance(frame, torch.Tensor):
            frame_array = frame.cpu().numpy()
        else:
            frame_array = frame
        frame_bytes = frame_array.tobytes()
        proc.stdin.write(frame_bytes)
    proc.stdin.close()
    await proc.wait()

    return out_video_path


async def extract_audio_from_video(
    video_path: str,
    out_audio_path: Optional[str] = None,
) -> str:
    """
    Extract audio from video.
    It is forced to WAV format for compatibility.
    44100 Hz, 16-bit PCM, stereo.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    if not out_audio_path:
        out_audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",  # Overwrite created output file
        "-i", video_path,
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # Convert audio to WAV
        "-ar", "44100",
        "-ac", "2",
        out_audio_path
    ]
    cmd = [c for c in cmd if c]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL)
    await proc.wait()

    return out_audio_path


async def save_video_audio(
    video_content: Union[bytes, str, List[Image.Image], np.ndarray],
    audio_path: str,
    fps: float = 30.0,
    out_video_path: Optional[str] = None,
    video_codec: str = DEFAULT_VIDEO_CODEC,
    audio_codec: str = DEFAULT_AUDIO_CODEC,
    output_format: str = "mp4",  # "mkv"
) -> str:
    """Merge audio and video."""
    if isinstance(video_content, (list, np.ndarray)):
        video_frames = video_content
        video_path = await save_video_frames(
            video_frames=video_frames,
            fps=fps,
            video_codec=video_codec,
            out_video_path=None,
            pix_fmt=DEFAULT_PIX_FMT,
            output_format=output_format)
    elif isinstance(video_content, bytes):
        video_binary = video_content
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
            temp_video.write(video_binary)
            temp_video.flush()
            video_path = temp_video.name
    elif isinstance(video_content, str):
        video_path = video_content
    else:
        raise TypeError(f"Expected bytes|str|list|np.ndarray for video_content, got {type(video_content)}")

    if not await aiofiles.os.path.exists(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
    if not await aiofiles.os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    if not out_video_path:
        out_video_path = tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False).name

    video_info = get_video_file_info_path(video_path)
    video_duration = video_info["video"]["duration_seconds"]

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",  # Overwrite created output file
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", audio_codec,
    ]
    cmd += [
        "-af", f"apad,atrim=0:{video_duration}",  # Align with the video
        "-shortest",
    ]
    cmd += [
        out_video_path
    ]
    cmd = [c for c in cmd if c]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL)
    await proc.wait()

    return out_video_path


def save_diffusers_video(
    video_frames: List[Image.Image],
    out_video_path: Optional[str] = None,
    fps: float = 30.0,
    quality: float = 5.0,
    bitrate: Optional[int] = None,
    macro_block_size: Optional[int] = 16,
) -> str:
    """
    Save a list of PIL Image frames as a video using the diffusers library.
    https://github.com/huggingface/diffusers/blob/main/src/diffusers/utils/export_utils.py#L141
    """
    if not video_frames:
        raise ValueError("No video frames provided.")
    if out_video_path is None:
        out_video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    with imageio.get_writer(
        out_video_path, fps=fps, quality=quality, bitrate=bitrate,
        macro_block_size=macro_block_size
    ) as writer:
        for frame in video_frames:
            frame_np = np.array(frame)
            writer.append_data(frame_np)  # type: ignore[attr-defined]
    return out_video_path


def save_bcthw_as_mp4(
    x: torch.Tensor,
    output_filename: str,
    fps: float = 30.0,
    video_codec: str = DEFAULT_VIDEO_CODEC,
) -> str:
    import einops

    # Version with no torchvision warning from
    # https://github.com/lllyasviel/FramePack/blob/main/diffusers_helper/utils.py#L266
    b, c, t, h, w = x.shape
    per_row = b
    for p in [6, 5, 4, 3, 2]:
        if b % p == 0:
            per_row = p
            break
    x = (torch.clamp(x.float(), -1., 1.) + 1) * 255 / 2
    x = x.detach().cpu().to(torch.uint8)
    x = einops.rearrange(x, '(m n) c t h w -> t (m h) (n w) c', n=per_row)
    # torchvision.io.write_video(output_filename, x, fps=fps, video_codec=codec, options={'crf': str(int(crf))})
    with imageio.get_writer(output_filename, fps=fps, codec=video_codec) as writer:
        for frame in x.numpy():
            writer.append_data(frame)  # type: ignore[attr-defined]
    return output_filename


def get_aligned_duration(
    duration: float,
    fps: float = 30.0,
    vae: int = 4
) -> float:
    """
    Aligns the duration to be compatible with diffusion models that use a VAE and FPS.
    This ensures that the number of frames is a multiple of the VAE's downsampling factor.
    VAE does: 1+4n
    """
    if duration == 0:
        return 0.0
    if duration < 0:
        raise ValueError("Duration must be positive.")
    if fps <= 0:
        raise ValueError("FPS must be positive.")
    if vae <= 0:
        raise ValueError("VAE must be positive.")
    num_frames = duration * fps
    lat_num_frames = int(math.ceil((num_frames - 1) / vae)) + 1
    aligned_num_frames = (lat_num_frames - 1) * vae + 1
    aligned_duration = aligned_num_frames / fps
    return aligned_duration


async def concatenate_videos(
    video_inputs: Union[List[bytes], List[str]],
    fast_copy: bool = True,
    video_codec: str = DEFAULT_VIDEO_CODEC,
    audio_codec: str = DEFAULT_AUDIO_CODEC,
    pix_fmt: str = DEFAULT_PIX_FMT,
) -> bytes:
    """
    Concatenate multiple videos (given as raw bytes) into a single video.
    Returns the final video as bytes.
    """
    if not video_inputs:
        raise ValueError("No videos provided")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_paths = []
        for i, video_input in enumerate(video_inputs):
            if isinstance(video_input, str):
                input_path = video_input
                if not await aiofiles.os.path.exists(input_path):
                    raise FileNotFoundError(f"Video file does not exist: {input_path}")
            elif isinstance(video_input, bytes):
                video_bytes = video_input
                input_path = f"{tmpdir}/input_{i}.mp4"
                async with aiofiles.open(input_path, "wb") as file:
                    await file.write(video_bytes)
            else:
                raise TypeError(f"Video input {i} is not bytes or str. Found: {type(video_input)}")
            input_paths.append(input_path)

        list_file = f"{tmpdir}/videos.txt"
        async with aiofiles.open(list_file, "w") as f:
            for path in input_paths:
                await f.write(f"file '{path}'\n")

        output_path = f"{tmpdir}/output.mp4"

        # Build ffmpeg command with stream copy (fast)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",  # Overwrite created output file
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
        ]
        if fast_copy:
            # This is fast but may leave the audio and video out of sync
            cmd += [
                "-c", "copy",
            ]
        else:
            cmd += [
                "-c:v", video_codec,
                "-c:a", audio_codec,
                "-pix_fmt", pix_fmt,
            ]
        cmd += [
            str(output_path)
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            err_msg = stderr.decode()
            if "Invalid data found when processing input" in err_msg:
                raise ValueError("One of the inputs is corrupted or has an unsupported format.")
            raise RuntimeError(f"ffmpeg failed: {err_msg}")

        concatenated_binary = await read_file_bytes(output_path)

        return concatenated_binary


DEFAULT_FONT = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def get_font_size(width: int, height: int) -> int:
    """Get the font size based on the frame dimensions."""
    if width >= 1280 or height >= 800:
        return 32
    if width >= 640 or height >= 400:
        return 24
    return 16


def get_font(
    font_path: str = DEFAULT_FONT,
    font_size: int = 32
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a truetype font, or default if not found."""
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        # apt-get install fonts-noto-color-emoji
        logging.warning(f"Could not load font at {font_path}. Using default font.")
        font = ImageFont.load_default()
    return font


def get_frame_with_text(
    width: int,
    height: int,
    text: str,
    output_type: str = "numpy",
    font_color: str = "white",
    font_size: int = 32,
    background_color: str = "black",
) -> Union[np.ndarray, torch.Tensor, Image.Image]:
    """
    Create a blank frame with multi-line text (including emojis) centered.
    Frame shape: [h, w, RGB].
    """
    # Convert to PIL for better text rendering
    img_pil = Image.new("RGB", (width, height), color=background_color)
    draw = ImageDraw.Draw(img_pil)

    font = get_font(font_size=font_size)

    # Handle multi-line text
    lines = text.split("\n")
    line_sizes = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        line_sizes.append((w, h))

    total_height = sum(h for _, h in line_sizes) + (len(lines) - 1) * 5  # line spacing

    # Start y so that block is vertically centered
    y = (height - total_height) // 2
    for line, (tw, th) in zip(lines, line_sizes):
        x = (width - tw) // 2
        draw.text((x, y), line, font=font, fill=font_color)
        y += th + 5

    # Convert back to desired format
    if output_type == "torch":
        return torch.from_numpy(np.array(img_pil)).permute(2, 0, 1)
    if output_type == "pil":
        return img_pil.convert("RGB")
    return np.array(img_pil)  # numpy


async def get_video_with_text(
    width: int,
    height: int,
    text: str,
    duration_seconds: float = 2.0,
    fps: float = 30.0,
    font_size: int = 32,
    font_color: str = "white",
) -> bytes:
    """
    Create a video with a single frame containing text.
    Returns the video as bytes.
    """
    video_frame = get_frame_with_text(
        width, height,
        text,
        font_size=font_size,
        font_color=font_color)
    # num_frames = int(math.ceil(duration_seconds * fps))
    num_frames = int(round(duration_seconds * fps))
    video_frames = [video_frame] * num_frames
    video_path = await save_video_frames(
        video_frames=video_frames,
        fps=fps)
    video_bytes = await read_file_bytes(video_path)
    if await aiofiles.os.path.exists(video_path):
        await aiofiles.os.unlink(video_path)
    return video_bytes


def get_text_position_coordinates(
    position: str,
    img_width: int,
    img_height: int,
    text_width: int,
    text_height: int,
    margin: int = 10
) -> Tuple[int, int]:
    """Get the (x, y) coordinates for placing text on an image based on the specified position."""
    if position == "top-left":
        x, y = margin, margin
    elif position == "top-right":
        x = img_width - text_width - margin
        y = margin
    elif position == "bottom-left":
        x = margin
        y = img_height - text_height - margin
    elif position == "bottom-right":
        x = img_width - text_width - margin
        y = img_height - text_height - margin
    elif position == "bottom-center":
        x = (img_width - text_width) // 2
        y = img_height - text_height - margin
    elif position == "center":
        x = (img_width - text_width) // 2
        y = (img_height - text_height) // 2
    else:
        raise ValueError(
            f"Invalid position: {position}. "
            f"Must be one of 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'center'.")
    return x, y


def split_text_lines(
    input_text: str,
    max_line_length: int
) -> List[str]:
    """Split text into multiple lines with a maximum line length."""
    return textwrap.wrap(
        input_text,
        width=max_line_length)


def add_text_to_frame(
    frame: Union[np.ndarray, torch.Tensor, Image.Image],
    text: str,
    font_size: int = 32,
    font_color: str = "white",
    position: Union[str, Tuple[int, int]] = "top-left",
) -> Union[np.ndarray, torch.Tensor, Image.Image]:
    """
    Add text to the frame.
    Frame should be [h, w, RGB].
    """
    if not text:
        logging.warning("No text provided to add to frame.")
        return frame
    if "\n" in text:
        logging.warning("Text contains newlines.")
        text = text.replace("\n", " ")

    output_type = "numpy"
    if isinstance(frame, torch.Tensor):
        output_type = "torch"
        frame_torch: torch.Tensor = frame
        frame_np = frame_torch.numpy()
        frame = frame_np.astype(np.uint8)
    if isinstance(frame, Image.Image):
        output_type = "pil"
        frame = np.array(frame.convert("RGB")).astype(np.uint8)
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"Unsupported frame type: {type(frame)}.")

    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)

    font = get_font(font_size=font_size)

    if isinstance(position, str):
        x, y = get_text_position_coordinates(
            position,
            img_width=img_pil.width,
            img_height=img_pil.height,
            text_width=int(draw.textlength(text, font=font)),
            text_height=font_size,
            margin=10)
    elif isinstance(position, tuple) and len(position) == 2:
        x, y = position
    else:
        raise ValueError("Position must be a string or (x, y) coordinates.")
    draw.text((x, y), text, font=font, fill=font_color)

    ret_frame = np.array(img_pil).astype(np.uint8)
    if output_type == "torch":
        return torch.from_numpy(ret_frame).permute(2, 0, 1)
    if output_type == "pil":
        return Image.fromarray(ret_frame).convert("RGB")
    return np.array(ret_frame)  # np.ndarray of shape (height, width, 3) with RGB values
