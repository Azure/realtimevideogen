import pytest

from datetime import timedelta

from quart_utils import json_pretty_filter
from quart_utils import format_datetime
from quart_utils import format_bytes
from quart_utils import format_string
from quart_utils import format_duration
from quart_utils import format_gpu_model
from quart_utils import format_url
from quart_utils import get_aspect_ratio
from quart_utils import get_class_emoji
from quart_utils import get_file_type_emoji
from quart_utils import get_file_type
from quart_utils import get_mime_type
from quart_utils import get_content_type_emoji
from quart_utils import get_friendly_region_name
from quart_utils import get_friendly_container_name
from quart_utils import is_rtgen_container
from quart_utils import get_docker_image
from quart_utils import parse_request_id


def test_json_pretty_filter() -> None:
    assert json_pretty_filter("{}") == "{}"
    assert json_pretty_filter("{'one': 1}") == "{'one': 1}"
    assert json_pretty_filter("{'two': [1, 2]}") == "{'two': [1, 2]}"
    assert json_pretty_filter("{'three': [1, 2, 'three']}") == "{'three': [1, 2, 'three']}"


def test_format_datetime() -> None:
    assert format_datetime(0) == "1970-01-01 00:00:00"
    assert format_datetime(1672531199) == "2022-12-31 23:59:59"
    assert format_datetime("abc") == "Invalid date"  # type: ignore[arg-type]


def test_format_bytes() -> None:
    assert format_bytes(0) == '<span class="text-muted">-</span>'
    assert format_bytes(500) == "500 B"
    assert format_bytes(1024) == "1.0 KiB"
    assert format_bytes(1536) == "1.5 KiB"
    assert format_bytes(1048576) == "1.0 MiB"
    assert format_bytes(1073741824) == "1.0 GiB"
    assert format_bytes(1099511627776) == "1.0 TiB"
    assert format_bytes(1125899906842624) == "1024.0 TiB"


def test_format_string() -> None:
    assert format_string("hello_world") == "Hello World"
    assert format_string("Test_String_Example") == "Test String Example"
    assert format_string("") == ""
    assert format_string("no_underscores") == "No Underscores"
    assert format_string("multiple___underscores") == "Multiple   Underscores"
    assert format_string(None) is None
    assert format_string(123) == 123  # type: ignore[arg-type]


def test_format_duration() -> None:
    assert format_duration(None) == "0"
    assert format_duration(timedelta(seconds=0)) == "0"
    assert format_duration(timedelta(seconds=47)) == "47 seconds"
    assert format_duration(timedelta(minutes=2, seconds=30)) == "2 minutes 30 seconds"
    assert format_duration(timedelta(hours=1, minutes=15, seconds=5)) == "1 hour 15 minutes 5 seconds"
    assert format_duration(timedelta(hours=0, minutes=0, seconds=5)) == "5 seconds"
    assert format_duration(timedelta(hours=0, minutes=3, seconds=0)) == "3 minutes 0 seconds"
    assert format_duration(timedelta(hours=2, minutes=0, seconds=0)) == "2 hours 0 minutes 0 seconds"


def test_format_url() -> None:
    assert format_url(None) is None
    assert format_url("http://example.com") == "example.com"
    assert format_url("https://10.0.0.1:8080") == "10.0.0.1:8080"


def test_format_gpu_model() -> None:
    # Raw NVIDIA GPU model strings
    assert format_gpu_model("NVIDIA A100-SXM4-80GB") == "A100 80GB"
    assert format_gpu_model("NVIDIA-A100-SXM4-80GB") == "A100 80GB"
    assert format_gpu_model("NVIDIA-A100-80GB-PCIe") == "A100 80GB"
    assert format_gpu_model("NVIDIA A100 80GB PCIe") == "A100 80GB"
    assert format_gpu_model("NVIDIA-H200") == "H200"
    assert format_gpu_model("NVIDIA H200") == "H200"
    assert format_gpu_model("NVIDIA-H100") == "H100"
    assert format_gpu_model("NVIDIA H100") == "H100"
    assert format_gpu_model("NVIDIA-H100-NVL") == "H100 NVL"
    assert format_gpu_model("NVIDIA H100 NVL") == "H100 NVL"
    assert format_gpu_model("NVIDIA-H100-80GB-HBM3") == "H100"
    assert format_gpu_model("NVIDIA H100 80GB HBM3") == "H100"
    assert format_gpu_model("Tesla-V100-PCIE-16GB") == "V100 16GB"
    assert format_gpu_model("Tesla V100-PCIE-16GB") == "V100 16GB"
    assert format_gpu_model("Tesla-V100-SXM2-32GB") == "V100 32GB"
    assert format_gpu_model("Tesla V100-SXM2-32GB") == "V100 32GB"
    # Azure VM SKU names — ND series (ND_A100_v4)
    assert format_gpu_model("Standard_ND96ams_A100_v4") == "A100 80GB"
    assert format_gpu_model("Standard_ND96amsr_A100_v4") == "A100 80GB"
    # Azure VM SKU names — ND series (ND_H100_v5)
    assert format_gpu_model("Standard_ND96isrf_H100_v5") == "H100"
    assert format_gpu_model("Standard_ND96isr_H100_v5") == "H100"
    # Azure VM SKU names — ND series (ND_MI300X_v5)
    assert format_gpu_model("Standard_ND96isr_MI300X_v5") == "MI300X"
    # Azure VM SKU names — ND series (ND_H200_v5)
    assert format_gpu_model("Standard_ND96isr_H200_v5") == "H200"
    # Azure VM SKU names — ND series (ND_GB200_v6 / ND_GB300_v6)
    assert format_gpu_model("Standard_ND128isr_GB300_v6") == "GB300"
    # Azure VM SKU names — NC series (NC_A100_v4)
    assert format_gpu_model("Standard_NC96ads_A100_v4") == "A100 80GB"
    # Azure VM SKU names — NC series (NCasT4_v3)
    assert format_gpu_model("Standard_NC4as_T4_v3") == "T4"
    assert format_gpu_model("Standard_NC8as_T4_v3") == "T4"
    assert format_gpu_model("Standard_NC64as_T4_v3") == "T4"
    # Azure VM SKU names — NC series (NCads_H100_v5)
    assert format_gpu_model("Standard_NC40ads_H100_v5") == "H100"
    # Azure VM SKU names — NV series (NVads_A10_v5)
    assert format_gpu_model("Standard_NV18ads_A10_v5") == "A10"
    assert format_gpu_model("Standard_NV36ads_A10_v5") == "A10"
    assert format_gpu_model("Standard_NV72ads_A10_v5") == "A10"
    # GB200 and GB300 raw strings
    assert format_gpu_model("NVIDIA GB200") == "GB200"
    assert format_gpu_model("NVIDIA-GB200") == "GB200"
    assert format_gpu_model("NVIDIA GB300") == "GB300"
    assert format_gpu_model("NVIDIA-GB300") == "GB300"
    # Azure VM SKU — unknown GPU identifier falls back to the GPU part (uppercased)
    assert format_gpu_model("Standard_NC6s_UnknownGPU_v3") == "UNKNOWNGPU"
    # Edge cases
    assert format_gpu_model("Some Other GPU") == "Some Other GPU"
    assert format_gpu_model("") == ""
    assert format_gpu_model(None) is None
    assert format_gpu_model(123) == 123  # type: ignore[arg-type]


def test_get_aspect_ratio() -> None:
    assert get_aspect_ratio(1) == "1:1"
    assert get_aspect_ratio(16 / 9) == "16:9"
    assert get_aspect_ratio(1280 / 720) == "16:9"
    assert get_aspect_ratio(1.77) == "16:9"
    assert get_aspect_ratio(16 / 10) == "16:10"
    assert get_aspect_ratio(1280 / 800) == "16:10"
    assert get_aspect_ratio(4 / 3) == "4:3"
    assert get_aspect_ratio(5 / 4) == "5:4"
    assert get_aspect_ratio(1280 / 1024) == "5:4"
    assert get_aspect_ratio(3 / 2) == "3:2"
    assert get_aspect_ratio(2 / 1) == "2:1"
    assert get_aspect_ratio(0) == "0.00:1"
    assert get_aspect_ratio(-1) == "-1.00:1"


@pytest.mark.asyncio
async def test_get_class_emoji() -> None:
    assert await get_class_emoji(None) == ""
    assert await get_class_emoji("flux") == "📄→🖼️"
    assert await get_class_emoji("hunyuanframepackf1") == "📄🖼️→🎬"
    assert await get_class_emoji("fantasytalking") == "📄🖼️🔉→🎬"
    assert await get_class_emoji("hunyuanframepackvae") == "🔢→🎬"
    assert await get_class_emoji("wan") == "📄🖼️→🎬"
    assert await get_class_emoji("kokoro") == "📄→🔉"
    assert await get_class_emoji("thinksound") == "🎬→🔉"
    assert await get_class_emoji("yolo") == "🖼️→🖼️"
    assert await get_class_emoji("mola") == "<span class='text-muted' title='mola'>❓</span>"
    assert await get_class_emoji("podcasttranscript") == "📄→📄"
    assert await get_class_emoji("gemma") == "📄→📄"


def test_get_file_type_emoji() -> None:
    assert get_file_type_emoji("image") == "🖼️"
    assert get_file_type_emoji("kernel") == "📊"
    assert get_file_type_emoji("tensor") == "📊"
    assert get_file_type_emoji("directory") == "📁"
    assert get_file_type_emoji("text") == "📄"
    assert get_file_type_emoji("archive") == "📦"
    assert get_file_type_emoji("audio") == "🎵"
    assert get_file_type_emoji("video") == "🎥"
    assert get_file_type_emoji("mola") == "❓ mola"
    assert get_file_type_emoji(None) == "❓"


def test_get_file_type() -> None:
    """Check the file type detection."""
    assert get_file_type("example.png") == "image"
    assert get_file_type("example.pdf") == "pdf"
    assert get_file_type("example.jsonl") == "jsonl"
    assert get_file_type("image.jpg") == "image"
    assert get_file_type("image.png") == "image"
    assert get_file_type("video.mp4") == "video"
    assert get_file_type("audio.mp3") == "audio"
    assert get_file_type("audio.wav") == "audio"
    assert get_file_type("document.pdf") == "pdf"
    assert get_file_type("document.pt") == "tensor"
    assert get_file_type("document.ptx") == "kernel"
    assert get_file_type("document.json") == "json"
    assert get_file_type("document.jsonl") == "jsonl"
    assert get_file_type("document.txt") == "text"
    assert get_file_type("unknown.xyz") == "unknown"


def test_get_mime_type() -> None:
    """Check the MIME type detection."""
    assert get_mime_type("example.png") == "image/png"
    assert get_mime_type("example.mp4") == "video/mp4"
    assert get_mime_type("example.wav") == "audio/x-wav"
    assert get_mime_type("image.jpg") == "image/jpeg"
    assert get_mime_type("image.png") == "image/png"
    assert get_mime_type("video.mp4") == "video/mp4"
    assert get_mime_type("audio.mp3") == "audio/mpeg"
    assert get_mime_type("audio.wav") == "audio/x-wav"
    assert get_mime_type("log.log") == "text/plain"
    assert get_mime_type("document.pdf") == "application/pdf"
    # assert get_mime_type("document.pt") == "application/octet-stream"
    # assert get_mime_type("document.ptx") == "application/octet-stream"
    assert get_mime_type("document.json") == "application/json"
    assert get_mime_type("document.jsonl") == "application/x-ndjson"
    assert get_mime_type("document.txt") == "text/plain"


def test_get_content_type_emoji() -> None:
    assert get_content_type_emoji("image/png") == "🖼️"
    assert get_content_type_emoji("video/mp4") == "🎥"
    assert get_content_type_emoji("text/plain") == "📄"
    assert get_content_type_emoji("application/json") == "📄"
    assert get_content_type_emoji("application/octet-stream") == "📦"
    assert get_content_type_emoji("application/mola") == "❓ application/mola"


def test_get_friendly_region_name() -> None:
    assert get_friendly_region_name("eastus2") == "East US 2"
    assert get_friendly_region_name("southeastasia") == "Southeast Asia"
    assert get_friendly_region_name("eastasia") == "East Asia"
    assert get_friendly_region_name("centralus") == "Central US"
    assert get_friendly_region_name("mola") == "Mola"
    assert get_friendly_region_name(None) == "N/A"


@pytest.mark.asyncio
async def test_get_friendly_container_name() -> None:
    assert await get_friendly_container_name("flux") == "FLUX"
    assert await get_friendly_container_name("fantasytalking") == "Fantasy Talking"
    assert await get_friendly_container_name("qwenimage") == "Qwen Image"
    assert await get_friendly_container_name("mola") == "mola"


@pytest.mark.asyncio
async def test_is_rtgen_container() -> None:
    assert await is_rtgen_container("flux") is True
    assert await is_rtgen_container("qwenimage") is True
    assert await is_rtgen_container("nonexistent") is False


@pytest.mark.asyncio
async def test_get_docker_image() -> None:
    docker_image = await get_docker_image("flux")
    assert docker_image is not None
    assert "/flux:v" in docker_image

    docker_image = await get_docker_image("llamagen")
    assert docker_image is not None
    assert "/llamagen:v" in docker_image

    docker_image = await get_docker_image("nonexistent")
    assert docker_image is None


@pytest.mark.asyncio
async def test_get_docker_image_custom_tag() -> None:
    docker_image = await get_docker_image("flux", tag="custom-tag")
    assert docker_image is not None
    assert docker_image.endswith(":custom-tag")
    assert "/flux:custom-tag" in docker_image

    docker_image = await get_docker_image("flux", tag="v9.9.9")
    assert docker_image is not None
    assert docker_image.endswith(":v9.9.9")

    # Custom tag on nonexistent container should still return None
    docker_image = await get_docker_image("nonexistent", tag="custom-tag")
    assert docker_image is None


def test_parse_request_id() -> None:
    assert parse_request_id("20250904T010335296_006_001_fantasytalking") == {
        'job_id': '20250904T010335296',
        'scene_id': '006',
        'service_name': 'fantasytalking',
        'sub_scene_id': '001',
        "task_id": "",
    }

    assert parse_request_id("20250904T010335296_flux") == {
        'job_id': '20250904T010335296',
        'scene_id': "",
        'service_name': 'flux',
        'sub_scene_id': "",
        "task_id": "",
    }

    assert parse_request_id("20260105T194652416_main_image_flux") == {
        'job_id': '20260105T194652416',
        'scene_id': "",
        'service_name': "flux",
        'sub_scene_id': "",
        "task_id": "main_image"
    }

    assert parse_request_id("") == {}
