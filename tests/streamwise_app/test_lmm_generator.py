#!/usr/bin/env python3
"""
Unit tests for LMM Generator.
"""

import os
import sys
import pytest

from PIL import Image

from unittest.mock import patch
from unittest.mock import MagicMock

from pydantic import ValidationError

# Add current path
sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path

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
    from file_utils import read_file_base64

    with temp_sys_path("apps"):
        from apps.lmm_generator import LMMGenerator

        from apps.client import ServiceRequest
        from apps.client import RequestStatus


@pytest.mark.asyncio
async def test_gen_image() -> None:
    service_manager = MagicMock()
    job_id = "gen_image"
    gen = LMMGenerator("streamwise_app", job_id, service_manager)
    try:
        # with pytest.raises(ServiceError):
        with pytest.raises(Exception, match="Service 'flux' not found"):
            await gen.gen_image(
                prompt="Test prompt",
                task_id="test_gen_image")
    finally:
        await gen.stop()
        del gen
        del service_manager


@pytest.mark.asyncio
async def test_gen_edit_image() -> None:
    service_manager = MagicMock()
    job_id = "gen_edit_image"
    gen = LMMGenerator("streamwise_app", job_id, service_manager)
    try:
        # with pytest.raises(ServiceError):
        with pytest.raises(Exception, match="Service 'fluxkontext' not found"):
            await gen.gen_edit_image(
                prompt="Test prompt",
                image=Image.new('RGB', (100, 100)),
                task_id="test_gen_edit_image")
    finally:
        await gen.stop()
        del gen
        del service_manager


@pytest.mark.asyncio
async def test_gen_extract_characters() -> None:
    service_manager = MagicMock()
    job_id = "job_test_gen_extract_characters"
    gen = LMMGenerator("streamwise_app", job_id, service_manager)
    try:
        # with pytest.raises(ServiceError):
        with pytest.raises(Exception, match="Service 'yolo' not found"):
            await gen.gen_extract_characters(
                image=Image.new('RGB', (100, 100)),
                task_id="gen_extract_characters")
    finally:
        await gen.stop()
        del gen
        del service_manager


@pytest.mark.asyncio
async def test_gen_video() -> None:
    service_manager = MagicMock()
    job_id = "job_test_gen_video"
    gen = LMMGenerator("streamwise_app", job_id, service_manager)
    try:
        # with pytest.raises(ServiceError):
        with pytest.raises(Exception, match="Service 'hunyuanframepackf1' not found"):
            await gen.gen_video(
                prompt="Test prompt",
                img=Image.new('RGB', (100, 100)),
                task_id="test_gen_video")
    finally:
        await gen.stop()
        del gen
        del service_manager


@pytest.mark.asyncio
async def test_gen_video_audio_from_img() -> None:
    service_manager = MagicMock()
    job_id = "job_test_gen_video_audio_from_img"
    gen = LMMGenerator("streamwise_app", job_id, service_manager)
    try:
        audio_base64 = await read_file_base64("tests/data/audio_4675.wav")
        # with pytest.raises(ServiceError):
        with pytest.raises(Exception, match="Service 'fantasytalking' not found"):
            await gen.gen_video_audio_from_img(
                prompt="Test prompt",
                audio_base64=audio_base64,
                img=Image.new('RGB', (100, 100)),
                task_id="test")
    finally:
        await gen.stop()
        del gen
        del service_manager


@pytest.mark.asyncio
async def test_gen_video_audio_from_video() -> None:
    service_manager = MagicMock()
    job_id = "job_test_gen_video_audio_from_video"
    gen = LMMGenerator("streamwise_app", job_id, service_manager)
    try:
        audio_base64 = await read_file_base64("tests/data/audio_4675.wav")
        # with pytest.raises(ServiceError):
        with pytest.raises(Exception, match="Service 'fantasytalking' not found"):
            await gen.gen_video_audio_from_video(
                prompt="Test prompt",
                audio_base64=audio_base64,
                video=[
                    Image.new('RGB', (100, 100)),
                    Image.new('RGB', (100, 100)),
                    Image.new('RGB', (100, 100)),
                ],
                task_id="test")
    finally:
        await gen.stop()
        del gen
        del service_manager


@pytest.mark.asyncio
async def test_get_video_last_latents() -> None:
    service_manager = MagicMock()
    job_id = "job_test_get_video_last_latents"
    gen = LMMGenerator("streamwise_app", job_id, service_manager)
    try:
        index, latents = await gen.get_video_last_latents(
            base_url="http://test:8080",
            task_id="test_get_video_last_latents")
        assert index == -1
        assert latents is None
    finally:
        await gen.stop()
        del gen
        del service_manager


@pytest.mark.asyncio
async def test_gen_video_from_latents() -> None:
    service_manager = MagicMock()
    job_id = "job_test_gen_video_from_latents"
    gen = LMMGenerator("streamwise_app", job_id, service_manager)
    try:
        with pytest.raises(ValidationError):
            await gen.gen_video_from_latents(
                latents=mock_torch.randn(1, 4, 8, 8),
                task_id="test")
    finally:
        await gen.stop()
        del gen
        del service_manager


@pytest.mark.asyncio
async def test_gen_intermediate_video_frames() -> None:
    job_id = "gen_intermediate_video_frames"
    service_manager = MagicMock()
    gen = LMMGenerator("streamwise_app", job_id, service_manager)
    try:
        request = ServiceRequest(
            request_id="request_gen_video",
            service_name="video-gen",
            payload_bytes=b"test")
        request.status = RequestStatus.COMPLETED

        async for frame in gen.gen_intermediate_video_frames(
            base_url="http://test:8080",
            video_gen_request=request,
            task_id="test_gen_intermediate_video_frames"
        ):
            assert frame is not None
    finally:
        await gen.stop()
        del gen
        del service_manager


@pytest.mark.asyncio
async def test_gen_image_upscale() -> None:
    service_manager = MagicMock()
    job_id = "gen_image_upscale"
    gen = LMMGenerator("streamwise_app", job_id, service_manager)
    try:
        # with pytest.raises(ServiceError):
        with pytest.raises(Exception, match="Service 'realesrgan' not found"):
            await gen.gen_image_upscale(
                image=Image.new('RGB', (40, 40)),
                task_id="test_gen_image_upscale")
    finally:
        await gen.stop()
        del gen
        del service_manager


@pytest.mark.asyncio
async def test_gen_video_upscale() -> None:
    service_manager = MagicMock()
    job_id = "gen_video_upscale"
    gen = LMMGenerator("streamcast", job_id, service_manager)
    try:
        # with pytest.raises(ServiceError):
        with pytest.raises(Exception, match="Service 'realesrgan' not found"):
            await gen.gen_video_upscale(
                video_binary=b"AAAAAAAA",
                task_id="test_gen_video_upscale")
    finally:
        await gen.stop()
        del gen
        del service_manager
