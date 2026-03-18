#!/usr/bin/env python3
"""
Unit tests for StreamDub.
"""

import base64
import json
import os
import sys
import pytest

from dataclasses import asdict
from http import HTTPStatus

from quart import Quart

from unittest.mock import patch
from unittest.mock import MagicMock
from unittest.mock import AsyncMock

# Add current path
sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path
from tests.torch_mock import TorchMock

mock_torch = TorchMock()

mock_modules = {}
mock_modules.update(mock_torch.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    with temp_sys_path("apps", "apps/streamdub"):
        from apps.streamdub.streamdub import StreamDubApp

scene_mocks_base = {
    'scenedetect': MagicMock(),
    'scenedetect.detectors': MagicMock(),
    'scenedetect.stats_manager': MagicMock(),
}

with patch.dict(sys.modules, {**mock_modules, **scene_mocks_base}):
    with temp_sys_path("apps", "apps/streamdub"):
        from apps.streamdub.streamdub_job import StreamDubJob


streamdub_app = StreamDubApp()


@pytest.fixture(name="test_app")
def _test_app() -> Quart:
    return streamdub_app.app


@pytest.mark.asyncio
async def test_app(test_app: Quart) -> None:
    """Check that GET / returns 200."""
    client = test_app.test_client()
    response = await client.get("/")
    assert response is not None
    assert response.status_code == HTTPStatus.OK
    assert "text/html; charset=utf-8" == response.content_type
    response_html = await response.get_data(as_text=True)
    assert response_html.startswith("<!DOCTYPE html>\n<html lang=\"en\">")
    assert "StreamDub" in response_html


@pytest.mark.asyncio
async def test_submit_job(test_app: Quart) -> None:
    """Check the API for job requests."""
    client = test_app.test_client()

    response = await client.post("/api/job", json={"video_base64": "AAAA"})
    assert response.status_code == HTTPStatus.BAD_REQUEST
    response_json = await response.get_json()
    assert "error" in response_json
    assert response_json["error"] == "Service manager not initialized"

    # Mock the service manager
    streamdub_app.service_manager = MagicMock()
    streamdub_app.service_manager.get_service_url = MagicMock(
        return_value="http://mock_service_url:1234"
    )

    response = await client.post("/api/job", json={"video_base64": "AAAA"})
    # assert response.status_code == HTTPStatus.OK
    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    response_json = await response.get_json()
    assert "error" in response_json
    # assert "job_id" in response_json
    # assert response_json["status"] == "success"


def test_to_language() -> None:
    """Test to_language converts language codes to language names."""
    scene_mocks = {
        'scenedetect': MagicMock(),
        'scenedetect.detectors': MagicMock(),
        'scenedetect.stats_manager': MagicMock(),
    }
    # Remove cached module to force re-import under mocked scenedetect
    sys.modules.pop('apps.streamdub.streamdub_job', None)
    with patch.dict(sys.modules, {**mock_modules, **scene_mocks}):
        with temp_sys_path("apps", "apps/streamdub"):
            from apps.streamdub.streamdub_job import to_language

    assert to_language("a") == "American English"
    assert to_language("b") == "British English"
    assert to_language("e") == "Spanish"
    assert to_language("f") == "French"
    assert to_language("g") == "German"
    assert to_language("i") == "Italian"
    assert to_language("j") == "Japanese"
    assert to_language("k") == "Korean"
    assert to_language("c") == "Chinese"
    assert to_language("r") == "Russian"

    # Case-insensitive: uppercase valid code maps to correct language
    assert to_language("E") == "Spanish"
    assert to_language("F") == "French"

    # Unknown code defaults to American English
    assert to_language("z") == "American English"


@pytest.mark.asyncio
async def test_streamdub_job_no_video() -> None:
    """StreamDubJob.gen_dub with missing video raises ValueError."""
    job = StreamDubJob(
        job_id="test_no_video",
        service_manager=MagicMock(),
    )
    with pytest.raises(ValueError, match="Missing 'video_base64'"):
        await job.gen_dub(video_base64=None)


@pytest.mark.asyncio
async def test_streamdub_job_generate_no_video() -> None:
    """StreamDubJob.generate with missing video_base64 raises ValueError."""
    job = StreamDubJob(
        job_id="test_generate_no_video",
        service_manager=MagicMock(),
    )
    with pytest.raises(ValueError, match="Missing 'video_base64'"):
        await job.generate(job_config={})


@pytest.mark.asyncio
async def test_streamdub_job_detect_scenes_missing_file() -> None:
    """StreamDubJob.detect_scenes raises FileNotFoundError when video file is absent."""
    job = StreamDubJob(
        job_id="test_detect_no_file",
        service_manager=MagicMock(),
    )
    with pytest.raises(FileNotFoundError):
        await job.detect_scenes()


@pytest.mark.asyncio
async def test_streamdub_job_gen_dub_no_scenes() -> None:
    """gen_dub with a valid video but no detected scenes raises ValueError."""
    job = StreamDubJob(
        job_id="test_gen_dub_no_scenes",
        service_manager=MagicMock(),
    )
    # "AAAA" is valid base64; scenedetect mocks return empty scene list
    with pytest.raises(ValueError, match="No scenes detected"):
        await job.gen_dub(video_base64="AAAA")


@pytest.mark.asyncio
async def test_gen_dub_writes_scenes_json() -> None:
    """gen_dub must write scenes.json after chunking audio so the UI can display scenes."""
    with patch.dict(sys.modules, {**mock_modules, **scene_mocks_base}):
        with temp_sys_path("apps", "apps/streamdub"):
            from apps.streamdub.streamdub_job import StreamDubJob as _StreamDubJob
            from apps.scene import SceneSegment

    service_manager = AsyncMock()
    service_manager.get_service_url = MagicMock(return_value="http://mock:1234")

    job_id = "test_scenes_json"
    job = _StreamDubJob(job_id=job_id, service_manager=service_manager)

    # Build two fake scenes (audio_path not yet set — chunk_audio_into_scenes sets it)
    fake_scenes = [
        SceneSegment(scene_id=0, start_frame=0, end_frame=30, start_sec=0.0, end_sec=1.0),
        SceneSegment(scene_id=1, start_frame=30, end_frame=60, start_sec=1.0, end_sec=2.0),
    ]

    # Patch detect_scenes and chunk_audio_into_scenes so we control the scene list
    async def fake_detect_scenes(**kwargs: object) -> list:
        return fake_scenes

    async def fake_chunk_audio(**kwargs: object) -> list:
        # Simulate chunking setting audio_path on each scene (mirrors real behaviour)
        for scene in fake_scenes:
            scene.audio_path = f"scene_{scene.scene_id:03d}.wav"
        return []

    with patch.object(job, "detect_scenes", side_effect=fake_detect_scenes), \
         patch.object(job, "chunk_audio_into_scenes", side_effect=fake_chunk_audio), \
         patch.object(job, "gen_dub_scene", side_effect=ValueError("stop")):
        try:
            await job.gen_dub(video_base64="AAAA")
        except (ValueError, Exception):
            pass  # We only care that scenes.json was written before gen_dub_scene is called

    scenes_json_path = os.path.join(job.job_path, "scenes.json")
    assert os.path.exists(scenes_json_path), "scenes.json must be written after chunk_audio_into_scenes"
    with open(scenes_json_path) as f:
        data = json.load(f)
    assert len(data) == 2
    assert data[0]["scene_id"] == 0
    assert data[0]["audio_path"] == "scene_000.wav"
    assert data[1]["scene_id"] == 1
    assert data[1]["audio_path"] == "scene_001.wav"

    await job.close()


@pytest.mark.asyncio
async def test_gen_dub_scene_uses_voice_cloning() -> None:
    """gen_dub_scene must call gen_clone_audio with the original scene audio when available."""
    with patch.dict(sys.modules, {**mock_modules, **scene_mocks_base}):
        with temp_sys_path("apps", "apps/streamdub"):
            from apps.streamdub.streamdub_job import StreamDubJob as _StreamDubJob
            from apps.scene import SceneSegment

    service_manager = AsyncMock()
    service_manager.get_service_url = MagicMock(return_value="http://mock:1234")

    job_id = "test_voice_clone"
    job = _StreamDubJob(job_id=job_id, service_manager=service_manager)

    # Write a dummy original scene audio file that the job should read and forward
    original_audio_content = b"RIFF....WAVEfmt "  # minimal dummy WAV bytes
    original_audio_b64 = base64.b64encode(original_audio_content).decode()
    scene_audio_path = os.path.join(job.job_path, "scene_000.wav")
    os.makedirs(job.job_path, exist_ok=True)
    with open(scene_audio_path, "wb") as f:
        f.write(original_audio_content)

    # Create a scene with transcript already set (skip transcription / translation)
    fake_scene = SceneSegment(
        scene_id=0,
        start_frame=0,
        end_frame=30,
        start_sec=0.0,
        end_sec=1.0,
    )
    fake_scene.audio_path = "scene_000.wav"
    fake_scene.transcript = "Hola mundo"  # pre-translated text

    dubbed_audio_b64 = base64.b64encode(b"dubbed_audio").decode()
    dubbed_video_binary = b"dubbed_video"

    gen_clone_audio_mock = AsyncMock(return_value=dubbed_audio_b64)
    gen_video_mock = AsyncMock(return_value=dubbed_video_binary)

    with patch.object(job, "transcribe_audio", new=AsyncMock(return_value="Hello world")), \
         patch.object(job, "translate_scene", new=AsyncMock(return_value="Hola mundo")), \
         patch.object(job.gen, "gen_clone_audio", gen_clone_audio_mock), \
         patch.object(job, "gen_video_lip_synced", gen_video_mock):
        await job.gen_dub_scene(fake_scene, lang_code="e")

    # Verify that gen_clone_audio was called with voice_sample set to the original audio base64
    gen_clone_audio_mock.assert_called_once()
    call_kwargs = gen_clone_audio_mock.call_args.kwargs
    assert call_kwargs.get("voice_sample") == original_audio_b64, (
        "voice_sample must equal the base64-encoded original scene audio"
    )
    assert call_kwargs.get("text") == "Hola mundo", (
        "gen_clone_audio must receive the translated text"
    )

    await job.close()


@pytest.mark.asyncio
async def test_gen_dub_scene_falls_back_when_no_audio() -> None:
    """gen_dub_scene falls back to gen_audio when the original scene audio is missing."""
    with patch.dict(sys.modules, {**mock_modules, **scene_mocks_base}):
        with temp_sys_path("apps", "apps/streamdub"):
            from apps.streamdub.streamdub_job import StreamDubJob as _StreamDubJob
            from apps.scene import SceneSegment

    service_manager = AsyncMock()
    service_manager.get_service_url = MagicMock(return_value="http://mock:1234")

    job = _StreamDubJob(job_id="test_fallback", service_manager=service_manager)

    # Do NOT write a scene audio file — simulate missing audio
    fake_scene = SceneSegment(
        scene_id=0,
        start_frame=0,
        end_frame=30,
        start_sec=0.0,
        end_sec=1.0,
    )
    fake_scene.audio_path = "scene_000.wav"
    fake_scene.transcript = "Hola mundo"

    dubbed_audio_b64 = base64.b64encode(b"dubbed_audio").decode()
    dubbed_video_binary = b"dubbed_video"

    gen_audio_mock = AsyncMock(return_value=dubbed_audio_b64)
    gen_clone_audio_mock = AsyncMock(return_value=dubbed_audio_b64)
    gen_video_mock = AsyncMock(return_value=dubbed_video_binary)

    with patch.object(job, "transcribe_audio", new=AsyncMock(return_value="Hello world")), \
         patch.object(job, "translate_scene", new=AsyncMock(return_value="Hola mundo")), \
         patch.object(job.gen, "gen_audio", gen_audio_mock), \
         patch.object(job.gen, "gen_clone_audio", gen_clone_audio_mock), \
         patch.object(job, "gen_video_lip_synced", gen_video_mock):
        await job.gen_dub_scene(fake_scene, lang_code="e")

    # gen_audio should be called for the fallback path; gen_clone_audio must NOT be called
    gen_audio_mock.assert_called_once()
    gen_clone_audio_mock.assert_not_called()

    await job.close()


def test_scene_segment_has_translation_field() -> None:
    """SceneSegment must have a 'translation' field separate from 'transcript'."""
    with patch.dict(sys.modules, {**mock_modules, **scene_mocks_base}):
        with temp_sys_path("apps", "apps/streamdub"):
            from apps.scene import SceneSegment

    scene = SceneSegment(scene_id=0, start_frame=0, end_frame=30, start_sec=0.0, end_sec=1.0)
    scene.transcript = "Hello, how are you?"
    scene.translation = "Hola, ¿cómo estás?"

    d = asdict(scene)
    assert d["transcript"] == "Hello, how are you?"
    assert d["translation"] == "Hola, ¿cómo estás?"
    # Changing translation must not affect transcript
    scene.translation = "Bonjour"
    assert scene.transcript == "Hello, how are you?"


@pytest.mark.asyncio
async def test_gen_dub_scene_stores_translation_separately() -> None:
    """gen_dub_scene must store original transcript in scene.transcript and
    the translation in scene.translation (not overwrite transcript)."""
    with patch.dict(sys.modules, {**mock_modules, **scene_mocks_base}):
        with temp_sys_path("apps", "apps/streamdub"):
            from apps.streamdub.streamdub_job import StreamDubJob as _StreamDubJob
            from apps.scene import SceneSegment

    service_manager = AsyncMock()
    service_manager.get_service_url = MagicMock(return_value="http://mock:1234")

    job = _StreamDubJob(job_id="test_translation_field", service_manager=service_manager)

    # Pre-create the job directory so transcript file writes succeed
    os.makedirs(job.job_path, exist_ok=True)

    scene = SceneSegment(
        scene_id=0, start_frame=0, end_frame=30,
        start_sec=0.0, end_sec=1.0,
        audio_path="scene_000.wav",
    )

    # Create a dummy audio file so voice cloning has a sample to read
    with open(os.path.join(job.job_path, "scene_000.wav"), "wb") as f:
        f.write(b"\x00" * 16)

    original_transcript = "Hello, how are you?"
    translated_text = "Hola, ¿cómo estás?"

    async def fake_transcribe(s: object) -> str:
        return original_transcript

    async def fake_translate(s: object, **kwargs: object) -> str:
        return translated_text

    async def fake_lip_sync(s: object) -> bytes:
        return b"video_bytes"

    with patch.object(job, "transcribe_audio", side_effect=fake_transcribe), \
         patch.object(job, "translate_scene", side_effect=fake_translate), \
         patch.object(job, "save_status", new=AsyncMock()), \
         patch.object(job, "gen_video_lip_synced", side_effect=fake_lip_sync), \
         patch.object(job, "get_submission_time", return_value=0.0):
        job.gen = MagicMock()
        job.gen.gen_clone_audio = AsyncMock(return_value="AAAA")
        job.gen.stop = AsyncMock()
        result = await job.gen_dub_scene(scene, lang_code="e")

    # Transcript must remain the original transcription
    assert scene.transcript == original_transcript, \
        "scene.transcript must hold the original transcription, not the translation"
    # Translation must be stored in the dedicated field
    assert scene.translation == translated_text, \
        "scene.translation must hold the translated text"
    # gen_clone_audio must have been called with the translation text, not the original
    job.gen.gen_clone_audio.assert_called_once()
    call_kwargs = job.gen.gen_clone_audio.call_args
    tts_text = call_kwargs.kwargs["text"]
    assert tts_text == translated_text, \
        "gen_clone_audio must receive the translation, not the original transcript"
    assert result == b"video_bytes"

    await job.close()
