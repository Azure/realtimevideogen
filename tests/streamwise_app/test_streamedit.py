#!/usr/bin/env python3
"""
Unit tests for StreamEdit.
"""

import base64
import os
import sys
import tempfile
import pytest

from http import HTTPStatus

from PIL import Image

from quart import Quart

from unittest.mock import patch
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

# Add current path
sys.path.append(os.getcwd())

from media_utils import video_frames_to_base64

from tests.test_utils import temp_sys_path
from tests.torch_mock import TorchMock

mock_torch = TorchMock()

mock_modules = {}
mock_modules.update(mock_torch.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    with temp_sys_path("apps", "apps/streamedit"):
        from apps.streamedit.streamedit import StreamEditApp

scene_mocks_base = {
    'scenedetect': MagicMock(),
    'scenedetect.detectors': MagicMock(),
    'scenedetect.stats_manager': MagicMock(),
}

with patch.dict(sys.modules, {**mock_modules, **scene_mocks_base}):
    with temp_sys_path("apps", "apps/streamedit"):
        from apps.streamedit.streamedit_job import StreamEditJob
        import apps.streamedit.streamedit_job as _sei_module  # keep reference for patch.object

with temp_sys_path("apps"):
    from scene import SceneSegment


streamedit_app = StreamEditApp()


@pytest.fixture(name="test_app")
def _test_app() -> Quart:
    return streamedit_app.app


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
    assert "StreamEdit" in response_html


@pytest.mark.asyncio
async def test_health(test_app: Quart) -> None:
    """Check /health."""
    client = test_app.test_client()
    response = await client.get("/health")
    assert response is not None
    assert response.status_code == HTTPStatus.OK
    response_json = await response.get_json()
    assert response_json == {
        "host": None,
        "jobs": {},
        "k8s_cluster": None,
        "port": None,
        "services": {},
        "status": "ok"
    }


@pytest.mark.asyncio
async def test_files(test_app: Quart) -> None:
    """Check /files endpoint."""
    client = test_app.test_client()

    if not os.path.exists("/tmp/streamedit"):
        os.makedirs("/tmp/streamedit", exist_ok=True)

    response = await client.get("/files")
    assert response.status_code == HTTPStatus.OK
    response_json = await response.get_json()
    assert "files" in response_json

    response = await client.get("/file/testfile.txt")
    assert response.status_code == HTTPStatus.NOT_FOUND
    response_json = await response.get_json()
    assert response_json == {"error": "File '/tmp/streamedit/testfile.txt' not found"}

    response = await client.get("/file_stream/job_id/testfile2.txt")
    assert response.status_code == HTTPStatus.NOT_FOUND
    response_json = await response.get_json()
    assert response_json == {"error": "File not found"}

    response = await client.get("/file_view/job_id/testfile3.txt")
    assert response.status_code == HTTPStatus.OK
    assert response.content_type == "text/html; charset=utf-8"
    response_text = await response.get_data(as_text=True)
    assert response_text.startswith("<!DOCTYPE html>\n<html>")
    assert "<title>File viewer: testfile3.txt</title>" in response_text


@pytest.mark.asyncio
async def test_unknown_route(test_app: Quart) -> None:
    """Check that an unknown route returns 404."""
    client = test_app.test_client()
    response = await client.get("/does-not-exist")
    assert response.status_code == HTTPStatus.NOT_FOUND


@pytest.mark.asyncio
async def test_job_submit_page(test_app: Quart) -> None:
    """Check the web page for job submission."""
    client = test_app.test_client()
    response = await client.get("/job")
    assert response.status_code == HTTPStatus.OK
    text = await response.get_data(as_text=True)
    assert len(text) > 0


@pytest.mark.asyncio
async def test_job_status_page(test_app: Quart) -> None:
    """Check the web page for job status."""
    client = test_app.test_client()
    job_id = "testjobid"
    response = await client.get(f"/job/{job_id}")
    assert response.status_code == HTTPStatus.OK
    text = await response.get_data(as_text=True)
    assert len(text) > 0


@pytest.mark.asyncio
async def test_api_job_status(test_app: Quart) -> None:
    """Check the API for job status (returns UNKNOWN for nonexistent jobs)."""
    client = test_app.test_client()
    job_id = "nonexistent_job_id"
    response = await client.get(f"/api/job/{job_id}/status")
    assert response.status_code == HTTPStatus.OK
    response_json = await response.get_json()
    assert "status" in response_json


@pytest.mark.asyncio
async def test_api_job_requests(test_app: Quart) -> None:
    """Check the API for job requests listing (returns empty for nonexistent jobs)."""
    client = test_app.test_client()
    job_id = "nonexistent_job_id"
    response = await client.get(f"/api/job/{job_id}/requests")
    assert response.status_code == HTTPStatus.OK


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
    streamedit_app.service_manager = MagicMock()
    streamedit_app.service_manager.get_service_url = MagicMock(
        return_value="http://mock_service_url:1234"
    )

    # Bad video data – scenedetect raises "Ensure file is valid video" (or similar)
    response = await client.post("/api/job", json={"video_base64": "AAAA"})
    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    response_json = await response.get_json()
    assert response_json["status"] == "error"
    assert "error" in response_json

    # Generate fake video for testing
    video_frames = [
        Image.new("RGB", (640, 480), color="blue")
        for _ in range(8)
    ]
    video_base64 = video_frames_to_base64(video_frames)

    response = await client.post("/api/job", json={"video_base64": video_base64})
    assert response.status_code == HTTPStatus.OK
    response_json = await response.get_json()
    assert response_json["status"] == "success"
    assert "job_id" in response_json


def test_edit_prompt() -> None:
    """Test that EDIT_PROMPT is imported and non-empty."""
    from apps.streamedit.edit_prompts import EDIT_PROMPT
    assert EDIT_PROMPT
    assert "video editor" in EDIT_PROMPT.lower() or "edit" in EDIT_PROMPT.lower()


@pytest.mark.asyncio
async def test_streamedit_job_no_video() -> None:
    """StreamEditJob.gen_edit with missing video raises ValueError."""
    job = StreamEditJob(
        job_id="test_no_video",
        service_manager=MagicMock(),
    )
    with pytest.raises(ValueError, match="Missing 'video_base64'"):
        await job.gen_edit(video_base64=None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_streamedit_job_detect_scenes_missing_file() -> None:
    """StreamEditJob.detect_scenes raises FileNotFoundError when video file is absent."""
    job = StreamEditJob(
        job_id="test_detect_no_file",
        service_manager=MagicMock(),
    )
    with pytest.raises(FileNotFoundError):
        await job.detect_scenes("/nonexistent/path/video.mp4")


@pytest.mark.asyncio
async def test_streamedit_job_chunk_audio_into_scenes() -> None:
    """chunk_audio_into_scenes saves scene_{id:03d}.wav and sets scene.audio_path."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        job = StreamEditJob(
            job_id="test_chunk_audio",
            service_manager=MagicMock(),
        )
        job.job_path = tmp_dir

        # Create a fake input video file
        video_path = f"{tmp_dir}/video.mp4"
        with open(video_path, "wb") as f:
            f.write(b"fake_video_data")

        fake_audio_base64 = base64.b64encode(b"dummy_audio").decode()

        job.scenes = [
            SceneSegment(scene_id=0, start_frame=0, end_frame=30, start_sec=0.0, end_sec=1.0),
            SceneSegment(scene_id=1, start_frame=30, end_frame=60, start_sec=1.0, end_sec=2.0),
        ]

        with patch.object(_sei_module, "extract_audio_from_video", new_callable=AsyncMock,
                          return_value=f"{tmp_dir}/audio.wav"), \
             patch.object(_sei_module, "read_file_base64", new_callable=AsyncMock,
                          return_value=fake_audio_base64), \
             patch.object(_sei_module, "chunk_audio_base64", return_value=fake_audio_base64):

            chunks = await job.chunk_audio_into_scenes()

        # Both scenes should have audio_path set with consistent naming
        assert job.scenes[0].audio_path == "scene_000.wav"
        assert job.scenes[1].audio_path == "scene_001.wav"

        # Per-scene WAV files must exist on disk
        assert os.path.exists(f"{tmp_dir}/scene_000.wav")
        assert os.path.exists(f"{tmp_dir}/scene_001.wav")

        # Two audio chunks returned
        assert len(chunks) == 2


@pytest.mark.asyncio
async def test_streamedit_job_gen_edit_scene_chunked() -> None:
    """_gen_edit_scene_chunked splits audio and generates sub-chunk videos."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        job = StreamEditJob(
            job_id="test_scene_chunked",
            service_manager=MagicMock(),
        )
        job.job_path = tmp_dir

        # Fake scene audio file
        fake_audio_base64 = base64.b64encode(b"dummy_audio_long").decode()
        with open(f"{tmp_dir}/scene_005.wav", "wb") as f:
            f.write(b"dummy_audio_long")

        scene = SceneSegment(
            scene_id=5,
            start_frame=0,
            end_frame=300,
            start_sec=0.0,
            end_sec=12.0,  # > MAX_FT_DURATION_SECS (~5.1 s)
            audio_path="scene_005.wav",
        )

        fake_frame = Image.new("RGB", (64, 64), color="red")
        fake_sub_edit = b"sub_edited_video"
        fake_concat = b"concatenated_video"

        with patch.object(_sei_module, "get_audio_chunks_by_silences",
                          return_value=[(0.0, 5.0), (5.0, 10.0), (10.0, 12.0)]), \
             patch.object(_sei_module, "chunk_video_binary", return_value=b"sub_scene_vid"), \
             patch.object(_sei_module, "get_video_frames", new_callable=AsyncMock,
                          return_value=[fake_frame] * 10), \
             patch.object(_sei_module, "chunk_audio_base64", return_value=fake_audio_base64), \
             patch.object(_sei_module, "concatenate_videos", new_callable=AsyncMock,
                          return_value=fake_concat), \
             patch.object(job.gen, "gen_video_audio_from_video", new_callable=AsyncMock,
                          return_value=fake_sub_edit):

            result = await job._gen_edit_scene_chunked(
                scene=scene,
                scene_binary=b"scene_vid",
                scene_audio_path=f"{tmp_dir}/scene_005.wav",
                scene_audio_base64=fake_audio_base64,
                edit_prompt="edit",
            )

        assert result == fake_concat


@pytest.mark.asyncio
async def test_streamedit_job_gen_edit_scene_uses_chunked_for_long_scene() -> None:
    """gen_edit_scene uses the chunked path when scene duration > MAX_FT_DURATION_SECS."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        job = StreamEditJob(
            job_id="test_long_scene",
            service_manager=MagicMock(),
        )
        job.job_path = tmp_dir

        # Fake input video file
        video_path = f"{tmp_dir}/video.mp4"
        with open(video_path, "wb") as f:
            f.write(b"fake_video_data")

        # Fake scene audio file
        fake_audio_base64 = base64.b64encode(b"dummy_audio_long").decode()
        with open(f"{tmp_dir}/scene_003.wav", "wb") as f:
            f.write(b"dummy_audio_long")

        scene = SceneSegment(
            scene_id=3,
            start_frame=0,
            end_frame=360,
            start_sec=0.0,
            end_sec=12.0,  # > MAX_FT_DURATION_SECS (~5.1 s)
            audio_path="scene_003.wav",
        )

        fake_concat = b"chunked_edit_result"

        chunked_mock = AsyncMock(return_value=fake_concat)
        with patch.object(_sei_module, "chunk_video_binary", return_value=b"scene_vid"), \
             patch.object(_sei_module, "read_file_base64", new_callable=AsyncMock,
                          return_value=fake_audio_base64), \
             patch.object(job, "_gen_edit_scene_chunked", chunked_mock):

            result_path = await job.gen_edit_scene(scene)

        # Chunked path must have been invoked
        chunked_mock.assert_called_once()

        # The edited file must be saved
        assert os.path.exists(f"{tmp_dir}/scene_003_edit.mp4")
        assert result_path == f"{tmp_dir}/scene_003_edit.mp4"


@pytest.mark.asyncio
async def test_streamedit_job_gen_edit_scene_saves_video() -> None:
    """gen_edit_scene reads audio from scene.audio_path and saves the edited video."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        job = StreamEditJob(
            job_id="test_scene_edit",
            service_manager=MagicMock(),
        )
        job.job_path = tmp_dir

        # Create a fake input video file so read_file_bytes can read it
        video_path = f"{tmp_dir}/video.mp4"
        with open(video_path, "wb") as f:
            f.write(b"fake_video_data")

        # Pre-create the per-scene audio file (as chunk_audio_into_scenes would)
        fake_audio_base64 = base64.b64encode(b"dummy_audio").decode()
        with open(f"{tmp_dir}/scene_002.wav", "wb") as f:
            f.write(b"dummy_audio")

        scene = SceneSegment(
            scene_id=2,
            start_frame=60,
            end_frame=120,
            start_sec=2.0,
            end_sec=4.0,
            audio_path="scene_002.wav",
        )

        with patch.object(_sei_module, "chunk_video_binary", return_value=b"scene_vid"), \
             patch.object(_sei_module, "get_video_frames", new_callable=AsyncMock, return_value=[]), \
             patch.object(_sei_module, "read_file_base64", new_callable=AsyncMock,
                          return_value=fake_audio_base64), \
             patch.object(job.gen, "gen_video_audio_from_video", new_callable=AsyncMock,
                          return_value=b"edited_video"):

            result_path = await job.gen_edit_scene(scene)

        # The edited scene file must be saved with consistent naming
        assert os.path.exists(f"{tmp_dir}/scene_002_edit.mp4")

        # The returned path must point to the edited scene file
        assert result_path == f"{tmp_dir}/scene_002_edit.mp4"


@pytest.mark.asyncio
async def test_extract_scene_frames_saves_pngs() -> None:
    """extract_scene_frames saves scene_{id:03d}_frame.png and populates frame_image_paths."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        job = StreamEditJob(
            job_id="test_extract_frames",
            service_manager=MagicMock(),
        )
        job.job_path = tmp_dir

        job.scenes = [
            SceneSegment(scene_id=0, start_frame=0, end_frame=30, start_sec=0.0, end_sec=1.0),
            SceneSegment(scene_id=1, start_frame=30, end_frame=60, start_sec=1.0, end_sec=2.0),
        ]

        # Patch cv2 so we don't need a real video file
        mock_cv2 = MagicMock()
        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, MagicMock())  # ok=True, dummy frame
        mock_cv2.VideoCapture.return_value = mock_cap

        def _fake_imwrite(path: str, _frame: object) -> bool:
            with open(path, "wb") as fh:
                fh.write(b"pngdata")
            return True

        mock_cv2.imwrite.side_effect = _fake_imwrite

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            await job.extract_scene_frames()

        # Each scene should now have exactly one image path
        assert job.scenes[0].frame_image_paths == ["scene_000_frame.png"]
        assert job.scenes[1].frame_image_paths == ["scene_001_frame.png"]


@pytest.mark.asyncio
async def test_extract_scene_frames_skips_failed_frame() -> None:
    """extract_scene_frames skips a scene when cv2 cannot read the frame."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        job = StreamEditJob(
            job_id="test_extract_frames_fail",
            service_manager=MagicMock(),
        )
        job.job_path = tmp_dir

        job.scenes = [
            SceneSegment(scene_id=0, start_frame=0, end_frame=30, start_sec=0.0, end_sec=1.0),
        ]

        mock_cv2 = MagicMock()
        mock_cap = MagicMock()
        mock_cap.read.return_value = (False, None)  # read fails
        mock_cv2.VideoCapture.return_value = mock_cap

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            await job.extract_scene_frames()

        # frame_image_paths must remain empty since reading failed
        assert job.scenes[0].frame_image_paths == []
