#!/usr/bin/env python3

import os
import pytest

from unittest import TestCase

from media_utils import empty_audio_file
from media_utils import get_audio_file_info
from media_utils import get_audio_duration
from file_utils import binary_to_base64
from file_utils import base64_to_binary
from media_utils import chunk_audio_base64
from media_utils import base64_to_audio_file
from file_utils import read_file_base64
from media_utils import fit_audio_to_duration
from media_utils import get_aligned_duration


class TestAudioUtils(TestCase):
    """Test cases for the empty_audio_file function."""

    def test_empty(self) -> None:
        """Test creating an empty audio file."""
        # Create an empty audio file
        audio_path = empty_audio_file(duration_seconds=0.5)

        # Verify file was created
        self.assertTrue(os.path.exists(audio_path))
        self.assertTrue(audio_path.endswith('.wav'))

        # Verify audio file info
        audio_info = get_audio_file_info(audio_path)
        self.assertEqual(audio_info['duration_seconds'], 0.5)
        self.assertEqual(audio_info['samplerate'], 44100)
        self.assertEqual(audio_info['channels'], 1)

        with self.assertRaises(FileNotFoundError):
            get_audio_file_info("nofile.wav")
        with self.assertRaises(TypeError):
            get_audio_file_info(["list", "of", "files"])  # type: ignore[arg-type]

        # Clean up the created file
        os.unlink(audio_path)

    def test_chunk(self) -> None:
        """Test chunking audio base64 data."""
        audio_path = empty_audio_file(duration_seconds=3.2)
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = binary_to_base64(audio_bytes)
        audio_base64 = chunk_audio_base64(
            audio_base64,
            start_seconds=1.0,
            end_seconds=2.0)
        audio_duration = get_audio_duration(audio_base64)
        self.assertAlmostEqual(audio_duration, 1.0, delta=0.1)

        audio_bytes = base64_to_binary(audio_base64)
        audio_duration = get_audio_duration(audio_bytes)
        self.assertAlmostEqual(audio_duration, 1.0, delta=0.1)

        with self.assertRaises(TypeError):
            chunk_audio_base64(audio_bytes)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            chunk_audio_base64(audio_base64, start_seconds=-1, end_seconds=0.2)
        with self.assertRaises(ValueError):
            chunk_audio_base64(audio_base64, start_seconds=1.0, end_seconds=5.0)

    def test_get_aligned_duration(self) -> None:
        VAE = 4
        FPS = 23
        self.assertEqual(get_aligned_duration(0.0, fps=FPS, vae=VAE), 0.0)
        self.assertAlmostEqual(get_aligned_duration(3.000, fps=FPS, vae=VAE), 3.000, delta=0.01)
        self.assertAlmostEqual(get_aligned_duration(3.001, fps=FPS, vae=VAE), 3.174, delta=0.01)
        self.assertAlmostEqual(get_aligned_duration(3.001, fps=FPS, vae=VAE), 3 + (VAE / FPS), delta=0.01)
        self.assertAlmostEqual(get_aligned_duration(4.500, fps=FPS, vae=VAE), 4.565, delta=0.01)
        self.assertAlmostEqual(get_aligned_duration(4.565, fps=FPS, vae=VAE), 4.565, delta=0.01)
        self.assertAlmostEqual(get_aligned_duration(4.566, fps=FPS, vae=VAE), 4.739, delta=0.01)


@pytest.mark.asyncio
async def test_base64() -> None:
    """Test converting audio file to base64 and back."""
    audio_path = empty_audio_file(duration_seconds=0.8)
    audio_base64 = await read_file_base64(audio_path)
    audio_duration = get_audio_duration(audio_base64)
    assert audio_duration == 0.8

    audio_path_2 = await base64_to_audio_file(audio_base64)

    with pytest.raises(TypeError):
        await read_file_base64(12345)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        await base64_to_audio_file(12345)  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        await read_file_base64("nonexisting.wav")

    # Clean up files
    os.unlink(audio_path)
    os.unlink(audio_path_2)


@pytest.mark.asyncio
async def test_resize_wav() -> None:
    audio_path = "tests/data/sample.wav"
    audio_base64 = await read_file_base64(audio_path)
    duration_secs = get_audio_duration(audio_base64)
    assert duration_secs == 24.025

    audio_shorter_path = fit_audio_to_duration(audio_path, 10.0)
    audio_shorter_base64 = await read_file_base64(audio_shorter_path)
    duration_shorter_secs = get_audio_duration(audio_shorter_base64)
    assert duration_shorter_secs == 10.0

    audio_longer_path = fit_audio_to_duration(audio_shorter_path, 10.6)
    audio_longer_base64 = await read_file_base64(audio_longer_path)
    duration_longer_secs = get_audio_duration(audio_longer_base64)
    assert duration_longer_secs == 10.6

    os.unlink(audio_shorter_path)
    os.unlink(audio_longer_path)

    with pytest.raises(TypeError):
        get_audio_duration(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        get_audio_duration([])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        fit_audio_to_duration(1, 2)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        fit_audio_to_duration(None, 1)  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        fit_audio_to_duration("nonexisting.wav", 10.6)
    with pytest.raises(ValueError):
        fit_audio_to_duration(audio_path, -0.2)
