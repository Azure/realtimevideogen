"""
Unit tests for tts_utils.py
"""
import os
import pytest

from media_utils import empty_audio_file
from file_utils import read_file_base64
from media_utils import get_audio_duration

from tts_utils import get_audio_chunks_by_silences
from tts_utils import split_into_sentences_max_duration
from tts_utils import strip_audio_file_silence
from tts_utils import generate_waveform_plt
from tts_utils import merge_chunks


def test_empy_audio_chunk() -> None:
    audio_path = empty_audio_file(duration_seconds=7.3)
    chunks = get_audio_chunks_by_silences(audio_path)
    assert len(chunks) == 2

    chunk0 = chunks[0]
    assert chunk0[0] == 0.0
    assert chunk0[1] == 3.65

    chunk1 = chunks[1]
    assert chunk1[0] == 3.65
    assert chunk1[1] == 7.3


def test_audio_sample_chunk() -> None:
    audio_path = "tests/data/audio_24secs.wav"
    if not os.path.exists(audio_path):
        return

    chunks = get_audio_chunks_by_silences(audio_path)
    assert len(chunks) == 5
    assert chunks[0][0] == 0.0
    for chunk_start, chunk_end in chunks:
        assert chunk_start < chunk_end


def test_split_sentences() -> None:
    text = ("This is a test. " * 20) + ("This sentence is way too long " * 50) + ("Short one. " * 10)
    chunks = split_into_sentences_max_duration(text, max_duration=3.0)
    assert len(chunks) > 1
    for chunk in chunks:
        assert 0 < len(chunk) < 5000  # Arbitrary limit to avoid extremely long chunks

    assert split_into_sentences_max_duration(None, max_duration=3.0) == []  # type: ignore[arg-type]
    assert split_into_sentences_max_duration("", max_duration=3.0) == []
    assert split_into_sentences_max_duration(" ", max_duration=3.0) == [" "]
    assert split_into_sentences_max_duration(" Test ", max_duration=3.0) == [" Test "]
    assert split_into_sentences_max_duration(" Test1. Test2. ", max_duration=3.0) == [" Test1. Test2. "]


@pytest.mark.asyncio
async def test_fit_audio_to_duration() -> None:
    audio_path = "tests/data/audio_24secs.wav"
    audio_base64 = await read_file_base64(audio_path)
    duration_secs = get_audio_duration(audio_base64)
    assert duration_secs == 24.025

    stripped_end_audio_path = strip_audio_file_silence(audio_path, strip_start=False, strip_end=True)
    stripped_end_audio_base64 = await read_file_base64(stripped_end_audio_path)
    assert stripped_end_audio_base64.startswith("UklGR")
    stripped_end_duration_secs = get_audio_duration(stripped_end_audio_base64)
    assert stripped_end_duration_secs < duration_secs
    os.unlink(stripped_end_audio_path)

    stripped_audio_path = strip_audio_file_silence(audio_path, strip_start=True, strip_end=True)
    stripped_audio_base64 = await read_file_base64(stripped_audio_path)
    assert stripped_audio_base64.startswith("UklGR")
    stripped_duration_secs = get_audio_duration(stripped_audio_base64)
    assert stripped_duration_secs < stripped_end_duration_secs
    os.unlink(stripped_audio_path)


def test_generate_waveform_plt() -> None:
    audio_path = "benchmark/samples/sample.wav"
    waveform_path = generate_waveform_plt(audio_path)
    assert waveform_path.endswith(".png")
    assert os.path.exists(waveform_path)
    assert os.path.getsize(waveform_path) > 0
    with open(waveform_path, "rb") as file:
        png_bytes = file.read()
        assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    os.unlink(waveform_path)


def test_merge_chunks() -> None:
    chunks = [(0.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0)]
    assert merge_chunks(chunks, max_duration_seconds=10.0) == [(0.0, 5.0)]
    assert merge_chunks(chunks, max_duration_seconds=5.0) == [(0.0, 5.0)]
    assert merge_chunks(chunks, max_duration_seconds=2.0) == [(0.0, 2.0), (2.0, 3.0), (3.0, 5.0)]
    assert merge_chunks(chunks, max_duration_seconds=1.0) == [(0.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0)]
    assert merge_chunks(chunks, max_duration_seconds=0.0) == [(0.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0)]
    assert merge_chunks(chunks, max_duration_seconds=-1.0) == [(0.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0)]

    assert merge_chunks(None) is None  # type: ignore[arg-type]
    assert merge_chunks([]) == []
    assert merge_chunks([(0.0, 1.5)]) == [(0.0, 1.5)]
