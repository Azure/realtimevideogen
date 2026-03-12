"""
Utility functions for TTS processing, including silence detection and audio chunking.
"""
import os
import re
import io
import base64
import math
import wave
import tempfile

import numpy as np

from typing import List
from typing import Tuple
from typing import Optional

from scipy.io import wavfile


SILENCE_THRESHOLDS_MS = [200, 100, 50, 10]  # thresholds for silence duration in milliseconds
# AMPLITUDE_THRESHOLD = 500  # amplitude threshold for silence
AMPLITUDE_THRESHOLD = 200  # amplitude threshold for silence
MIN_SILENCE_DURATION = 0.2  # seconds


def detect_silences(
    data: np.ndarray,
    rate: int,
    amp_silence_threshold: float = AMPLITUDE_THRESHOLD,
    min_silence_duration_seconds: float = MIN_SILENCE_DURATION
) -> List[Tuple[float, float]]:
    """
    Detect silences by identifying continuous regions.
    """
    is_silent = np.abs(data) < amp_silence_threshold
    min_silence_samples = int(rate * min_silence_duration_seconds)

    silences = []
    start = None
    for i, silent in enumerate(is_silent):
        if silent and start is None:
            start = i
        elif not silent and start is not None:
            if i - start >= min_silence_samples:
                start_seconds = start / rate
                end_seconds = i / rate
                silences.append((
                    start_seconds,
                    end_seconds))
            start = None
    if start is not None and len(data) - start >= min_silence_samples:
        start_seconds = start / rate
        end_seconds = len(data) / rate
        silences.append((start_seconds, end_seconds))
    return silences


def get_time_in_period(
    start: float,
    end: float,
    method: str = "middle"  # 'start', 'end', or 'middle'
) -> float:
    """Get a time point in the given period based on the method."""
    if start >= end:
        raise ValueError("Start time must be less than end time.")
    if method == "start":
        return start
    elif method == "end":
        return end
    elif method == "middle":
        return (start + end) / 2.0
    return (start + end) / 2.0


def is_audio_silence(
    audio_data: np.ndarray,
    amp_silence_threshold: float = AMPLITUDE_THRESHOLD
) -> bool:
    """Check if the audio data is silence based on amplitude threshold."""
    if not isinstance(audio_data, np.ndarray):
        raise TypeError(f"Expected np.ndarray for audio_data, got {type(audio_data)}")
    return bool(np.all(np.abs(audio_data) < amp_silence_threshold))


def merge_trailing_chunks(
    chunks: List[Tuple[float, float]],
    rate: int = 16000,
    data: Optional[np.ndarray] = None
) -> List[Tuple[float, float]]:
    """Merge chunks if it is into the previous chunk."""
    if len(chunks) < 2:
        return chunks
    last_start, last_end = chunks[-1]
    last_start_idx = int(last_start * rate)
    last_end_idx = int(last_end * rate)
    last_chunk = data[last_start_idx:last_end_idx]
    if is_audio_silence(last_chunk):
        chunks[-2] = (chunks[-2][0], last_end)
        chunks.pop()
    return chunks


def merge_chunks(
    chunks: List[Tuple[float, float]],
    max_duration_seconds: float = 5.0,
) -> List[Tuple[float, float]]:
    """Merge chunks if the combined duration is under max_duration_seconds."""
    if not chunks:
        return chunks
    if max_duration_seconds <= 0:
        return chunks
    if len(chunks) < 2:
        return chunks

    merged_chunks = []
    current_start, current_end = chunks[0]
    for start, end in chunks[1:]:
        if (end - current_start) <= max_duration_seconds:
            current_end = end  # Extend the current chunk
        else:
            merged_chunks.append((current_start, current_end))
            current_start, current_end = start, end
    merged_chunks.append((current_start, current_end))  # Add the last chunk
    return merged_chunks


def align_chunks(
    chunks: List[Tuple[float, float]],
    chunk_alignment_seconds: float = 1 / 30.0,  # Align to 30 FPS
) -> List[Tuple[float, float]]:
    """Align chunk boundaries to the next frame (ceil) based on FPS."""
    if chunk_alignment_seconds <= 0 or not chunks:
        return chunks

    aligned_chunks = []
    for start, end in chunks:
        aligned_start = math.ceil(start / chunk_alignment_seconds) * chunk_alignment_seconds
        aligned_end = math.ceil(end / chunk_alignment_seconds) * chunk_alignment_seconds
        if aligned_start >= aligned_end:
            aligned_end = aligned_start + chunk_alignment_seconds
        aligned_chunks.append((aligned_start, min(aligned_end, chunks[-1][1])))
    return aligned_chunks


def get_audio_chunks_by_silences_greedy_new(
    audio_path: str,
    max_duration_seconds: float = 5.0,  # 5 seconds is what fantasy talking allows
    chunk_alignment_seconds: float = 1 / 30.0,  # Align to 30 FPS
    min_chunk_duration_seconds: float = 0.5,
    method: str = "start",  # 'start', 'end', or 'middle'
) -> List[Tuple[float, float]]:
    """
    Chunk audio using hierarchical greedy silence-based splitting:
    - Prefer longer silences first
    - Chunks <= max_duration_seconds
    - Merge tiny intermediate and trailing chunks
    - Boundaries aligned to chunk_alignment_seconds
    """
    if not isinstance(audio_path, str):
        raise TypeError(f"Expected str for audio_path, got {type(audio_path)}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"WAV file does not exist: {audio_path}")

    rate, data = wavfile.read(audio_path)
    if data.ndim > 1:
        data = data.mean(axis=1).astype(data.dtype)

    total_duration = len(data) / rate
    chunk_start = 0.0
    chunks: List[Tuple[float, float]] = []

    sorted_thresholds_ms = sorted(SILENCE_THRESHOLDS_MS, reverse=True)

    # Precompute silences for all thresholds
    silence_map = {
        threshold_ms: detect_silences(
            data,
            rate,
            min_silence_duration_seconds=threshold_ms / 1000.0)
        for threshold_ms in sorted_thresholds_ms
    }

    while chunk_start < total_duration:
        chunk_deadline = min(chunk_start + max_duration_seconds, total_duration)
        chunk_end = None

        # Try longer silences first
        for threshold_ms in sorted_thresholds_ms:
            candidate_silences = [
                (s, e) for s, e in silence_map[threshold_ms]
                if chunk_start < s <= chunk_deadline
            ]
            if candidate_silences:
                # Pick the last silence in period
                last_silence = max(candidate_silences, key=lambda se: get_time_in_period(se[0], se[1], method))
                chunk_end = get_time_in_period(last_silence[0], last_silence[1], method)
                break

        # No silence found → use max_duration
        if chunk_end is None:
            chunk_end = chunk_deadline

        # Merge tiny chunks (intermediate or trailing)
        if chunks and (chunk_end - chunk_start) < min_chunk_duration_seconds:
            prev_start, prev_end = chunks.pop()
            chunk_start = prev_start
            chunk_end = max(prev_end, chunk_end)

        # Align boundaries
        aligned_start = math.floor(chunk_start / chunk_alignment_seconds) * chunk_alignment_seconds
        aligned_end = math.ceil(chunk_end / chunk_alignment_seconds) * chunk_alignment_seconds
        aligned_end = min(aligned_end, total_duration)

        chunks.append((aligned_start, aligned_end))
        chunk_start = chunk_end

    return chunks


def get_audio_chunks_by_silences_greedy(
    audio_path: str,
    max_duration_seconds: float = 5.0,  # 5 seconds is what fantasy talking allows
    chunk_alignment_seconds: float = 1 / 30.0,  # Align to 30 FPS
    min_chunk_duration_seconds: float = 0.5,
    method: str = "start",  # 'start', 'end', or 'middle'
) -> List[Tuple[float, float]]:
    """
    Chunk audio using greedy silence-based splitting with contiguous segments.
    Each chunk ends at the last silence under max_duration.
    Prevents creating multiple small chunks unnecessarily.
    """
    if not isinstance(audio_path, str):
        raise TypeError(f"Expected str for audio_path, got {type(audio_path)}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"WAV file does not exist: {audio_path}")
    rate, data = wavfile.read(audio_path)

    if data.ndim > 1:  # Convert stereo to mono if needed
        data = data.mean(axis=1).astype(data.dtype)

    total_duration = len(data) / rate
    chunk_start_seconds = 0.0
    chunks = []

    # Precompute silences at all thresholds
    silence_map = {
        duration_ms: detect_silences(
            data, rate,
            min_silence_duration_seconds=duration_ms / 1000.0)
        for duration_ms in SILENCE_THRESHOLDS_MS
    }

    # Find within the silences
    while chunk_start_seconds < total_duration:
        chunk_deadline = min(chunk_start_seconds + max_duration_seconds, total_duration)

        # Find the last silence before the chunk deadline
        best_silence_end = None
        for duration_ms in SILENCE_THRESHOLDS_MS:
            silences = silence_map[duration_ms]
            candidate_silences = [
                (start, end) for start, end in silences
                if chunk_start_seconds < start <= chunk_deadline
            ]
            if candidate_silences:
                # Take the last silence under deadline
                candidate_time = max(
                    get_time_in_period(start, end, method)
                    for start, end in candidate_silences
                )
                if candidate_time > (chunk_deadline - 0.5):  # prefer silence close to end
                    best_silence_end = candidate_time
                    break
                elif not best_silence_end:
                    # fallback: keep the last available silence if no better found
                    best_silence_end = candidate_time

        # If no silence or too early, just go to max duration
        if best_silence_end is None or best_silence_end < chunk_start_seconds + 0.5:
            chunk_end_seconds = chunk_deadline
        else:
            chunk_end_seconds = best_silence_end

        chunks.append((chunk_start_seconds, chunk_end_seconds))
        chunk_start_seconds = chunk_end_seconds

    merged_chunks = merge_trailing_chunks(chunks, rate, data)
    merged_chunks = merge_chunks(merged_chunks, max_duration_seconds)
    aligned_chunks = align_chunks(merged_chunks, chunk_alignment_seconds)

    # Final checks
    # If not enough chunks
    min_num_chunks = math.ceil(total_duration / max_duration_seconds)
    if len(aligned_chunks) < min_num_chunks:
        return get_audio_chunks_hard_cutoff(
            total_duration,
            max_duration_seconds)
    # If any chunk is longer than max_duration_seconds
    if any((end - start) > max_duration_seconds for start, end in aligned_chunks):
        return get_audio_chunks_hard_cutoff(
            total_duration,
            max_duration_seconds)

    return aligned_chunks


def get_audio_chunks_hard_cutoff(
    total_duration: float,
    max_duration_seconds: float = 5.0,
) -> List[Tuple[float, float]]:
    """Chunk audio using hard cutoff at max_duration_seconds."""
    chunks = []
    num_chunks = math.ceil(total_duration / max_duration_seconds)
    chunk_duration = total_duration / num_chunks
    for i in range(num_chunks):
        start = i * chunk_duration
        end = min((i + 1) * chunk_duration, total_duration)
        chunks.append((start, end))
    return chunks


def get_audio_chunks_by_silences_binary(
    audio_path: str,
    max_duration_seconds: float = 5.0,
    chunk_alignment_seconds: float = 1 / 30.0,  # Align to 30 FPS
    method: str = "middle",  # 'start', 'end', or 'middle'
) -> List[Tuple[float, float]]:
    """
    Chunk audio using recursive binary search.
    If a segment is longer than max_duration_seconds, split it at the longest
    silence inside (according to `method`). Falls back to hard cutoff.
    Returns a list of (start_time, end_time) in seconds.
    """
    if not isinstance(audio_path, str):
        raise TypeError(f"Expected str for audio_path, got {type(audio_path)}")

    rate, data = wavfile.read(audio_path)

    # Convert stereo to mono if needed
    if data.ndim > 1:
        data = data.mean(axis=1).astype(data.dtype)

    total_duration = len(data) / rate

    # Detect all silences once and sort by length (longest first)
    silences = detect_silences(data, rate)
    silences = sorted(silences, key=lambda s: (s[1] - s[0]), reverse=True)

    def split_segment(start: float, end: float) -> List[Tuple[float, float]]:
        duration = end - start
        if duration <= max_duration_seconds:
            return [(start, end)]

        # Find longest silence inside this segment
        candidate = None
        for s_start, s_end in silences:
            if start < s_start and s_end < end:
                candidate = (s_start, s_end)
                break  # longest first due to sorting

        if candidate:
            split_point = get_time_in_period(candidate[0], candidate[1], method)
            return split_segment(start, split_point) + split_segment(split_point, end)

        # No silence inside -> hard cutoff
        cutoff = min(start + max_duration_seconds, end)
        left = [(start, cutoff)]
        right = split_segment(cutoff, end) if cutoff < end else []
        return left + right

    chunks = split_segment(0.0, total_duration)

    merged_chunks = merge_trailing_chunks(chunks, rate, data)
    merged_chunks = merge_chunks(merged_chunks, max_duration_seconds)
    aligned_chunks = align_chunks(merged_chunks, chunk_alignment_seconds)

    return aligned_chunks


def get_audio_chunks_by_silences(
    audio_path: str,
    max_duration_seconds: float = 5.0,  # 5 seconds is what fantasy talking allows
    chunk_alignment_seconds: float = 1 / 30.0,  # Align to 30 FPS
    min_chunk_duration_seconds: float = 0.5,
    method: str = "start",  # 'start', 'end', or 'middle'
) -> List[Tuple[float, float]]:
    # get_audio_chunks_by_silences_binary
    # get_audio_chunks_by_silences_greedy
    return get_audio_chunks_by_silences_greedy(
        audio_path,
        max_duration_seconds,
        chunk_alignment_seconds,
        min_chunk_duration_seconds,
        method)


def is_audio_path_silence(audio_path: str) -> bool:
    """
    Check if the audio is silence based on amplitude threshold.
    Returns True if the audio is considered silence, False otherwise.
    """
    if not isinstance(audio_path, str):
        raise TypeError(f"Expected str for audio_path, got {type(audio_path)}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"WAV file does not exist: {audio_path}")
    _, data = wavfile.read(audio_path)
    if data.ndim > 1:
        data = data.mean(axis=1).astype(data.dtype)
    return is_audio_silence(data)


def is_audio_base64_silence(audio_base64: str) -> bool:
    """
    Check if the base64-encoded audio is silence based on amplitude threshold.
    """
    if not isinstance(audio_base64, str):
        raise TypeError(f"Expected str for audio_base64, got {type(audio_base64)}")
    audio_bytes = base64.b64decode(audio_base64)
    buffer = io.BytesIO(audio_bytes)
    _, data = wavfile.read(buffer)
    if data.ndim > 1:
        data = data.mean(axis=1).astype(data.dtype)
    return is_audio_silence(data)


def strip_audio_file_silence(
    input_path: str,
    strip_start: bool = False,
    strip_end: bool = True,
    output_path: Optional[str] = None,
    amp_silence_threshold: float = AMPLITUDE_THRESHOLD
) -> str:
    """
    Strip silence from the start and/or end of a WAV audio file.
    """
    if not isinstance(input_path, str):
        raise TypeError(f"Expected str for input_path, got {type(input_path)}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input WAV file does not exist: {input_path}")
    if not input_path.lower().endswith(".wav"):
        raise ValueError(f"Input file must be a WAV file: {input_path}")

    with wave.open(input_path, "rb") as wf:
        params = wf.getparams()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        nframes = wf.getnframes()
        frames = wf.readframes(nframes)

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sampwidth]
    audio = np.frombuffer(frames, dtype=dtype)

    # Handle multi-channel audio: reshape so axis=1 is channels
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)
        amplitude = np.max(np.abs(audio), axis=1)  # collapse channels
    else:
        amplitude = np.abs(audio)

    # Identify non-silence indices
    non_silent_indices = np.where(amplitude >= amp_silence_threshold)[0]

    if non_silent_indices.size == 0:
        # Entire file is silent: return original or empty file
        if not output_path:
            output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        with wave.open(output_path, "wb") as wf:
            wf.setparams(params._replace(nframes=0))
            wf.writeframes(b"")
        return output_path

    start_idx, end_idx = 0, len(audio)

    if strip_start:
        start_idx = non_silent_indices[0]
    if strip_end:
        end_idx = non_silent_indices[-1] + 1

    trimmed = audio[start_idx:end_idx]

    if not output_path:
        output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

    with wave.open(output_path, "wb") as wf:
        wf.setparams(params._replace(nframes=len(trimmed)))
        wf.writeframes(trimmed.tobytes())

    return output_path


def get_sentences(text: str) -> list[str]:
    """
    Extract sentences from a text including the punctuation.
    """
    if not text:
        return []
    # Simple regex to split sentences while keeping the punctuation
    sentences = re.split(r'([.!?]+)', text)
    sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty strings
    # Combine punctuation with the sentence
    return [''.join(sentences[i:i + 2]).strip() for i in range(0, len(sentences), 2)]


def estimate_num_words_from_audio_duration(duration_seconds: float, speed: float = 1.0) -> int:
    """
    Estimate number of words based on audio duration.
    0.4 seconds per word on average (16 * num_words / 40.0).
    Data in "audio_duration.csv".
    """
    num_words = int(math.ceil((duration_seconds * speed) / 0.4))  # seconds -> words
    return max(1, num_words)


def estimate_audio_duration_from_words(num_words: int, speed: float = 1.0) -> float:
    """
    Estimate audio duration based on the number of words.
    0.4 seconds per word on average (16 * num_words / 40.0).
    Data in "audio_duration.csv".
    """
    duration_seconds = 0.4 * num_words  # words -> seconds
    return duration_seconds / speed


def estimate_audio_duration_from_chars(num_chars: int, speed: float = 1.0) -> float:
    """
    Estimate audio duration based on the number of characters.
    0.064 seconds per char on average (13 * num_chars / 200.0).
    Data in "audio_duration.csv".
    """
    duration_seconds = 0.065 * num_chars  # chars -> seconds
    return duration_seconds / speed


def estimate_audio_duration(text: str, speed: float = 1.0) -> float:
    """
    Estimate audio duration based on the number of words and characters.
    """
    num_chars = len(text)
    num_words = len(text.strip().split())
    # sub_sentences = get_sentences(text)
    # num_sentences = len(sub_sentences)

    duration_seconds = max(
        estimate_audio_duration_from_chars(num_chars, speed),
        estimate_audio_duration_from_words(num_words, speed),
    )  # seconds
    return duration_seconds


def split_into_sentences_max_duration(
    text: str,
    max_duration: float = 5.0,
) -> list[str]:
    """
    Split the text into sub-sentences with a maximum estimated audio duration.
    If a sentence exceeds the limit, it is further split by words.
    """
    if not text:
        return []

    if estimate_audio_duration(text) <= max_duration:
        return [text]

    sentences = get_sentences(text)
    if not sentences:
        return [text]

    chunks = []
    current_chunk = ""
    current_duration = 0.0  # seconds

    for sentence in sentences:
        sentence_duration = estimate_audio_duration(sentence)

        if sentence_duration <= max_duration:
            # Sentence fits within max duration
            if current_duration + sentence_duration > max_duration:
                # Commit current chunk and start a new one
                current_chunk = current_chunk.strip()
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
                current_duration = sentence_duration
            else:
                # Append sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_duration += sentence_duration

        else:
            # Sentence too long -> split by words
            words = sentence.split()
            word_chunk = ""
            word_duration = 0.0

            for word in words:
                duration = estimate_audio_duration(word + " ")
                if word_duration + duration > max_duration:
                    # Commit the chunk of words
                    word_chunk = word_chunk.strip()
                    if word_chunk:
                        chunks.append(word_chunk)
                    word_chunk = word
                    word_duration = duration
                else:
                    word_chunk += " " + word if word_chunk else word
                    word_duration += duration

            word_chunk = word_chunk.strip()
            if word_chunk:
                chunks.append(word_chunk)

            # Reset main chunk tracking after forced split
            current_chunk = ""
            current_duration = 0.0

    current_chunk = current_chunk.strip()
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def generate_waveform_plt(wav_file_name: str) -> str:
    """Generate a waveform PNG image from a WAV audio file using matplotlib."""
    import matplotlib.pyplot as plt

    # Open the file
    # Read the WAV file to generate the waveform
    rate, data = wavfile.read(wav_file_name)
    if data.ndim > 1:  # Stereo or multi-channel audio to mono
        data = data.mean(axis=1).astype(data.dtype)
    data_plt = data / np.max(np.abs(data)) if np.max(np.abs(data)) != 0 else data

    # Generate and add silences
    silences = {}
    for silence_duration_ms in [200, 100, 50, 10]:
        silences[silence_duration_ms] = detect_silences(
            data, rate,
            min_silence_duration_seconds=silence_duration_ms / 1000.0)

    waveform_path = f"{wav_file_name}-waveform.png"
    times = np.arange(len(data)) / rate
    plt.figure(figsize=(14, 4))
    plt.plot(times, data_plt, label="Waveform", color="steelblue")

    color_map = {
        200: "red",
        100: "orange",
        50: "yellow",
        10: "green"
    }
    for silence_duration, start_end in reversed(silences.items()):
        for silence_start, silence_end in start_end:
            label = None
            if silence_start == start_end[0][0]:
                label = f"Silence {silence_duration} ms"
            plt.axvspan(
                silence_start, silence_end,
                color=color_map[silence_duration],
                alpha=0.5,
                label=label)

    duration_seconds = len(data) / rate
    plt.xlim(0, duration_seconds)
    plt.ylim(-1, 1)
    plt.title("Waveform with silences highlighted")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (normalized)")
    plt.legend(loc="lower left")
    plt.savefig(waveform_path, bbox_inches="tight", dpi=300)
    plt.close()

    return waveform_path
