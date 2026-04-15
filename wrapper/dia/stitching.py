"""
Example from:
https://github.com/nari-labs/dia/issues/35

Check also:
https://github.com/nari-labs/dia/pull/159/files

Run with:
python stitching.py --input long_dialogue_test.txt \
    --output long_test_out.mp3 \
    --speed 0.98 \
    --device cuda --audio-prompt example_prompt.mp3 \
    --text-prompt="[S1] Open weights text to dialogue model.\n[S2] You get full control over scripts and voices."
"""

from dia.model import Dia

import argparse
import os
import re
import time
import logging
import soundfile
import numpy as np

from typing import Iterator
from typing import Tuple

from np.typing import NDArray

import torch
from torch import device

from pathlib import Path


def chunk_text(
    text: str,
    audio_prompt_text: str,
    max_words: int = 50,
    overlap_speakers: bool = False,
) -> Iterator[Tuple[str, bool]]:
    """Split text into speaker-aware chunks.
    Args:
        text (str): The input text to chunk
        audio_prompt_text (str): The audio prompt text to prepend to each chunk for the voice prompt fragment
        max_words (int): Maximum words per chunk
        overlap_speakers (bool): Whether to overlap speakers across chunks

    Yields:
        Tuple[str, bool]: (text chunk, silence flag)
        silence flag is True if the dialogue chunk has not been cut due to the max_words limit,
        otherwise False to stitch both dialogue split fragments with no pause.
    """
    lines = text.split('\n')
    # remove empty lines
    lines = [line for line in lines if len(line) > 0]
    word_count = 0
    last_speaker = None

    num_lines = len(lines)
    for i in range(0, num_lines, 2):
        # concatenate dialogue pairs (S1 / S2) into a single sequence
        sequence = lines[i] + "\n" + lines[i + 1]
        # count non-whitespace sequences as words
        word_count = len(re.findall(r'\S+', sequence))
        # check if the dialogue pair sequence is too long
        if word_count > max_words * 1.1:
            all_words = sequence.split(" ")
            shorten_sequence = " ".join(all_words[:max_words])
            residual_words = " ".join(all_words[max_words:])

            # Check for speaker markers [Speaker]
            speaker_matches = re.findall(r"\[(S\d+)](?:.*?transcript.*?|.*?transcription.*?)", shorten_sequence)

            if speaker_matches:
                try:
                    last_speaker = speaker_matches[1]
                except Exception:
                    last_speaker = "S1"
            # concatenate the audio prompt fragment with the split dialogue sequence
            # End sequence with the next speaker tag to prevent audio shortening
            next_speaker = [c for c in ["S1", "S2"] if c != last_speaker][0]
            first_prompted_sequence = audio_prompt_text + "\n" + shorten_sequence + f"\n[{next_speaker}]"
            # yield the first (shortened) chunk immediately — no silence between split fragments
            yield (first_prompted_sequence, False)

            sequence = residual_words
            if last_speaker == "S1":
                prompted_sequence = audio_prompt_text + "\n[S1] " + sequence + \
                    "\n[S1]"  # S2 starts inside the residual words sequence
            else:
                # S2 is already talking in the residual words sequence
                prompted_sequence = audio_prompt_text + "\n[S2] " + sequence + "\n[S1]"
        else:
            prompted_sequence = audio_prompt_text + "\n" + sequence + "\n[S1]"

        # Check for ellipsis at the end of the sequence to deny silence between chunks
        if sequence.endswith("..."):
            yield (prompted_sequence, False)
        else:
            yield (prompted_sequence, True)


def add_silence(
    audio: NDArray[np.floating],
    duration_sec: float = 0.5,
    sample_rate: int = 44100,
) -> NDArray[np.floating]:
    """Add silence to the end of an audio segment"""
    silence_samples = int(duration_sec * sample_rate)
    silence = np.zeros(silence_samples, dtype=audio.dtype)
    return np.concatenate([audio, silence])


def detect_device() -> device:
    """Detect the best available device for inference"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_duration(start: float) -> Tuple[int, int]:
    # Helper to calculate duration of the audio generation
    end = time.time()
    elapsed = end - start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    return minutes, seconds


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate long-form TTS using chunked text")
    parser.add_argument("--input", type=str, required=True, help="Input text file")
    parser.add_argument("--output", type=str, required=True, help="Output WAV file")
    parser.add_argument("--chunk-size", type=int, default=50, help="Max words per chunk")
    parser.add_argument("--speed", type=float, default=0.9, help="Speech speed factor (lower is slower)")
    parser.add_argument("--silence", type=float, default=0.3, help="Silence between chunks (seconds)")
    parser.add_argument("--device", type=str, default=None, help="Device for inference (cuda, mps, cpu)")
    parser.add_argument("--model", type=str, default="nari-labs/Dia-1.6B", help="Model name or path")
    parser.add_argument("--tokens-per-chunk", type=int, default=3072, help="Maximum tokens per audio chunk")
    parser.add_argument("--cfg-scale", type=float, default=3.0, help="Classifier-free guidance scale")
    parser.add_argument("--temperature", type=float, default=1.3, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling top-p value")
    parser.add_argument("--cfg-filter-top-k", type=int, default=30, help="Top k filter for CFG guidance")
    parser.add_argument("--audio-prompt", type=str, default=None, help="Optional audio prompt for voice cloning")
    parser.add_argument("--text-prompt", type=str, default=None, help="Optional text prompt for voice cloning")
    parser.add_argument("--tmp-dir", type=str, default="./tmp_audio", help="Directory for temporary files")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep temporary audio files")
    parser.add_argument("--overlap-speakers", action="store_true", default=True,
                        help="Overlap speaker identities across chunks")

    args = parser.parse_args()

    # Setup device
    device = detect_device() if args.device is None else torch.device(args.device)
    logging.info(f"[yellow]Using device: {device}[/]")

    # Create temporary directory if needed
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(exist_ok=True, parents=True)

    # Read input text
    input_path = Path(args.input)
    if not input_path.exists():
        logging.error(f"Input file {args.input} does not exist")
        return 1

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # check if audio prompt and text prompt are both provided (both required for voice control)
    if args.audio_prompt and args.text_prompt:
        text_prompt = args.text_prompt
        len_prompt_words = len(re.findall(r'\S+', text_prompt))
        len_prompt_char = len(text_prompt) + 1
    else:
        text_prompt = ""
        len_prompt_words = 0
        len_prompt_char = 0

    # Load model (do this after chunking to start inference as quickly as possible)
    logging.info(f"Loading Dia model from {args.model}...[/]")
    start_time = time.time()
    try:
        model = Dia.from_pretrained(args.model, device=device)
        logging.info(f"[yellow]Model loaded in {time.time() - start_time:.2f} seconds")
    except Exception as ex:
        logging.error(f"Loading model: {ex}")
        return 1

    # Generate audio for each chunk
    tmp_files = []
    logging.info("Generating audio for each chunk...")
    start = time.time()
    start_time = time.time()

    # Split text into chunks
    for i, (chunk, silence_flag) in enumerate(chunk_text(text, text_prompt, args.chunk_size, args.overlap_speakers)):
        chunk_file = tmp_dir / f"chunk_{i:03d}.wav"
        tmp_files.append(chunk_file)

        # Print chunk info removing the prompt words if any
        words = len(re.findall(r'\S+', chunk)) - len_prompt_words
        logging.info(f"[yellow]\nChunk {i + 1} ({words} words)[/]")
        # remind if the chunk was split due to max_words limit
        if not silence_flag:
            logging.info(
                "Dialogue split due to max_words limit or silence removed due to [...] "
                "detected at the end of the sentence")
        log_msg0 = chunk[len_prompt_char:200 + len_prompt_char]
        log_msg1 = ' [...]' if len(chunk) > 200 + len_prompt_char else ''
        logging.info(f"{'=' * 40}\n{log_msg0}{log_msg1}\n{'=' * 40}")

        # Generate audio for this chunk
        try:
            # Determine if audio prompt
            if text_prompt == "":
                audio_prompt = None
            else:
                audio_prompt = args.audio_prompt

            # Audio generation with retry logic and voice control option
            audio = model.generate(
                text=chunk,
                max_tokens=args.tokens_per_chunk,
                cfg_scale=args.cfg_scale,
                temperature=args.temperature,
                top_p=args.top_p,
                use_cfg_filter=True,
                cfg_filter_top_k=args.cfg_filter_top_k,
                audio_prompt_path=audio_prompt,
                use_torch_compile=False,
            )

            # Apply speed adjustment
            if args.speed != 1.0:
                orig_len = len(audio)
                target_len = int(orig_len / args.speed)
                x_orig = np.arange(orig_len)
                x_new = np.linspace(0, orig_len - 1, target_len)
                audio = np.interp(x_new, x_orig, audio)

            # Add silence at the end of the audio fragment
            if args.silence > 0 and silence_flag:
                audio = add_silence(audio, args.silence)

            # Save temporary file
            soundfile.write(chunk_file, audio, 44100)

            # generation statistics
            minutes, seconds = get_duration(start_time)
            if minutes > 0:
                logging.info(
                    f"Generated {chunk_file} (duration: {len(audio) / 44100:.2f} seconds) - "
                    f"Processed in {minutes} minutes and {seconds} seconds.")
            else:
                logging.info(
                    f"Generated {chunk_file} (duration: {len(audio) / 44100:.2f} seconds) - "
                    f"Processed in {seconds} seconds.")

        except Exception as ex:
            logging.error(f"Error processing chunk {i}: {ex}\n")
            # Continue with next chunk
        start_time = time.time()

    # Combine all audio files
    logging.info(f"Combining {len(tmp_files)} audio segments...")
    all_audio = []

    for tmp_file in tmp_files:
        if tmp_file.exists():
            audio, sr = soundfile.read(tmp_file)
            all_audio.append(audio)

    if not all_audio:
        logging.error("No audio was generated")
        return 1

    # Concatenate and save the final output
    final_audio = np.concatenate(all_audio)
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    logging.info(f"[yellow]Saving final audio to {output_path} ({len(final_audio) / 44100:.2f} seconds)")
    soundfile.write(output_path, final_audio, 44100)

    # Clean up temporary files
    if not args.keep_tmp:
        logging.info("Cleaning up temporary files...")
        for tmp_file in tmp_files:
            if tmp_file.exists():
                tmp_file.unlink()
        if not os.listdir(tmp_dir):
            tmp_dir.rmdir()

    minutes, seconds = get_duration(start)
    logging.info(
        f"Done! Final audio saved to {output_path} - Total processing time {minutes} minutes and {seconds} seconds")
    return 0


if __name__ == "__main__":
    main()
