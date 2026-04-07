"""
Wrapper for VibeVoice model.
"""
import os
import base64
import logging
import asyncio
import tempfile

import torch
from torch import inference_mode

from typing import override
from typing import Optional
from typing import Dict
from typing import Union
from typing import Any

# Copy from transformers PR: https://github.com/huggingface/transformers/pull/40546/files
from vibevoice_processor import VibeVoiceProcessor
from modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference

from wrapper_model import ModelGeneration


class VoiceMapper:
    """Maps speaker names to voice file paths"""

    def __init__(self) -> None:
        self.setup_voice_presets()
        new_dict = {}
        for name, path in self.voice_presets.items():
            if '_' in name:
                name = name.split('_')[0]
            if '-' in name:
                name = name.split('-')[-1]
            new_dict[name] = path
        self.voice_presets.update(new_dict)

    def setup_voice_presets(self) -> None:
        """Setup voice presets by scanning the voices directory."""
        voices_dir = os.path.join(os.path.dirname(__file__), "voices")
        if not os.path.exists(voices_dir):
            logging.warning(f"Voices directory not found at {voices_dir}")
            self.voice_presets: dict[str, str] = {}
            self.available_voices: dict[str, str] = {}
            return

        wav_files = [
            f for f in os.listdir(voices_dir)
            if f.lower().endswith('.wav') and os.path.isfile(os.path.join(voices_dir, f))]

        self.voice_presets = {}
        for wav_file in wav_files:
            name = os.path.splitext(wav_file)[0]
            full_path = os.path.join(voices_dir, wav_file)
            self.voice_presets[name] = full_path

        self.voice_presets = dict(sorted(self.voice_presets.items()))

        # Filter out voices that don't exist (this is now redundant but kept for safety)
        self.available_voices = {
            name: path for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }

        logging.info(f"Found {len(self.available_voices)} voice files in {voices_dir}")
        logging.info(f"Available voices: {', '.join(self.available_voices.keys())}")

    def get_voice_path(self, speaker_name: str) -> str:
        """Get voice file path for a given speaker name"""
        if speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]
        speaker_lower = speaker_name.lower()
        for preset_name, path in self.voice_presets.items():
            if preset_name.lower() in speaker_lower or speaker_lower in preset_name.lower():
                return path
        voices_list = list(self.voice_presets.values())
        if not voices_list:
            raise ValueError("No voice presets available.")
        default_voice = voices_list[0]
        logging.warning(f"No voice preset found for '{speaker_name}', using default voice: {default_voice}")
        return default_voice


class VibeVoiceGeneration(ModelGeneration):
    """Handle audio generation using the VibeVoice model.
    https://github.com/microsoft/VibeVoice"""

    HF_MODEL_NAME = "microsoft/VibeVoice-1.5B"

    def __init__(
        self,
        model_name: str = "vibevoice",
        param_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(model_name)
        self.param_dtype = param_dtype
        # https://github.com/microsoft/VibeVoice/tree/main/demo/voices
        self.voice_mapper: Optional[VoiceMapper] = None
        self.processor: Optional[VibeVoiceProcessor] = None
        self.vibevoice: Optional[VibeVoiceForConditionalGenerationInference] = None

    def __del__(self) -> None:
        if self.processor is not None:
            self.processor = None
        if self.vibevoice is not None:
            self.vibevoice = None

    def load_model(self) -> None:
        self.load_timer.start("voice_mapper")
        self.voice_mapper = VoiceMapper()
        logging.info("Available voices:")
        for voice_name, voice_path in self.voice_mapper.available_voices.items():
            logging.info(f"  {voice_name}: {voice_path}")
        self.load_timer.end("voice_mapper")

        self.load_timer.start("processor")
        self.processor = VibeVoiceProcessor.from_pretrained(self.HF_MODEL_NAME)
        # TODO can we move this to the GPU device?
        # self.processor.to(self.device)
        # self.processor.tokenizer = self.processor.tokenizer.to(self.device)
        # self.audio_processor = self.processor.audio_processor.to(self.device)
        self.load_timer.end("processor")

        self.load_timer.start("vibevoice")
        self.vibevoice = VibeVoiceForConditionalGenerationInference.from_pretrained(
            self.HF_MODEL_NAME,
            torch_dtype=self.param_dtype,
            attn_implementation="flash_attention_2")
        if not self.vibevoice:
            raise ValueError("Failed to load VibeVoice model")
        self.vibevoice.eval()
        self.vibevoice.set_ddpm_inference_steps(num_steps=10)
        self.vibevoice.to(self.device)
        self.load_timer.end("vibevoice")

    def init_parallelism(self) -> None:
        self.load_timer.start("torch_dist")

        self.rank = int(os.getenv("RANK", 0))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))

        self.device_id = self.local_rank

        if not torch.cuda.is_available():
            self.device_id = 0
            self.device = torch.device("cpu")
            logging.warning("CUDA is not available. Running on CPU.")
            self.load_timer.end("torch_dist")
            return  # CPU mode, no parallelism needed

        self.device = torch.device(f"cuda:{self.device_id}")

        torch.cuda.set_device(self.local_rank)

        if self.world_size > 1:
            logging.warning("VibeVoice does not support multi-GPU setups (yet).")

    def init_model_parallelism(self) -> None:
        if self.world_size > 1:
            logging.warning("VibeVoice does not support multi-GPU setups (yet).")

    def model_compile(self) -> None:
        """Compile the model components for optimized performance.
        self.processor cannot be compiled due to dynamic input shapes.
        """
        if not self.torch_compile:
            return

        self.load_timer.start("vibevoice_compile")
        self.vibevoice = torch.compile(
            self.vibevoice,
            mode="max-autotune-no-cudagraphs",
        )
        self.load_timer.end("vibevoice_compile")

    def _assert_model_init(self) -> None:
        super()._assert_model_init()
        if not self.voice_mapper:
            raise ValueError("VoiceMapper not initialized")
        if not self.processor:
            raise ValueError("Processor not initialized")
        if not self.vibevoice:
            raise ValueError("VibeVoice model not initialized")

    @inference_mode()
    async def warmup(self) -> None:
        logging.info("Warmup for VibeVoice generation.")
        await self.generate(text="Warmup")

    def _decode_voice_sample_to_tmp_file(self, voice_sample: str) -> str:
        """Decode a base64-encoded WAV voice sample to a temporary file.

        The caller is responsible for deleting the file when done.

        Args:
            voice_sample: Base64-encoded WAV audio.

        Returns:
            Path to the temporary WAV file.
        """
        audio_bytes = base64.b64decode(voice_sample)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
            tmp_f.write(audio_bytes)
            tmp_path = tmp_f.name
        logging.info(f"Decoded voice_sample to temporary file: {tmp_path} ({len(audio_bytes)} bytes).")
        return tmp_path

    def _cleanup_tmp_voice_file(self, tmp_path: Optional[str]) -> None:
        """Remove a temporary voice file created by _decode_voice_sample_to_tmp_file.

        Silently ignores errors so that cleanup never raises inside a finally block.

        Args:
            tmp_path: Path returned by _decode_voice_sample_to_tmp_file, or None (no-op).
        """
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError as e:
                logging.warning(f"Could not remove temporary voice file {tmp_path}: {e}")

    @override
    @inference_mode()
    async def generate(
        self,
        text: str,
        voice: str = "woman_000",
        voice_sample: Optional[str] = None,
        cfg_scale: float = 1.3,
        job_id: Optional[str] = None,
        output_type: str = "audio_path",
    ) -> str:
        """Generate speech audio from text.

        Args:
            text: The text to synthesise.
            voice: Name of a built-in voice preset (used when *voice_sample* is not provided).
            voice_sample: Base64-encoded WAV audio to clone the voice from.  When supplied
                the model uses this audio as the reference speaker instead of a preset.
            cfg_scale: Classifier-free guidance scale.
            job_id: Optional job identifier used for output file naming.
            output_type: Output format selector (currently only "audio_path" is supported).
        """
        gen_timer = self._new_gen_timer(job_id)

        self._assert_model_init()

        self.running = True

        _tmp_voice_path: Optional[str] = None
        try:
            if not self.voice_mapper:
                raise ValueError("VoiceMapper not initialized")

            if voice_sample is not None:
                if job_id is not None:
                    # Save the input audio for debugging, mirroring how other wrappers
                    # save their inputs (e.g. /tmp/{job_id}.png for images).
                    # Use basename to strip any path-traversal characters from job_id.
                    safe_job_id = os.path.basename(job_id)
                    debug_voice_path = f"/tmp/{safe_job_id}_voice_sample.wav"
                    audio_bytes = base64.b64decode(voice_sample)
                    with open(debug_voice_path, "wb") as out_f:
                        out_f.write(audio_bytes)
                    logging.info(
                        f"Saved voice_sample to {debug_voice_path} ({len(audio_bytes)} bytes)."
                    )
                    voice_path = debug_voice_path
                    # _tmp_voice_path stays None: debug file is not cleaned up so it
                    # remains available for post-mortem inspection.
                else:
                    # No job_id available (e.g. during warmup): fall back to a temp file
                    # that is cleaned up in the finally block.
                    _tmp_voice_path = self._decode_voice_sample_to_tmp_file(voice_sample)
                    voice_path = _tmp_voice_path
                logging.info("Using cloned voice from provided voice_sample.")
            else:
                voice_path = self.voice_mapper.get_voice_path(voice)
                logging.info(f"Using voice: {voice} -> {voice_path}")
            voice_samples = [voice_path]

            # https://github.com/microsoft/VibeVoice/blob/main/demo/inference_from_file.py
            if not self.processor:
                raise ValueError("Processor not initialized")
            gen_timer.start("processor")
            inputs = self.processor(
                text=["Speaker 0:" + text + "\n"],  # Wrap in list for batch processing
                voice_samples=[voice_samples],  # Wrap in list for batch processing
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            inputs = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in inputs.items()
            }
            gen_timer.end("processor")

            gen_timer.start("vibevoice")
            if self.vibevoice is None:
                raise RuntimeError("VibeVoice model not loaded")
            outputs = await asyncio.to_thread(
                self.vibevoice.generate,
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={'do_sample': False},
                verbose=True,
            )
            output_path = "/tmp/file.wav"
            if job_id is not None:
                output_path = f"/tmp/file_{job_id}.wav"
            if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                speech_output = outputs.speech_outputs[0]
                await asyncio.to_thread(
                    self.processor.save_audio,
                    speech_output,
                    output_path=output_path,
                )
            gen_timer.end("vibevoice")

            return output_path
        finally:
            self.running = False
            gen_timer.end("total")
            self._cleanup_tmp_voice_file(_tmp_voice_path)

    async def get_rest_args(
        self,
        data_json: Dict[str, Union[str, int, float]]
    ) -> Dict[str, Any]:
        """Get REST argguments."""
        if data_json is None:
            raise ValueError("Missing JSON body")

        job_id = data_json.get("job_id", None)

        text = data_json.get("text", None)
        if text is None:
            raise ValueError("Missing 'text' parameter")
        voice = data_json.get("voice", "af_heart")
        voice_sample = data_json.get("voice_sample", None)
        args: Dict[str, Any] = {
            "job_id": job_id,
            "text": text,
            "voice": voice,
        }
        if voice_sample is not None:
            args["voice_sample"] = voice_sample
        return {
            "task": self.model_name,
            "args": args,
        }
