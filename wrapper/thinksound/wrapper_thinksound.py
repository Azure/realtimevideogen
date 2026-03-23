"""
ThinkSound wrapper for video-to-audio generation.
Based on: https://github.com/FunAudioLLM/ThinkSound

https://github.com/FunAudioLLM/ThinkSound/blob/master/extract_latents.py
https://github.com/FunAudioLLM/ThinkSound/blob/master/predict.py
"""

import logging
import os
import tempfile
import shutil
import uuid
import cv2
import subprocess
import wave
import io
import aiofiles

import numpy as np

from typing import override
from typing import Optional
from typing import Dict
from typing import Any
from typing import Tuple

import torch
import torch.distributed as dist
from torch import inference_mode

from wrapper_model import ModelGeneration

from data_utils.v2a_utils.vggsound_224_no_audio import VGGSound
from data_utils.v2a_utils.feature_utils_224 import FeaturesUtils

from ThinkSound.models import create_model_from_config
from ThinkSound.models.utils import load_ckpt_state_dict


class ThinkSoundGeneration(ModelGeneration):
    """ThinkSound video-to-audio generation wrapper."""

    def __init__(
        self,
        model_name: str = "thinksound",
        param_dtype: torch.dtype = torch.float16,
        use_half: bool = True,
        sample_rate: int = 44100,
    ) -> None:
        super().__init__(model_name)

        self.param_dtype = param_dtype
        self.use_half = use_half
        self.sample_rate = sample_rate

        # ThinkSound specific configuration
        self.synchformer_ckpt = "FunAudioLLM/ThinkSound/synchformer_state_dict.pth"
        self.diffusion_objective = "rectified_flow"  # or "v"
        self.cfg_scale = 5.0
        self.num_steps = 24

        # Model components
        self.feature_extractor: Optional[OptimizedFeaturesUtils] = None
        self.diffusion_model: Optional[torch.nn.Module] = None

        # GPU info
        self.GPU = None
        if torch.cuda.is_available():
            self.GPU = torch.cuda.get_device_name(0)

    def __del__(self) -> None:
        if self.feature_extractor is not None:
            self.feature_extractor = None
        if self.diffusion_model is not None:
            self.diffusion_model = None
        if dist.is_initialized():
            dist.destroy_process_group()

    def init_parallelism(self) -> None:
        self.load_timer.start("torch_dist")

        self.rank = int(os.getenv("RANK", 0))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))

        self.device_id = self.local_rank
        self.device = torch.device(f"cuda:{self.device_id}")

        torch.cuda.set_device(self.local_rank)

        if self.world_size > 1:
            logging.warning("ThinkSound is not optimized for multi-GPU setups (yet).")
            self.world_size = 1

        self.load_timer.end("torch_dist")

    def load_model(self) -> None:
        assert torch.cuda.is_available()

        # Setup PyTorch optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.load_timer.start("feature_extractor")
        # Initialize feature extractor with custom modifications
        self.feature_extractor = OptimizedFeaturesUtils(
            self,
            vae_ckpt=None,
            vae_config=None,
            enable_conditions=True,
            synchformer_ckpt=self.synchformer_ckpt,
            use_half=self.use_half
        )
        self.load_timer.end("feature_extractor")

        self.load_timer.start("diffusion_model")
        # https://github.com/FunAudioLLM/ThinkSound/blob/master/ThinkSound/configs/model_configs/thinksound.json
        duration = 9.0  # TODO fix harcoding
        model_config: Dict[str, Any] = {}
        model_config["sample_size"] = duration * model_config["sample_rate"]
        model_config["model"]["diffusion"]["config"]["sync_seq_len"] = 24 * int(duration)
        model_config["model"]["diffusion"]["config"]["clip_seq_len"] = 8 * int(duration)
        model_config["model"]["diffusion"]["config"]["latent_seq_len"] = round(44100 / 64 / 32 * duration)
        model = create_model_from_config(model_config)
        ckpt_dir = "FunAudioLLM/ThinkSound"
        model.load_state_dict(torch.load(ckpt_dir, weights_only=False))  # nosec B614 - trusted model
        load_vae_state = load_ckpt_state_dict("FunAudioLLM/ThinkSound/vae.ckpt")
        model.pretransform.load_state_dict(load_vae_state)
        model = model.to(self.device)
        self.diffusion_model = model
        self.load_timer.end("diffusion_model")

        logging.info(f"Loaded ThinkSound with feature extractor and diffusion model on device: {self.device}")

    def init_model_parallelism(self) -> None:
        if self.world_size > 1:
            logging.warning("ThinkSound does not support model parallelism yet.")

    def model_compile(self) -> None:
        if not self.torch_compile:
            return
        self.load_timer.start("compile")
        self.model = torch.compile(self.model)  # type: ignore[has-type]
        self.load_timer.end("compile")

    def _assert_model_init(self) -> None:
        super()._assert_model_init()
        assert self.feature_extractor is not None
        # assert self.diffusion_model is not None

    def _assert_args(self, duration_sec: float) -> None:
        if duration_sec <= 0:
            raise ValueError(f"Duration must be positive, got {duration_sec}")
        if duration_sec > 60:  # Reasonable limit
            raise ValueError(f"Duration too long: {duration_sec}s. Maximum supported is 60s.")

    async def _prepare_video_file(
        self,
        video_binary: bytes,
        session_dir: str
    ) -> Tuple[str, float]:
        """Convert video binary to MP4 file and return path and duration."""
        videos_dir = os.path.join(session_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)

        # Save video binary to temporary file
        temp_video_path = os.path.join(videos_dir, "input_video.mp4")
        async with aiofiles.open(temp_video_path, "wb") as file:
            await file.write(video_binary)

        # Convert to MP4 if needed and get duration
        final_video_path = os.path.join(videos_dir, "demo.mp4")

        # Use ffmpeg to ensure proper format and get duration
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", temp_video_path,
                "-preset", "fast",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-strict", "experimental",
                final_video_path
            ],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to convert video: {result.stderr}")

        # Get video duration using opencv
        cap = cv2.VideoCapture(final_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 1
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        duration_sec = frames / fps

        return final_video_path, duration_sec

    async def _extract_features(
        self,
        video_path: str,
        caption: str,
        caption_cot: str,
        duration_sec: float,
        session_dir: str
    ) -> str:
        """Extract features from video and text using the feature extractor."""
        assert self.feature_extractor is not None, "Feature extractor is not initialized"
        cot_dir = os.path.join(session_dir, "cot_coarse")
        os.makedirs(cot_dir, exist_ok=True)
        csv_path = os.path.join(cot_dir, "cot.csv")

        caption_cot_escaped = caption_cot.replace('"', "'")
        async with aiofiles.open(csv_path, "w", encoding="utf-8") as file:
            await file.write("id,caption,caption_cot\n")
            await file.write(f"demo,{caption},\"{caption_cot_escaped}\"\n")

        # Create dataset
        results_dir = os.path.join(session_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        dataset = VGGSound(
            root=os.path.dirname(video_path),
            tsv_path=csv_path,
            sample_rate=self.sample_rate,
            duration_sec=duration_sec,
            audio_samples=int(self.sample_rate * duration_sec),
            start_row=0,
            end_row=None,
            save_dir=results_dir
        )

        # Process single item
        data = dataset[0]  # Get the first (and only) item

        with torch.no_grad():
            output = {
                'caption': str(data['caption']),
                'caption_cot': str(data['caption_cot'])
            }

            # Extract video features
            clip_video = data['clip_video']
            clip_features = self.feature_extractor.encode_video_with_clip(clip_video.unsqueeze(0))
            output['metaclip_features'] = clip_features

            sync_video = data['sync_video']
            sync_features = self.feature_extractor.encode_video_with_sync(sync_video.unsqueeze(0))
            output['sync_features'] = sync_features

            # Extract text features
            caption_list = [data['caption']]
            metaclip_global_text_features, metaclip_text_features = self.feature_extractor.encode_text(caption_list)
            output['metaclip_global_text_features'] = metaclip_global_text_features
            output['metaclip_text_features'] = metaclip_text_features

            caption_cot_list = [data['caption_cot']]
            t5_features = self.feature_extractor.encode_t5_text(caption_cot_list)
            output['t5_features'] = t5_features

            # Convert tensors to numpy and save
            sample_output = {
                'id': 'demo',
                'caption': output['caption'],
                'caption_cot': output['caption_cot'],
                'metaclip_features': output['metaclip_features'][0],
                'sync_features': output['sync_features'][0],
                'metaclip_global_text_features': output['metaclip_global_text_features'][0],
                'metaclip_text_features': output['metaclip_text_features'][0],
                't5_features': output['t5_features'][0],
            }

            for k, v in sample_output.items():
                if isinstance(v, torch.Tensor):
                    sample_output[k] = v.float().cpu().numpy()

            features_path = os.path.join(results_dir, 'demo.npz')
            np.savez(features_path, **sample_output)

            return features_path

    @inference_mode()
    def _generate_audio_from_features(self, features_path: str, duration_sec: float) -> bytes:
        """Generate audio from extracted features using the diffusion model."""

        if self.diffusion_model is None:
            raise RuntimeError("Diffusion model not loaded")

        # Load features
        npz_data = np.load(features_path, allow_pickle=True)
        data = {
            key: npz_data[key]
            for key in npz_data.files
        }

        # Convert to tensors and prepare metadata
        metadata = [{
            'id': 'demo',
            'video_exist': torch.tensor(True),
            **{k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
               for k, v in data.items()}
        }]

        # Create dummy audio tensor (latent length calculation)
        latent_length = round(self.sample_rate / 64 / 32 * duration_sec)
        reals = torch.zeros((1, 64, latent_length), dtype=torch.float32)

        batch = (reals, metadata)

        # Generate audio using the diffusion model
        with torch.no_grad():
            audios = self._predict_step(batch)

        # Convert to audio bytes (WAV format)
        audio_np = audios[0].numpy()  # Get first audio from batch
        audio_bytes = self._numpy_to_wav_bytes(audio_np, self.sample_rate)

        return audio_bytes

    def _predict_step(
        self,
        batch: Tuple[torch.Tensor, Any]
    ) -> torch.Tensor:
        """Run the diffusion model prediction."""
        # This would be the actual prediction logic from predict.py
        # For now, return dummy audio
        reals, _ = batch  # reals, metadata
        batch_size = reals.shape[0]
        # length = reals.shape[2]

        # Generate dummy audio for testing
        dummy_audio = torch.randint(
            -32767,
            32767,
            (batch_size, int(self.sample_rate * 5)),
            dtype=torch.int16)
        return dummy_audio

    def _numpy_to_wav_bytes(
        self,
        audio_np: np.ndarray,
        sample_rate: int
    ) -> bytes:
        """Convert numpy audio array to WAV bytes."""
        # Ensure audio is in the right format
        if audio_np.dtype != np.int16:
            audio_np = (audio_np * 32767).astype(np.int16)

        # Create WAV bytes
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_np.tobytes())

        return wav_buffer.getvalue()

    @inference_mode()
    async def warmup(self) -> None:
        logging.info(f"[{self.rank}] Warmup for ThinkSound generation.")
        # Create a small dummy video for warmup
        dummy_video = np.random.randint(0, 255, (30, 224, 224, 3), dtype=np.uint8)  # 1 second at 30fps

        # Convert to video bytes (this would need a proper implementation)
        dummy_video_bytes = dummy_video  # Placeholder

        await self.generate(
            video_binary=dummy_video_bytes,
            caption="A warmup audio generation",
            caption_cot="This is a warmup to initialize the model components"
        )

    @override
    @inference_mode()
    async def generate(
        self,
        video_binary: bytes,
        caption: str = "",
        caption_cot: str = "",
        max_duration_sec: Optional[float] = None,
        job_id: Optional[str] = None,
    ) -> bytes:
        gen_timer = self._new_gen_timer(job_id)

        self._assert_model_init()

        self.running = True  # Mark running to avoid concurrent calls

        # Create temporary session directory
        unique_id = uuid.uuid4().hex[:8]
        session_dir = tempfile.mkdtemp(prefix=f"thinksound_{unique_id}_")

        try:
            # Prepare video file and get duration
            gen_timer.start("video_preparation")
            video_path, duration_sec = await self._prepare_video_file(video_binary, session_dir)

            if max_duration_sec and duration_sec > max_duration_sec:
                duration_sec = max_duration_sec

            self._assert_args(duration_sec)
            gen_timer.end("video_preparation")

            logging.info(f"Generating audio for {duration_sec:.2f}s video with caption: '{caption}'")

            # Extract features
            gen_timer.start("feature_extraction")
            features_path = await self._extract_features(
                video_path,
                caption,
                caption_cot,
                duration_sec,
                session_dir)
            gen_timer.end("feature_extraction")

            # Generate audio
            gen_timer.start("audio_generation")
            audio_bytes = self._generate_audio_from_features(features_path, duration_sec)
            gen_timer.end("audio_generation")

            return audio_bytes
        finally:
            self.running = False
            gen_timer.end("total")
            # Clean up temporary files
            shutil.rmtree(session_dir, ignore_errors=True)

    def get_health(self) -> Dict[str, Any]:
        ret = super().get_health()
        ret.update({
            "gpu": self.GPU,
            "rank": self.rank,
            "world_size": self.world_size,
            "dtype": str(self.param_dtype),
            "use_half": self.use_half,
            "sample_rate": self.sample_rate,
            "diffusion_objective": self.diffusion_objective,
        })
        return ret

    def get_rest_args(
        self,
        data_json: Dict[str, str]
    ) -> Dict[str, Any]:
        if data_json is None or not isinstance(data_json, dict):
            raise ValueError("Missing JSON body")

        video_base64 = data_json.get("video", None)
        if video_base64 is None:
            raise ValueError("Missing 'video' parameter")

        # Convert base64 video to binary
        import base64
        video_binary = base64.b64decode(video_base64)

        caption = data_json.get("caption", "")
        caption_cot = data_json.get("caption_cot", "")
        max_duration_sec: Optional[float] = None
        raw_duration = data_json.get("max_duration_sec", None)
        if raw_duration is not None:
            max_duration_sec = float(raw_duration)

        return {
            "task": self.model_name,
            "args": {
                "video_binary": video_binary,
                "caption": caption,
                "caption_cot": caption_cot,
                "max_duration_sec": max_duration_sec,
            }
        }


class OptimizedFeaturesUtils(FeaturesUtils):
    """Optimized FeaturesUtils for ThinkSound with half precision and CUDA support."""

    def __init__(
        self,
        parent_instance: Any,
        *args: Any,
        use_half: bool = True,
        **kwargs: Any
    ) -> None:
        self.parent = parent_instance
        _prev_device = torch.device("cpu")

        try:
            torch.set_default_device("cuda")
            super().__init__(*args, **kwargs)
        finally:
            torch.set_default_device(_prev_device)

        self.use_half = use_half
        if self.use_half:
            logging.info("Using half precision for models to save memory")

        # Load models to CUDA
        if self.clip_model is not None:  # type: ignore[has-type]
            self.clip_model = self._load_to_cuda(self.clip_model)  # type: ignore[has-type]

        if hasattr(self, 't5_model') and self.t5_model is not None:  # type: ignore[has-type]
            self.t5_model = self._load_to_cuda(self.t5_model)  # type: ignore[has-type]

        if self.synchformer is not None:  # type: ignore[has-type]
            self.synchformer = self._load_to_cuda(self.synchformer)  # type: ignore[has-type]

    def _load_to_cuda(self, model: Any) -> Any:
        if self.use_half:
            model = model.half()
        return model.to(self.parent.device)

    @inference_mode()
    def encode_video_with_clip(self, x: Any, batch_size: int = -1) -> Any:
        out = super().encode_video_with_clip(x.to(self.parent.device), batch_size)
        torch.cuda.empty_cache()
        return out

    @inference_mode()
    def encode_video_with_sync(self, x: Any, batch_size: int = -1) -> Any:
        x = x.to(self.parent.device)
        if self.use_half:
            x = x.half()
        out = super().encode_video_with_sync(x, batch_size)
        torch.cuda.empty_cache()
        return out

    @inference_mode()
    def encode_text(self, text_list: Any) -> Any:
        out = super().encode_text(text_list)
        torch.cuda.empty_cache()
        return out

    @inference_mode()
    def encode_t5_text(self, text: list[str]) -> torch.Tensor:
        assert self.t5_model is not None, 'T5 model is not loaded'
        assert self.t5_tokenizer is not None, 'T5 Tokenizer is not loaded'

        inputs = self.t5_tokenizer(
            text,
            truncation=True,
            max_length=77,
            padding="max_length",
            return_tensors="pt")

        inputs = {
            k: v.to(self.parent.device)
            for k, v in inputs.items()
        }

        return self.t5_model(**inputs).last_hidden_state
