"""
Wrapper for Hunyuan-Avatar video generation.
"""
import os
import einops
import librosa
import logging
import datetime
import tempfile
import aiofiles
import numpy as np

from typing_extensions import override  # for Python 3.11 compatibility
from typing import Union
from typing import List
from typing import Optional
from typing import Dict
from typing import Any

import torch
import torch.distributed as dist
from torch import inference_mode

from hymm_sp.config import parse_args
from hymm_sp.data_kits.face_align import AlignImage
from hymm_sp.modules.parallel_states import nccl_info
from hymm_sp.modules.parallel_states import initialize_sequence_parallel_state
from sample_inference_audio import HunyuanVideoSampler
from encode_data import VideoAudioTextLoaderVal

from transformers import WhisperModel
from transformers import AutoFeatureExtractor

from PIL import Image

from xfuser.config import EngineConfig

from model_timing import LoadTimer
from model_timing import GenTimer

from wrapper_usp import USPGeneration

from image_utils import base64_to_img
from media_utils import base64_to_audio_file
from media_utils import empty_audio_file
from media_utils import save_video_audio


class HunyuanAvatarGeneration(USPGeneration):
    """
    Hunyuan-Avatar video generation wrapper.
    """

    def __init__(
        self,
        engine_config: EngineConfig = None,
        param_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(
            "hunyuanavatar",
            engine_config,
            param_dtype)

        # Model components
        self.hunyuan_video_sampler: Optional[HunyuanVideoSampler] = None
        self.wav2vec: Optional[WhisperModel] = None
        self.align_instance: Optional[AlignImage] = None
        self.feature_extractor: Optional[AutoFeatureExtractor] = None
        self.text_encoder: Optional[torch.nn.Module] = None
        self.text_encoder_2: Optional[torch.nn.Module] = None

        # Model features
        self.image_size = 704
        self.vae_stride = (4, 8, 8)  # time, height, width
        self.MAX_FRAMES = 1 + 80
        self.FPS = 12.5  # only support either 12.5 or 25 fps

        # Load models
        self.models_root_path = '/hunyuanavatar/weights/ckpts'
        os.environ['MODEL_BASE'] = '/hunyuanavatar/weights'

        self.args = parse_args()
        self.args.ckpt = f"{self.models_root_path}/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt"
        self.args.prompt_template_video = None

    def __del__(self) -> None:
        # Ensure the model is properly cleaned up
        if self.hunyuan_video_sampler is not None:
            self.hunyuan_video_sampler = None
        if self.wav2vec is not None:
            self.wav2vec = None
        if self.align_instance is not None:
            self.align_instance = None
        if self.feature_extractor is not None:
            self.feature_extractor = None
        if self.text_encoder is not None:
            self.text_encoder = None
        if self.text_encoder_2 is not None:
            self.text_encoder_2 = None

        super().__del__()

    def init_parallelism(self) -> None:
        self.load_timer.start("torch_dist")  # type: ignore[has-type]

        logging.info("Initializing distributed environment...")

        if "MASTER_ADDR" not in os.environ:
            logging.info("MASTER_ADDR not set, skipping distributed initialization.")
            return

        self.rank = int(os.getenv("RANK", 0))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))

        self.device_id = self.local_rank
        self.device = torch.device(f"cuda:{self.device_id}")

        torch.cuda.set_device(self.local_rank)

        if self.world_size <= 1:
            logging.info("World size is 1, skipping distributed initialization.")
            self.load_timer.end("torch_dist")  # type: ignore[has-type]
            return

        if not dist.is_initialized():
            logging.info(
                f"Initializing distributed environment with local_rank: {self.rank}, world_size: {self.world_size}.")
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=datetime.timedelta(seconds=2**31 - 1),
                world_size=self.world_size,
                rank=self.rank)
        else:
            logging.info("Distributed environment already initialized.")

        torch.manual_seed(self.base_seed)
        torch.cuda.manual_seed_all(self.base_seed)

        logging.info(
            f"Distributed environment initialized with rank: {dist.get_rank()}, world_size: {dist.get_world_size()}.")

        initialize_sequence_parallel_state(self.world_size)

        self.load_timer.end("torch_dist")  # type: ignore[has-type]

    def load_model(self) -> None:
        self.rank = 0
        self.vae_dtype = torch.float16
        self.device = torch.device("cuda")
        if nccl_info.sp_size > 1:
            self.device = torch.device(f"cuda:{dist.get_rank()}")
            self.rank = dist.get_rank()

        self.load_timer = LoadTimer()

        self.load_timer.start("hunyuan_video_sampler")
        self.hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
            f"{self.models_root_path}/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
            args=self.args,
            device=self.device)
        self.load_timer.end("hunyuan_video_sampler")

        # Get the updated args
        self.args = self.hunyuan_video_sampler.args

        # Load the wav2vec model
        self.load_timer.start("wav2vec")
        self.wav2vec = WhisperModel.from_pretrained(
            f"{self.models_root_path}/whisper-tiny/"  # nosec B615 - local path
        ).to(device=self.device, dtype=torch.float32)
        self.wav2vec.requires_grad_(False)
        self.load_timer.end("wav2vec")

        # Load the align instance
        self.load_timer.start("align_instance")
        det_path = f"{self.models_root_path}/det_align/detface.pt"
        self.align_instance = AlignImage("cuda", det_path=det_path)
        self.load_timer.end("align_instance")

        # Load the feature extractor
        self.load_timer.start("feature_extractor")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            f"{self.models_root_path}/whisper-tiny/")  # nosec B615 - local path
        self.load_timer.end("feature_extractor")

        self.text_encoder = self.hunyuan_video_sampler.text_encoder
        self.text_encoder_2 = self.hunyuan_video_sampler.text_encoder_2

        self.data_loader = VideoAudioTextLoaderVal(
            image_size=self.image_size,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            feature_extractor=self.feature_extractor,
        )

    def _assert_model_init(self) -> None:
        assert self.hunyuan_video_sampler is not None, "HunyuanVideoSampler is not initialized."
        assert self.wav2vec is not None, "Wav2Vec model is not initialized."
        assert self.align_instance is not None, "AlignImage instance is not initialized."
        assert self.feature_extractor is not None, "Feature extractor is not initialized."
        assert self.text_encoder is not None, "Text encoder is not initialized."
        assert self.text_encoder_2 is not None, "Text encoder 2 is not initialized."

    @inference_mode()
    async def warmup(self) -> None:
        logging.info(f"[{self.rank}] Warmup for Hunyuan Avatar generation.")
        empty_img = Image.new("RGB", (640, 480), color=(255, 255, 255))
        empty_audio_path = empty_audio_file(0.5)  # 0.5 seconds of silence
        await self.generate(
            img=empty_img,
            audio_path=empty_audio_path,
            prompt="A warmup generation",
            width=640,
            height=480,
            sampling_steps=2)
        if os.path.exists(empty_audio_path):
            os.remove(empty_audio_path)

    @override
    @inference_mode()
    async def generate(
        self,
        img: Image.Image,
        audio_path: str,
        prompt: str,
        width: int = 1280,
        height: int = 720,
        sampling_steps: int = 30,  # 10
        audio_scale: float = 1.0,  # 1.0 for audio, 0.0 for no audio influence TODO not used
        cfg_scale: float = 5.0,  # prompt
        audio_cfg_scale: float = 5.0,  # TODO not used
        job_id: Optional[str] = None,
        output_type: str = "pil",  # "pil", "video_binary", "video_path"
    ) -> List[Image.Image]:
        """
        Generate a video from an image, a piece of audio, and a text prompt.
        Args:
            img (Image.Image): Input image.
            audio_path (str): Path to the audio file.
            prompt (str): Text prompt for video generation.
            neg_prompt (str, optional): Negative text prompt. Defaults to "".
            width (int, optional): Width of the output video. Defaults to 1280.
            height (int, optional): Height of the output video. Defaults to 720.
            sampling_steps (int, optional): Number of sampling steps. Defaults to 30.
        """
        gen_timer = self._new_gen_timer(job_id)

        self.running = True  # Mark running to avoid concurrent calls

        try:
            self._assert_model_init()

            audio_duration = librosa.get_duration(path=audio_path)  # TODO figure audio
            num_frames = int(self.FPS * audio_duration // self.vae_stride[0]) * self.vae_stride[0] + 5
            if num_frames > self.MAX_FRAMES:
                raise ValueError(f"Audio {audio_duration:.2f}s exceeds {self.MAX_FRAMES} frames")
            # num_frames = min(num_frames, self.MAX_FRAMES)
            lat_num_frames = (num_frames - 1) // self.vae_stride[0] + 1

            if self.rank == 0:
                logging.info(
                    f"[{self.rank}] Audio:{audio_duration:.2f}s #frames:{num_frames} #lat_frames:{lat_num_frames}.")

            # Prepare the data
            gen_timer.start("encoding_inputs")
            img_resized = img.resize((width, height), Image.LANCZOS)
            results = self.data_loader.encode_data(
                ref_image=img_resized,
                audio_path=audio_path,
                prompt=prompt,
                fps=self.FPS,
            )
            gen_timer.end("encoding_inputs")

            self.args.cfg_scale = cfg_scale
            self.args.infer_steps = sampling_steps
            gen_timer.start("hunyuanavatar_generation")
            assert self.hunyuan_video_sampler is not None
            samples = self.hunyuan_video_sampler.predict(
                self.args,
                results,
                self.wav2vec,
                self.feature_extractor,
                self.align_instance)
            gen_timer.end("hunyuanavatar_generation")

            sample = samples['samples'][0].unsqueeze(0)  # de-noised latent, (bs, 16, t//4, h//8, w//8)
            # sample = sample[:, :, :results["audio_len"][0]]
            sample = sample[:, :, :results["audio_len"]]

            logging.info(f"[{self.rank}] Sample shape after slicing: {sample.shape}.")
            video = einops.rearrange(sample[0], "c f h w -> f h w c")
            video = (video * 255.).data.cpu().numpy().astype(np.uint8)  # （f h w c)

            torch.cuda.empty_cache()

            final_frames = []
            for frame in video:
                final_frames.append(frame)
            final_frames = np.stack(final_frames, axis=0)

            return await self._output_video(  # type: ignore[return-value]
                job_id,
                gen_timer,
                audio_path,
                final_frames,
                output_type)
        finally:
            self.running = False
            gen_timer.end("total")

    async def _output_video(
        self,
        job_id: Optional[str],
        gen_timer: GenTimer,
        audio_path: str,
        video_frames: np.ndarray,
        output_type: str = "pil",  # "pil", "video_binary", "video_path"
    ) -> Union[List[Image.Image], str, bytes, None]:
        gen_timer.start("output")
        try:
            if output_type == "pil":
                return video_frames

            if output_type in ("video_binary", "video_path"):
                if not job_id:
                    video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                else:
                    video_path = f"/tmp/{job_id}.mp4"
                video_path = await save_video_audio(
                    video_content=video_frames,
                    audio_path=audio_path,
                    out_video_path=video_path,
                    fps=self.FPS)
                if output_type == "video_path":
                    return video_path

                # video_binary
                async with aiofiles.open(video_path, "rb") as file:
                    video_binary = await file.read()
                return video_binary

            logging.error(f"Unknown output type: {output_type}")
            return None  # type: ignore[return-value]
        finally:
            gen_timer.end("output")

    async def get_rest_args(self, data_json: Dict[str, str]) -> Dict[str, Any]:
        if data_json is None:
            raise ValueError("Missing JSON body")

        job_id = data_json.get("job_id", None)

        img_base64 = data_json.get("img", None)
        if img_base64 is None:
            raise ValueError("Missing 'img' parameter")
        img = base64_to_img(img_base64)

        audio_base64 = data_json.get("audio", None)
        if audio_base64 is None:
            raise ValueError("Missing 'audio' parameter")
        audio_path = None
        if not job_id:
            audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        else:
            audio_path = f"/tmp/{job_id}.wav"
        audio_path = await base64_to_audio_file(
            audio_base64,
            audio_path=audio_path)

        prompt = data_json.get("prompt", None)
        if prompt is None:
            raise ValueError("Missing 'prompt' parameter")

        height = int(data_json.get("height", 480))
        width = int(data_json.get("width", 640))
        steps = int(data_json.get("sampling_steps", 10))
        audio_scale = float(data_json.get("audio_scale", 1.0))
        cfg_scale = float(data_json.get("cfg_scale", 5.0))
        audio_cfg_scale = float(data_json.get("audio_cfg_scale", 5.0))
        return {
            "task": self.model_name,
            "args": {
                "job_id": job_id,
                "img": img,
                "prompt": prompt,
                "audio_path": audio_path,
                "width": width,
                "height": height,
                "sampling_steps": steps,
                "audio_scale": audio_scale,
                "cfg_scale": cfg_scale,
                "audio_cfg_scale": audio_cfg_scale,
            }
        }
