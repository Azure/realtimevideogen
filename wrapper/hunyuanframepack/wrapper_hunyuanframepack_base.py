import logging
import math
import time
import tempfile
import aiofiles

import numpy as np

from typing import Union
from typing import Optional
from typing import Dict
from typing import Any
from typing import Tuple

import torch
import torch.distributed as dist

from torch import inference_mode

from model_timing import GenTimer
from wrapper_usp import USPGeneration
from media_utils import save_bcthw_as_mp4

from PIL import Image

from diffusers import AutoencoderKLHunyuanVideo
from diffusers import HunyuanVideoFramepackPipeline
from diffusers import FlowMatchEulerDiscreteScheduler

from diffusers_helper.hunyuan import vae_decode
from diffusers_helper.hunyuan import vae_encode
from diffusers_helper.hunyuan import encode_prompt_conds
from diffusers_helper.utils import crop_or_pad_yield_mask
from diffusers_helper.utils import repeat_to_batch_size
from diffusers_helper.utils import resize_and_center_crop
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import get_flux_sigmas_from_mu
from diffusers_helper.k_diffusion.uni_pc_fm import FlowMatchUniPC
from diffusers_helper.k_diffusion.wrapper import fm_wrapper

from transformers import LlamaModel
from transformers import CLIPTextModel
from transformers import LlamaTokenizerFast
from transformers import CLIPTokenizer
from transformers import SiglipImageProcessor
from transformers import SiglipVisionModel

from image_utils import base64_to_img

if torch.cuda.is_available():
    from hunyuanframepack_xfuser import parallelize_transformer

from xfuser.config import EngineConfig
from xfuser.core.distributed import initialize_runtime_state
from xfuser.core.distributed import get_runtime_state


def get_hidden_size(
    height: int,
    width: int,
) -> int:
    lat_h = height / 8
    lat_w = width / 8
    lat_h_pad4 = (lat_h + 3) // 4 * 4
    lat_w_pad4 = (lat_w + 3) // 4 * 4
    lat_h_pad8 = (lat_h + 7) // 8 * 8
    lat_w_pad8 = (lat_w + 7) // 8 * 8
    dim = int(9 * lat_h / 2 * lat_w / 2 + lat_h * lat_w / 2 + lat_h_pad4
              / 4 * lat_w_pad4 / 4 + 4 * lat_h_pad8 / 8 * lat_w_pad8 / 8)
    return dim


class HunyuanFramePackBase(USPGeneration):
    def __init__(
        self,
        model_name: str = "hunyuanframepack",
        framepack_model_name: str = "lllyasviel/FramePackI2V_HY",
        engine_config: EngineConfig = None,
        param_dtype: torch.dtype = torch.bfloat16,
        enable_tiling: bool = False,
        enable_slicing: bool = False,
    ) -> None:
        super().__init__(
            model_name=model_name,
            engine_config=engine_config,
            param_dtype=param_dtype,
        )

        self.enable_tiling = enable_tiling
        self.enable_slicing = enable_slicing

        # Model components
        self.text_encoder: Optional[LlamaModel] = None
        self.text_encoder_2: Optional[CLIPTextModel] = None
        self.tokenizer: Optional[LlamaTokenizerFast] = None
        self.tokenizer_2: Optional[CLIPTokenizer] = None
        self.vae: Optional[AutoencoderKLHunyuanVideo] = None
        self.scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None
        self.feature_extractor: Optional[SiglipImageProcessor] = None
        self.image_encoder: Optional[SiglipVisionModel] = None
        self.transformer: Optional[HunyuanVideoTransformer3DModelPacked] = None

        # Model features
        self.framepack_model_name = framepack_model_name
        self.shift = 3.0
        self.strength = 1.0
        self.vae_stride = (4, 8, 8)  # time, height, width
        self.LAT_CHANNELS = 16
        self.num_heads = 24
        self.FPS = 30  # This is technically a constant for the model

    def __del__(self) -> None:
        # Clean models
        if self.text_encoder is not None:
            del self.text_encoder
        if self.text_encoder_2 is not None:
            del self.text_encoder_2
        if self.tokenizer is not None:
            del self.tokenizer
        if self.tokenizer_2 is not None:
            del self.tokenizer_2
        if self.vae is not None:
            del self.vae
        if self.scheduler is not None:
            del self.scheduler
        if self.feature_extractor is not None:
            del self.feature_extractor
        if self.image_encoder is not None:
            del self.image_encoder
        if self.transformer is not None:
            del self.transformer
        super().__del__()

    def load_model(self) -> None:
        assert torch.cuda.is_available()

        prev_memory = torch.cuda.memory_allocated()
        self.load_timer.start("text_encoder")
        self.text_encoder = LlamaModel.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            subfolder="text_encoder",
            torch_dtype=torch.float16).to(self.device)  # nosec B615
        self.text_encoder.eval().requires_grad_(False)
        self.load_timer.end("text_encoder")
        diff_memory = torch.cuda.memory_allocated() - prev_memory
        logging.info(f"[{self.rank}] Text memory allocated: {diff_memory / 1024 / 1024 ** 2:.2f} GB.")

        self.text_encoder_2 = CLIPTextModel.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            subfolder='text_encoder_2',
            torch_dtype=torch.float16).to(self.device)  # nosec B615
        assert self.text_encoder_2 is not None
        self.text_encoder_2.eval().requires_grad_(False)

        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            subfolder='tokenizer')  # nosec B615

        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            subfolder='tokenizer_2')  # nosec B615

        prev_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        self.load_timer.start("vae")
        self.vae = AutoencoderKLHunyuanVideo.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            subfolder='vae',
            torch_dtype=torch.float16).to(self.device)
        assert self.vae is not None
        self.vae.eval().requires_grad_(False)

        if not self.enable_tiling:
            logging.info(f"[{self.rank}] Disabling tiling for VAE.")
            self.vae.disable_tiling()
        else:
            logging.info(f"[{self.rank}] Enabling tiling for VAE.")
            self.vae.enable_tiling()

        if not self.enable_slicing:
            logging.info(f"[{self.rank}] Disabling slicing for VAE.")
            self.vae.disable_slicing()
        else:
            logging.info(f"[{self.rank}] Enabling slicing for VAE.")
            self.vae.enable_slicing()
        self.load_timer.end("vae")

        diff_memory = torch.cuda.memory_allocated() - prev_memory
        logging.info(f"[{self.rank}] VAE memory allocated: {diff_memory / 1024 / 1024 ** 2:.2f} GB.")

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            subfolder='scheduler',
            torch_dtype=torch.float16)

        self.feature_extractor = SiglipImageProcessor.from_pretrained(
            "lllyasviel/flux_redux_bfl",
            subfolder='feature_extractor')  # nosec B615

        prev_memory = torch.cuda.memory_allocated()
        self.load_timer.start("image_encoder")
        self.image_encoder = SiglipVisionModel.from_pretrained(
            "lllyasviel/flux_redux_bfl",
            subfolder='image_encoder',
            torch_dtype=torch.float16).to(self.device)  # nosec B615
        self.image_encoder.eval().requires_grad_(False)
        self.load_timer.end("image_encoder")
        diff_memory = torch.cuda.memory_allocated() - prev_memory
        logging.info(f"[{self.rank}] Img memory allocated: {diff_memory / 1024 / 1024 ** 2:.2f} GB.")

        prev_memory = torch.cuda.memory_allocated()
        self.load_timer.start("dit")
        self.transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
            self.framepack_model_name,
            torch_dtype=self.param_dtype
        )
        self.transformer = self.transformer.to(self.device)
        self.transformer.eval().requires_grad_(False)
        self.load_timer.end("dit")
        diff_memory = torch.cuda.memory_allocated() - prev_memory
        logging.info(f"[{self.rank}] FramePack memory allocated: {diff_memory / 1024 / 1024 ** 2:.2f} GB.")

    def init_model_parallelism(self) -> None:
        if not dist.is_initialized() or self.world_size <= 1:
            return

        self.load_timer.start("dit_parallel")
        assert self.transformer is not None
        assert self.vae is not None
        temp_pipeline = HunyuanVideoFramepackPipeline(
            self.text_encoder,
            self.tokenizer,
            self.transformer,
            self.vae,
            self.scheduler,
            self.text_encoder_2,
            self.tokenizer_2,
            self.image_encoder,
            self.feature_extractor,
        )
        initialize_runtime_state(temp_pipeline, self.engine_config)
        get_runtime_state().set_video_input_parameters(
            batch_size=1,
        )
        parallelize_transformer(temp_pipeline)
        self.load_timer.end("dit_parallel")

    def model_compile(self) -> None:
        if not self.torch_compile:
            return

        # This started to happen with torch 2.8.0
        # Skipping dit_compile as it fails with: KeyError: op23 Set TORCHDYNAMO_VERBOSE=1"
        logging.info(f"[{self.rank}] Compiling DiT with torch.compile().")
        self.load_timer.start("dit_compile")
        self.transformer = torch.compile(
            self.transformer,
            mode="max-autotune-no-cudagraphs",
        )
        self.load_timer.end("dit_compile")

        logging.info(f"[{self.rank}] Compiling VAE with torch.compile().")
        self.load_timer.start("vae_compile")
        assert self.vae is not None
        self.vae = torch.compile(  # type: ignore[assignment]
            self.vae,
            mode="max-autotune-no-cudagraphs",
        )
        self.load_timer.end("vae_compile")

    def _assert_model_init(self) -> None:
        super()._assert_model_init()
        assert self.text_encoder is not None
        assert self.image_encoder is not None
        assert self.vae is not None
        assert self.transformer is not None

    def _assert_args(
        self,
        height: int,
        width: int,
    ) -> None:
        """Check if the image size is supported for the current parallelism."""
        if height % self.vae_stride[1] != 0:
            raise ValueError(f"Height {height} must be divisible by {self.vae_stride[1]}.")
        if width % self.vae_stride[2] != 0:
            raise ValueError(f"Width {width} must be divisible by {self.vae_stride[2]}.")
        hidden_size = get_hidden_size(height, width)
        if hidden_size % self.world_size != 0:
            raise ValueError(f"{height}x{width} ({hidden_size}) not supported for {self.world_size} GPUs.")

    @inference_mode()
    async def warmup(self) -> None:
        logging.info(f"[{self.rank}] Warmup for Hunyuan FramePack ({self.model_name}) generation.")
        await self.generate(
            job_id="warmup",
            img=Image.new("RGB", (768, 512), (255, 255, 255)),
            prompt="Warmup prompt",
            neg_prompt="",
            height=512,
            width=768,
            num_frames=1 + 4,
            sampling_steps=5)

    def _encode_text(
        self,
        gen_timer: GenTimer,
        prompt: str,
        neg_prompt: str,
        cfg: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Text encoder for Hunyuan FramePack generation."""
        gen_timer.start("text_encoder")
        llama_vec, clip_l_pooler = encode_prompt_conds(
            prompt,
            self.text_encoder,
            self.text_encoder_2,
            self.tokenizer,
            self.tokenizer_2)
        if cfg == 1:
            llama_vec_n = torch.zeros_like(llama_vec)
            clip_l_pooler_n = torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
                neg_prompt,
                self.text_encoder,
                self.text_encoder_2,
                self.tokenizer,
                self.tokenizer_2)
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        assert self.transformer is not None
        llama_vec = llama_vec.to(self.transformer.dtype)
        llama_vec_n = llama_vec_n.to(self.transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(self.transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(self.transformer.dtype)
        gen_timer.end("text_encoder")

        return llama_vec, llama_attention_mask, clip_l_pooler, llama_vec_n, llama_attention_mask_n, clip_l_pooler_n

    def _process_image(
        self,
        gen_timer: GenTimer,
        img: Image.Image,
        height: int,
        width: int,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """ Process image for Hunyuan FramePack generation."""
        # h,w,RGB -> 1,RGB,1,h,w ([544,704,3] -> [1,3,1,544,704])
        gen_timer.start("image_preprocess")
        # We may resize image for parallelism; 1024x720 works with SP=8
        img_resized = img.resize((width, height), Image.Resampling.LANCZOS)
        input_image = np.array(img_resized)
        t0_img = time.time()
        H, W, C = input_image.shape
        if C != 3:
            raise ValueError(f"Input image must be RGB: {input_image.shape}")
        # The model works with other resolutions, skip buckets
        # height, width = find_nearest_bucket(H, W, resolution=640) # 720x1280 -> 480x832
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        input_image_pt = torch.from_numpy(input_image_np).float() / (255.0 / 2.0) - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
        gen_timer.end("image_preprocess")
        if self.rank == 0:
            logging.info(
                f"[{self.rank}] Image processing time: {time.time() - t0_img:.3f} seconds "
                f"img:{W}x{H}, video:{width}x{height}.")
        return input_image_np, input_image_pt

    def _clip_vision(
        self,
        gen_timer: GenTimer,
        input_image_np: np.ndarray,
    ) -> torch.Tensor:
        """ CLIP Vision encoder for Hunyuan FramePack generation."""
        # (1, 729, 1152)
        gen_timer.start("image_encoder")
        t0_clip = time.time()
        image_encoder_output = hf_clip_vision_encode(input_image_np, self.feature_extractor, self.image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        assert self.transformer is not None
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(self.transformer.dtype)
        gen_timer.end("image_encoder")
        if self.rank == 0:
            logging.info(f"[{self.rank}] CLIP Vision encoding time: {time.time() - t0_clip:.3f} seconds.")
        return image_encoder_last_hidden_state

    @inference_mode()
    async def generate(
        self,
        img: Image.Image,
        prompt: str,
        neg_prompt: str = "",
        height: int = 512,
        width: int = 768,
        num_frames: int = 1 + 80,
        sampling_steps: int = 25,  # 10
        # latent frames for every Hunyuan Video window: 9->36 pixel frames -> 1.2 seconds
        latent_window_size: int = 9,
        cfg: float = 1.0,
        distilled_guidance_scale: float = 10.0,
        guidance_rescale: int = 0,
        save_intermediate: Optional[str] = None,
        job_id: Optional[str] = None,
        output_type: str = "tensor"
    ) -> torch.Tensor:
        raise NotImplementedError("Implement generate() in subclasses.")

    @inference_mode()
    def vae_decode(
        self,
        latents: torch.Tensor,
        job_id: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Latent -> Pixels
        """
        gen_timer = self._new_gen_timer(job_id)

        assert self.vae is not None
        assert latents is not None
        assert isinstance(latents, torch.Tensor)
        if latents.ndim != 5:  # B, C, T, H, W
            raise ValueError(f"Latents must be a 5D tensor (B, C, T, H, W), got {latents.ndim}D.")
        if latents.shape[1] != 16:
            raise ValueError(f"Latents must have 16 channels, got {latents.shape[1]} channels.")

        try:
            gen_timer.start("vae_decoder")
            latents = latents.to(self.device, dtype=self.param_dtype)
            pixels = vae_decode(latents, self.vae)
            gen_timer.end("vae_decoder")
            return pixels
        finally:
            gen_timer.end("total")

    @inference_mode()
    def vae_encode(
        self,
        pixels: torch.Tensor,
        job_id: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Pixels -> Latent
        """
        gen_timer = self._new_gen_timer(job_id)

        assert self.vae is not None
        assert pixels is not None
        assert isinstance(pixels, torch.Tensor)
        if pixels.ndim != 5:
            raise ValueError(f"Pixels must be a 5D tensor (B, C, T, H, W), got {pixels.ndim}D.")
        if pixels.shape[1] != 3:
            raise ValueError(f"Pixels must have 3 channels (RGB), got {pixels.shape[1]} channels.")

        try:
            gen_timer.start("vae_encoder")
            pixels = pixels.to(self.device, dtype=self.param_dtype)
            latents = vae_encode(pixels, self.vae)
            gen_timer.end("vae_encoder")
            return latents
        finally:
            gen_timer.end("total")

    @inference_mode()
    def _sample_hunyuan(
        self,
        it0: int,
        gen_timer: GenTimer,
        initial_latent: Optional[torch.Tensor] = None,
        concat_latent: Optional[torch.Tensor] = None,
        strength: float = 1.0,
        width: int = 512,
        height: int = 512,
        frames: int = 16,
        real_guidance_scale: float = 1.0,
        distilled_guidance_scale: float = 6.0,
        guidance_rescale: float = 0.0,
        num_inference_steps: int = 25,
        batch_size: Optional[int] = None,
        generator: Optional[Any] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        prompt_poolers: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_poolers: Optional[torch.Tensor] = None,
        negative_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if batch_size is None:
            assert prompt_embeds is not None
            batch_size = int(prompt_embeds.shape[0])

        # Random noise
        # B, C, T, H, W (1, 16, X, 8, 8)
        LAT_CHANNELS = 16
        assert generator is not None
        latents = torch.randn(
            (
                batch_size,
                LAT_CHANNELS,
                (frames + 3) // self.vae_stride[0],
                height // self.vae_stride[1],
                width // self.vae_stride[2]
            ),
            generator=generator, device=generator.device
        ).to(device=self.device, dtype=torch.float32)

        mu = math.log(self.shift)

        sigmas = get_flux_sigmas_from_mu(num_inference_steps, mu).to(self.device)

        k_model = fm_wrapper(self.transformer)

        if initial_latent is not None:
            sigmas = sigmas * strength
            first_sigma = sigmas[0].to(device=self.device, dtype=torch.float32)
            initial_latent = initial_latent.to(device=self.device, dtype=torch.float32)
            latents = initial_latent.float() * (1.0 - first_sigma) + latents.float() * first_sigma

        if concat_latent is not None:
            concat_latent = concat_latent.to(latents)

        distilled_guidance = torch.tensor([distilled_guidance_scale * 1000.0]
                                          * batch_size).to(device=self.device, dtype=self.param_dtype)

        prompt_embeds = repeat_to_batch_size(prompt_embeds, batch_size)
        prompt_embeds_mask = repeat_to_batch_size(prompt_embeds_mask, batch_size)
        prompt_poolers = repeat_to_batch_size(prompt_poolers, batch_size)
        negative_prompt_embeds = repeat_to_batch_size(negative_prompt_embeds, batch_size)
        negative_prompt_embeds_mask = repeat_to_batch_size(negative_prompt_embeds_mask, batch_size)
        negative_prompt_poolers = repeat_to_batch_size(negative_prompt_poolers, batch_size)
        concat_latent = repeat_to_batch_size(concat_latent, batch_size)

        sampler_kwargs = dict(
            dtype=self.param_dtype,
            cfg_scale=real_guidance_scale,
            cfg_rescale=guidance_rescale,
            concat_latent=concat_latent,
            positive=dict(
                pooled_projections=prompt_poolers,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_embeds_mask,
                guidance=distilled_guidance,
                **kwargs,
            ),
            negative=dict(
                pooled_projections=negative_prompt_poolers,
                encoder_hidden_states=negative_prompt_embeds,
                encoder_attention_mask=negative_prompt_embeds_mask,
                guidance=distilled_guidance,
                **(kwargs if negative_kwargs is None else {**kwargs, **negative_kwargs}),
            )
        )

        sampler = FlowMatchUniPC(k_model, extra_args=sampler_kwargs)

        # generated_latents = sampler.sample(latents, sigmas=sigmas)
        order = min(3, len(sigmas) - 2)
        model_prev_list, t_prev_list = [], []
        for it1 in range(len(sigmas) - 1):
            gen_timer.start(f"dit_{it0:03d}_{it1:03d}")
            vec_t = sigmas[it1].expand(latents.shape[0])
            if it1 == 0:
                model_prev_list = [sampler.model_fn(latents, vec_t)]
                t_prev_list = [vec_t]
            elif it1 < order:
                init_order = it1
                latents, model_x = sampler.update_fn(latents, model_prev_list, t_prev_list, vec_t, init_order)
                model_prev_list.append(model_x)
                t_prev_list.append(vec_t)
            else:
                latents, model_x = sampler.update_fn(latents, model_prev_list, t_prev_list, vec_t, order)
                model_prev_list.append(model_x)
                t_prev_list.append(vec_t)
            model_prev_list = model_prev_list[-order:]
            t_prev_list = t_prev_list[-order:]
            gen_timer.end(f"dit_{it0:03d}_{it1:03d}")
        generated_latents = model_prev_list[-1]

        return generated_latents

    async def _output_video(
        self,
        job_id: Optional[str],
        gen_timer: GenTimer,
        pixels: torch.Tensor,
        output_type: str = "tensor",  # "tensor", "video_binary", "video_path"
    ) -> Union[torch.Tensor, str, bytes, None]:
        gen_timer.start("output")
        try:
            if output_type == "tensor":
                return pixels

            if output_type in ("video_binary", "video_path"):
                if not job_id:
                    video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                else:
                    video_path = f"/tmp/{job_id}.mp4"
                video_path = save_bcthw_as_mp4(
                    pixels,
                    video_path,
                    fps=self.FPS)
                if output_type == "video_path":
                    return video_path

                # video_binary
                async with aiofiles.open(video_path, "rb") as file:
                    video_binary = await file.read()
                return video_binary

            logging.error(f"Unknown output type: {output_type}")
            return None
        finally:
            gen_timer.end("output")

    async def get_rest_args(
        self,
        data_json: Dict[str, Union[str, int, float]]
    ) -> Dict[str, Any]:
        if data_json is None or not isinstance(data_json, dict):
            raise ValueError("Missing JSON body")

        job_id = data_json.get("job_id", None)

        img_base64 = data_json.get("img", None)
        if img_base64 is None:
            raise ValueError("Missing 'img' parameter")
        if not isinstance(img_base64, str):
            raise ValueError("'img' parameter must be a base64 string")
        img = base64_to_img(img_base64)

        prompt = data_json.get("prompt", None)
        if prompt is None:
            raise ValueError("Missing 'prompt' parameter")

        neg_prompt = data_json.get("neg_prompt", "")

        height = int(data_json.get("height", 480))
        width = int(data_json.get("width", 640))
        num_frames = int(data_json.get("num_frames", 1 + 16))
        steps = int(data_json.get("sampling_steps", 5))
        latent_window_size = int(data_json.get("latent_window_size", 9))
        cfg = float(data_json.get("cfg", 1.0))
        distilled_guidance_scale = float(data_json.get("distilled_guidance_scale", 10.0))
        guidance_rescale = int(data_json.get("guidance_rescale", 0.0))
        save_intermediate = data_json.get("save_intermediate", None)
        output_type = data_json.get("output_type", "tensor")

        video_seconds = data_json.get("video_seconds", None)
        if video_seconds is not None:
            VAE_FRAMES = self.vae_stride[0]
            num_frames = int(video_seconds * self.FPS)
            num_frames = 1 + ((num_frames - 1) // VAE_FRAMES) * VAE_FRAMES  # 4n + 1

        return {
            "task": self.model_name,
            "args": {
                "job_id": job_id,
                "img": img,
                "prompt": prompt,
                "neg_prompt": neg_prompt,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "sampling_steps": steps,
                "latent_window_size": latent_window_size,
                "cfg": cfg,
                "distilled_guidance_scale": distilled_guidance_scale,
                "guidance_rescale": guidance_rescale,
                "save_intermediate": save_intermediate,
                "output_type": output_type,
            }
        }
