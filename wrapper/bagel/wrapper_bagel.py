import logging
import os
import asyncio

from typing import override
from typing import List
from typing import Optional
from typing import Dict
from typing import Any

from PIL import Image
from copy import deepcopy

import torch
import torch.distributed as dist
from torch import inference_mode

from wrapper_model import ModelGeneration
from image_utils import base64_to_img

from modeling.bagel import BagelConfig
from modeling.bagel import Bagel
from modeling.bagel import Qwen2Config
from modeling.bagel import Qwen2ForCausalLM
from modeling.bagel import SiglipVisionConfig
from modeling.bagel import SiglipVisionModel
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae
from modeling.qwen2 import Qwen2Tokenizer

from data.transforms import ImageTransform
from data.data_utils import add_special_tokens

from accelerate import load_checkpoint_and_dispatch
from accelerate import init_empty_weights

from xfuser.config import EngineConfig


class BagelGeneration(ModelGeneration):
    def __init__(
        self,
        engine_config: EngineConfig = None,
        param_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__("bagel")

        self.engine_config = engine_config
        self.param_dtype = param_dtype

        # Parallelism
        self.gpu = torch.cuda.get_device_name(0)
        self.rank = -1
        self.world_size = -1
        self.local_rank = -1
        self.device: Optional[torch.device] = None

        # Model components
        self.tokenizer: Optional[Qwen2Tokenizer] = None
        self.new_token_ids: Optional[List[int]] = None
        self.model: Optional[Bagel] = None
        self.language_model: Optional[Qwen2ForCausalLM] = None
        self.vit_model: Optional[SiglipVisionModel] = None
        self.vae_model: Optional[Any] = None
        self.vae_transform: Optional[ImageTransform] = None
        self.vit_transform: Optional[ImageTransform] = None

    def __del__(self) -> None:
        if self.tokenizer is not None:
            self.tokenizer = None
        if self.model is not None:
            self.model = None
        if self.language_model is not None:
            self.language_model = None
        if self.vit_model is not None:
            self.vit_model = None
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
        self.load_timer.end("torch_dist")

        # TODO implement xfuser parallelism
        if self.world_size > 1:
            logging.warning("Parallelism is not supported in Bagel generation, running on single device.")

    def load_model(self) -> None:
        assert torch.cuda.is_available()

        model_path = "BAGEL-7B-MoT"  # TODO temporary

        # VAE
        # TODO check the dtype
        self.load_timer.start("vae")
        self.vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
        self.vae_model.to(self.device)
        self.load_timer.end("vae")

        # LLM
        self.load_timer.start("llm")
        llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"
        self.load_timer.end("llm")

        # ViT
        self.load_timer.start("vit")
        vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
        self.load_timer.end("vit")

        # Transformer
        self.load_timer.start("bagel")
        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=2,
            max_latent_size=64,
        )

        with init_empty_weights():
            self.language_model = Qwen2ForCausalLM(llm_config)
            self.vit_model = SiglipVisionModel(vit_config)
            self.model = Bagel(self.language_model, self.vit_model, config)
            self.model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        self.model = load_checkpoint_and_dispatch(
            self.model,
            checkpoint=os.path.join(model_path, "ema.safetensors"),
            offload_buffers=True,
            dtype=self.param_dtype,
            device_map={"": self.device_id},  # TODO fix
            force_hooks=True,
            offload_folder="/tmp/offload"
        )
        # self.model.to(self.device)  # done through accelerate#load_checkpoint_and_dispatch()
        self.model.eval()
        # self.model.require_grad_(False)
        self.load_timer.end("bagel")

        # Tokenizer
        self.load_timer.start("tokenizer")
        self.tokenizer = Qwen2Tokenizer.from_pretrained(
            model_path,
            torch_dtype=self.param_dtype,
        )
        # self.tokenizer.to(self.device)
        self.tokenizer, self.new_token_ids, _ = add_special_tokens(self.tokenizer)
        self.load_timer.end("tokenizer")

        # Image Transform
        self.vae_transform = ImageTransform(1024, 512, 16)
        self.vit_transform = ImageTransform(980, 224, 14)

    def init_model_parallelism(self) -> None:
        if self.world_size > 1:
            logging.warning("Parallelism not supported.")

    def model_compile(self) -> None:
        if not self.torch_compile:
            return

        self.load_timer.start("dit_compile")
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.model = torch.compile(
            self.model,
            mode="max-autotune-no-cudagraphs"
        )
        self.load_timer.end("dit_compile")

    @inference_mode()
    def update_context_text(self, text: Any, gen_context: Dict[str, Any]) -> Dict[str, Any]:
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_prompts(  # type: ignore[union-attr]
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            prompts=[text],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        # TODO make the tokenizer stuff the right way
        generation_input["text_token_lens"] = generation_input["text_token_lens"].to(self.device)
        generation_input["packed_text_ids"] = generation_input["packed_text_ids"].to(self.device)
        generation_input["packed_text_position_ids"] = generation_input["packed_text_position_ids"].to(self.device)
        generation_input["packed_text_indexes"] = generation_input["packed_text_indexes"].to(self.device)
        generation_input["packed_key_value_indexes"] = generation_input["packed_key_value_indexes"].to(self.device)
        generation_input["key_values_lens"] = generation_input["key_values_lens"].to(self.device)

        past_key_values = self.model.forward_cache_update_text(  # type: ignore[union-attr]
            past_key_values, **generation_input)
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        return gen_context

    @inference_mode()
    def update_context_images(
        self, images: Any, gen_context: Dict[str, Any], vae: bool = True, vit: bool = True
    ) -> Dict[str, Any]:
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        # VAE
        generation_input, kv_lens, ropes = self.model.prepare_vae_images(  # type: ignore[union-attr]
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            images=images,
            transforms=self.vae_transform,
            new_token_ids=self.new_token_ids,
        )
        generation_input["padded_images"] = generation_input["padded_images"].to(self.device)
        past_key_values = self.model.forward_cache_update_vae(  # type: ignore[union-attr]
            self.vae_model, past_key_values, **generation_input)

        # ViT
        generation_input, kv_lens, ropes = self.model.prepare_vit_images(  # type: ignore[union-attr]
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            images=images,
            transforms=self.vit_transform,
            new_token_ids=self.new_token_ids,
        )
        past_key_values = self.model.forward_cache_update_vit(  # type: ignore[union-attr]
            past_key_values, **generation_input)

        # Output
        gen_context['past_key_values'] = past_key_values
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes

        return gen_context

    def decode_image(self, latent: Any, image_shape: Any) -> Image.Image:
        H, W = image_shape
        model = self.model  # type: ignore[union-attr]
        vae = self.vae_model  # type: ignore[union-attr]
        h, w = H // model.latent_downsample, W // model.latent_downsample
        latent = latent.reshape(1, h, w, model.latent_patch_size,
                                model.latent_patch_size, model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, model.latent_channel, h
                                * model.latent_patch_size, w * model.latent_patch_size)
        # TODO do this right away instead of float32 to bfloat16
        latent = latent.to(self.param_dtype).to(self.device)  # Ensure dtype matches model
        image = vae.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())
        return image

    @inference_mode()
    async def warmup(self) -> None:
        logging.info(f"[{self.rank}] Warmup for Bagel generation")
        empty_img_0 = Image.new("RGB", (512, 512), (255, 255, 255))
        empty_img_1 = Image.new("RGB", (512, 512), (255, 255, 255))
        await self.generate(
            width=1024,
            height=1024,
            prompt="Warmup.",
            neg_prompt="",
            imgs=[empty_img_0, empty_img_1],
            sampling_steps=2)

    @override
    @inference_mode()
    async def generate(
        self,
        height: int,
        width: int,
        prompt: str,
        neg_prompt: str = "",  # TODO not used
        imgs: List[Image.Image] = [],
        sampling_steps: int = 50,
        understanding_output: bool = False,
        job_id: Optional[str] = None,
    ) -> Image.Image:
        gen_timer = self._new_gen_timer(job_id)

        self.running = True  # Mark running to avoid concurrent calls

        try:
            # Other arguments
            cfg_renorm_min = 0.0
            cfg_renorm_type = "global"
            cfg_text_scale = 3.0
            cfg_img_scale = 1.5
            cfg_type = "parallel"
            cfg_interval = (0.4, 1.0)
            timestep_shift = 3.0

            image_shape = (height, width)

            # https://github.com/ByteDance-Seed/Bagel/blob/main/inferencer.py#L119
            num_hidden_layers = self.model.config.llm_config.num_hidden_layers  # type: ignore[union-attr]
            # num_hidden_layers = 32  # Set this up properly
            gen_context = {
                'kv_lens': [0],
                'ropes': [0],
                'past_key_values': NaiveCache(num_hidden_layers),
            }
            cfg_text_context = deepcopy(gen_context)
            cfg_img_context = deepcopy(gen_context)

            with torch.autocast(device_type="cuda", enabled=True, dtype=self.param_dtype):
                if imgs is not None and len(imgs) > 0:
                    img_latents = []
                    for img in imgs:
                        img_rgb = img.convert("RGB")
                        img_latent = self.vae_transform.resize_transform(img_rgb)  # type: ignore[union-attr]
                        img_latents.append(img_latent)
                    gen_context = self.update_context_images(img_latents, gen_context, vae=not understanding_output)
                    # image_shapes = img_latent.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                # TODO figure why the order of this matters
                # Text (we can add multiple)
                cfg_text_context = deepcopy(gen_context)
                gen_context = self.update_context_text(prompt, gen_context)
                cfg_img_context = self.update_context_text(prompt, cfg_img_context)

                # VAE latent
                gen_timer.start("vae_encoder")
                past_key_values = gen_context['past_key_values']
                kv_lens = gen_context['kv_lens']
                ropes = gen_context['ropes']
                generation_input = self.model.prepare_vae_latent(  # type: ignore[union-attr]
                    curr_kvlens=kv_lens,
                    curr_rope=ropes,
                    image_sizes=[image_shape],
                    new_token_ids=self.new_token_ids,
                )
                packed_vae_token_indexes = generation_input['packed_vae_token_indexes']
                packed_vae_position_ids = generation_input['packed_vae_position_ids']
                packed_text_ids = generation_input['packed_text_ids']
                packed_text_indexes = generation_input['packed_text_indexes']
                packed_position_ids = generation_input['packed_position_ids']
                packed_indexes = generation_input['packed_indexes']
                packed_seqlens = generation_input['packed_seqlens']
                key_values_lens = generation_input['key_values_lens']
                packed_key_value_indexes = generation_input['packed_key_value_indexes']
                gen_timer.end("vae_encoder")

                # Text cfg
                gen_timer.start("text_encoder")
                cfg_text_past_key_values = cfg_text_context['past_key_values']
                kv_lens_cfg = cfg_text_context['kv_lens']
                ropes_cfg = cfg_text_context['ropes']
                generation_input_cfg_text = self.model.prepare_vae_latent_cfg(  # type: ignore[union-attr]
                    curr_kvlens=kv_lens_cfg,
                    curr_rope=ropes_cfg,
                    image_sizes=[image_shape],
                )
                gen_timer.end("text_encoder")

                # Image cfg
                gen_timer.start("image_encoder")
                cfg_img_past_key_values = cfg_img_context['past_key_values']
                kv_lens_cfg = cfg_img_context['kv_lens']
                ropes_cfg = cfg_img_context['ropes']
                generation_input_cfg_img = self.model.prepare_vae_latent_cfg(  # type: ignore[union-attr]
                    curr_kvlens=kv_lens_cfg,
                    curr_rope=ropes_cfg,
                    image_sizes=[image_shape],
                )
                # TODO Try to get the VAE to generate in the GPU directly
                x_t = generation_input['packed_init_noises']
                x_t = x_t.to(self.device)
                x_t = x_t.to(self.param_dtype)
                gen_timer.end("image_encoder")

                # Diffusion sampling
                # https://github.com/ByteDance-Seed/Bagel/blob/main/modeling/bagel/bagel.py#L643
                timesteps = torch.linspace(1, 0, sampling_steps, device=x_t.device)
                timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
                dts = timesteps[:-1] - timesteps[1:]
                timesteps = timesteps[:-1]

                for it, timestep in enumerate(timesteps):
                    gen_timer.start(f"dit_{it:03d}")
                    timestep_tensor = torch.tensor([timestep] * x_t.shape[0], device=x_t.device)
                    if timestep > cfg_interval[0] and timestep <= cfg_interval[1]:
                        cfg_text_scale_ = cfg_text_scale
                        cfg_img_scale_ = cfg_img_scale
                    else:
                        cfg_text_scale_ = 1.0
                        cfg_img_scale_ = 1.0
                    v_t = self.model._forward_flow(  # type: ignore[union-attr]
                        x_t=x_t,
                        timestep=timestep_tensor,
                        packed_vae_token_indexes=packed_vae_token_indexes,
                        packed_vae_position_ids=packed_vae_position_ids,
                        packed_text_ids=packed_text_ids,
                        packed_text_indexes=packed_text_indexes,
                        packed_position_ids=packed_position_ids,
                        packed_indexes=packed_indexes,
                        packed_seqlens=packed_seqlens,
                        key_values_lens=key_values_lens,
                        past_key_values=past_key_values,
                        packed_key_value_indexes=packed_key_value_indexes,
                        cfg_renorm_min=cfg_renorm_min,
                        cfg_renorm_type=cfg_renorm_type,
                        # cfg_text
                        cfg_text_scale=cfg_text_scale_,
                        cfg_text_past_key_values=cfg_text_past_key_values,
                        # cfg_text_packed_position_ids,
                        cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
                        # cfg_text_packed_query_indexes,
                        cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
                        # cfg_text_key_values_lens,
                        cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
                        # cfg_text_packed_key_value_indexes,
                        cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
                        # cfg_img
                        cfg_img_scale=cfg_img_scale_,
                        cfg_img_past_key_values=cfg_img_past_key_values,
                        # cfg_img_packed_position_ids,
                        cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
                        # cfg_img_packed_query_indexes,
                        cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
                        # cfg_img_key_values_lens,
                        cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
                        # cfg_img_packed_key_value_indexes,
                        cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
                        cfg_type=cfg_type,
                    )
                    x_t = x_t - v_t.to(x_t.device) * dts[it]  # velocity pointing from data to noise
                    gen_timer.end(f"dit_{it:03d}")

                unpacked_latent = x_t.split((packed_seqlens - 2).tolist())

                # Decode the image
                gen_timer.start("image_decoder")
                output_image = self.decode_image(unpacked_latent[0], image_shape)
                gen_timer.end("image_decoder")
            return output_image
        finally:
            self.running = False
            gen_timer.end("total")

    def get_health(self) -> Dict[str, Any]:
        ret = super().get_health()
        ret.update({
            "gpu": self.gpu,
            "rank": self.rank,
            "world_size": self.world_size,
            "dtype": str(self.param_dtype),
        })
        return ret

    async def get_rest_args(self, data_json: Dict[str, str]) -> Dict[str, Any]:
        if data_json is None:
            raise ValueError("Missing JSON body")
        imgs_base64 = data_json.get("imgs", None)
        imgs = []
        if imgs_base64 is not None:
            for img_base64 in imgs_base64:
                img = base64_to_img(img_base64)
                imgs.append(img)
        prompt = data_json.get("prompt", None)
        if prompt is None:
            raise ValueError("Missing 'prompt' parameter")
        neg_prompt = data_json.get("neg_prompt", "")
        height = int(data_json.get("height", 400))
        width = int(data_json.get("width", 640))
        steps = int(data_json.get("sampling_steps", 20))
        return {
            "task": self.model_name,
            "args": {
                "imgs": [img] if img is not None else [],
                "prompt": prompt,
                "neg_prompt": neg_prompt,
                "height": height,
                "width": width,
                "sampling_steps": steps,
            }
        }


async def main() -> None:
    bagel = BagelGeneration()

    img_size = (720, 1280)
    img_size = (1024, 1024)
    img_size = (512, 512)
    num_steps = 50

    # Generate original image 1
    img_prompt = "A full-body photo of a young woman standing on a white background. "
    img_prompt += "She has straight blonde hair, parted slightly to the side, and is smiling at the camera. "
    img_prompt += "She is wearing a white sleeveless tank top, blue skinny jeans, and blue socks. "
    img_prompt += "Her arms are relaxed at her sides, and she is standing with her feet together. "
    img_prompt += "The lighting is bright and even, with no visible shadows, giving the image a clean studio feel."

    output_image_0 = await bagel.generate(
        width=img_size[0],
        height=img_size[1],
        prompt=img_prompt,
        imgs=[],
        sampling_steps=num_steps,
    )
    output_image_0.save("output/output_image_0.png")

    # Generate original image 2
    img_prompt = "A full-body photo of a young man standing on a white studio background. "
    img_prompt += "He has short black hair styled upward and is smiling slightly at the camera. "
    img_prompt += "He is wearing a fitted light gray V-neck T-shirt, dark blue jeans, and brown suede shoes. "
    img_prompt += "His arms are relaxed by his sides, and his stance is casual with feet slightly apart. "
    img_prompt += "The lighting is bright and even, with no shadows, creating a clean and professional look."
    output_image_1 = await bagel.generate(
        width=img_size[0],
        height=img_size[1],
        prompt=img_prompt,
        imgs=[],
        sampling_steps=num_steps,
    )
    output_image_1.save("output/output_image_1.png")

    # Merge the two images
    img_prompt += "Using the two people wearing from the original two images."
    img_prompt += "A photorealistic podcast setup featuring the woman in the right and the man in the left sitting "
    img_prompt += "across each other at a wooden table in a professional recording studio."
    img_prompt += "The studio has red acoustic panels on the walls, warm lighting, and a large off screen in the "
    img_prompt += "background."
    img_prompt += "Both wear headphones and speak into high-quality podcast microphones mounted on adjustable arms."
    img_prompt += "Scene captured from a slightly elevated front-facing perspective, showing their upper bodies and "
    img_prompt += "expressive gestures as they engage in conversation."
    img_prompt += "Table equipped with coffee mugs, water bottles, and recording equipment."

    output_image_2 = await bagel.generate(
        img_size[0],
        img_size[1],
        img_prompt,
        imgs=[
            output_image_0,
            output_image_1,
        ],
        sampling_steps=num_steps,
    )
    output_image_2.save("output/output_image_2.png")


if __name__ == "__main__":
    asyncio.run(main())
