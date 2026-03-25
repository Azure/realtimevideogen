# mypy: ignore-errors
"""
xFuser FramePack for Hunyuan Video.
Extends the Hunyuan Video Framepack Pipeline to support xFuser's long context attention and sequence parallelism.
Sources:
* https://github.com/lllyasviel/FramePack
* https://github.com/xdit-project/xDiT
"""
# mypy: ignore-errors
# Source: https://github.com/lllyasviel/FramePack and https://github.com/xdit-project/xDiT
import torch
import functools
import einops

from typing import Tuple
from typing import Any
from typing import Optional
from typing import Dict

from diffusers import HunyuanVideoFramepackPipeline

from diffusers.models.attention import Attention
from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoAttnProcessor2_0
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput

from flash_attn import flash_attn_varlen_func

from xfuser.envs import PACKAGES_CHECKER
from xfuser.core.cache_manager.cache_manager import get_cache_manager
from xfuser.core.distributed import get_sp_group
from xfuser.core.distributed import get_sequence_parallel_world_size
from xfuser.core.distributed import get_sequence_parallel_rank
from xfuser.core.distributed import get_classifier_free_guidance_world_size
from xfuser.core.distributed import get_classifier_free_guidance_rank
from xfuser.core.distributed import get_runtime_state
from xfuser.core.long_ctx_attention import xFuserLongContextAttention
from xfuser.model_executor.layers.attention_processor import xFuserAttentionProcessorRegister


# Class for the attention extending:
# https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/layers/attention_processor.py
env_info = PACKAGES_CHECKER.get_packages_info()
HAS_LONG_CTX_ATTN = env_info["has_long_ctx_attn"]
HAS_FLASH_ATTN = env_info["has_flash_attn"]


@xFuserAttentionProcessorRegister.register(HunyuanVideoAttnProcessor2_0)
class xFuserFramepackSingleHunyuanVideoAttnProcessor2_0(HunyuanVideoAttnProcessor2_0):
    def __init__(self) -> None:
        super().__init__()

        assert get_sequence_parallel_world_size() > 1
        assert HAS_LONG_CTX_ATTN is True
        assert HAS_FLASH_ATTN is True

        self.hybrid_seq_parallel_attn = xFuserLongContextAttention(use_kv_cache=True)

    # HunyuanAttnProcessorFlashAttnSingle
    def __call__(
        self,
        attn: HunyuanVideoAttnProcessor2_0,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv = attention_mask
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        txt_length = encoder_hidden_states.shape[1]
        query = torch.cat([apply_rotary_emb_transposed(query[:, :-txt_length],
                          image_rotary_emb), query[:, -txt_length:]], dim=1)
        key = torch.cat([apply_rotary_emb_transposed(key[:, :-txt_length],
                        image_rotary_emb), key[:, -txt_length:]], dim=1)
        hidden_states = attn_varlen_func(query, key, value, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv,
                                         self.hybrid_seq_parallel_attn, attn)
        hidden_states = hidden_states.flatten(-2)
        hidden_states, encoder_hidden_states = hidden_states[:, :-txt_length], hidden_states[:, -txt_length:]
        return hidden_states, encoder_hidden_states


def attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    hybrid_seq_parallel_attn: Any = None,
    attn: Any = None
) -> torch.Tensor:
    if cu_seqlens_q is None and cu_seqlens_kv is None and max_seqlen_q is None and max_seqlen_kv is None:
        # Needed for XDiT
        x = hybrid_seq_parallel_attn(attn, q, k, v, dropout_p=0.0, causal=False, joint_strategy="none")
        # Original
        # x = flash_attn_func(q, k, v)
        return x

    B, L, H, C = q.shape

    q = q.flatten(0, 1)
    k = k.flatten(0, 1)
    v = v.flatten(0, 1)

    x = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)

    x = x.unflatten(0, (B, L))

    return x


def apply_rotary_emb_transposed(
    x: torch.Tensor,
    freqs_cis: torch.Tensor
) -> torch.Tensor:
    # https://github.com/lllyasviel/FramePack/blob/c5d375661a2557383f0b8da9d11d14c23b0c4eaf/diffusers_helper/models/hunyuan_video_packed.py#L190
    cos, sin = freqs_cis.unsqueeze(-2).chunk(2, dim=-1)
    x_real, x_imag = x.unflatten(-1, (-1, 2)).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    out = x.float() * cos + x_rotated.float() * sin
    out = out.to(x)
    return out


@xFuserAttentionProcessorRegister.register(HunyuanVideoAttnProcessor2_0)
class xFuserFramepackDoubleHunyuanVideoAttnProcessor2_0(HunyuanVideoAttnProcessor2_0):
    def __init__(self) -> None:
        super().__init__()

        assert get_sequence_parallel_world_size() > 1
        assert HAS_LONG_CTX_ATTN is True
        assert HAS_FLASH_ATTN is True

        self.hybrid_seq_parallel_attn = xFuserLongContextAttention(use_kv_cache=True)

    # HunyuanAttnProcessorFlashAttnDouble
    def __call__(
        self,
        attn: HunyuanVideoAttnProcessor2_0,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv = attention_mask
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = apply_rotary_emb_transposed(query, image_rotary_emb)
        key = apply_rotary_emb_transposed(key, image_rotary_emb)

        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

        encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
        encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
        encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

        encoder_query = attn.norm_added_q(encoder_query)
        encoder_key = attn.norm_added_k(encoder_key)

        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        hidden_states = attn_varlen_func(query, key, value, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv,
                                         self.hybrid_seq_parallel_attn, attn)
        hidden_states = hidden_states.flatten(-2)
        txt_length = encoder_hidden_states.shape[1]
        hidden_states, encoder_hidden_states = hidden_states[:, :-txt_length], hidden_states[:, -txt_length:]
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        return hidden_states, encoder_hidden_states


def parallelize_transformer(pipe_hunyuan: HunyuanVideoFramepackPipeline) -> HunyuanVideoFramepackPipeline:
    transformer = pipe_hunyuan.transformer

    """
    Parallelize the transformer.
    """
    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self: Any,
        hidden_states: torch.Tensor,  # shape: [1, 16, 9, 80, 76]
        timestep: Any,
        encoder_hidden_states: torch.Tensor,  # shape: [1, 512, 4096]
        encoder_attention_mask: torch.Tensor,  # shape: [1, 512]
        pooled_projections: Any,
        guidance: Any,
        latent_indices: Any = None,
        clean_latents: Optional[torch.Tensor] = None,            # shape: [1, 16,  2, 80, 76]
        clean_latent_indices: Optional[torch.Tensor] = None,     # shape: [1, 2]
        clean_latents_2x: Optional[torch.Tensor] = None,         # shape: [1, 16,  2, 80, 76]
        clean_latent_2x_indices: Optional[torch.Tensor] = None,  # shape: [1, 2]
        clean_latents_4x: Optional[torch.Tensor] = None,         # shape: [1, 16, 16, 80, 76]
        clean_latent_4x_indices: Optional[torch.Tensor] = None,  # shape: [1, 16]
        image_embeddings: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict] = None,
        return_dict: bool = True
    ) -> Any:
        """
        Started from https://github.com/lllyasviel/FramePack/blob/main/diffusers_helper/models/hunyuan_video_packed.py
        Aligned with https://github.com/xdit-project/xDiT/blob/main/examples/hunyuan_video_usp_example.py
        """
        if attention_kwargs is None:
            attention_kwargs = {}

        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        p, p_t = self.config['patch_size'], self.config['patch_size_t']  # 2, 1
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p

        original_context_length = post_patch_num_frames * post_patch_height * post_patch_width

        # 1. FramePack + RoPE
        # image_rotary_emb ~= rope_freqs
        # hidden_states: [1, 16, 9, 80, 76] -> [1, 17500, 3072] = [1, 7*5*5*5*5*4, 3*2*512]
        hidden_states, rope_freqs = self.process_input_hidden_states(
            hidden_states, latent_indices,
            clean_latents, clean_latent_indices,
            clean_latents_2x, clean_latent_2x_indices,
            clean_latents_4x, clean_latent_4x_indices
        )

        # 2. Conditional embeddings
        temb = self.time_text_embed(timestep, guidance, pooled_projections)
        # [1, 512, 4096] -> [1, 512, 3072]
        encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

        extra_encoder_hidden_states = self.image_projection(image_embeddings)
        extra_attention_mask = torch.ones(
            (batch_size, extra_encoder_hidden_states.shape[1]),
            dtype=encoder_attention_mask.dtype,
            device=encoder_attention_mask.device
        )
        encoder_hidden_states = torch.cat([extra_encoder_hidden_states, encoder_hidden_states], dim=1)
        encoder_attention_mask = torch.cat([extra_attention_mask, encoder_attention_mask], dim=1)

        text_len = encoder_attention_mask.sum().item()
        # 1, 1241, 3072 -> 1, text_len(742), 3072
        encoder_hidden_states = encoder_hidden_states[:, :text_len]
        attention_mask = None, None, None, None

        # Sequence-parallel chunking
        sp_world = get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()
        cfg_world = get_classifier_free_guidance_world_size()
        cfg_rank = get_classifier_free_guidance_rank()

        # Chunk hidden_states and rope_freqs identically along the sequence dimension
        # [1, 17226, 3072] -> [1, 17226 / sp_world, 3072]
        # [1, 17500, 3072] -> [1, 17500 / sp_world, 3072]
        if hidden_states.shape[-2] % sp_world != 0:
            # This will cause a noisy output after chunking
            raise RuntimeError(f"hidden_states {hidden_states.shape} is not divisible by sp_world {sp_world}.")
        hidden_states = torch.chunk(hidden_states, sp_world, dim=-2)[sp_rank]

        # [1, 17226, 256] -> [1, 17226 / sp_world, 256]
        if rope_freqs.shape[-2] % sp_world != 0:
            # This will cause a noisy output after chunking
            raise RuntimeError(f"rope_freqs {rope_freqs.shape} is not divisible by sp_world {sp_world}.")
        rope_freqs = torch.chunk(rope_freqs, sp_world, dim=-2)[sp_rank]

        # Chunk encoder_hidden_states [1, 742, 3072] -> [1, 742 / sp_world, 3072]
        if encoder_hidden_states.shape[-2] % sp_world != 0:
            get_runtime_state().split_text_embed_in_sp = False
        else:
            get_runtime_state().split_text_embed_in_sp = True

        encoder_hidden_states = torch.chunk(
            encoder_hidden_states,
            cfg_world,
            dim=0)[cfg_rank]

        if get_runtime_state().split_text_embed_in_sp:
            encoder_hidden_states = torch.chunk(
                encoder_hidden_states,
                sp_world,
                dim=-2)[sp_rank]

        # 3. Transformer blocks
        # https://github.com/lllyasviel/FramePack/blob/main/diffusers_helper/models/hunyuan_video_packed.py#HunyuanVideoTransformerBlock
        for block in self.transformer_blocks + self.single_transformer_blocks:
            # HunyuanVideoSingleTransformerBlock
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                rope_freqs
            )

        # Output projection
        # This is torch.Size([1, 8613, 3072])
        hidden_states = self.norm_out(hidden_states, temb)
        # Ensure dtype matches proj_out weights
        hidden_states = hidden_states.to(self.proj_out.weight.dtype)
        hidden_states = self.proj_out(hidden_states)
        # shape: ([1, 8613, 64])
        # Gather and reshape like xDiT
        hidden_states = get_sp_group().all_gather(hidden_states, dim=-2)
        hidden_states = hidden_states[:, -original_context_length:, :]
        hidden_states = einops.rearrange(hidden_states, 'b (t h w) (c pt ph pw) -> b c (t pt) (h ph) (w pw)',
                                         t=post_patch_num_frames, h=post_patch_height, w=post_patch_width,
                                         pt=p_t, ph=p, pw=p)
        # Rearranged hidden_states [1, 16, 9, 80, 76]

        if return_dict:
            return Transformer2DModelOutput(sample=hidden_states)

        return hidden_states,

    new_forward = new_forward.__get__(transformer)  # type: ignore[attr-defined]
    transformer.forward = new_forward

    # Apply the xFuser attention processor to all transformer blocks
    for block in transformer.transformer_blocks:
        block.attn.processor = xFuserFramepackDoubleHunyuanVideoAttnProcessor2_0()
    for block in transformer.single_transformer_blocks:
        block.attn.processor = xFuserFramepackSingleHunyuanVideoAttnProcessor2_0()

    # Register the attention layers to the cache manager
    # This is needed for sequence parallelism to work correctly and allow each chunk to attend to the entire sequence
    for block in transformer.transformer_blocks:
        for submodule in block.modules():
            if isinstance(submodule, Attention):
                get_cache_manager().register_cache_entry(submodule, "attn", "sequence_parallel_attn_cache")
    for block in transformer.single_transformer_blocks:
        for submodule in block.modules():
            if isinstance(submodule, Attention):
                get_cache_manager().register_cache_entry(submodule, "attn", "sequence_parallel_attn_cache")

    return pipe_hunyuan
