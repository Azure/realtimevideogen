# from https://github.com/xdit-project/xDiT/blob/main/examples/flux_usp_example.py
# from https://github.com/chengzeyi/ParaAttention/blob/main/examples/run_flux.py

import torch
import functools
import types

from typing import Optional
from typing import Any

from diffusers import DiffusionPipeline

from xfuser.core.distributed import get_runtime_state
from xfuser.core.distributed import get_classifier_free_guidance_world_size
from xfuser.core.distributed import get_classifier_free_guidance_rank
from xfuser.core.distributed import get_cfg_group
from xfuser.core.distributed import get_sequence_parallel_world_size
from xfuser.core.distributed import get_sequence_parallel_rank
from xfuser.core.distributed import get_sp_group

from xfuser.model_executor.models.transformers.transformer_flux import xFuserFluxAttnProcessor


def parallelize_transformer(pipe: DiffusionPipeline) -> DiffusionPipeline:
    transformer = getattr(pipe, "transformer")
    assert transformer is not None, "pipe has no transformer attribute"
    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        *args: Any,
        timestep: Optional[torch.LongTensor] = None,
        img_ids: Optional[torch.Tensor] = None,
        txt_ids: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        cfg_world = get_classifier_free_guidance_world_size()
        cfg_rank = get_classifier_free_guidance_rank()
        sp_world = get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()
        if hidden_states.shape[0] % cfg_world != 0:
            raise ValueError(f"Cannot split {hidden_states.shape[0]} ({hidden_states.shape}) into {cfg_world} parts")
        if hidden_states.shape[-2] % sp_world != 0:
            raise ValueError(f"Cannot split {hidden_states.shape[-2]} ({hidden_states.shape}) into {sp_world} parts")
        assert encoder_hidden_states is not None, "encoder_hidden_states must not be None"
        if encoder_hidden_states.shape[-2] % sp_world != 0:
            get_runtime_state().split_text_embed_in_sp = False
        else:
            get_runtime_state().split_text_embed_in_sp = True

        if isinstance(timestep, torch.Tensor) and timestep.ndim != 0 and timestep.shape[0] == hidden_states.shape[0]:
            timestep = torch.chunk(timestep, cfg_world, dim=0)[cfg_rank]
        hidden_states = torch.chunk(hidden_states, cfg_world, dim=0)[cfg_rank]
        hidden_states = torch.chunk(hidden_states, sp_world, dim=-2)[sp_rank]
        encoder_hidden_states = torch.chunk(encoder_hidden_states, cfg_world, dim=0)[cfg_rank]
        if get_runtime_state().split_text_embed_in_sp:
            encoder_hidden_states = torch.chunk(encoder_hidden_states, sp_world, dim=-2)[sp_rank]
        assert img_ids is not None, "img_ids must not be None"
        img_ids = torch.chunk(img_ids, sp_world, dim=-2)[sp_rank]
        if get_runtime_state().split_text_embed_in_sp:
            assert txt_ids is not None, "txt_ids must not be None when split_text_embed_in_sp is True"
            txt_ids = torch.chunk(txt_ids, sp_world, dim=-2)[sp_rank]

        for block in transformer.transformer_blocks + transformer.single_transformer_blocks:
            block.attn.processor = xFuserFluxAttnProcessor()

        output = original_forward(
            hidden_states,
            encoder_hidden_states,
            *args,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            **kwargs,
        )

        return_dict = not isinstance(output, tuple)
        sample = output[0]
        sample = get_sp_group().all_gather(sample, dim=-2)
        sample = get_cfg_group().all_gather(sample, dim=0)
        if return_dict:
            return output.__class__(sample, *output[1:])
        return (sample, *output[1:])

    bound_forward: Any = types.MethodType(new_forward, transformer)
    transformer.forward = bound_forward

    return pipe
