import logging

from typing import Any
from typing import List
from typing import Union
from typing import Optional
from typing import Callable

import torch
import torch.amp as amp
from torch import Tensor
from torch.nn import Module

from xfuser.core.distributed import get_sequence_parallel_rank
from xfuser.core.distributed import get_sequence_parallel_world_size
from xfuser.core.distributed import get_sp_group

from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d


def usp_fantasytalking_forward(
    self: Module,
    x: List[Tensor],  # [C, T, H, W]
    timestep: Tensor,  # [B]
    context: List[Tensor],  # [L, C]
    seq_len: Union[int, Tensor],  # [B] or scalar
    clip_fea: Optional[Tensor] = None,
    y: Optional[List[Tensor]] = None,
    use_gradient_checkpointing: bool = False,
    audio_proj: Optional[Module] = None,
    audio_context_lens: Optional[Tensor] = None,  # [B] length per example
    latents_num_frames: Optional[int] = None,
    audio_scale: float = 1.0,
    **kwargs: Any,
) -> Tensor:
    """
    Copied from:
    https://github.com/Fantasy-AMAP/fantasy-talking/blob/main/diffsynth/models/wan_video_dit.py
    And adjusted based on:
    https://github.com/Wan-Video/Wan2.1/blob/main/wan/distributed/xdit_context_parallel.py
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == "i2v":
        assert clip_fea is not None and y is not None
    # params
    device = x[0].device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [
        self.patch_embedding(u.unsqueeze(0))  # type: ignore[operator]
        for u in x
    ]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
    )  # [B,2]
    x = [u.flatten(2).transpose(1, 2) for u in x]  # [[C, L, T],,]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([  # type: ignore[assignment]
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)  # type: ignore[arg-type]
        for u in x
    ])

    # time embeddings
    with amp.autocast(dtype=torch.float32, device_type="cuda"):
        e = self.time_embedding(  # type: ignore[operator]
            sinusoidal_embedding_1d(self.freq_dim, timestep).float()
        )
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))  # type: ignore[operator]
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(  # type: ignore[operator]
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])  # type: ignore[operator, arg-type]
            for u in context
        ])
    )

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)  # type: ignore[operator]
        context = torch.concat([context_clip, context], dim=1)  # type: ignore[assignment, list-item]

    # Context Parallel
    sp_world = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()
    if x.shape[1] % sp_world != 0:  # type: ignore[attr-defined]
        raise ValueError(
            f"Input sequence length {x.shape} is not divisible "  # type: ignore[attr-defined]
            f"by sequence parallel {sp_world}"
        )
    logging.debug(f"Input sequence length {x.shape} and sequence parallel {sp_world}.")  # type: ignore[attr-defined]
    x = torch.chunk(x, sp_world, dim=1)[sp_rank]  # type: ignore[arg-type]
    logging.debug(f"Input sequence length after chunking {x.shape}.")  # type: ignore[attr-defined]

    """
    # https://github.com/Fantasy-AMAP/fantasy-talking/issues/52
    # audio chunking along the #frames dimension doesn't work -> black frames
    if audio_proj is not None:
        # chunking audio_proj based on sequence parallel rank
        logging.info(f"Audio projection shape {audio_proj.shape} and sequence parallel {sp_world}.")
        if audio_proj.shape[2] % sp_world != 0:
            # insert silence frames evenly to each chunk to make it divisible by sequence parallel
            # e.g. audio_proj.shape = [1, 4, 15, 2048] and sp_world = 4 -> [1, 4, 16, 2048]
            # calculate how many frames we need to pad
            num_frames = audio_proj.shape[2]
            pad_frames = math.ceil(num_frames / sp_world) * sp_world - num_frames

            pad_shape = list(audio_proj.shape)
            pad_shape[2] = pad_frames
            silence = torch.zeros(pad_shape, dtype=audio_proj.dtype, device=audio_proj.device)
            # audio_proj = torch.cat([audio_proj, silence], dim=2)
            logging.info(f"Audio projection shape after padding {audio_proj.shape}.")
            audio_proj = torch.chunk(audio_proj, sp_world, dim=2)[sp_rank]
        else:
            audio_proj = torch.chunk(audio_proj, sp_world, dim=2)[sp_rank]
        logging.info(f"Audio projection shape after chunking {audio_proj.shape}.")
    """

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
        audio_proj=audio_proj,
        audio_context_lens=audio_context_lens,
        latents_num_frames=latents_num_frames,
        audio_scale=audio_scale,
    )

    def create_custom_forward(module: Module) -> Callable[..., Tensor]:
        def custom_forward(*inputs: Tensor, **kwargs: Any) -> Tensor:
            return module(*inputs, **kwargs)
        return custom_forward

    for block in self.blocks:  # type: ignore[union-attr]
        if self.training and use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x,
                **kwargs,
                use_reentrant=False,
            )
        else:
            x = block(x, **kwargs)

    # head
    x = self.head(x, e)  # type: ignore[operator]

    # Context Parallel
    x = get_sp_group().all_gather(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)  # type: ignore[operator]
    x = torch.stack(x).float()  # type: ignore[assignment]
    return x  # type: ignore[return-value]
