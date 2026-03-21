"""
This script is to benchmark the VAE decoder and see how to optimize it.
Trying to make the code parallel across 8 GPUs.
"""

from wan.utils.utils import cache_video
from wan.modules.vae import WanVAE

from typing import Any
from typing import Iterator
from typing import Optional
from typing import Tuple
from typing import Union

import os
import time

import torch
import torch.amp as amp
import torch.distributed as dist


ScalePair = Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]]


# Original
def decode_old(
    vae_model: Any,
    z: torch.Tensor,
    scale: ScalePair
) -> torch.Tensor:
    vae_model.clear_cache()
    # z: [b,c,t,h,w]
    if isinstance(scale[0], torch.Tensor):
        assert isinstance(scale[1], torch.Tensor)
        z = z / scale[1].view(1, vae_model.z_dim, 1, 1, 1) + scale[0].view(
            1, vae_model.z_dim, 1, 1, 1)
    else:
        z = z / scale[1] + scale[0]
    iter_ = z.shape[2]
    x = vae_model.conv2(z)
    for i in range(iter_):
        vae_model._conv_idx = [0]
        if i == 0:
            out = vae_model.decoder(
                x[:, :, i:i + 1, :, :],
                feat_cache=vae_model._feat_map,
                feat_idx=vae_model._conv_idx)
        else:
            out_ = vae_model.decoder(
                x[:, :, i:i + 1, :, :],
                feat_cache=vae_model._feat_map,
                feat_idx=vae_model._conv_idx)
            out = torch.cat([out, out_], 2)
    vae_model.clear_cache()
    return out


# New version of decode
def decode(
    vae_model: Any,
    z: torch.Tensor,
    scale: ScalePair,
) -> torch.Tensor:
    rank = dist.get_rank()
    vae_model.clear_cache()
    # z: [b,c,t,h,w]
    if isinstance(scale[0], torch.Tensor):
        assert isinstance(scale[1], torch.Tensor)
        scale_0 = scale[0].view(1, vae_model.z_dim, 1, 1, 1)
        scale_1 = scale[1].view(1, vae_model.z_dim, 1, 1, 1)
        z = z / scale_1 + scale_0
    else:
        z = z / scale[1] + scale[0]
    iter_ = z.shape[2]
    x = vae_model.conv2(z)
    out_list = []
    for i in range(iter_):
        vae_model._conv_idx = [0]
        out_ = vae_model.decoder(
            x[:, :, i:i + 1, :, :],
            feat_cache=vae_model._feat_map,
            feat_idx=vae_model._conv_idx)
        if rank == 0:
            print("DECODE", x.shape, out_.shape)
            # 21 x torch.Size([1, 16, 1, 68, 90]) -> torch.Size([1, 3, 4, 544, 720])
        out_list.append(out_)
    out = torch.cat(out_list, dim=2)
    vae_model.clear_cache()
    return out


# Distributed version
# TODO need to remove the hard coding and clean it up
def decode_parallel(
    vae_model: Any,
    z: torch.Tensor,
    scale: ScalePair,
) -> Optional[torch.Tensor]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    vae_model.clear_cache()
    # z: [b,c,t,h,w]
    if isinstance(scale[0], torch.Tensor):
        assert isinstance(scale[1], torch.Tensor)
        scale_0 = scale[0].view(1, vae_model.z_dim, 1, 1, 1)
        scale_1 = scale[1].view(1, vae_model.z_dim, 1, 1, 1)
        z = z / scale_1 + scale_0
    else:
        z = z / scale[1] + scale[0]

    x = vae_model.conv2(z)  # [b, c, t, h, w]

    total_t = x.shape[2]
    # slice_t = total_t // world_size
    start = rank * total_t // world_size
    start = max(0, start - 2)  # We process the previous 2 frames for the convolution -> TODO take only the right output
    end = (rank + 1) * total_t // world_size
    local_x = x[:, :, start:end, :, :]  # Local slice: [16, 21, 68, 90] -> [16, 2-3, 68, 90]
    local_t = local_x.shape[2]

    # Decode local slice
    local_outs = []
    for i in range(local_t):
        vae_model._conv_idx = [0]
        # TODO this cannot be split like this because it has dependencies
        # [2] local_x:([1, 16, 2, 68, 90]) local_result:([1, 3, 5, 544, 720])
        # [4] local_x:([1, 16, 3, 68, 90]) local_result:([1, 3, 9, 544, 720])
        # conv -> middle -> upsamples -> head
        out_ = vae_model.decoder(
            local_x[:, :, i:i + 1, :, :],
            feat_cache=vae_model._feat_map,
            feat_idx=vae_model._conv_idx)
        # print(f"[{rank}] DECODE each frame of", local_x.shape, "->", out_.shape)
        # It should be [1, 16, 1(N), 68, 90] -> [1, 3, 4, 544, 720])
        # but it does                        -> [1, 3, 1, 544, 720])
        if rank == 0:
            local_outs.append(out_)
        elif i >= 2:  # TODO this is a hack to avoid the first 2 frames
            local_outs.append(out_)
    local_result = torch.cat(local_outs, dim=2)

    print(f"[{rank}] local_x:{local_x.shape} local_result: {local_result.shape}")  # [b, c, slice_t, h, w]
    # local_x:torch.Size([1, 16, 3, 68, 90]) local_result: torch.Size([1, 3, 3, 544, 720])
    # local_x:torch.Size([1, 16, 2, 68, 90]) local_result: torch.Size([1, 3, 2, 544, 720])

    # TODO The chunks have different sizes, so we need to gather them

    device = None  # TODO
    # Extend tensor of size [1, 3, 5-12, 544, 720] to max size [1, 3, 12, 544, 720] with zeros
    local_result_zeroes = torch.zeros(1, 3, 12, 544, 720).to(device)  # [1, 3, 12, 544, 720]
    local_result_zeroes[:, :, 0:local_result.shape[2], :, :] = local_result
    local_result = local_result_zeroes

    # Rank 0 gathers everything
    # 1+2*10 -> 1+8*10
    if rank == 0:
        # gather_list = [torch.zeros_like(local_result) for _ in range(world_size)]
        gather_list = [torch.zeros(1, 3, 12, 544, 720).to(device) for _ in range(world_size)]  # TODO max size
    else:
        gather_list = None
    dist.gather(local_result, gather_list=gather_list, dst=0)  # This might be fast enough but needs warm up

    final = None  # TODO return something empty at least
    if rank == 0:
        assert gather_list is not None
        # TODO avoid concatenating the 0s for padding
        for gather_rank in range(len(gather_list)):
            # TODO avoid the hardcoding
            gather_size = 12
            if gather_rank == 0:
                gather_size = 5
            elif gather_rank == 2 or gather_rank == 6:
                gather_size = 8
            gather_list[gather_rank] = gather_list[gather_rank][:, :, 0:gather_size, :, :]  # TODO remove the 0s

        final = torch.cat(gather_list, dim=2)  # Concatenate on time axis
        # [c, total_t, h, w] -> [3, 40!, 544, 720] Should be [3, 81, 544, 720]
        print(f"[{rank}] final shape: {final.shape}")

    vae_model.clear_cache()
    return final


def decode_stream(
    vae_model: Any,
    z: torch.Tensor,
    scale: ScalePair,
    start_frame: int = 0
) -> Iterator[torch.Tensor]:
    # TODO start_frame
    vae_model.clear_cache()
    # z: [b, c, #lat_frames, lat_h, lat_w]
    if isinstance(scale[0], torch.Tensor):
        assert isinstance(scale[1], torch.Tensor)
        scale_0 = scale[0].view(1, vae_model.z_dim, 1, 1, 1)
        scale_1 = scale[1].view(1, vae_model.z_dim, 1, 1, 1)
        z = z / scale_1 + scale_0
    else:
        z = z / scale[1] + scale[0]
    iter_ = z.shape[2]
    x = vae_model.conv2(z)
    for i in range(iter_):
        vae_model._conv_idx = [0]
        out = vae_model.decoder(
            x[:, :, i:i + 1, :, :],
            feat_cache=vae_model._feat_map,
            feat_idx=vae_model._conv_idx)
        num_frames = out.shape[2]
        print("out shape", out.shape, num_frames)
        for frame_ix in range(num_frames):
            yield out[:, :, frame_ix, :, :].unsqueeze(2)  # #frames x [b, c, h, w]
    vae_model.clear_cache()


def main() -> None:
    # Setup distributed env
    t0 = time.time()
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print(f"[{rank}] Setting up distributed environment...")

    device_id = local_rank
    device = torch.device(f"cuda:{device_id}")

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    print(f"[{rank}] Distributed environment setup in {time.time() - t0:.3f} seconds")

    # Load VAE
    t0 = time.time()
    print(f"[{rank}] Loading WanVAE...")
    ckpt_dir = "Wan2.1/Wan2.1-I2V-14B-480P"
    vae = WanVAE(
        vae_pth=os.path.join(ckpt_dir, 'Wan2.1_VAE.pth'),
        device=device,
    )
    print(f"[{rank}] Loaded WanVAE in {time.time() - t0:.3f} seconds")  # ~9 seconds

    # 0 DEBUG: VAE decode z shape torch.Size([1, 16, 21, 68, 90])
    x0 = torch.load("tensor_x0.pt").to(device)  # [16, 21, 68, 90]
    print(f"[{rank}] x0 shape: {x0.shape}")
    x0 = [x0]

    # Warmup run (loading models fully, etc)
    t0 = time.time()
    print(f"[{rank}] Warmup decoding x0...")
    videos = vae.decode(x0)
    print(f"[{rank}] Decoded x0 in {time.time() - t0:.3f} seconds")  # ~10.5 seconds
    video = videos[0]
    sum_val = video.sum().item()
    print(f"[{rank}] video sum:", sum_val)  # -33767824
    print(f"[{rank}] videos shape: {video.shape}")  # [3, 81, 544, 720]

    # Sync just in case
    dist.barrier()

    # Actual run
    t0 = time.time()
    print(f"[{rank}] Decoding x0...")
    videos = vae.decode(x0)
    print(f"[{rank}] Decoded x0 in {time.time() - t0:.3f} seconds")  # ~4.5 seconds
    # Checking sizes
    video = videos[0]
    sum_val = video.sum().item()
    print(f"[{rank}] video sum:", sum_val)  # -33767824
    print(f"[{rank}] videos shape: {video.shape}")  # [3, 81, 544, 720]

    cache_video(
        tensor=video[None],
        save_file="video_debug_original.mp4",
        fps=16,
        nrow=1,
    )

    # Inner code for decode distributed
    for retry in range(1):
        print(f"[{rank}] {retry} ==========================")
        zs = x0
        t0 = time.time()
        with amp.autocast('cuda', dtype=vae.dtype):
            '''
            videos = [
                # Replaced with the function
                # vae.model.decode(u.unsqueeze(0), vae.scale).float().clamp_(-1, 1).squeeze(0)
                decode(vae.model, u, vae.scale).float().clamp_(-1, 1).squeeze(0)
                for u in zs
            ]
            '''
            videos = []
            for u in zs:
                video = decode_parallel(vae.model, u, vae.scale)
                # video = decode(vae.model, u, vae.scale)
                if rank == 0:
                    assert video is not None
                    videos.append(video.float().clamp_(-1, 1).squeeze(0))
        print(f"[{rank}] Parallelized inner decode in {time.time() - t0:.3f} seconds")  # ~4.4 seconds
        if rank == 0:
            video = videos[0]
            sum_val = video.sum().item()
            print(f"[{rank}@{retry}] video sum:", sum_val)  # -33351406.0 != -33767824
            print(f"[{rank}@{retry}] videos shape: {video.shape}")  # [3, 81, 544, 720]
            # cache_video(
            #    tensor=video[None],
            #    save_file=f"video_debug_parallel_{retry}.mp4",
            #    fps=16,
            #    nrow=1)

    # Inner code for decode yield
    for retry in range(1):
        print(f"[{rank}] {retry} ==========================")
        zs = x0
        t0 = time.time()
        with amp.autocast('cuda', dtype=vae.dtype):
            video_frames = []
            frame_id = 0
            for video_latent in zs:
                # video_latent shape [16, 21, 68, 90]
                for video_frame in vae.decode_stream(video_latent):
                    # video_frame shape [3, 544, 720]
                    if rank == 0:
                        video_frames.append(video_frame)
                        print(frame_id, "video_frame", video_frame.shape)
                        frame_id += 1
        print(f"[{rank}] Inner decode with yield in {time.time() - t0:.3f} seconds")  # ~4.4 seconds
        if rank == 0:
            video = torch.stack(video_frames)  # 81 * [3, 544, 720] -> [81, 3, 544, 720]
            video = video.permute(1, 0, 2, 3)  # [81, 3, 544, 720] -> [3, 81, 544, 720]
            sum_val = video.sum().item()
            print(f"[{rank}@{retry}] video sum:", sum_val)  # -33351406.0 != -33767824
            print(f"[{rank}@{retry}] videos shape: {video.shape}")  # [3, 81, 544, 720]

            cache_video(
                tensor=video[None],
                save_file=f"video_debug_yield_{retry}.mp4",
                fps=16,
                nrow=1,
            )


if __name__ == "__main__":
    main()
