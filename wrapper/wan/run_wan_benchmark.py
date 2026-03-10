import os
import time
import logging
import torch

from PIL import Image

from torch.profiler import profile
from torch.profiler import ProfilerActivity

from wrapper_wan21 import Wan21VideoGeneration

from wan.utils.utils import cache_video


def main() -> None:
    NUM_GPUS = 8
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    RANK = int(os.environ.get("RANK", 1))
    GPU = torch.cuda.get_device_name(0)

    TIMING_LOG_FILE = "timing.csv"

    # Load the Wan model
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]
    with profile(activities=activities, profile_memory=True, record_shapes=True) as prof:
        t0 = time.time()

        parallel_mode = "mixed"
        # Multiserver setup: ulysses in server and ring across
        ulysses_size = NUM_GPUS if WORLD_SIZE > NUM_GPUS else WORLD_SIZE
        ring_size = WORLD_SIZE // NUM_GPUS if WORLD_SIZE > NUM_GPUS else 1

        if parallel_mode == "ulysses":
            # Ulysses all GPUs in the cluster
            ulysses_size = WORLD_SIZE
            ring_size = 1
        elif parallel_mode == "ring":
            # Ring all GPUs in the cluster
            ulysses_size = 1
            ring_size = WORLD_SIZE

        print(f"[{RANK:03d}] Parallel setup: {WORLD_SIZE} GPUs, {ulysses_size} Ulysses, {ring_size} Ring")
        video_gen = Wan21VideoGeneration(
            # ulysses_size=ulysses_size,
            # ring_size=ring_size,
            param_dtype=torch.bfloat16,
            # param_dtype=torch.float32,
        )
        print(f"[{video_gen.rank}] Loaded model in {time.time() - t0:.3f} seconds {video_gen.load_timer}")

    # Show profiling data
    if video_gen.rank == 0:
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

    # Translated from "wan/configs/shared_config.py"
    input_neg_prompt = "Gorgeous colors, overexposed, static, blurred details, subtitles, style, artwork, "
    input_neg_prompt += "painting, picture, still, overall gray, worst quality, low quality, "
    input_neg_prompt += "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    input_neg_prompt += "poorly drawn faces, deformed, disfigured, deformed limbs, fused fingers, "
    input_neg_prompt += "still picture, cluttered background, three legs, many people in the background, "
    input_neg_prompt += "walking backwards"

    # Query 1
    input_image = Image.open("/mnt/inigog/generated_image.png").convert("RGB")  # 640x480
    input_prompt = "The woman on the right speaks while the man on the left listens."
    input_neg_prompt = "No camera movement."

    # Query 2
    # input_image = Image.open("generated_image_man.png").convert("RGB")
    # input_image = Image.open("generated_image_man_highres.png").convert("RGB")
    # input_prompt = "The man is explaining something to another person in the podcast studio."
    # input_neg_prompt = "No camera movement."

    # Warm-up run (loading models fully, etc)
    video = video_gen.generate(
        img=Image.open("/mnt/inigog/warmup_image.png").convert("RGB"),
        prompt="Warmup prompt",
        neg_prompt="Warmup prompt",
        num_frames=4 + 1,
        sampling_steps=1,
    )

    if video_gen.rank == 0:
        with open(TIMING_LOG_FILE, "a") as file_timing:
            file_timing.write(
                "#run_id,hw,world_size,ulysses_size,ring_size,batch_size,num_frames,sampling_steps,colors,"
                "frames,height,width,txt_enc,img_enc,vae_enc,sched_setup,dit,sched,vae_dec,total\n")
            video_size_str = ",".join(map(str, video.size()))
            file_timing.write(
                f"w,{GPU},{WORLD_SIZE},{ulysses_size},{ring_size},1,5,1,{video_size_str},{video_gen.gen_timer[-1]}\n")

    # Timing multiple configurations
    for run_id in range(1):
        # for num_frames in [80+1, 4+1, 8+1, 20+1, 40+1, 60+1, 80+1]: # 4n+1
        for num_frames in [80 + 1]:  # 4n+1
            # for sampling_steps in [10, 1, 2, 4, 5, 6, 8, 10, 20, 50]:
            for sampling_steps in [10, 50]:
                t0 = time.time()
                video = video_gen.generate(
                    img=input_image,
                    prompt=input_prompt,
                    neg_prompt=input_neg_prompt,
                    num_frames=num_frames,
                    sampling_steps=sampling_steps,
                )
                total_time_seconds = time.time() - t0
                logging.info(
                    f"Video generated in {total_time_seconds:.3f} seconds with {num_frames} frames "
                    f"and {sampling_steps} steps.")

                if video_gen.rank == 0:
                    with open(TIMING_LOG_FILE, "a") as file_timing:
                        video_size_str = ",".join(map(str, video.size()))
                        file_timing.write(
                            f"{run_id},{GPU},{WORLD_SIZE},{ulysses_size},{ring_size},1,{num_frames},"
                            f"{sampling_steps},{video_size_str},{video_gen.gen_timer[-1]}\n")

                if video_gen.rank == 0:
                    video_file_name = f"gen_video_world{WORLD_SIZE}_u{ulysses_size}_r{ring_size}_" + \
                                      f"frames{num_frames}_steps{sampling_steps}_run{run_id}.mp4"
                    cache_video(
                        tensor=video[None],
                        save_file=video_file_name,
                        fps=video_gen.FPS,
                        nrow=1,
                        normalize=True,
                        value_range=(-1, 1))

    del video_gen


if __name__ == "__main__":
    main()
