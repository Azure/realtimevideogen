import os
import torch

from typing import List

from PIL import Image
from PIL.Image import Resampling

from wrapper_wan21 import Wan21VideoGeneration

from wan.utils.utils import cache_video


def main() -> None:
    NUM_GPUS = 8
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    # RANK = int(os.environ.get("RANK", 1))
    GPU = torch.cuda.get_device_name(0)

    DEBUG = False

    ulysses_size = NUM_GPUS if WORLD_SIZE > NUM_GPUS else WORLD_SIZE
    ring_size = WORLD_SIZE // NUM_GPUS if WORLD_SIZE > NUM_GPUS else 1

    video_gen = Wan21VideoGeneration(
        # ulysses_size=ulysses_size,
        # ring_size=ring_size,
    )

    # Size of the frame taken by Wan 2.1 480p
    height = 544
    width = 720

    input_images: List[Image.Image] = [
        Image.open("generated_image_20250415T165936.png"),
        Image.open("generated_image_20250416T162934.png"),
        Image.open("generated_image_flux_20250416T190517.png"),
        Image.open("generated_image_hidream_20250416T171929.png"),
        Image.open("generated_image_hidream_20250416T172445.png"),
        Image.open("generated_image_hidream_20250416T190419.png"),
        Image.open("generated_image_hidream_20250416T174426.png"),
        Image.open("person1_generated_image_hidream_20250416T171929.png"),
        Image.open("person0_generated_image_hidream_20250416T171929.png"),
    ]
    input_images = [
        input_image.convert("RGB").resize((width, height), Resampling.LANCZOS)
        for input_image in input_images
    ]

    input_prompts = [
        "The person is speaking.",
        "The person is covering their face.",
        "The person is raising their arms.",
        "The person is yawning.",
        "The person is standing up.",
        "The person is pointing.",
        "The person is sitting down.",
        "The person is waving.",
        "The person is clapping.",
        "The person is turning their head.",
    ]

    # Warm-up run (loading models fully, etc)
    video = video_gen.generate(
        img=Image.open("warmup_image.png").convert("RGB"),
        prompt="Warmup prompt",
        neg_prompt="Warmup prompt",
        num_frames=1 + 4,
        sampling_steps=1,
    )

    if video_gen.rank == 0:
        TIMING_LOG_FILE = "timing.csv"
        with open(TIMING_LOG_FILE, "a") as file_timing:
            file_timing.write(
                "#run_id,hw,world_size,ulysses_size,ring_size,batch_size,num_frames,sampling_steps,"
                "colors,frames,height,width,txt_enc,img_enc,vae_enc,sched_setup,dit,sched,vae_dec,total\n")
            video_size_str = ",".join(map(str, video.size()))
            file_timing.write(
                f"w,{GPU},{WORLD_SIZE},{ulysses_size},{ring_size},1,5,1,{video_size_str},{video_gen.gen_timer[-1]}\n")

    # Run without batching to set the baseline
    video = video_gen.generate(
        img=input_images[0],
        prompt=input_prompts[0],
        neg_prompt="",
        num_frames=81,
        sampling_steps=10,
    )
    if video_gen.rank == 0:
        with open(TIMING_LOG_FILE, "a") as file_timing:
            video_size_str = ",".join(map(str, video.size()))
            file_timing.write(
                f"b,{GPU},{WORLD_SIZE},{ulysses_size},{ring_size},1,81,10,{video_size_str},{video_gen.gen_timer[-1]}\n")

        if DEBUG:
            video_file_name = f"gen_video_world{WORLD_SIZE}_u{ulysses_size}_r{ring_size}_frames81_steps10_run0.mp4"
            cache_video(
                tensor=video[None],
                save_file=video_file_name,
                fps=video_gen.FPS,
                nrow=1,
                normalize=True,
                value_range=(-1, 1)
            )

    # Running batches of different sizes
    for run_id in range(1):
        num_frames = 1 + 80
        sampling_steps = 10
        for batch_size in range(1, len(input_images) + 1):
            batch_img = input_images[0:batch_size]
            batch_prompt = input_prompts[0:batch_size]
            batch_neg_prompt = [""] * batch_size
            batch_num_frames = [num_frames] * batch_size
            batch_start_frames = [1] * batch_size

            videos = video_gen.generate_batch(
                batch_img=batch_img,
                batch_prompt=batch_prompt,
                batch_neg_prompt=batch_neg_prompt,
                batch_num_frames=batch_num_frames,
                batch_start_frames=batch_start_frames,
                sampling_steps=sampling_steps,
            )
            if video_gen.rank == 0:
                with open(TIMING_LOG_FILE, "a") as file_timing:
                    video_size_str = ",".join(map(str, videos[0].size()))
                    file_timing.write(
                        f"{run_id},{GPU},{WORLD_SIZE},{ulysses_size},{ring_size},{batch_size},{num_frames},"
                        f"{sampling_steps},{video_size_str},{video_gen.gen_timer[-1]}\n")

                # Save videos for debugging
                if DEBUG:
                    for batch_id, video in enumerate(videos):
                        video_file_name = f"gen_video_world{WORLD_SIZE}_u{ulysses_size}_" + \
                            f"r{ring_size}_batch{batch_id}_frames{num_frames}_steps{sampling_steps}_run{run_id}.mp4"
                        cache_video(
                            tensor=video[None],
                            save_file=video_file_name,
                            fps=video_gen.FPS,
                            nrow=1,
                            normalize=True,
                            value_range=(-1, 1)
                        )


if __name__ == "__main__":
    main()
