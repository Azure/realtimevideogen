import os
import requests
import logging
import random
import argparse
import re

from typing import Optional

from benchmark_commons import HEADERS_JSON
from benchmark_commons import setup_logging
from benchmark_commons import log_and_print
from benchmark_commons import ServiceRequestInfo

from media_utils import get_video_file_info
from image_utils import img_to_base64

from PIL import Image


class ServiceHunyuanFramePackRequestInfo(ServiceRequestInfo):
    def __init__(self, data_json: dict):
        if "hunyuanframepack" not in data_json:
            logging.error("Missing 'hunyuanframepack' key in JSON response")
            return
        data_json = data_json["hunyuanframepack"]
        self.gpu_model = data_json["gpu"]
        self.dtype = data_json["dtype"]
        self.ring_size = data_json["ring_size"] if "ring_size" in data_json else -1
        self.ulysses_size = data_json["ulysses_size"] if "ulysses_size" in data_json else -1
        self.world_size = data_json["world_size"] if "world_size" in data_json else -1
        self.torch_compile = data_json["torch_compile"] if "torch_compile" in data_json else False
        gen_timer = data_json["gen_timer"][-1]
        self.total_time = gen_timer["total"]
        # dit_XXX dit_XXX_YYY image_encoder text_encoder vae_decoder vae_encoder
        gen_timer_dit_outter_steps = {k: v for k, v in gen_timer.items() if re.match(r"^dit_\d+$", k)}
        gen_timer_dit_inner_steps = {k: v for k, v in gen_timer.items() if re.match(r"^dit_\d+_\d+$", k)}
        self.image_encoder_time = gen_timer.get("image_encoder", 0.0)
        self.text_encoder_time = gen_timer.get("text_encoder", 0.0)
        self.vae_encoder_time = gen_timer.get("vae_encoder", 0.0)
        self.num_steps = len(gen_timer_dit_inner_steps) / \
            len(gen_timer_dit_outter_steps) if len(gen_timer_dit_outter_steps) > 0 else 0
        self.total_steps_time = sum(gen_timer_dit_inner_steps.values())
        self.avg_steps_time = 0
        if gen_timer_dit_inner_steps:
            self.total_steps_time / len(gen_timer_dit_inner_steps)
        self.vae_decoder_time = gen_timer.get("vae_decoder", 0.0)

    def to_csv_str(self) -> str:
        return f"{self.gpu_model},{self.dtype}," \
            f"{self.ring_size},{self.ulysses_size}," \
            f"{self.world_size},{self.torch_compile}," \
            f"{self.num_steps},{self.total_steps_time:.2f}," \
            f"{self.avg_steps_time:.2f}," \
            f"{self.total_time:.2f}"

    @staticmethod
    def get_csv_header() -> str:
        return "gpu_model,dtype,ring_size,ulysses_size,world_size," \
            "torch_compile,num_steps,total_steps_time,avg_steps_time,total_time"
        #    "torch_compile,num_steps,total_steps_time,avg_steps_time,total_time"


def get_server_request_info(container_ip: str, container_port: int) -> Optional[ServiceHunyuanFramePackRequestInfo]:
    url_health = f"http://{container_ip}:{container_port}/health"
    response_health = requests.get(url_health, timeout=10)
    if response_health.ok:
        data_json = response_health.json()
        return ServiceHunyuanFramePackRequestInfo(data_json)
    return None


RANDOM_PROMPTS = [
    "A cat jumps over a fence",
    "A person riding a bicycle in the city",
    "A dog playing with a ball in the park",
    "A car driving through a mountain road",
    "A bird flying over a lake at sunrise",
]

RANDOM_NEG_PROMPTS = [
    "blurry",
    "low quality",
    "noisy",
    "distorted",
    "unrealistic",
    "strange shapes",
    "chaotic",
]

# Benchmark parameters
NUM_RUNS = 3

# FramePack reduces the sizes a lot
# https://github.com/lllyasviel/FramePack/blob/main/diffusers_helper/bucket_tools.py
# Large frame sizes run out of memory
TEST_VIDEO_SIZES = [
    # 8 GPUs
    (384, 256),  # 3:2, 0.10 MP
    (512, 384),  # 4:3, 0.20 MP
    (768, 512),  # 3:2, 0.39 MP
    # 4 GPUs
    (320, 240),  # 4:3, 0.08 MP
    (640, 480),  # 4:3, 0.31 MP
    # (640, 608),  # ~1, 0.39 MP
    # 2 GPUs
    (704, 544),  # ~4:3, 0.38 MP
    (832, 480),  # ~16:9, 0.40 MP
]
TEST_STEPS = [
    1,
    # 2,
    5,
    10,
    20,
    50,
]
FPS = 30.0
TEST_NUM_FRAMES = [
    36,  # 1.2 seconds
    72,  # 2.4 seconds
    # 108,  # 3.6 seconds
    144,  # 4.8 seconds
    324,  # 10.8 seconds
]

setup_logging()

if not os.path.exists("output"):
    os.makedirs("output")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Hunyuan FramePack server performance")
    parser.add_argument("--container_ip", type=str, default=None)
    parser.add_argument("--container_port", type=int, default=-1)
    parser.add_argument("--container", type=str, default=None)
    parser.add_argument("--input_img", type=str, default=None, help="Path to input image (will be resized)")
    parser.add_argument("--output_csv", type=str, default="hunyuanframepack.csv",
                          help="Output CSV file for results")
    args = parser.parse_args()

    container_ip = args.container_ip
    container_port = args.container_port
    if args.container:
        match = re.match(r"^(http://)?(\d+\.\d+\.\d+\.\d+):(\d+)$", args.container)
        if match:
            container_ip = match.group(2)
            container_port = int(match.group(3))
    url = f"http://{container_ip}:{container_port}/hunyuanframepack"
    logging.info(f"Using Hunyuan FramePack at URL: {url}")

    # Use a sample image
    if not args.input_img or not os.path.exists(args.input_img):
        raise ValueError("Please provide a valid input image path with --input_img")
    img = Image.open(args.input_img).convert("RGB")
    img_base64 = img_to_base64(img)

    # Output CSV file
    output_csv = args.output_csv

    # Warmup run
    payload_warmup = {
        "img": img_base64,
        "prompt": "Warmup video",
        "neg_prompt": "",
        "width": 768,
        "height": 512,
        "num_frames": 9,
        "sampling_steps": 2,
    }
    response_warmup = requests.post(url, json=payload_warmup, headers=HEADERS_JSON, timeout=600)
    if not response_warmup.ok:
        raise RuntimeError(f"Warmup request failed: {response_warmup.status_code} {response_warmup.text}")
    logging.info(f"Warmed up in {response_warmup.elapsed.total_seconds()} seconds")

    # Print CSV header
    server_csv_header = ServiceHunyuanFramePackRequestInfo.get_csv_header()
    line_csv = f"#run_num,steps,frames,width,height,{server_csv_header},http_time"
    log_and_print(output_csv, line_csv)

    # Benchmark parameters
    for num_run in range(NUM_RUNS):
        for test_steps in TEST_STEPS:
            for test_frames in TEST_NUM_FRAMES:
                for test_w, test_h in TEST_VIDEO_SIZES:
                    payload = {
                        "img": img_base64,
                        "prompt": random.choice(RANDOM_PROMPTS),
                        "neg_prompt": random.choice(RANDOM_NEG_PROMPTS),
                        "width": test_w,
                        "height": test_h,
                        "num_frames": test_frames,
                        "sampling_steps": test_steps,
                    }

                    response = requests.post(url, json=payload, headers=HEADERS_JSON, timeout=600)

                    server_req_info = get_server_request_info(container_ip, container_port)
                    if server_req_info is None:
                        logging.error("Failed to get server request info, skipping run")
                        continue
                    server_req_info_csv = server_req_info.to_csv_str()

                    if response.ok and "video/mp4" in response.headers.get("Content-Type", ""):
                        http_time = response.elapsed.total_seconds()
                        video_binary = response.content

                        """
                        # Save output for visual inspection
                        video_path =
                        f"output/hunyuanframepack_benchmark_{test_steps}_frames_{test_frames}_size_"
                        f"{test_w}x{test_h}_{num_run}.mp4"
                        with open(video_path, "wb") as f:
                            f.write(video_binary)

                        # Optionally, extract first frame for quick check
                        if with_video:
                            try:
                                vid = imageio.get_reader(video_path, "ffmpeg")
                                first_frame = vid.get_data(0)
                                imageio.imwrite(video_path.replace(".mp4", "_frame0.png"), first_frame)
                            except Exception as e:
                                logging.warning(f"Could not extract frame: {e}")
                        """
                        video_file_info = get_video_file_info(video_binary)
                        video_info = video_file_info["video"]
                        response_w, response_h = video_info["width"], video_info["height"]
                        video_num_frames = video_info["num_frames"]
                        frame_count_mismatch = (
                            video_num_frames is not None and
                            (video_num_frames < test_frames - 1 or video_num_frames > test_frames + 1)
                        )

                        # TODO verify there is some audio
                        # TODO verify there is some video

                        if response_w != test_w or response_h != test_h:
                            logging.error(f"Expected video size {test_w}x{test_h}, but got {response_w}x{response_h}")
                            line_csv = f"{num_run},{test_steps},{test_frames},{test_w},{test_h}," + \
                                       f"{server_req_info_csv},-1"
                            log_and_print(output_csv, line_csv)
                        elif server_req_info.num_steps != test_steps:
                            logging.error(f"Expected {test_steps} steps, but got {server_req_info.num_steps} steps")
                            line_csv = f"{num_run},{test_steps},{test_frames},{test_w},{test_h}," + \
                                       f"{server_req_info_csv},-1"
                            log_and_print(output_csv, line_csv)
                        elif frame_count_mismatch:
                            # Add some margin because of the audio length
                            logging.error(f"Expected {test_frames} frames, but got {video_num_frames} frames")
                            line_csv = f"{num_run},{test_steps},{test_frames},{test_w},{test_h}," + \
                                       f"{server_req_info_csv},-1"
                            print(line_csv)
                            with open(output_csv, "a") as file_csv:
                                file_csv.write(line_csv + "\n")
                        else:
                            line_csv = f"{num_run},{test_steps},{test_frames},{test_w},{test_h}," + \
                                       f"{server_req_info_csv},{http_time:.2f}"
                            log_and_print(output_csv, line_csv)
                    # CUDA out of memory. Tried to allocate 6.64 GiB.
                    # GPU 0 has a total capacity of 79.21 GiB of which 4.91 GiB is free.
                    elif "CUDA out of memory. Tried to allocate " in response.text:
                        response_json = response.json()
                        logging.error(f"Out of memory error: {response_json}")
                        line_csv = f"{num_run},{test_steps},{test_frames},{test_w},{test_h},{server_req_info_csv},-1"
                        log_and_print(output_csv, line_csv)
                    elif "exceeds maximum frames" in response.text:
                        response_json = response.json()
                        error_msg = response_json["error"]
                        logging.error(f"Maximum frames: {error_msg}")
                        line_csv = f"{num_run},{test_steps},{test_frames},{test_w},{test_h},{server_req_info_csv},-1"
                        log_and_print(output_csv, line_csv)
                    elif "is not divisible by sp_world" in response.text:
                        # 832x480 -> hidden_states torch.Size([1, 17226, 3072]) is not divisible by sp_world 4.
                        response_json = response.json()
                        error_msg = response_json["error"]
                        logging.error(f"Bad parallelism: {error_msg}")
                        line_csv = f"{num_run},{test_steps},{test_frames},{test_w},{test_h},{server_req_info_csv},-1"
                        log_and_print(output_csv, line_csv)
                    elif "not supported for" in response.text:
                        # 480x832 (17966) not supported for 8 GPUs.
                        response_json = response.json()
                        error_msg = response_json["error"]
                        logging.error(f"Bad parallelism: {error_msg}")
                        line_csv = f"{num_run},{test_steps},{test_frames},{test_w},{test_h},{server_req_info_csv},-1"
                        log_and_print(output_csv, line_csv)
                    else:
                        logging.error(f"Failed to get video from response: {response.text}" if hasattr(
                            response, "text") else response)
                        line_csv = f"{num_run},{test_steps},{test_frames},{test_w},{test_h},{server_req_info_csv},-1"
                        log_and_print(output_csv, line_csv)
