import os
import asyncio
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

from file_utils import read_file_base64

from media_utils import get_video_file_info
from image_utils import img_to_base64
from media_utils import chunk_audio_base64
from media_utils import get_audio_duration

from PIL import Image


class ServiceFantasyTalkingRequestInfo(ServiceRequestInfo):
    """Fantasy Talking service request info extracted from /health endpoint JSON response."""

    def __init__(
        self,
        data_json: dict
    ) -> None:
        if "fantasytalking" not in data_json:
            logging.error("Missing 'fantasytalking' key in JSON response")
            return
        data_ft_json = data_json["fantasytalking"]
        self.gpu_model = data_ft_json["gpu"]
        self.dtype = data_ft_json["dtype"]
        self.ring_size = data_ft_json["ring_size"] if "ring_size" in data_ft_json else -1
        self.ulysses_size = data_ft_json["ulysses_size"] if "ulysses_size" in data_ft_json else -1
        self.world_size = data_ft_json["world_size"] if "world_size" in data_ft_json else -1
        self.torch_compile = data_ft_json["torch_compile"] if "torch_compile" in data_ft_json else False
        gen_timer = data_ft_json["gen_timer"][-1]
        self.total_time = gen_timer["total"]
        # TODO add VAE and others
        gen_timer_steps = {k: v for k, v in gen_timer.items() if k.startswith("dit_")}
        self.num_steps = len(gen_timer_steps)
        self.total_steps_time = sum(gen_timer_steps.values())
        self.avg_steps_time = self.total_steps_time / len(gen_timer_steps) if gen_timer_steps else 0

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


def get_server_request_info(container_ip: str, container_port: int) -> Optional[ServiceFantasyTalkingRequestInfo]:
    url_health = f"http://{container_ip}:{container_port}/health"
    response_health = requests.get(url_health, timeout=10)
    if response_health.ok:
        data_json = response_health.json()
        return ServiceFantasyTalkingRequestInfo(data_json)
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

FPS = 16.0  # Hunyuan FramePack

# Benchmark parameters
NUM_RUNS = 3

TEST_VIDEO_SIZES = [
    (320, 240),  # does not work with 8 GPUs
    (640, 480),
    (1280, 720),
    # Too much memory after this
    # (1920, 1072), # (1920, 1080) breaks
    # (2560, 1440),
    # (3840, 2160), # 4k
]
TEST_STEPS = [
    1,
    # 2,
    5,
    10,
    20,
    50,
]
TEST_NUM_FRAMES = [
    1 + 8,   # (VAE stride 4, so 1+4*n)
    1 + 20,
    1 + 40,
    1 + 60,
    1 + 76,
    1 + 80,  # Maximum size
]

setup_logging()

if not os.path.exists("output"):
    os.makedirs("output")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Fantasy Talking server performance")
    parser.add_argument("--container_ip", type=str, default=None)
    parser.add_argument("--container_port", type=int, default=-1)
    parser.add_argument("--container", type=str, default=None)
    parser.add_argument("--input_img", type=str, default=None, help="Path to input image (will be resized)")
    parser.add_argument("--input_audio", type=str, default=None, help="Path to input image (will be resized)")
    parser.add_argument("--output_csv", type=str, default="fantasytalking.csv", help="Output CSV file for results")
    args = parser.parse_args()

    container_ip = args.container_ip
    container_port = args.container_port
    if args.container:
        match = re.match(r"^(http://)?(\d+\.\d+\.\d+\.\d+):(\d+)$", args.container)
        if match:
            container_ip = match.group(2)
            container_port = int(match.group(3))
    url = f"http://{container_ip}:{container_port}/fantasytalking"
    logging.info(f"Using Fantasy Talking at URL: {url}")

    # Use a sample image
    if not args.input_img or not os.path.exists(args.input_img):
        raise ValueError("Please provide a valid input image path with --input_img")
    img = Image.open(args.input_img).convert("RGB")
    img_base64 = img_to_base64(img)

    # Use a sample audio
    if not args.input_audio or not os.path.exists(args.input_audio):
        raise ValueError("Please provide a valid input audio path with --input_audio")
    audio_base64 = asyncio.run(read_file_base64(args.input_audio))
    audio_seconds = get_audio_duration(audio_base64)  # Ensure audio is valid
    warmup_audio_base64 = chunk_audio_base64(audio_base64, 0, 0.1)

    # Output CSV file
    output_csv = args.output_csv

    # Warmup run
    payload_warmup = {
        "img": img_base64,
        "audio": warmup_audio_base64,
        "prompt": "Warmup video",
        "neg_prompt": "",
        "width": 640,
        "height": 480,
        "sampling_steps": 2,
    }
    response_warmup = requests.post(url, json=payload_warmup, headers=HEADERS_JSON, timeout=600)
    if not response_warmup.ok:
        raise RuntimeError(f"Warmup request failed: {response_warmup.status_code} {response_warmup.text}")
    logging.info(f"Warmed up in {response_warmup.elapsed.total_seconds():.2f} seconds")

    # CSV header
    server_csv_header = ServiceFantasyTalkingRequestInfo.get_csv_header()
    line_csv = f"#run_num,steps,frames,width,height,{server_csv_header},http_time"
    log_and_print(output_csv, line_csv)

    # Benchmark parameters
    for num_run in range(NUM_RUNS):
        for test_steps in TEST_STEPS:
            for test_frames in TEST_NUM_FRAMES:
                # #frames -> audio seconds -> video length
                chunk_audio_seconds = 1.0 * (test_frames - 1) / FPS
                if chunk_audio_seconds > audio_seconds:
                    logging.warning(
                        f"Requested {test_frames} frames ({chunk_audio_seconds:.2f}s) exceeds audio length "
                        f"({audio_seconds:.2f}s), using full audio")
                    chunk_audio_seconds = audio_seconds
                chunk_audio_str = chunk_audio_base64(audio_base64, 0, chunk_audio_seconds)
                for test_w, test_h in TEST_VIDEO_SIZES:
                    payload = {
                        "img": img_base64,  # TODO resize input image?
                        "audio": chunk_audio_str,
                        "prompt": random.choice(RANDOM_PROMPTS),
                        "neg_prompt": random.choice(RANDOM_NEG_PROMPTS),
                        "width": test_w,
                        "height": test_h,
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

                        video_file_info = get_video_file_info(video_binary)
                        video_info = video_file_info["video"]
                        response_w, response_h = video_info["width"], video_info["height"]
                        video_num_frames = video_info["num_frames"]
                        frame_count_mismatch = (
                            video_num_frames is not None
                            and (video_num_frames < test_frames - 1 or video_num_frames > test_frames + 1)
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
                            # Add 1 frame margin because of the audio length
                            logging.error(f"Expected {test_frames} frames, but got {video_num_frames} frames")
                            line_csv = f"{num_run},{test_steps},{test_frames},{test_w},{test_h}," + \
                                f"{server_req_info_csv},-1"
                            print(line_csv)
                            with open(output_csv, "a") as file_csv:
                                file_csv.write(line_csv + "\n")
                        else:
                            # Log successful run
                            line_csv = f"{num_run},{test_steps},{test_frames},{test_w},{test_h}," + \
                                f"{server_req_info_csv},{http_time:.2f}"
                            log_and_print(output_csv, line_csv)
                    elif "CUDA out of memory. Tried to allocate " in response.text:
                        response_json = response.json()
                        error_msg = response_json["error"]
                        logging.error(f"Out of memory error: {error_msg}")
                        line_csv = f"{num_run},{test_steps},{test_frames},{test_w},{test_h},{server_req_info_csv},-1"
                        log_and_print(output_csv, line_csv)
                    elif "exceeds maximum frames" in response.text:
                        response_json = response.json()
                        error_msg = response_json["error"]
                        logging.error(f"Maximum frames: {error_msg}")
                        line_csv = f"{num_run},{test_steps},{test_frames},{test_w},{test_h},{server_req_info_csv},-1"
                        log_and_print(output_csv, line_csv)
                    elif "is invalid for input of size" in response.text:
                        response_json = response.json()
                        error_msg = response_json["error"]
                        logging.error(f"Invalid size: {error_msg}")
                        line_csv = f"{num_run},{test_steps},{test_frames},{test_w},{test_h},{server_req_info_csv},-1"
                        log_and_print(output_csv, line_csv)
                    else:
                        logging.error(f"Failed to get video from response: {response.text}" if hasattr(
                            response, "text") else response)
                        line_csv = f"{num_run},{test_steps},{test_frames},{test_w},{test_h},{server_req_info_csv},-1"
                        log_and_print(output_csv, line_csv)
