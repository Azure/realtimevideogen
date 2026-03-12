import os
import requests
import logging
import random
import argparse

from typing import Optional

from benchmark_commons import HEADERS_JSON
from benchmark_commons import setup_logging
from benchmark_commons import log_and_print
from benchmark_commons import ServiceRequestInfo

from io import BytesIO
from PIL import Image


class ServiceFluxRequestInfo(ServiceRequestInfo):
    def __init__(self, data_json: dict):
        if "flux" not in data_json:
            logging.error("Missing 'flux' key in JSON response")
            return
        data_flux_json = data_json["flux"]
        self.gpu_model = data_flux_json["gpu"]
        self.dtype = data_flux_json["dtype"]
        self.ring_size = data_flux_json["ring_size"]
        self.ulysses_size = data_flux_json["ulysses_size"]
        self.world_size = data_flux_json["world_size"]
        self.torch_compile = data_flux_json["torch_compile"]
        gen_timer = data_flux_json["gen_timer"][-1]
        self.total_time = gen_timer["total"]
        gen_timer_steps = {k: v for k, v in gen_timer.items() if k.startswith("step_")}
        self.num_steps = len(gen_timer_steps)
        self.total_steps_time = sum(gen_timer_steps.values())
        self.avg_steps_time = 0
        if self.num_steps > 0:
            self.avg_steps_time = self.total_steps_time / self.num_steps

    def to_csv_str(self) -> str:
        return f"{self.gpu_model},{self.dtype}," \
            f"{self.ring_size},{self.ulysses_size}," \
            f"{self.world_size},{self.torch_compile}," \
            f"{self.num_steps},{self.total_steps_time:.2f}," \
            f"{self.avg_steps_time:.2f},{self.total_time:.2f}"

    @staticmethod
    def get_csv_header() -> str:
        return "gpu_model,dtype,ring_size,ulysses_size,world_size," \
            "torch_compile,num_steps,total_steps_time,avg_steps_time,total_time"


def get_server_request_info(container_ip: str, container_port: int) -> Optional[ServiceFluxRequestInfo]:
    """Get server request detailed info from the health endpoint."""
    url_health = f"http://{container_ip}:{container_port}/health"
    response_health = requests.get(url_health)
    if response_health.ok:
        data_json = response_health.json()
        return ServiceFluxRequestInfo(data_json)
    return None


RANDOM_PROMPTS = [
    "A beautiful sunset over the mountains",
    "A futuristic cityscape at night",
    "A serene forest with a river flowing through it",
    "A majestic lion resting under a tree",
    "A vibrant coral reef teeming with marine life",
    "A cozy cabin in the snow with smoke coming from the chimney",
    "A bustling market in a small village",
    "A peaceful beach with gentle waves lapping at the shore",
    "A grand castle on a hilltop surrounded by mist",
    "A colorful hot air balloon floating in the sky",
    "A mysterious ancient temple hidden in the jungle",
    "A sleek sports car racing down a winding road",
    "A group of friends enjoying a picnic in the park",
    "A majestic eagle soaring through the sky",
    "A tranquil lake reflecting the surrounding mountains",
    "A vibrant autumn forest with leaves falling",
    "A futuristic robot exploring a distant planet",
    "A cozy coffee shop with people chatting and working",
]

RANDOM_NEG_PROMPTS = [
    "blurry",
    "low quality",
    "overexposed",
    "underexposed",
    "noisy",
    "distorted",
    "pixelated",
    "unfocused",
    "bad lighting",
    "weird colors",
    "strange shapes",
    "unrealistic",
    "chaotic",
]

# Benchmark parameters
NUM_RUNS = 5
TEST_IMG_SIZES = [
    (320, 240),  # Too small for 8 GPUs
    (640, 480),
    (1280, 720),
    (1920, 1072),  # (1920, 1080) breaks
    (2560, 1440),
    (3840, 2160),  # 4k
]
TEST_STEPS = [
    1,
    5,
    10,
    20,
    50,
]

setup_logging()

if not os.path.exists("output"):
    os.makedirs("output")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Benchmark Flux server performance")
    arg_parser.add_argument("--container_ip", type=str, default="10.244.2.13")
    arg_parser.add_argument("--container_port", type=int, default=8080)
    arg_parser.add_argument("--output_csv", type=str, default="flux.csv", help="Output CSV file for results")
    args = arg_parser.parse_args()

    container_ip = args.container_ip
    container_port = args.container_port
    url = f"http://{container_ip}:{container_port}/flux"
    logging.info(f"Using Flux at URL: {url}")

    # Output CSV file
    output_csv = args.output_csv

    # Warmup run
    payload_warmup = {
        "prompt": "Test image",
        "neg_prompt": "",
        "width": 1280,
        "height": 720,
        "sampling_steps": 1,
    }
    response_warmup = requests.post(url, json=payload_warmup, headers=HEADERS_JSON)
    if not response_warmup.ok:
        raise RuntimeError(f"Warmup request failed: {response_warmup.status_code} {response_warmup.text}")
    logging.info(f"Warmed up in {response_warmup.elapsed.total_seconds()} seconds")

    # CSV header
    server_csv_header = ServiceFluxRequestInfo.get_csv_header()
    line_csv = f"#run_num,steps,width,height,{server_csv_header},http_time"
    log_and_print(output_csv, line_csv)

    # Benchmark parameters
    for num_run in range(NUM_RUNS):
        for test_steps in TEST_STEPS:
            for test_img_size in TEST_IMG_SIZES:
                test_w, test_h = test_img_size
                payload = {
                    "prompt": random.choice(RANDOM_PROMPTS),
                    "neg_prompt": random.choice(RANDOM_NEG_PROMPTS),
                    "width": test_w,
                    "height": test_h,
                    "sampling_steps": test_steps,
                }

                # Actually send the request to the REST Flux endpoint
                response = requests.post(url, json=payload, headers=HEADERS_JSON)

                server_req_info = get_server_request_info(container_ip, container_port)
                server_req_info_csv = server_req_info.to_csv_str()

                if response.ok and "image/png" in response.headers["Content-Type"]:
                    http_time = response.elapsed.total_seconds()

                    image_binary = response.content
                    image_bytes = BytesIO(image_binary)
                    image = Image.open(image_bytes)

                    # Save output for visual inspection
                    image_path = f"output/flux_benchmark_{test_steps}_size_{test_w}x{test_h}_{num_run}.png"
                    image.save(image_path, format="PNG")

                    response_w, response_h = image.size

                    if response_w != test_w or response_h != test_h:
                        logging.error(f"Expected image size {test_w}x{test_h}, but got {response_w}x{response_h}")
                        line_csv = f"{num_run},{test_steps},{test_w},{test_h},{server_req_info_csv},-1"
                        log_and_print(output_csv, line_csv)
                    elif server_req_info.num_steps != test_steps:
                        logging.error(f"Expected {test_steps} steps, but got {server_req_info.num_steps} steps")
                        line_csv = f"{num_run},{test_steps},{test_w},{test_h},{server_req_info_csv},-1"
                        log_and_print(output_csv, line_csv)
                    else:
                        line_csv = f"{num_run},{test_steps},{test_w},{test_h},{server_req_info_csv},{http_time:.2f}"
                        log_and_print(output_csv, line_csv)
                else:
                    if "assert hidden_states.shape[-2]" in response.text:
                        logging.error("Cannot split the image into multiple GPUs, too small image size")
                    elif f"{test_h}x{test_w} not supported for " in response.text:
                        logging.error("Wrong image size")
                    else:
                        logging.error("Failed to generate image: {response.text}")
                    line_csv = f"{num_run},{test_steps},{test_w},{test_h},{server_req_info_csv},-1"
                    log_and_print(output_csv, line_csv)
