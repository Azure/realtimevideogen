from __future__ import annotations

import math

from sim_types import WorkflowConfig
from sim_types import GPUType
from sim_types import Model


SECONDS_IN_MINUTE = 60.0
SECONDS_IN_HOUR = 60.0 * 60.0

# Video resolution constants (16:10)
NUM_PIXELS_ORIGINAL = 1280 * 800
NUM_PIXELS_ORIGINAL_FLUX = 1280 * 800
NUM_PIXELS_ORIGINAL_HF = 512 * 320
NUM_PIXELS_ORIGINAL_FT = 640 * 400
NUM_PIXELS_ORIGINAL_UPSCALER = 1280 * 800

NUM_PIXELS_MEDIUM = 640 * 400
NUM_PIXELS_MEDIUM_FLUX = 640 * 400
NUM_PIXELS_MEDIUM_HF = 256 * 160
NUM_PIXELS_MEDIUM_FT = 320 * 200
NUM_PIXELS_MEDIUM_UPSCALER = 640 * 400

NUM_PIXELS_LOW = 320 * 200
NUM_PIXELS_LOW_FLUX = 320 * 200
NUM_PIXELS_LOW_HF = 128 * 80
NUM_PIXELS_LOW_FT = 160 * 100
NUM_PIXELS_LOW_UPSCALER = 320 * 200

# StreamCast constants
TOTAL_INPUT_TOKENS = 20 * 1024  # 20K tokens for instructions, PDFs, etc.
TOTAL_VIDEO_SECONDS = 10 * 60  # 10 minutes video
TOTAL_SUBSCENES = 172  # each subscene is 3.5 seconds -> limited by fantasytalking 81 frames at 23 FPS
TOTAL_SCENES = 43  # each scene is 4 subscenes
FPS: dict[Model, float] = {
    Model.HF: 30,
    Model.FT: 23,
}
NUM_STEPS: dict[Model, int] = {
    Model.FLUX: 25,
    Model.HF: 10,
    Model.FT: 10,
}
FRAMES_OPTIONS: dict[Model, list[int]] = {
    Model.HF: [36, 72, 108, 144, 324],
    Model.FT: [9, 21, 41, 61, 77],
}
FRAMES_PER_STEP_IDX = 4

DEFAULT_WORKFLOW_CONFIG = WorkflowConfig(
    total_video_seconds=TOTAL_VIDEO_SECONDS,
    total_scenes=TOTAL_SCENES,
    total_frames={
        Model.HF: math.ceil(TOTAL_VIDEO_SECONDS * FPS[Model.HF]),
        Model.FT: math.ceil(TOTAL_VIDEO_SECONDS * FPS[Model.FT]),
    },
    total_subscenes=TOTAL_SUBSCENES,
    per_subscene_frames={
        Model.HF: math.ceil(TOTAL_VIDEO_SECONDS * FPS[Model.HF] / TOTAL_SUBSCENES),
        Model.FT: math.ceil(TOTAL_VIDEO_SECONDS * FPS[Model.FT] / TOTAL_SUBSCENES),
    },
    # default per-frame number of denoising steps
    num_steps=dict(NUM_STEPS),
    # supported number of generation frames
    hf_frames=FRAMES_OPTIONS[Model.HF],
    ft_frames=FRAMES_OPTIONS[Model.FT],
    frames_per_step_idx=FRAMES_PER_STEP_IDX,
    total_input_tokens=TOTAL_INPUT_TOKENS,
)

# Available device counts for scaling
# Tensor parallelism (TP) or sequence parallelism (SP)
DEVICE_OPTIONS = {
    Model.GEMMA: [1, 2, 4, 8],
    Model.FLUX: [1, 2, 4, 8, 16],
    Model.OTHERS: [1],  # Single GPU, no parallelism
    Model.HF: [1, 2, 4, 8, 10, 16, 20, 24, 32, 40],
    Model.HF_VAE: [1],  # Single GPU, no parallelism
    Model.FT: [1, 2, 4, 8, 10, 16, 20, 24, 32, 40],
    Model.FT_VAE: [1],  # Single GPU, no parallelism
    Model.UPSCALER: [1, 2, 4, 8],  # Single GPU, no parallelism
}

# Models that only have one instance in the system, so not scaling them across GPU types
SINGLE_INSTANCE_MODELS = [
    Model.GEMMA,
    Model.FLUX,
    Model.OTHERS,
]

# Models that can only be run on a single GPU
SINGLE_DEVICE_MODELS = [
    Model.OTHERS,
    Model.HF_VAE,
    Model.FT_VAE,
]


NUM_GPUS_PER_SERVER = {
    GPUType.A100: 8,
    GPUType.H100: 8,
    GPUType.H200: 8,
    GPUType.GB200: 8,  # This is technically 4 GPUs per server, but nothing fits
}


POWER_GPU_IDLE = {
    GPUType.A100: 65.0,  # Watts
    GPUType.H100: 80.0,  # Watts TODO placeholder value
    GPUType.H200: 80.0,  # Watts TODO placeholder value
    GPUType.GB200: 170.0,  # Watts
}


POWER_GPU_TDP = {
    GPUType.A100: 400.0,  # Watts
    GPUType.H100: 700.0,  # Watts
    GPUType.H200: 700.0,  # Watts
    GPUType.GB200: 1200.0,  # Watts
}


# Cost per GPU
GPU_SPOT_COST = {
    # $ / hour (Spot prices)
    GPUType.A100: 1.07,  # $8.56 for 8 GPUs
    GPUType.H100: 4.03,  # $32.24 for 8 GPUs
    GPUType.H200: 4.22,  # $33.76 for 8 GPUs
    GPUType.GB200: 10.76  # $43.04 for 4 GPUs
}

GPU_RESERVED_COST = {
    # $ / hour (Reserved prices)
    GPUType.A100: 3.4,  # $27.2 for 8 GPUs
    GPUType.H100: 5.39,  # $43.12 for 8 GPUs
    GPUType.H200: 5.64,  # $45.12 for 8 GPUs
    GPUType.GB200: 14.42  # $57.68 for 4 GPUs
}

GPU_COST = GPU_SPOT_COST
