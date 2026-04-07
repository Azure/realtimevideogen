import math

from enum import Enum

# FPS
WAN_FPS = 16.0
FANTASYTALKING_FPS = 23.0
HUNYUANFRAMEPACK_FPS = 30.0
# VAE
VAE_T = 4
FANTASYTALKING_VAE_T = VAE_T
HUNYUANFRAMEPACK_VAE_T = VAE_T
VAE_STRIDE = (VAE_T, 8, 8)  # (T, H, W)

# Quality as number of steps:
#  Low: 10
#  Medium: 20
#  High: 30
NUM_STEPS = 20

# This duration is the mismatch with 1 + (frames - 1 // 4) * 4
# We can probably extend this number beyond 1+80 frames
# MAX_FT_DURATION_SECS = (1 + 80) / 23.0 # 1+10 frames / 23 FPS
MAX_FT_DURATION_SECS = (1 + 116) / 23.0  # 1+10 frames / 23 FPS


class VideoQuality(Enum):
    """Quality of the video."""
    LOW = "low"  # 5 steps
    MEDIUM = "medium"  # 15 steps
    HIGH = "high"  # 25 steps


QUALITY_TO_NUM_STEPS = {
    VideoQuality.LOW.value: 10,
    VideoQuality.MEDIUM.value: 15,
    VideoQuality.HIGH.value: 25,
}


def get_num_video_frames_from_duration(
    duration_seconds: float,
    fps: float = FANTASYTALKING_FPS,
    vae_t: int = FANTASYTALKING_VAE_T
) -> int:
    """
    Get number of frames for the video based on audio duration and FPS.
    This is based on what Fantasy Talking does.
    This rounds based on the latent of the VAE.
    """
    audio_num_frames = int(math.ceil(duration_seconds * fps))
    num_video_frames = int(1 + math.ceil((audio_num_frames - 1) / vae_t) * vae_t)  # Round up to VAE (1+4n)
    return num_video_frames


def to_num_latent_frames(
    num_frames: int,
    vae_t: int = FANTASYTALKING_VAE_T
) -> int:
    return (num_frames - 1) // vae_t + 1


def to_num_frames(
    num_latent_frames: int,
    vae_t: int = FANTASYTALKING_VAE_T
) -> int:
    return num_latent_frames * vae_t + 1
