"""
Module for loading latency and power consumption data from CSV files.
"""

import pandas as pd

from pathlib import Path

from sim_types import LatencyData
from sim_types import PowerData
from sim_types import GPUType
from sim_types import LatencyGPUTypeData
from sim_types import PowerGPUTypeData
from sim_types import QualityLevel

from constants import NUM_PIXELS_ORIGINAL_UPSCALER
from constants import NUM_PIXELS_ORIGINAL_FT
from constants import NUM_PIXELS_ORIGINAL_HF
from constants import NUM_PIXELS_ORIGINAL_FLUX
from constants import NUM_PIXELS_LOW_FT
from constants import NUM_PIXELS_LOW_HF
from constants import NUM_PIXELS_LOW_FLUX
from constants import NUM_PIXELS_LOW_UPSCALER
from constants import NUM_PIXELS_MEDIUM_FT
from constants import NUM_PIXELS_MEDIUM_HF
from constants import NUM_PIXELS_MEDIUM_UPSCALER
from constants import NUM_PIXELS_MEDIUM_FLUX
from constants import POWER_GPU_IDLE
from constants import POWER_GPU_TDP

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


def load_latency_data(
    data_dir: str | Path = _DEFAULT_DATA_DIR,
) -> LatencyData:
    """
    Load latency and throughput mapping data from CSV files.

    Args:
        data_dir: The directory where the CSV files are stored.
    Returns:
        LatencyData: An object containing all loaded latency data.
    """
    data_path = Path(data_dir)

    data = LatencyData(gpus={})
    for gpu_type in GPUType:
        data.gpus[gpu_type] = LatencyGPUTypeData(gpu_type=gpu_type)

        # Flux time -> per image generation
        csv_flux_path = data_path / f"latency_flux_mapping_{gpu_type.value.lower()}.csv"
        df_flux = pd.read_csv(csv_flux_path, comment='#')
        data[gpu_type].flux = dict(zip(
            df_flux["world_size"],
            df_flux["avg_steps_time"]))

        # Hunyuan Framepack per step time -> [36, 72, 108, 144, 324] frames generation
        csv_hf_path = data_path / f"latency_hf_mapping_{gpu_type.value.lower()}.csv"
        df_hf = pd.read_csv(csv_hf_path, comment='#')
        data[gpu_type].hf = dict(zip(
            df_hf["world_size"],
            df_hf["avg_steps_time"]))

        # Hunyuan Framepack VAE time -> per inference iteration
        # Derived: steps * avg_step_time * vae_pct(vae_time / total_time)
        data[gpu_type].hf_vae = dict(zip(
            df_hf["world_size"],
            df_hf["vae_time"]))

        # Fantasy Talking per step time -> [9, 21, 41, 61, 77] frames generation
        csv_ft_path = data_path / f"latency_ft_mapping_{gpu_type.value.lower()}.csv"
        df_ft = pd.read_csv(csv_ft_path, comment='#')
        data[gpu_type].ft = dict(zip(
            df_ft["world_size"],
            df_ft["avg_steps_time"]))

        # Fantasy Talking VAE time -> per inference iteration
        # Derived: steps * avg_step_time * vae_pct(vae_time / total_time)
        data[gpu_type].ft_vae = dict(zip(
            df_ft["world_size"],
            df_ft["vae_time"]))

        # Upscaler time -> per image frame
        csv_upscaler_path = data_path / f"latency_upscaler_{gpu_type.value.lower()}.csv"
        df_upscaler = pd.read_csv(csv_upscaler_path, comment='#')
        data[gpu_type].upscaler = dict(zip(
            df_upscaler['world_size'],
            df_upscaler['avg_steps_time']))

        # Gemma time -> first scene and per scene
        csv_gemma_path = data_path / f"latency_gemma_{gpu_type.value.lower()}.csv"
        df_gemma = pd.read_csv(csv_gemma_path, comment='#')
        data[gpu_type].gemma_first_scene = dict(zip(
            df_gemma['tp'],
            df_gemma['first_scene_time']))
        data[gpu_type].gemma_per_scene = dict(zip(
            df_gemma['tp'],
            df_gemma['per_scene_time']))

        # Others time -> kokoro and other overheads -> time per scene
        csv_others_path = data_path / f"latency_others_{gpu_type.value.lower()}.csv"
        df_others = pd.read_csv(csv_others_path, comment='#')
        data[gpu_type].others = dict(zip(
            df_others['world_size'],
            df_others['time']))

    return data


def load_power_data(
    data_dir: str | Path = _DEFAULT_DATA_DIR
) -> PowerData:
    """
    Load power consumption data from CSV files.

    Args:
        data_dir: The directory where the CSV files are stored.
    Returns:
        PowerData: An object containing all loaded power consumption data.
    """
    data_path = Path(data_dir)

    data = PowerData(gpus={})
    for gpu_type in GPUType:
        data.gpus[gpu_type] = PowerGPUTypeData(gpu_type=gpu_type)

        # Flux power profile
        power_flux_file_name = data_path / f'power_flux_mapping_{gpu_type.value.lower()}.csv'
        power_flux_df = pd.read_csv(power_flux_file_name, comment='#')
        data[gpu_type].flux = dict(zip(
            power_flux_df['world_size'],
            power_flux_df['power_watts']))

        # Hunyuan Framepack 640x400 power profile
        power_hf_file_name = data_path / f'power_hf_mapping_{gpu_type.value.lower()}.csv'
        power_hf_df = pd.read_csv(power_hf_file_name, comment='#')
        data[gpu_type].hf = dict(zip(
            power_hf_df['world_size'],
            power_hf_df['power_watts']))

        # Hunyuan Framepack 1280x800 power profile
        power_hf_file_name_high = data_path / f'power_hf_mapping_{gpu_type.value.lower()}_high.csv'
        power_hf_high_df = pd.read_csv(power_hf_file_name_high, comment='#')
        data[gpu_type].hf_high = dict(zip(
            power_hf_high_df['world_size'],
            power_hf_high_df['power_watts']))

        # Hunyuan Framepack VAE power profile
        power_hf_vae_file_name = data_path / f'power_hf_vae_{gpu_type.value.lower()}.csv'
        power_hf_vae_df = pd.read_csv(power_hf_vae_file_name, comment='#')
        data[gpu_type].hf_vae = dict(zip(
            power_hf_vae_df['world_size'],
            power_hf_vae_df['power_watts']))

        # Hunyuan Framepack VAE high power profile
        power_hf_vae_high_file_name = data_path / f'power_hf_vae_{gpu_type.value.lower()}_high.csv'
        power_hf_vae_high_df = pd.read_csv(power_hf_vae_high_file_name, comment='#')
        data[gpu_type].hf_vae_high = dict(zip(
            power_hf_vae_high_df['world_size'],
            power_hf_vae_high_df['power_watts']))

        # Fantasy Talking 640x400 power profile
        power_ft_file_name = data_path / f'power_ft_mapping_{gpu_type.value.lower()}.csv'
        power_ft_df = pd.read_csv(power_ft_file_name, comment='#')
        data[gpu_type].ft = dict(zip(
            power_ft_df['world_size'],
            power_ft_df['power_watts']))

        # Fantasy Talking 1280x800 power profile
        power_ft_high_file_name = data_path / f'power_ft_mapping_{gpu_type.value.lower()}_high.csv'
        power_ft_high_df = pd.read_csv(power_ft_high_file_name, comment='#')
        data[gpu_type].ft_high = dict(zip(
            power_ft_high_df['world_size'],
            power_ft_high_df['power_watts']))

        # Fantasy Talking VAE mapping
        power_ft_vae_file_name = data_path / f'power_ft_vae_mapping_{gpu_type.value.lower()}.csv'
        power_ft_vae_df = pd.read_csv(power_ft_vae_file_name, comment='#')
        data[gpu_type].ft_vae = dict(zip(
            power_ft_vae_df['world_size'],
            power_ft_vae_df['power_watts']))

        # Fantasy Talking VAE high mapping
        power_ft_vae_high_file_name = data_path / f'power_ft_vae_mapping_{gpu_type.value.lower()}_high.csv'
        power_ft_vae_high_df = pd.read_csv(power_ft_vae_high_file_name, comment='#')
        data[gpu_type].ft_vae_high = dict(zip(
            power_ft_vae_high_df['world_size'],
            power_ft_vae_high_df['power_watts']))

        # Upscaler power profile
        power_upscaler_file_name = data_path / f'power_upscaler_{gpu_type.value.lower()}.csv'
        power_upscaler_df = pd.read_csv(power_upscaler_file_name, comment='#')
        data[gpu_type].upscaler = dict(zip(
            power_upscaler_df['world_size'],
            power_upscaler_df['power_watts']))

        # Gemma power profile
        power_gemma_first_scene_file_name = data_path / f'power_gemma_first_scene_{gpu_type.value.lower()}.csv'
        power_gemma_per_scene_file_name = data_path / f'power_gemma_per_scene_{gpu_type.value.lower()}.csv'
        power_gemma_first_scene_df = pd.read_csv(power_gemma_first_scene_file_name, comment='#')
        power_gemma_per_scene_df = pd.read_csv(power_gemma_per_scene_file_name, comment='#')
        data[gpu_type].gemma_first_scene = dict(zip(
            power_gemma_first_scene_df['world_size'],
            power_gemma_first_scene_df['power_watts']
        ))
        data[gpu_type].gemma_per_scene = dict(zip(
            power_gemma_per_scene_df['world_size'],
            power_gemma_per_scene_df['power_watts']
        ))

    # Idle and TDP power profiles
    for gpu_type in GPUType:
        data[gpu_type].idle = POWER_GPU_IDLE[gpu_type]
        data[gpu_type].tdp = POWER_GPU_TDP[gpu_type]

    return data


def load_adaptive_quality_data(
    data_dir: str | Path,
    level: QualityLevel,
) -> LatencyData:
    """Load latency data for adaptive quality."""
    assert isinstance(level, QualityLevel)

    latency_data = load_latency_data(data_dir=data_dir)

    if level == QualityLevel.ORIGINAL or level == QualityLevel.HIGH:
        return latency_data

    if level == QualityLevel.MEDIUM:
        ratio_flux = NUM_PIXELS_MEDIUM_FLUX / NUM_PIXELS_ORIGINAL_FLUX
        ratio_hf = NUM_PIXELS_MEDIUM_HF / NUM_PIXELS_ORIGINAL_HF
        ratio_hf_vae = NUM_PIXELS_MEDIUM_HF / NUM_PIXELS_ORIGINAL_HF
        ratio_ft = NUM_PIXELS_MEDIUM_FT / NUM_PIXELS_ORIGINAL_FT
        ratio_ft_vae = NUM_PIXELS_MEDIUM_FT / NUM_PIXELS_ORIGINAL_FT
        ratio_upscaler = NUM_PIXELS_MEDIUM_UPSCALER / NUM_PIXELS_ORIGINAL_UPSCALER
        for gpu_type in GPUType:
            latency_data[gpu_type].flux = {
                k: v * ratio_flux
                for k, v in latency_data[gpu_type].flux.items()
            }
            latency_data[gpu_type].hf = {
                k: v * ratio_hf
                for k, v in latency_data[gpu_type].hf.items()
            }
            latency_data[gpu_type].hf_vae = {
                k: v * ratio_hf_vae
                for k, v in latency_data[gpu_type].hf_vae.items()
            }
            latency_data[gpu_type].ft = {
                k: v * ratio_ft
                for k, v in latency_data[gpu_type].ft.items()
            }
            latency_data[gpu_type].ft_vae = {
                k: v * ratio_ft_vae
                for k, v in latency_data[gpu_type].ft_vae.items()
            }
            latency_data[gpu_type].upscaler = {
                k: v * ratio_upscaler
                for k, v in latency_data[gpu_type].upscaler.items()
            }
        return latency_data

    if level == QualityLevel.LOW:
        ratio_flux = NUM_PIXELS_LOW_FLUX / NUM_PIXELS_ORIGINAL_FLUX
        ratio_hf = NUM_PIXELS_LOW_HF / NUM_PIXELS_ORIGINAL_HF
        ratio_hf_vae = NUM_PIXELS_LOW_HF / NUM_PIXELS_ORIGINAL_HF
        ratio_ft = NUM_PIXELS_LOW_FT / NUM_PIXELS_ORIGINAL_FT
        ratio_ft_vae = NUM_PIXELS_LOW_FT / NUM_PIXELS_ORIGINAL_FT
        ratio_upscaler = NUM_PIXELS_LOW_UPSCALER / NUM_PIXELS_ORIGINAL_UPSCALER
        for gpu_type in GPUType:
            latency_data[gpu_type].flux = {
                k: v * ratio_flux
                for k, v in latency_data[gpu_type].flux.items()
            }
            latency_data[gpu_type].hf = {
                k: v * ratio_hf
                for k, v in latency_data[gpu_type].hf.items()
            }
            latency_data[gpu_type].hf_vae = {
                k: v * ratio_hf_vae
                for k, v in latency_data[gpu_type].hf_vae.items()
            }
            latency_data[gpu_type].ft = {
                k: v * ratio_ft
                for k, v in latency_data[gpu_type].ft.items()
            }
            latency_data[gpu_type].ft_vae = {
                k: v * ratio_ft_vae
                for k, v in latency_data[gpu_type].ft_vae.items()
            }
            latency_data[gpu_type].upscaler = {
                k: v * ratio_upscaler
                for k, v in latency_data[gpu_type].upscaler.items()
            }
        return latency_data

    return latency_data
