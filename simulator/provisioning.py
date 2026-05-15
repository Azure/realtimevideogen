"""
Provisioning simulation module.
"""
from __future__ import annotations

import os
import sys

# Ensure streamwise/ and simulator/ are on sys.path so model_provisioner
# imports work in child processes spawned by ProcessPoolExecutor.
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_STREAMWISE_DIR = os.path.join(_REPO_ROOT, "streamwise")
_SIMULATOR_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_REPO_ROOT, _STREAMWISE_DIR, _SIMULATOR_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tqdm.auto import tqdm

import logging

from typing import Optional

from itertools import product
from itertools import combinations_with_replacement

from functools import partial

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError
from concurrent.futures import as_completed

from model_provisioner.sim_types import WorkflowConfig
from model_provisioner.sim_types import GPUType
from model_provisioner.sim_types import LatencyData
from model_provisioner.sim_types import Provision
from model_provisioner.sim_types import ProvisioningResult
from model_provisioner.sim_types import Model
from model_provisioner.sim_types import ModelAllocation
from model_provisioner.sim_types import PowerData
from model_provisioner.sim_types import QualityLevel
from model_provisioner.sim_types import Policy
from model_provisioner.sim_types import Result
from model_provisioner.sim_types import num_gpus_to_str

from model_provisioner.auto_model_allocator import AutoModelAllocator

from model_provisioner.policies import STREAMWISE_POLICY

from model_provisioner.constants import SECONDS_IN_HOUR


GPU_PROVISIONS: list[int] = [
    0,
    8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88,
    96, 104, 112, 128, 144, 160, 176, 192, 208,
    224, 240, 256, 288, 304, 320, 352, 384, 400, 448, 480,
    512, 576, 624, 640, 672, 704, 768, 832, 864, 896, 960,
    1024, 1152, 1280, 1408, 1536, 1664, 1792, 2048, 2304, 2560, 2880,
    3200, 3584, 4096,
]


# Trimmed down
GPU_PROVISIONS_SHORT: list[int] = [
    0,
    8, 16, 32, 48, 64, 72,
    96, 128, 192,
    256, 480, 512, 768, 1024, 2048, 4096,
]


def get_provisions(
    gpus_types: list[GPUType],
    limits_pairs: bool = True,
    short_list: bool = False,
) -> list[Provision]:
    """
    Generate a list of provisioning options for the given GPU types.
    If limits_pairs=True, we only support pairs.
    """
    assert 0 < len(gpus_types)

    if len(gpus_types) <= 2 or not limits_pairs:
        return get_provisions_internal(
            gpus_types,
            short_list=short_list)

    # Generate all the pairs of GPU types
    provisions: list[Provision] = []
    for gpu_type_pair in combinations_with_replacement(gpus_types, r=2):
        assert len(gpu_type_pair) == 2
        if gpu_type_pair[0] == gpu_type_pair[1]:
            continue  # single GPU type
        pair_provisions = get_provisions_internal(
            list(gpu_type_pair),
            short_list=short_list)
        provisions.extend(pair_provisions)
    provisions = remove_duplicate_provisions(provisions)
    return provisions


def get_provisions_internal(
    gpus_types: list[GPUType],
    short_list: bool = False,
) -> list[Provision]:
    """
    Generate a list of provisioning options for the given GPU types.
    """
    provisions: list[Provision] = []

    for counts in product(GPU_PROVISIONS_SHORT if short_list else GPU_PROVISIONS, repeat=len(gpus_types)):
        num_gpus = {
            gpu_type: count
            for gpu_type, count in zip(gpus_types, counts)
            if count > 0
        }
        if len(num_gpus) > 0:
            provisions.append(Provision(num_gpus=num_gpus))

    provisions = remove_duplicate_provisions(provisions)

    return provisions


def remove_duplicate_provisions(
    provisions: list[Provision],
) -> list[Provision]:
    unique_provisions_dict: dict[str, Provision] = {}
    for provision in provisions:
        key = ','.join([
            f"{gpu_type.value}:{count}"
            for gpu_type, count in sorted(provision.num_gpus.items())
        ])
        unique_provisions_dict[key] = provision

    return list(unique_provisions_dict.values())


def _process_provision(
    provision: Provision,
    workflow: WorkflowConfig,
    latency_data: LatencyData,
    power_data: Optional[PowerData],
    policy: Policy,
    verbose: bool,
) -> tuple[Provision, Optional[Result]]:
    """Worker function to process a single provision."""
    gpu_types = [
        gpu_type
        for gpu_type, count in provision.num_gpus.items()
        if count > 0
    ]
    assert 0 < len(gpu_types), f"No GPUs provisioned: {provision.num_gpus}."
    assert len(gpu_types) <= 2, f"Only support up to 2 GPU types in a provision: {provision.num_gpus}."

    try:
        allocator = AutoModelAllocator(
            workflow=workflow,
            latency_data=latency_data,
            power_data=power_data,
            policy=policy,
        )
        result = allocator.allocate(
            num_gpus=provision.num_gpus,
            verbose=verbose,
        )
        if verbose:
            logging.info(
                f"Total time for {provision}: "
                f"{result.total_time_s:.2f} seconds ({result.total_time_s / SECONDS_IN_HOUR:.2f} hours)")
        return (provision, result)
    except KeyError as key_ex:
        logging.error(f"Error processing provision {provision}: {key_ex}", exc_info=True)
        return (provision, None)


def get_provisioning_results(
    workflow: WorkflowConfig,
    latency_data: LatencyData,
    power_data: Optional[PowerData] = None,
    policy: Policy = STREAMWISE_POLICY,
    provisions: Optional[list[Provision]] = None,
    verbose: bool = False,
    max_workers: Optional[int] = None,
    timeout: float = 10.0,  # 10 seconds max per provision
    short_list: bool = False,
) -> ProvisioningResult:
    """
    Get provisioning results for a list of GPU options.

    Args:
        max_workers: Maximum number of worker processes. None uses all available CPUs.
        timeout: Timeout in seconds for each provision task (default: 10.0).
    """
    times: list[float] = []
    costs: list[float] = []
    energies: list[float] = []
    ttffs: list[float] = []
    tbfs: list[float] = []

    actual_provision: list[dict[GPUType, int]] = []
    config_provision: list[dict[GPUType, int]] = []
    model_provision: list[dict[GPUType, dict[Model, list[ModelAllocation]]]] = []

    if provisions is None:
        provisions = get_provisions(
            policy.hardware,
            short_list=short_list)

    worker_func = partial(
        _process_provision,
        workflow=workflow,
        latency_data=latency_data,
        power_data=power_data,
        policy=policy,
        verbose=verbose,
    )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(worker_func, provision): provision
            for provision in provisions
        }
        with tqdm(total=len(provisions), desc=f"{policy.name} policy") as pbar:
            for future in as_completed(futures):
                provision = futures[future]
                try:
                    provision_result, result = future.result(timeout=timeout)
                    if result is None:
                        logging.error(f"Skipping provision {provision} due to errors.")
                    else:
                        times.append(result.total_time_s)
                        costs.append(result.cost)
                        energies.append(result.total_energy)
                        ttffs.append(result.ttff_s)
                        tbfs.append(result.tbf_s)
                        actual_provision.append(result.gpus_used.copy())
                        config_provision.append(provision_result.num_gpus.copy())
                        model_provision.append(result.models)
                except TimeoutError:
                    logging.warning(f"Provision {provision} timed out after {timeout} seconds.")
                finally:
                    pbar.update(1)

    # Sort results by the order of input provisions
    provision_to_index = {num_gpus_to_str(p.num_gpus): i for i, p in enumerate(provisions)}
    sorted_indices = sorted(
        range(len(config_provision)),
        key=lambda i: provision_to_index.get(num_gpus_to_str(config_provision[i]), float('inf'))
    )
    times = [times[i] for i in sorted_indices]
    costs = [costs[i] for i in sorted_indices]
    energies = [energies[i] for i in sorted_indices]
    ttffs = [ttffs[i] for i in sorted_indices]
    tbfs = [tbfs[i] for i in sorted_indices]
    actual_provision = [actual_provision[i] for i in sorted_indices]
    config_provision = [config_provision[i] for i in sorted_indices]
    model_provision = [model_provision[i] for i in sorted_indices]

    return ProvisioningResult(
        latencies=times,
        costs=costs,
        energies=energies,
        ttffs=ttffs,
        tbfs=tbfs,
        actual_provision=actual_provision,
        config_provision=config_provision,
        model_provision=model_provision,
    )


def get_provisioning_adaptive_results(
    workflow_config: WorkflowConfig,
    provisioning_qualities: dict[QualityLevel, ProvisioningResult],
    video_seconds: int = 10 * 60,
) -> ProvisioningResult:
    """
    Get provisioning results for adaptive quality policy.
    """
    assert len(provisioning_qualities) == 3

    num_provisions = len(provisioning_qualities[QualityLevel.HIGH].costs)

    assert num_provisions == len(provisioning_qualities[QualityLevel.HIGH].actual_provision), \
        f"High: {num_provisions} != {len(provisioning_qualities[QualityLevel.HIGH].actual_provision)}"
    assert num_provisions == len(provisioning_qualities[QualityLevel.MEDIUM].actual_provision), \
        f"Medium: {num_provisions} != {len(provisioning_qualities[QualityLevel.MEDIUM].actual_provision)}"
    assert num_provisions == len(provisioning_qualities[QualityLevel.LOW].actual_provision), \
        f"Low: {num_provisions} != {len(provisioning_qualities[QualityLevel.LOW].actual_provision)}"

    total_frames_ft = workflow_config.total_frames[Model.FT]

    # for each provisioning option, get the adaptive policy cost and TTFF
    costs: list[float] = []
    energies: list[float] = []
    latencies: list[float] = []
    ttffs: list[float] = []
    tbfs: list[float] = []
    qualities: list[float] = []

    actual_provision: list[dict[GPUType, int]] = []
    config_provision: list[dict[GPUType, int]] = []
    model_provision: list[dict[GPUType, dict[Model, list[ModelAllocation]]]] = []

    for idx in range(num_provisions):
        num_gpus = provisioning_qualities[QualityLevel.HIGH].config_provision[idx]
        # Initial check
        gpu_types = [
            gpu_type
            for gpu_type, count in num_gpus.items()
            if count > 0
        ]
        assert 0 < len(gpu_types), f"No GPUs provisioned: {num_gpus}."
        assert len(gpu_types) <= 2, f"Only support up to 2 GPU types in a provision: {num_gpus}."

        config_provision.append(num_gpus.copy())

        # check the actual provision
        gpus_high = provisioning_qualities[QualityLevel.HIGH].actual_provision[idx]
        gpus_medium = provisioning_qualities[QualityLevel.MEDIUM].actual_provision[idx]
        gpus_low = provisioning_qualities[QualityLevel.LOW].actual_provision[idx]

        # check if the TTFF of low is less than total time of video
        ttff_low = provisioning_qualities[QualityLevel.LOW].ttffs[idx]
        ttff_med = provisioning_qualities[QualityLevel.MEDIUM].ttffs[idx]
        if ttff_low > video_seconds:
            logging.warning(
                f"Cannot apply policy for {num_gpus_to_str(num_gpus)}. "
                f"TTFF low {ttff_low:.2f} > Video {video_seconds}.")
            raise ValueError(f"Low quality TTFF ({ttff_low:.2f}) exceeds video length ({video_seconds}).")

        # the portion of the low, medium, and high quality video
        portion_low = int(ttff_low / video_seconds * total_frames_ft)
        portion_medium = int(ttff_med / video_seconds * total_frames_ft)
        if portion_medium > total_frames_ft:
            portion_medium = total_frames_ft - portion_low
        portion_high = total_frames_ft - portion_low - portion_medium

        logging.debug(
            f"Adaptive policy for {num_gpus_to_str(num_gpus)}, "
            f"Portions Low:{portion_low} Medium:{portion_medium} High:{portion_high}")

        # Options to calculate the adaptive cost:
        # 1. Calculate the adaptive cost proportionally (not used)
        adaptive_cost = (
            portion_low / total_frames_ft * provisioning_qualities[QualityLevel.LOW].costs[idx]
            + portion_medium / total_frames_ft * provisioning_qualities[QualityLevel.MEDIUM].costs[idx]
            + portion_high / total_frames_ft * provisioning_qualities[QualityLevel.HIGH].costs[idx]
        )
        # 2. Sum of all costs (not used)
        adaptive_cost = (
            provisioning_qualities[QualityLevel.LOW].costs[idx]
            + provisioning_qualities[QualityLevel.MEDIUM].costs[idx]
            + provisioning_qualities[QualityLevel.HIGH].costs[idx]
        )
        # 3. Calculate the adaptive cost with the most expensive setting (not used)
        if portion_high > 0:
            adaptive_cost = provisioning_qualities[QualityLevel.HIGH].costs[idx]
        elif portion_medium > 0:
            adaptive_cost = provisioning_qualities[QualityLevel.MEDIUM].costs[idx]
        else:
            adaptive_cost = provisioning_qualities[QualityLevel.LOW].costs[idx]
        # 4. Highest cost
        adaptive_cost = provisioning_qualities[QualityLevel.HIGH].costs[idx]

        adaptive_energy = (
            portion_low / total_frames_ft * provisioning_qualities[QualityLevel.LOW].energies[idx]
            + portion_medium / total_frames_ft * provisioning_qualities[QualityLevel.MEDIUM].energies[idx]
            + portion_high / total_frames_ft * provisioning_qualities[QualityLevel.HIGH].energies[idx]
        )

        costs.append(adaptive_cost)
        energies.append(adaptive_energy)
        latencies.append(provisioning_qualities[QualityLevel.LOW].latencies[idx])
        ttffs.append(ttff_low)
        tbfs.append(provisioning_qualities[QualityLevel.LOW].tbfs[idx])
        weighted_quality = (
            portion_low / total_frames_ft * 1
            + portion_medium / total_frames_ft * 2
            + portion_high / total_frames_ft * 3
        )
        qualities.append(weighted_quality)

        actual_provision.append({
            gpu_type: max(
                gpus_high.get(gpu_type, 0),
                gpus_medium.get(gpu_type, 0),
                gpus_low.get(gpu_type, 0))
            for gpu_type in gpu_types
        })
        model_provision.append(provisioning_qualities[QualityLevel.HIGH].model_provision[idx])

    return ProvisioningResult(
        latencies=latencies,
        costs=costs,
        energies=energies,
        ttffs=ttffs,
        tbfs=tbfs,
        actual_provision=actual_provision,
        config_provision=config_provision,
        qualities=qualities,
        model_provision=model_provision,
    )
