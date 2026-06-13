"""Tests for shared StreamWise container configuration."""

from __future__ import annotations

from tests.test_utils import temp_sys_path

with temp_sys_path("streamwise", "simulator"):
    from container_config import CONTAINER_RESOURCES
    from container_config import get_minimum_service_container_specs


def test_minimum_service_specs_include_expected_gpu_defaults() -> None:
    """GPU defaults should preserve current minimal /api/service behavior."""
    specs = get_minimum_service_container_specs(max_gpus=8)

    for container_name in CONTAINER_RESOURCES:
        assert container_name in specs

    assert specs["gemma"].gpu == 2
    assert specs["kokoro"].gpu == "1g.10gb"
    assert specs["hunyuanframepackvae"].gpu == 1
    assert specs["podcasttranscript"].gpu == 0


def test_minimum_service_specs_reuse_container_resources_values() -> None:
    """CPU/memory/storage values should come from shared CONTAINER_RESOURCES."""
    specs = get_minimum_service_container_specs(max_gpus=2)
    gemma = specs["gemma"]

    assert gemma.cpu == CONTAINER_RESOURCES["gemma"].cpu
    assert gemma.memory_gib == CONTAINER_RESOURCES["gemma"].memory_gib
    assert gemma.ephemeral_storage_gib == CONTAINER_RESOURCES["gemma"].ephemeral_storage_gib


def test_minimum_service_specs_gpu_scaling_edge_cases() -> None:
    """Scalable services should clamp GPU defaults and treat negative max_gpus as zero."""
    specs_zero = get_minimum_service_container_specs(max_gpus=0)
    specs_one = get_minimum_service_container_specs(max_gpus=1)
    specs_negative = get_minimum_service_container_specs(max_gpus=-2)

    assert specs_zero["gemma"].gpu == 0
    assert specs_one["gemma"].gpu == 1
    assert specs_negative["gemma"].gpu == 0
    # MIG containers always keep their MIG profile regardless of max_gpus.
    assert specs_zero["yolo"].gpu == "1g.10gb"
    assert specs_zero["kokoro"].gpu == "1g.10gb"
    assert specs_zero["realesrgan"].gpu == "1g.10gb"
