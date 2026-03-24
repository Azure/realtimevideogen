#!/usr/bin/env python3

import sys
import pytest

from PIL import Image
from unittest.mock import patch
from unittest.mock import MagicMock

from typing import override
from typing import Dict
from typing import Any
from typing import Optional

sys.path.append("wrapper")

from wrapper_model import ModelGeneration


class MockModelGeneration(ModelGeneration):
    def __init__(
        self,
        output_type: str = "str",
    ) -> None:
        super().__init__("test_model")
        self.output_type = output_type

    async def get_rest_args(self, data_json: Dict[str, str]) -> Dict[str, Any]:
        if data_json is None or not isinstance(data_json, dict):
            raise ValueError("Missing JSON body")
        job_id = data_json.get("job_id", "default_job")
        ret = {
            "task": self.model_name,
            "args": {
                "job_id": job_id,
            }
        }
        return ret

    async def warmup(self) -> None:
        return await self.generate()

    @override
    async def generate(
        self,
        job_id: Optional[str] = None,
    ) -> Any:
        gen_timer = self._new_gen_timer(job_id)
        try:
            self._assert_model_init()
            if self.output_type == "str":
                return "reply"
            if self.output_type == "bytes":
                return b"reply"
            if self.output_type == "list_str":
                return ["reply1", "reply2"]
            if self.output_type == "list_bytes":
                return [b"reply1", b"reply2"]
            if self.output_type == "pillow":
                return Image.new('RGB', (64, 64), color="red")
            if self.output_type == "list_pillow":
                return [
                    Image.new('RGB', (64, 64), color="red"),
                    Image.new('RGB', (64, 64), color="blue")
                ]
            return "reply"
        finally:
            gen_timer.end()


def test_abstract_wrapper() -> None:
    with pytest.raises(TypeError):
        ModelGeneration()


@pytest.mark.asyncio
async def test_wrapper_model() -> None:
    model = MockModelGeneration()
    assert model is not None
    assert model.model_name == "test_model"

    model.init()
    assert model.status == "ok"

    health = model.get_health()
    assert health is not None
    timestamps = model.get_timestamps()
    assert timestamps is not None

    gen_args = await model.get_rest_args({})
    assert gen_args is not None

    with pytest.raises(ValueError, match="Missing JSON body"):
        await model.get_rest_args(None)  # type: ignore[arg-type]

    await model.warmup()

    resp = await model.generate()
    assert resp == "reply"

    resp = await model.generate("job0")
    assert resp == "reply"

    resp = await model.generate("job1")
    assert resp == "reply"

    # Data types
    model.output_type = "str"
    resp = await model.generate("job2")
    assert resp == "reply"

    model.output_type = "bytes"
    resp = await model.generate("job3")
    assert resp == b"reply"

    model.output_type = "list_str"
    resp = await model.generate("job4")
    assert resp == ["reply1", "reply2"]

    model.output_type = "pillow"
    resp = await model.generate("job5")
    assert resp.size == (64, 64)

    model.output_type = "list_pillow"
    resp = await model.generate("job6")
    assert len(resp) == 2
    assert resp[0].size == (64, 64)
    assert resp[1].size == (64, 64)

    model.output_type = "list_bytes"
    resp = await model.generate("job7")
    assert len(resp) == 2
    assert resp[0] == b"reply1"
    assert resp[1] == b"reply2"

    model.output_type = "unknown"
    resp = await model.generate("job8")
    assert resp == "reply"

    # Other methods
    health = model.get_health()
    assert health is not None
    timestamps = model.get_timestamps()
    assert timestamps is not None

    model.interrupt()

    model.get_gpu_info()


def test_get_gpu_info_mig() -> None:
    """Test that get_gpu_info handles MIG (Not Supported) errors gracefully."""
    import nvidia_smi
    import torch
    from nvidia_smi import NVMLError

    model = MockModelGeneration()
    model.init()

    mock_handle = MagicMock()
    mock_mem_info = MagicMock()
    mock_mem_info.used = 4 * 1024 ** 3   # 4 GiB
    mock_mem_info.total = 40 * 1024 ** 3  # 40 GiB

    # Simulate a MIG instance: memory info works, but utilization / power /
    # temperature / clock queries all raise NVMLError("Not Supported").
    with patch.object(nvidia_smi, 'nvmlInit'), \
         patch.object(nvidia_smi, 'nvmlShutdown'), \
         patch.object(nvidia_smi, 'nvmlDeviceGetCount', return_value=1), \
         patch.object(nvidia_smi, 'nvmlDeviceGetHandleByIndex', return_value=mock_handle), \
         patch.object(nvidia_smi, 'nvmlDeviceGetName', return_value="MIG 1g.5gb"), \
         patch.object(nvidia_smi, 'nvmlDeviceGetMemoryInfo', return_value=mock_mem_info), \
         patch.object(nvidia_smi, 'nvmlDeviceGetUtilizationRates', side_effect=NVMLError("Not Supported")), \
         patch.object(nvidia_smi, 'nvmlDeviceGetTemperature', side_effect=NVMLError("Not Supported")), \
         patch.object(nvidia_smi, 'nvmlDeviceGetPowerUsage', side_effect=NVMLError("Not Supported")), \
         patch.object(nvidia_smi, 'nvmlDeviceGetEnforcedPowerLimit', side_effect=NVMLError("Not Supported")), \
         patch.object(nvidia_smi, 'nvmlDeviceGetClockInfo', side_effect=NVMLError("Not Supported")), \
         patch.object(torch.cuda, 'current_device', return_value=0):

        gpu_info = model.get_gpu_info()

    assert gpu_info is not None, "get_gpu_info should return results even when some NVML calls are not supported"
    assert len(gpu_info) == 1

    g = gpu_info[0]
    assert g["name"] == "MIG 1g.5gb"
    assert g["mem_gib_used"] == pytest.approx(4.0)
    assert g["mem_gib_total"] == pytest.approx(40.0)

    # Fields not supported by MIG instances should be None (not raise an exception)
    assert g["sm_util"] is None
    assert g["mem_util"] is None
    assert g["temp"] is None
    assert g["power_draw_watts"] is None
    assert g["power_limit_watts"] is None
    assert g["graphics_clock"] is None
    assert g["sm_clock"] is None
    assert g["mem_clock"] is None

    # gpu_setup must remain True so future calls are not suppressed
    assert model.gpu_setup is True


@pytest.mark.asyncio
async def test_wrapper_health() -> None:
    model = MockModelGeneration()
    assert model is not None
    assert model.model_name == "test_model"

    health = model.get_health()
    assert health is not None
