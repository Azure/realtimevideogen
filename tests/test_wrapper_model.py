#!/usr/bin/env python3

import sys
import pytest

from PIL import Image

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
    async def generate(  # type: ignore[override]
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
        ModelGeneration()  # type: ignore[abstract]


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


@pytest.mark.asyncio
async def test_wrapper_health() -> None:
    model = MockModelGeneration()
    assert model is not None
    assert model.model_name == "test_model"

    health = model.get_health()
    assert health is not None
