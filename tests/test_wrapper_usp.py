#!/usr/bin/env python3

import sys
import gc
import pytest

from typing import override
from typing import Dict
from typing import Optional
from typing import Union
from typing import Any

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.torch_mock import TorchMock

mock_torch = TorchMock()

mock_modules = {
    'nvidia_smi': MagicMock(),
    'torch': mock_torch,
    'xfuser': MagicMock(),
    'xfuser.config': MagicMock(),
    'xfuser.core': MagicMock(),
    'xfuser.core.distributed': MagicMock(),
}
mock_modules.update(mock_torch.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from wrapper_usp import USPGeneration


class MockUSPGeneration(USPGeneration):
    def __init__(self) -> None:
        super().__init__("test_usp")

    async def get_rest_args(
        self,
        data_json: Dict[str, Union[str, int, float]]
    ) -> Dict[str, Any]:
        return data_json

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
            return await super().generate()
        finally:
            gen_timer.end()


@pytest.mark.asyncio
async def test_wrapper_usp() -> None:
    model = MockUSPGeneration()
    assert model is not None
    assert model.model_name == "test_usp"

    model.init()
    assert model.status == "ok"

    health = model.get_health()
    assert health is not None
    timestamps = model.get_timestamps()
    assert timestamps is not None

    await model.get_rest_args({})

    with pytest.raises(NotImplementedError):
        await model.warmup()
    with pytest.raises(NotImplementedError):
        await model.generate()
    with pytest.raises(NotImplementedError):
        await model.generate("job0")

    health = model.get_health()
    assert health is not None

    del model
    gc.collect()
