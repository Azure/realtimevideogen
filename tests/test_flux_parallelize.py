#!/usr/bin/env python3

import sys

from typing import Tuple

from unittest.mock import MagicMock
from unittest.mock import patch

from tests.torch_mock import TorchMock
from tests.diffusers_mock import DiffusersMock

mock_torch = TorchMock()
mock_diffusers = DiffusersMock()

sys.path.append("wrapper")
sys.path.append("wrapper/flux")

mock_modules = {
    'xfuser': MagicMock(),
    'xfuser.config': MagicMock(),
    'xfuser.core': MagicMock(),
    'xfuser.core.distributed': MagicMock(),
    'xfuser.model_executor': MagicMock(),
    'xfuser.model_executor.models': MagicMock(),
    'xfuser.model_executor.models.transformers.transformer_flux': MagicMock(),
    'xfuser.model_executor.layers': MagicMock(),
    'xfuser.model_executor.layers.attention_processor': MagicMock(),
}
mock_modules.update(mock_torch.get_sub_modules())
mock_modules.update(mock_diffusers.get_sub_modules())

with patch.dict(sys.modules, mock_modules):
    from flux.flux_xfuser import parallelize_transformer
    import flux.flux_xfuser as _flux_xfuser_mod

# Re-register the module so patch() targets the same object as
# parallelize_transformer.__globals__ (patch.dict restores sys.modules
# on exit, removing the module entry that was added during import).
sys.modules['flux.flux_xfuser'] = _flux_xfuser_mod


class DummyTransformer:
    def __init__(self) -> None:
        self.transformer_blocks: list[MagicMock] = []
        self.single_transformer_blocks: list[MagicMock] = []

    def forward(self, *args: object, **kwargs: object) -> Tuple:
        return (mock_torch.randn(2, 4, 8), "extra")


def test_parallelize_transformer() -> None:
    transformer = DummyTransformer()
    pipeline = MagicMock()  # DiffusionPipeline
    pipeline.transformer = transformer

    # Patch xfuser distributed utils so they return trivial values
    with (
        patch("flux.flux_xfuser.get_classifier_free_guidance_world_size", return_value=1),
        patch("flux.flux_xfuser.get_classifier_free_guidance_rank", return_value=0),
        patch("flux.flux_xfuser.get_sequence_parallel_world_size", return_value=1),
        patch("flux.flux_xfuser.get_sequence_parallel_rank", return_value=0),
        patch("flux.flux_xfuser.get_runtime_state") as mock_runtime,
        patch("flux.flux_xfuser.get_sp_group") as mock_sp_group,
        patch("flux.flux_xfuser.get_cfg_group") as mock_cfg_group
    ):

        runtime_state = MagicMock()
        runtime_state.split_text_embed_in_sp = True
        mock_runtime.return_value = runtime_state

        mock_sp_group.return_value.all_gather = lambda x, dim: x
        mock_cfg_group.return_value.all_gather = lambda x, dim: x

        hidden = mock_torch.randn(2, 4, 8)
        enc = mock_torch.randn(2, 4, 8)
        img_ids = mock_torch.ones(2, 4, 1)
        txt_ids = mock_torch.ones(2, 4, 1)
        timestep = mock_torch.tensor(1)

        parallel_pipeline = parallelize_transformer(pipeline)

        result = parallel_pipeline.transformer.forward(
            hidden,
            enc,
            img_ids=img_ids,
            txt_ids=txt_ids,
            timestep=timestep)

        assert isinstance(result, tuple)
