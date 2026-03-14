#!/usr/bin/env python3

import sys
import pytest

from typing import Tuple

from unittest.mock import MagicMock
from unittest.mock import patch

from tests.torch_mock import TorchMock

mock_torch = TorchMock()

sys.path.append("wrapper")
sys.path.append("wrapper/flux")

with patch.dict(sys.modules, {
    'torch': mock_torch,
    'xfuser': MagicMock(),
    'xfuser.config': MagicMock(),
    'xfuser.core': MagicMock(),
    'xfuser.core.distributed': MagicMock(),
    'xfuser.model_executor': MagicMock(),
    'xfuser.model_executor.models': MagicMock(),
    'xfuser.model_executor.models.transformers.transformer_flux': MagicMock(),
    'xfuser.model_executor.layers': MagicMock(),
    'xfuser.model_executor.layers.attention_processor': MagicMock(),
    'diffusers': MagicMock(),
}):
    from flux.flux_xfuser import parallelize_transformer


class DummyTransformer:
    def __init__(self) -> None:
        self.transformer_blocks = []
        self.single_transformer_blocks = []
        self.forward = MagicMock(return_value=(mock_torch.randn(2, 4, 8), "extra"))

    def forward(self, *args, **kwargs) -> Tuple:
        return (mock_torch.randn(2, 4), "extra")


def test_parallelize_transformer() -> None:
    transformer = DummyTransformer()
    pipeline = MagicMock()  # DiffusionPipeline
    pipeline.transformer = transformer

    # Patch xfuser distributed utils so they return trivial values
    with (
        patch.dict(
            sys.modules, {
                "torch": mock_torch,
                "xfuser": MagicMock(),
                "xfuser.config": MagicMock(),
                "xfuser.core": MagicMock(),
                "xfuser.core.distributed": MagicMock(),
                "xfuser.model_executor": MagicMock(),
                "xfuser.model_executor.models": MagicMock(),
                "xfuser.model_executor.models.transformers.transformer_flux": MagicMock(),
                "xfuser.model_executor.layers": MagicMock(),
                "xfuser.model_executor.layers.attention_processor": MagicMock(),
                "diffusers": MagicMock(),
            }
        ),
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
        # assert parallel_pipeline.transformer.forward is not transformer.forward

        with pytest.raises(ValueError):
            result = parallel_pipeline.transformer.forward(
                hidden,
                enc,
                img_ids=img_ids,
                txt_ids=txt_ids,
                timestep=timestep)

            transformer.forward.assert_called()
            assert isinstance(result, tuple)
            assert result[0].shape == (1, 4, 8)
            assert result[1] == "extra"
