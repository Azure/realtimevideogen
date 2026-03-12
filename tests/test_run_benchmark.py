#!/usr/bin/env python3

import sys
import pytest

from unittest.mock import patch
from unittest.mock import MagicMock

from tests.torch_mock import TorchMock

mock_torch = TorchMock()

sys.path.append("wrapper")
sys.path.append("wrapper/flux")
sys.path.append("wrapper/wan")

with patch.dict(sys.modules, {
    'torch': mock_torch,
    'torch.profiler': MagicMock(),
    'torch.hub': MagicMock(),
    'torch.version': MagicMock(),
    'torchvision': MagicMock(),
    'torch.distributed': MagicMock(),
    'torch.amp': MagicMock(),
    'torchvision.transforms': MagicMock(),
    'torchvision.transforms.functional': MagicMock(),
    'xfuser': MagicMock(),
    'xfuser.config': MagicMock(),
    'xfuser.core': MagicMock(),
    'xfuser.core.distributed': MagicMock(),
    'xfuser.model_executor': MagicMock(),
    'xfuser.model_executor.layers': MagicMock(),
    'xfuser.model_executor.layers.attention_processor': MagicMock(),
    'diffusers': MagicMock(),
    'wan.modules': MagicMock(),
    'wan.modules.t5': MagicMock(),
    'wan.modules.clip': MagicMock(),
    'wan.modules.vae': MagicMock(),
    'wan.modules.model': MagicMock(),
    'wan.utils': MagicMock(),
    'wan.utils.utils': MagicMock(),
    'wan.utils.fm_solvers_unipc': MagicMock(),
    'wan.distributed': MagicMock(),
    'wan.distributed.fsdp': MagicMock(),
    'wan.distributed.xdit_context_parallel': MagicMock(),
}):
    from flux.run_flux_benchmark import main as run_flux_benchmark_main
    from wan.run_wan_benchmark import main as run_wan_benchmark_main
    from wan.run_wan_benchmark_batching import main as run_wan_benchmark_batching_main
    # import wan.run_wan_vae_decoder_benchmark


def test_run_flux_benchmark() -> None:
    with pytest.raises(ValueError):  # engine_config, input_config = engine_args.create_config()
        run_flux_benchmark_main()


def test_run_wan_benchmark() -> None:
    with pytest.raises(FileNotFoundError):  # No such file or directory: 'generated_image.png'
        run_wan_benchmark_main()


def test_run_wan_benchmark_batching_main() -> None:
    with pytest.raises(FileNotFoundError):  # No such file or directory: 'generated_image.png'
        run_wan_benchmark_batching_main()
