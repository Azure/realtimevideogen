from unittest.mock import MagicMock

import importlib.util

from typing import Dict
from typing import Any
from typing import Tuple


class DiffusersMock(MagicMock):
    def __init__(
        self,
        *args: Tuple,
        **kwargs: Dict
    ) -> None:
        super().__init__(*args, **kwargs)

        self.__spec__ = importlib.util.spec_from_loader("diffusers", loader=None)

    def get_sub_modules(self) -> Dict[str, Any]:
        return {
            "diffusers": self,
            "diffusers.models": MagicMock(),
            "diffusers.models.autoencoders": MagicMock(),
            "diffusers.models.autoencoders.autoencoder_kl": MagicMock(),
            "diffusers.models.activations": MagicMock(),
            "diffusers.models.attention_processor": MagicMock(),
            "diffusers.models.unets": MagicMock(),
            "diffusers.models.unets.unet_2d_blocks": MagicMock(),
            "diffusers.models.normalization": MagicMock(),
            "diffusers.models.transformers": MagicMock(),
            "diffusers.models.transformers.dual_transformer_2d": MagicMock(),
            "diffusers.models.transformers.transformer_2d": MagicMock(),
            "diffusers.models.downsampling": MagicMock(),
            "diffusers.models.upsampling": MagicMock(),
            "diffusers.models.resnet": MagicMock(),
            "diffusers.pipelines": MagicMock(),
            "diffusers.pipelines.pipeline_utils": MagicMock(),
            "diffusers.utils": MagicMock(),
            "diffusers.utils.torch_utils": MagicMock(),
        }
